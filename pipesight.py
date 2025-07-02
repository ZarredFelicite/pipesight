#!/usr/bin/env python3

import os
import time
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2
#import csv
import statistics
import numpy as np
import hdbscan  # switched from DBSCAN to HDBSCAN for better hierarchical clustering
import argparse
#import xlsxwriter
from pymail import send_mail
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import onnxruntime
import bisect
from pprint import pprint
import requests
from requests import Response
import json

# ------------------------------------------------------------
# Conditional verbose logger (available early in module)
# ------------------------------------------------------------

def vprint(*_args, **_kwargs):
    """Print only when -v/--verbose flag is supplied."""
    if globals().get("args", None) is not None and getattr(args, "verbose", False):
        print(*_args, **_kwargs)

def draw_label(draw, box, text, color):
  size = font.getlength(text)
  box_width = size + 2
  box_height = 10
  box_position = (box[0], box[1] - box_height)
  draw.rectangle(box, outline=colors[color], width=1)
  draw.rectangle([box_position, (box_position[0] + box_width, box_position[1] + box_height)], fill=colors[color])
  draw.text((box_position[0], box_position[1]-4), text, fill=(0,0,0), font=font)

def contains(object, region):
  if (object[0] > region[0]) and (object[1] > region[1]) and (object[2] < region[2]) and (object[3] < region[3]):
    return True
  else:
    return False

def center_contains(object, region):
  if (object[0] > region[0]) and (object[1] > region[1]) and (object[0] < region[2]) and (object[1] < region[3]):
    return True
  else:
    return False

def filter_det(result, classes, conf):
  indices = {v: k for k, v in result.names.items()}
  return [obj for obj in result.boxes if obj.cls[0] in [indices[c] for c in classes] and obj.conf > conf]

def convert_bbox_wh(bbox):
  return [ int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2), int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2) ]

def size_filter(bbox):
  if (bbox[2][2]/bbox[2][3] < 0.6) or (bbox[2][2]/bbox[2][3] > 1.4):
    return False
  else:
    return True

def hough_circles(draw, img):
  cvimg = np.array(img.convert("L"))
  #cvimg = cv2.convertScaleAbs(cvimg, alpha=3.0, beta=0)
  cvimg = cv2.GaussianBlur(cvimg, (5,5), cv2.BORDER_DEFAULT)
  cvimg = cv2.equalizeHist(cvimg)
  #cvimg = cv2.equalizeHist(cvimg)
  #canny = cv2.Canny(cvimg, 70, 100, apertureSize=3)
  #cvimg = cv2.adaptiveThreshold(cvimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21n, 2)
  ret, cvimg = cv2.threshold(cvimg, 0, 200, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #cvimg = cv2.GaussianBlur(cvimg, (5,5), cv2.BORDER_DEFAULT)
  #cvimg = cv2.Canny(cvimg, 30, 50, apertureSize=3)
  cv2.imwrite('pre_hough.jpg', cvimg)
  circles = cv2.HoughCircles(cvimg, cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=60, param2=40, minRadius=3, maxRadius=70)
  circles = np.uint16(np.around(circles)).tolist()
  for circle in circles[0]:
    center_x, center_y, radius = circle
    top_left = (center_x - radius, center_y - radius)
    bottom_right = (center_x + radius, center_y + radius)
    draw.ellipse([top_left, bottom_right], outline="red", width=2)
    print(circle)
  return draw

def hough_circle(img, detection):
  crop = np.array(img.crop(convert_bbox_wh(detection[2])).convert("L"))
  #crop = cv2.medianBlur(crop, 5)
  #hist = cv2.equalizeHist(crop)
  crop = cv2.GaussianBlur(crop, (5,5), cv2.BORDER_DEFAULT)
  circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1, 5, param1= 50, param2=30, minRadius=1, maxRadius=40)
  if circles is not None:
    radius = np.uint16(np.around(circles))[0][0]
    print(radius)
    radius = radius[2]
    #if int(radius[2]) == 23:
    #  print(radius[2])
    #  cv2.imwrite('raw_crop.jpg', crop)
    #  cv2.circle(crop, (radius[0],radius[1]), radius[2], (255,255,255), 1)
    #  cv2.imwrite('crop.jpg', crop)
    #  quit()
  else:
    radius = 0
  return radius


def postprocess(output, conf, iou):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= conf:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    boxes = list(zip(class_ids, scores, boxes))
    boxes = [boxes[i] for i in indices if size_filter(boxes[i])]
    return boxes

def preprocess(img, imgsz=640):
  width, height = img.size
  img = img.crop((int(width*0.38), int(height*0.5), int(width-width*0.5), int(height-height*0.22)))
  img.thumbnail((imgsz,imgsz))
  bg = Image.new('RGB', (imgsz, imgsz), (0, 0, 0))
  offset = ((imgsz - img.width) // 2, (imgsz - img.height) // 2)
  bg.paste(img, offset)
  img = bg
  enhancer = ImageEnhance.Contrast(img)
  img = enhancer.enhance(3.0)
  if args.verbose:
    img.save('img_input.jpg')
  img_input = (np.array(img).transpose(2, 0, 1) / 255.0).astype('float32').reshape(1, 3, 640, 640)
  return img_input, img

def excel(results):
  workbook = xlsxwriter.Workbook('output/inventory.xlsx')
  worksheet = workbook.add_worksheet()
  number_format = workbook.add_format({'num_format': '0'})
  title_format = workbook.add_format({'bold': True, 'font_color': 'blue'})
  worksheet.write("A1", "Part", title_format)
  worksheet.write("B1", "Min", title_format)
  worksheet.write("C1", "Status", title_format)
  worksheet.write("D1", "Count", title_format)
  worksheet.write("E1", "Images", title_format)
  worksheet.write("F1", "Count", title_format)
  worksheet.write("G1", "Images", title_format)
  with open('parts.csv', 'r') as file:
    csvFile = list(csv.reader(file))
    parts = {part[0]: part[1] for part in csvFile}

  print("Stock Levels")
  #data = dict(sorted(data.items(), key=lambda item: item[1]))
  for i,(k,v) in enumerate(results.items()):
    if 0 in v["count"]:
      status = "NO STOCK"
    elif any([count < int(parts[k]) for count in v["count"]]):
      status = "LOW STOCK"
    else:
      status = "OK"
    chars = ['D','E','F','G','H','I']
    worksheet.write(f"A{i+2}", k)
    worksheet.write(f"B{i+2}", parts[k], number_format)
    worksheet.write(f"C{i+2}", status)
    list1 = [0,2,4]
    list2 = [1,3,5]
    for j in range(max([len(list(v["regions"])),len(list(v['count']))])):
      print(k)
      if k == "HM081":
        print(v)
      worksheet.write(f"{chars[list1[j]]}{i+2}", str(v['count'][j]) if j<len(list(v['count'])) else '', number_format)
      if len(list(v["regions"])) > j:
        worksheet.write(f"{chars[list2[j]]}{i+2}", '=HYPERLINK("' + list(v["regions"])[j].replace('output/','') + '")')
  workbook.close()

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if os.path.exists(UPLOAD_FOLDER):
  shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
font = ImageFont.load_default(size=14)

@app.route('/upload', methods=['POST'])
def upload_file():
  if 'file' not in request.files:
    return jsonify({'message': 'No file part'}), 400
  file = request.files['file']
  if file.filename == '':
    return jsonify({'message': 'No selected file'}), 400
  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    result = process([filepath])
    #if app.config["SAVE"]:
    #  draw_result(res, filepath)
    if result:
      return jsonify({'message': result}), 200
    else:
      return jsonify({'message': "Failed"}), 400
  else:
    return jsonify({'message': 'Invalid file type'}), 400

def process(files):
  imgs = [Image.open(img) for img in files]
  if os.path.exists("output"):
    shutil.rmtree("output")
  os.mkdir("./output")
  #os.mkdir("./output/regions")

  session = onnxruntime.InferenceSession('./models/last.onnx', None)
  input_name = session.get_inputs()[0].name
  for img_indx, img in enumerate(imgs):
    img_input, img = preprocess(img)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    start = time.time()
    raw_result = session.run([], {input_name: img_input})
    #draw = hough_circles(draw, img_draw)
    if args.profile:
      print('Inference time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    start = time.time()
    results = postprocess(raw_result, args.conf, args.iou)
    vprint(f"Detections after NMS: {len(results)} | confidences: {[round(r[1], 2) for r in results]}")
    if args.profile:
      print('NMS time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    # data drift detection
    #mean_x = sum([detection[2][0] for detection in results])/len(results)
    #mean_y = sum([detection[2][1] for detection in results])/len(results)
    #print(mean_x)
    #print(mean_y)
    # Apply DBSCAN
    start = time.time()

    #radii = [[hough_circle(img, detection)] for detection in results]
    # ------------------------------------------------------------
    # CLUSTER DETECTIONS WITH HDBSCAN
    # ------------------------------------------------------------
    # Build feature vector:  [bbox_width , bbox_height * h_weight , y_position * y_weight]
    # Width remains the dominant factor; height and y-position are given smaller weights.
    pipe_widths = [int(det[2][2]) for det in results]
    pipe_heights = [int(det[2][3]) for det in results]
    y_weight = 0.1  # weight for y-coordinate influence (moderate)
    x_weight = 0 # weight for x-coordinate influence (smaller than y)
    h_weight = 1  # weight for height influence (smaller than width)
    w_weight = 1  # weight for height influence (smaller than width)
    features = np.array([[pipe_widths[i] * w_weight,
                         pipe_heights[i] * h_weight,
                         int(det[2][1]) * y_weight,
                         int(det[2][0]) * x_weight]
                        for i, det in enumerate(results)], dtype=float)

    # Perform clustering – HDBSCAN will return -1 for noise points
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
    groups_size = clusterer.fit_predict(features)
    unique_labels, counts_labels = np.unique(groups_size, return_counts=True)
    vprint("HDBSCAN label distribution:", dict(zip(unique_labels.tolist(), counts_labels.tolist())))

    # Median width of each cluster → maps cluster label → canonical width
    cluster_medians = {}
    for idx_det, label in enumerate(groups_size):
        # Ignore noise points (label == -1) when computing cluster medians
        if label == -1:
            continue
        cluster_medians.setdefault(label, []).append(pipe_widths[idx_det])
    for label in cluster_medians:
        cluster_medians[label] = statistics.median(cluster_medians[label])
    vprint("Cluster median widths:", cluster_medians)

    if args.profile:
      print('HDBSCAN time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    with open('sizes.txt', 'r') as f:
      parts = f.readlines()
    parts = [ part.replace('\n','').split(',') for part in parts ]
    catalog_sizes = [float(part[1]) for part in parts]
    data = { part : { 'count': int(offset), 'box': [1000,1000,0,0], 'size': float(size) } for part, size, offset in parts }
    data['undefined'] = {'count': 0, 'box': [1000,1000,0,0], 'size': 0}

    if args.save_raw:
        # Draw all bounding boxes after NMS but before size filtering
        final_confidence_scores = []
        for result in results:
            xywh = result[2]
            coord = convert_bbox_wh(xywh)
            draw.rectangle(coord, outline=(0, 255, 0), width=1) # Draw in green for raw
            final_confidence_scores.append(result[1])  # Collect confidence score
    else:
        # Apply size filtering and draw processed bounding boxes and labels
        final_confidence_scores = []
        for det_idx, (result, group_size) in enumerate(zip(results, groups_size)):
            result = result + (convert_bbox_wh(result[2]),)

            # ------------------------------------------------------------
            # SHORT-CIRCUIT: noise detections remain undefined
            # ------------------------------------------------------------
            if group_size == -1:
                idx = len(parts)  # maps to 'undefined'
            # ------------------------------------------------------------
            # Low-quality / aspect-ratio fallback path
            # ------------------------------------------------------------
            elif abs(1 - result[2][2]/result[2][3]) > 0.05 or result[1] < 0.3:
                closest_score = float('inf')
                closest_group = None
                current_area = result[2][2] * result[2][3]
                for close_result, close_group_size in zip(results, groups_size):
                    if close_result[1] > 0.5 and (close_result[2][0]-result[2][0])**2 + (close_result[2][1]-result[2][1])**2 != 0:
                        center_dist = np.sqrt((close_result[2][0]-result[2][0])**2 + (close_result[2][1]-result[2][1])**2)
                        close_area = close_result[2][2] * close_result[2][3]
                        size_diff = abs(current_area - close_area)
                        # Normalize and combine (simple approach: sum of normalized values)
                        # Need to consider how to normalize appropriately based on expected ranges
                        # Weight size similarity based on aspect ratio (more weight for square boxes)
                        aspect_ratio_weight = 1 - abs(1 - result[2][2]/result[2][3])
                        combined_score = center_dist + (size_diff * aspect_ratio_weight)*0.5
                        if combined_score < closest_score:
                            closest_score = combined_score
                            closest_group = close_group_size

                if closest_group is not None and closest_group in cluster_medians:
                    idx = bisect.bisect_left(catalog_sizes, cluster_medians[closest_group])
                else:
                    # Fallback to undefined
                    idx = len(parts)  # undefined
            else:
                # Confident detection inside a valid cluster
                idx = bisect.bisect_left(catalog_sizes, cluster_medians.get(group_size, pipe_widths[det_idx]))

            # ------------------------------------------------------------
            # Update inventory counts (including undefined)
            # ------------------------------------------------------------
            part_key = parts[idx][0] if idx < len(parts) else 'undefined'
            data[part_key]['count'] += 1

            if part_key != 'undefined':
                # Update bounding box for visualisation
                if result[3][0] < data[part_key]['box'][0]:
                    data[part_key]['box'][0] = int(result[3][0])
                if result[3][1] < data[part_key]['box'][1]:
                    data[part_key]['box'][1] = int(result[3][1])
                if result[3][2] > data[part_key]['box'][2]:
                    data[part_key]['box'][2] = int(result[3][2])
                if result[3][3] > data[part_key]['box'][3]:
                    data[part_key]['box'][3] = int(result[3][3])

                xywh = result[2]
                coord = convert_bbox_wh(xywh)
                #if center_contains(xy, results[part]["box"]):
                if args.verbose:
                    draw_label(draw, coord, str(round(result[1], 2)), idx)
                else:
                    draw.rectangle(coord, outline=colors[idx], width=3)
                #draw_label(draw, coord, str(int(result[2][2])*int(result[2][3])), group)
                #data[part]["count"] += 1
                #draw.rectangle(results[part]["box"], outline=(250,0,0), width=6)
                final_confidence_scores.append(result[1])  # Collect confidence score
        for part in data.keys():
            if data[part]['box'][0] != 1000:
                draw_label(draw, [data[part]['box'][0]-10, data[part]['box'][1]-10, data[part]['box'][2]+10, data[part]['box'][3]+10], part, -1)

    # Calculate and display average confidence score
    if final_confidence_scores:
        avg_confidence = sum(final_confidence_scores) / len(final_confidence_scores)
        confidence_text = f"Avg Confidence: {avg_confidence:.3f}"
        
        # Create a background rectangle for the text
        text_bbox = draw.textbbox((10, 10), confidence_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background rectangle
        draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill=(0, 0, 0, 128))
        
        # Draw the text
        draw.text((10, 10), confidence_text, fill=(255, 255, 255), font=font)
    else:
        # No detections found
        no_detection_text = "No detections found"
        text_bbox = draw.textbbox((10, 10), no_detection_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background rectangle
        draw.rectangle([(5, 5), (15 + text_width, 15 + text_height)], fill=(0, 0, 0, 128))
        
        # Draw the text
        draw.text((10, 10), no_detection_text, fill=(255, 255, 255), font=font)

    img_draw.save(os.path.join("output", f"result-{img_indx}.jpg"), quality=100)

  temp_response_text = ""
  if args.mail:
    with open("./output/result-0.jpg", "rb") as f:
      files_upload = { "file": f}
      temp_response = requests.post("https://temp.sh/upload", files=files_upload)
    temp_response_text = temp_response.text
    print(temp_response_text)

  #excel(results)
  email = "Pipe Inventory Report:<br><br>"
  for part in sorted(list(data.keys())):
    #if part == 'ALP101616':
    #  data[part]['count'] -= 1
    email += f"{part}: {data[part]['count']}<br>"
  email += f"<br>preview: {temp_response_text}<br>"
  if args.mail: # Only send mail if args.mail is true
    send_mail(email)
  print(email)
  # Create a new dictionary with only part numbers and counts
  part_counts = {part: data[part]['count'] for part in data.keys()}

  # Write the part counts to a JSON file in the output directory
  output_file_path = os.path.join("output", "results.json")
  with open(output_file_path, 'w') as json_file:
    json.dump(part_counts, json_file, indent=4)

  # ------------------------------------------------------------
  # Optional: visualize clusters only
  # ------------------------------------------------------------
  if args.clusters:
      img_clusters = img.copy()
      draw_clusters = ImageDraw.Draw(img_clusters)
      for det_idx, (result, cluster_label) in enumerate(zip(results, groups_size)):
          coord = convert_bbox_wh(result[2])
          # White for noise, otherwise cycle through predefined colors
          outline_col = (255, 255, 255) if cluster_label == -1 else colors[cluster_label % len(colors)]
          draw_clusters.rectangle(coord, outline=outline_col, width=3)
          if args.verbose:
              # Label with cluster id for easier debugging
              draw_label(draw_clusters, coord, str(cluster_label), cluster_label % len(colors) if cluster_label != -1 else 0)
      clusters_path = os.path.join("output", f"clusters-{img_indx}.jpg")
      img_clusters.save(clusters_path, quality=100)
      vprint(f"Clusters image saved to {clusters_path}")

  return output_file_path

def run_tests():
    test_images_dir = "test/images"
    true_results_dir = "test/results"
    output_dir = "output"

    # Clean up previous output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    total_pipes = 0
    correct_classifications = 0

    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        # Assuming the JSON file has the same name as the image file but with a .json extension
        true_result_file = os.path.splitext(image_file)[0] + ".json"
        true_result_path = os.path.join(true_results_dir, true_result_file)

        if not os.path.exists(true_result_path):
            print(f"Warning: True result file not found for {image_file}. Skipping.")
            continue

        # Process the image using the process function
        try:
            # The process function now writes to output/results.json and returns the path
            generated_result_path = process([image_path])
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

        # Read generated and true results
        if not os.path.exists(generated_result_path):
            print(f"Error: Generated result file not found for {image_file}. Skipping comparison.")
            continue

        with open(generated_result_path, 'r') as f:
            generated_results = json.load(f)

        with open(true_result_path, 'r') as f:
            true_results = json.load(f)

        # Compare results
        # Assuming the keys (part numbers) are the same in both dictionaries
        for part, true_count in true_results.items():
            generated_count = generated_results.get(part, 0)
            total_pipes += true_count
            # For simplicity, considering a classification correct if the count for a part matches
            # A more sophisticated comparison might be needed depending on requirements
            if generated_count == true_count:
                correct_classifications += true_count
            # else:
                # If counts don't match, we need to figure out how many were misclassified.
                # This simple approach counts all pipes for that part as incorrect if the total count doesn't match.
                # A more accurate method would involve comparing individual pipe classifications if available.
                # pass # Simple count comparison is sufficient for this basic test

        # Clean up the generated results.json for the current image
        if os.path.exists(generated_result_path):
             os.remove(generated_result_path)


    # Calculate accuracy
    accuracy = (correct_classifications / total_pipes) if total_pipes > 0 else 0

    print(f"\nOverall Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--imgs', nargs='+')
  parser.add_argument('-s', '--serve', action='store_true')
  parser.add_argument('-w', '--write', action='store_true')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('-r', '--save_raw', action='store_true')
  parser.add_argument('-p', '--profile', action='store_true')
  parser.add_argument('-m', '--mail', action='store_true')
  parser.add_argument('-d', '--device', default="cuda:0")
  parser.add_argument('-c', '--conf', type=float, default=0.5)
  parser.add_argument('-u', '--iou', type=float, default=0.5)
  parser.add_argument('--test', action='store_true', help='Run tests on images in test/images')
  parser.add_argument('--clusters', action='store_true', help='Save an additional image showing only HDBSCAN clusters')
  args = parser.parse_args()

  colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 0, 255), # Magenta
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 255, 0), # Yellow
  ]
  if args.serve:
    app.config["SAVE"] = args.write
    app.run(debug=False, host='0.0.0.0', port=5001)
  elif args.test:
      run_tests()
  elif args.imgs:
    result = process(args.imgs)
    print(result)
