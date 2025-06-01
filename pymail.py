#!/usr/bin/env python3

import os
import smtplib
import zipfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def send_mail(text, folder_path=None):
  if folder_path:
    zip_path = "/tmp/email.zip"
    zip_folder(folder_path, zip_path)
  message = MIMEMultipart("alternative")
  message["Subject"] = "Pipes Inventory Report"
  message["From"] = "robotmon3@gmail.com"
  message["To"] = "zarredf@hiltonmfg.com.au"
  password = "roqglsvdkmdqlajc"
  # Create the plain-text and HTML version of your message
  html = f"""\
  <html>
    <body>
      <p>{text}
      </p>
    </body>
  </html>
  """
  message.attach(MIMEText(text, "plain"))
  message.attach(MIMEText(html, "html"))
  if folder_path:
    with open(zip_path, 'rb') as f:
        message.attach(MIMEApplication(f.read(), Name=os.path.basename(zip_path)))
  # Send email
  with smtplib.SMTP('smtp.gmail.com', 587) as server:
      server.starttls()  # Secure the connection
      server.login(message["From"], password)
      server.sendmail(message["From"], message["To"], message.as_string())
