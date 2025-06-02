#!/usr/bin/env python3

import os
import smtplib
import zipfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

load_dotenv()

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def send_mail(text, folder_path=None):
  sender_email = os.environ.get("SENDER_EMAIL")
  recipient_email = os.environ.get("RECIPIENT_EMAIL")
  email_password = os.environ.get("EMAIL_PASSWORD")

  if not sender_email or not recipient_email or not email_password:
      print("Error: Email credentials not set in environment variables.")
      return # Or raise an exception

  if folder_path:
    zip_path = "/tmp/email.zip"
    zip_folder(folder_path, zip_path)
  message = MIMEMultipart("alternative")
  message["Subject"] = "Pipes Inventory Report"
  message["From"] = sender_email
  message["To"] = recipient_email
  password = email_password
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
      server.login(sender_email, email_password)
      server.sendmail(sender_email, recipient_email, message.as_string())
