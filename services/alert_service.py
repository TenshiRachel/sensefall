import os
import smtplib
import threading
import time

from email.message import EmailMessage
from datetime import datetime

EMAIL_USER = os.getenv("SMARTFALL_EMAIL_USER")
EMAIL_PASS = os.getenv("SMARTFALL_EMAIL_PASS")
EMAIL_TO = os.getenv("SMARTFALL_EMAIL_TO")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

_last_sent_time = 0


def send_fall_alert(confidence=0.0, trigger_source="unknown", extra_notes=""):
	def task():
		global _last_sent_time

		now = time.time()
			
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		msg = EmailMessage()
		msg["Subject"] = "SmartFall Alert: Fall Detected"
		msg["From"] = EMAIL_USER
		msg["To"] = EMAIL_TO
		
		body = f"""FALL DETECTED
	
			Time: {timestamp}
			Confidence: {confidence:.2f}
			
			Trigger: {trigger_source}
			
			 
			Please check immediately.
			
			""".strip()

		msg.set_content(body)
		
		try:
			with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
				smtp.login(EMAIL_USER, EMAIL_PASS)
				smtp.send_message(msg)
			
			_last_sent_time = now
			print("Alert email sent.")
		
		except Exception as e:
			print("Email failed:", e)
		
	threading.Thread(target=task).start()
		

