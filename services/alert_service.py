import os
import smtplib
import threading
import time

from dotenv import load_dotenv
load_dotenv()

from email.message import EmailMessage
from datetime import datetime

EMAIL_USER = os.getenv("SMARTFALL_EMAIL_USER")
EMAIL_PASS = os.getenv("SMARTFALL_EMAIL_PASS")
EMAIL_TO = os.getenv("SMARTFALL_EMAIL_TO")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

_last_sent_time = 0

def format_sensor_scores(sensor_scores):
    if not sensor_scores or not isinstance(sensor_scores, dict):
        return "N/A"

    parts = []
    for name, value in sensor_scores.items():
        if value is None:
            continue
        try:
            parts.append(f"{name}={float(value):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{name}={value}")

    return ", ".join(parts) if parts else "N/A"

def send_fall_alert(confidence=0.0, trigger_source="unknown", event_time=None, sensor="unknown", sensor_scores=None, device_id=None, location="home",extra_notes=""):
	def task():
		global _last_sent_time

		now = time.time()
			
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		device_label = device_id or os.getenv("CLOUD_DEVICE_ID", "home_pi_01")
		sensor_score_text = format_sensor_scores(sensor_scores)
		
		msg = EmailMessage()
		msg["Subject"] = "SmartFall Alert: Fall Detected"
		msg["From"] = EMAIL_USER
		msg["To"] = EMAIL_TO
		
		body = f"""FALL DETECTED
	
			Time: {timestamp}
			Location: {location}
			Device: {device_label}
			Confidence: {float(confidence):.2f}
			
			Trigger: {trigger_source}
			Sensor Mode: {sensor}
			Sensor Scores: {sensor_score_text}
			Notes: {extra_notes if extra_notes else "No additional notes."}
			
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
		

