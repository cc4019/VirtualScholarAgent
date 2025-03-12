#!/usr/bin/env python
import os
import sys
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_sender.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("email_sender")

def send_email(recipient_email, subject, body, attachments=None):
    """Send an email with optional attachments"""
    # Load environment variables
    load_dotenv()
    
    # Get email credentials from environment variables
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        logger.error("Missing email credentials. Set SENDER_EMAIL and SENDER_PASSWORD in .env file")
        return False
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    
    # Attach body
    msg.attach(MIMEText(body, 'html'))
    
    # Attach files if provided
    if attachments:
        for attachment_path in attachments:
            path = Path(attachment_path)
            if not path.exists():
                logger.warning(f"Attachment not found: {attachment_path}")
                continue
                
            with open(path, 'rb') as file:
                attachment = MIMEApplication(file.read(), Name=path.name)
                attachment['Content-Disposition'] = f'attachment; filename="{path.name}"'
                msg.attach(attachment)
    
    # Send email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False

def find_latest_summary():
    """Find the most recent research application summary file"""
    project_root = Path(__file__).resolve().parent.parent
    summary_dir = project_root / "data" / "paper_retrieval_results"
    
    if not summary_dir.exists():
        logger.error(f"Summary directory not found: {summary_dir}")
        return None
    
    # Find all markdown summary files
    summary_files = list(summary_dir.glob("research_application_summary*.json"))
    
    if not summary_files:
        logger.warning("No summary files found")
        return None
    
    # Sort by modification time (newest first)
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest summary: {latest_summary}")
    return latest_summary


def main():
    """Main function to find and email the latest summary"""
    # Load environment variables
    load_dotenv()
    
    # Get recipient email from environment or command line
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    if len(sys.argv) > 1:
        recipient_email = sys.argv[1]
    
    if not recipient_email:
        logger.error("No recipient email provided. Set RECIPIENT_EMAIL in .env file or provide as command line argument")
        return False
    
    # Find the latest summary file
    latest_summary = find_latest_summary()
    if not latest_summary:
        logger.error("No summary file found to send")
        return False
    
    # Read the summary content
    try:
        with open(latest_summary, 'r', encoding='utf-8') as file:
            summary_content = json.load(file)
    except Exception as e:
        logger.error(f"Failed to read summary file: {str(e)}")
        return False
    
    # Extract papers
    today = summary_content.get("summary_date", datetime.now().strftime("%Y-%m-%d"))
    papers = summary_content.get("papers", [])

    # Constructing HTML body
    paper_html = ""
    for paper in papers:
        research_applications = "".join([f"<li>{item}</li>" for item in paper.get("research_applications", [])])
        strengths = "".join([f"<li>{item}</li>" for item in paper.get("strengths", [])])
        limitations = "".join([f"<li>{item}</li>" for item in paper.get("limitations", [])])
        key_relevance_points = "".join([f"<li>{item}</li>" for item in paper.get("key_relevance_points", [])])

        paper_html += f"""
        <tr>
            <td style="padding:10px;border-bottom:1px solid #ddd;">
                <strong>{paper["title"]}</strong><br>
                <a href="https://arxiv.org/abs/{paper["arxiv_id"]}" style="color:#3498db;">{paper["arxiv_id"]}</a><br>
                <strong>Score:</strong> {paper["overall_score"]}
            </td>
            <td style="padding:10px;border-bottom:1px solid #ddd;">
                <strong>Abstract:</strong>
                <p style="margin-top:5px;">{paper["abstract"]}</p>
                <br>
                <strong>Research Applications:</strong>
                <ul>{research_applications}</ul>
                <strong>Strengths:</strong>
                <ul>{strengths}</ul>
                <strong>Limitations:</strong>
                <ul>{limitations}</ul>
                <strong>Key Relevance Points:</strong>
                <ul>{key_relevance_points}</ul>
            </td>
        </tr>
        """

    # Create email body
    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            h1 {{ color: #2c3e50; }}
            .container {{ width: 80%; margin: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            .footer {{ color: #7f8c8d; font-size: 0.8em; margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Research Paper Summary</h1>
            <p>Here is your latest research paper summary for {today}.</p>
            <table>
                <tr>
                    <th>Title & Score</th>
                    <th>Details</th>
                </tr>
                {paper_html}
            </table>
            <div class="footer">
                <p>This email was automatically generated by your research paper analysis system.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Send the email with the summary as both content and attachment
    success = send_email(
        recipient_email=recipient_email,
        subject=f"Research Paper Summary - {today}",
        body=body,
        attachments=[latest_summary]
    )
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 