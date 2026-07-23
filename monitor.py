import sys
import os
import pandas as pd
import requests
from pathlib import Path

from dotenv import load_dotenv

# إضافة المسار الحالي للمشروع إلى sys.path لضمان رؤية المجلدات (مثل src)
sys.path.append(os.getcwd())

from src.logger.logger import logger

# التأكد من استيراد Evidently بشكل صحيح ومتوافق مع الإصدارات الحديثة
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
except ImportError:
    try:
        from evidently import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError as e:
        logger.error(f"Evidently library import error: {e}")
        sys.exit(1)

# ضع الـ Webhook URL الخاص بـ Slack هنا
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
def send_slack_alert(message: str):
    """Sends a notification to Slack channel."""
    payload = {"text": f"🚨 *ML Monitoring Alert* 🚨\n{message}"}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            logger.info("Slack alert sent successfully.")
        else:
            logger.error(f"Failed to send Slack alert. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error connecting to Slack: {str(e)}")

def run_monitoring(
    ref_path: str = "artifacts/features/engineered_data.csv", 
    curr_path: str = "artifacts/logs/inference_logs.csv",
    output_dir: str = "reports"
):
    """
    Performs comprehensive data analysis and sends alerts if drift is detected.
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("--- Starting Professional Monitoring Process ---")
        
        # التأكد من وجود الملفات قبل القراءة
        if not os.path.exists(ref_path) or not os.path.exists(curr_path):
            logger.error(f"Data files not found at {ref_path} or {curr_path}")
            return
            
        reference = pd.read_csv(ref_path)
        
        # قراءة الـ inference logs بدون هيدر مع معالجة الأعمدة تلقائياً
        try:
            current_raw = pd.read_csv(curr_path, header=None)
        except Exception as e:
            logger.error(f"Failed to read inference logs: {e}")
            return
            
        # مطابقة عدد الأعمدة بقدر ما لدينا في الـ reference
        num_features = reference.shape[1]
        
        if current_raw.shape[1] >= num_features:
            current = current_raw.iloc[:, :num_features]
            current.columns = reference.columns
        else:
            logger.error(f"Error: Inference logs columns count ({current_raw.shape[1]}) is less than reference columns ({num_features})!")
            return
            
        logger.info(f"Successfully aligned {num_features} columns dynamically.")
        
        logger.info("Analyzing data quality and drift...")
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference, current_data=current)
        
        report_file = output_path / "drift_report.html"
        report.save_html(str(report_file))
        logger.info(f"Report generated at: {report_file}")
        
        # Check for drift and trigger alert
        report_dict = report.as_dict()
        drift_detected = report_dict.get('metrics', [{}])[0].get('result', {}).get('drift_detected', False)
        
        if drift_detected:
            msg = f"⚠️ *Data Drift Detected!* Please review the report at: {report_file.absolute()}"
            logger.warning(msg)
            send_slack_alert(msg)
        else:
            logger.info("Monitoring complete: No drift detected.")
            
        print("✅ Monitoring script finished successfully and report is ready!")
        
    except Exception as e:
        logger.error(f"Monitoring process failed: {str(e)}")

if __name__ == "__main__":
    run_monitoring()