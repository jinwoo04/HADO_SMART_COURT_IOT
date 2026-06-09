"""
One-time helper to export YOLOv8n-pose to NCNN format.
Run this on your Mac (faster than on Pi), then copy the output folder to Pi.

  $ python export_ncnn.py
  → creates yolov8n-pose_ncnn_model/

Copy to Pi:
  $ scp -r yolov8n-pose_ncnn_model pi@<PI_IP>:~/HADO_iot/pi/
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")
    model.export(format="ncnn", half=True, imgsz=320)
    print("✅ NCNN export complete → yolov8n-pose_ncnn_model/")
