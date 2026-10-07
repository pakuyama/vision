"""
YOLO26 웹캠 실시간 Object Detection
설치: pip install -U ultralytics opencv-python
실행: python webcam_detect.py
종료: 'q' 키
"""
import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"  # n/s/m/l/x 중 선택 (처음 실행 시 자동 다운로드)
CAMERA_INDEX = 0           # 기본 웹캠
CONF_THRESHOLD = 0.5       # 신뢰도 임계값


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows에서 빠른 초기화
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated = results[0].plot()

            # 탐지된 객체 수 및 FPS 표시
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            num_objects = len(results[0].boxes)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {num_objects}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLO26 Webcam Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
