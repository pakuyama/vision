"""
YOLO26 웹캠 실시간 Instance Segmentation
설치: pip install -U ultralytics opencv-python
실행: python webcam_segment.py
종료: 'q' 키
"""
import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n-seg.pt"    # n/s/m/l/x 중 선택 (처음 실행 시 자동 다운로드)
CAMERA_INDEX = 0                 # 기본 웹캠
CONF_THRESHOLD = 0.5             # 신뢰도 임계값
SHOW_BOXES = True                # 마스크와 함께 박스/라벨 표시 여부


def open_camera(preferred_index, max_index=5):
    """지정한 인덱스를 먼저 시도하고, 실패하면 다른 인덱스에서 웹캠을 찾는다."""
    indices = [preferred_index] + [i for i in range(max_index) if i != preferred_index]
    for index in indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Windows에서 빠른 초기화
        if cap.isOpened() and cap.read()[0]:
            print(f"웹캠({index}) 연결됨")
            return cap
        cap.release()
    raise RuntimeError("사용 가능한 웹캠을 찾을 수 없습니다.")


def main():
    model = YOLO(MODEL_PATH)

    cap = open_camera(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            # retina_masks=True: 마스크를 원본 해상도로 출력해 경계가 더 매끄러움
            results = model(frame, conf=CONF_THRESHOLD, retina_masks=True, verbose=False)
            result = results[0]
            annotated = result.plot(boxes=SHOW_BOXES, masks=True)

            # 분할된 객체 수 및 FPS 표시
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            num_masks = 0 if result.masks is None else len(result.masks)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Masks: {num_masks}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLO26 Webcam Segmentation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
