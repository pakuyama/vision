"""
YOLO26 웹캠 실시간 Object Tracking
설치: pip install -U ultralytics opencv-python
실행: python webcam_track.py
종료: 'q' 키
"""
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"        # n/s/m/l/x 중 선택 (처음 실행 시 자동 다운로드)
TRACKER = "bytetrack.yaml"       # "bytetrack.yaml" 또는 "botsort.yaml"
CAMERA_INDEX = 0                 # 기본 웹캠
CONF_THRESHOLD = 0.5             # 신뢰도 임계값
TRAIL_LENGTH = 30                # 이동 궤적으로 남길 프레임 수


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

    # 트랙 ID별 중심점 이동 기록
    track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            # persist=True: 프레임 간 트랙 ID 유지
            results = model.track(frame, persist=True, tracker=TRACKER,
                                  conf=CONF_THRESHOLD, verbose=False)
            result = results[0]
            annotated = result.plot()  # 박스 + 클래스 + 트랙 ID 표시

            boxes = result.boxes
            active_ids = set()
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                centers = boxes.xywh.cpu().tolist()
                for track_id, (x, y, _, _) in zip(ids, centers):
                    active_ids.add(track_id)
                    trail = track_history[track_id]
                    trail.append((int(x), int(y)))

                    # 이동 궤적 그리기
                    for i in range(1, len(trail)):
                        thickness = max(1, int(4 * i / len(trail)))
                        cv2.line(annotated, trail[i - 1], trail[i], (0, 255, 255), thickness)

            # 화면에서 사라진 트랙의 기록 정리
            for track_id in list(track_history):
                if track_id not in active_ids:
                    del track_history[track_id]

            # 추적 중인 객체 수 및 FPS 표시
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}  Tracks: {len(active_ids)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLO26 Webcam Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
