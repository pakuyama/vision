# YOLO26으로 웹캠 실시간 비전 실습

> 강의노트 · 2026-10-07
> 웹캠 영상 하나로 **탐지(Detection) → 추적(Tracking) → 분할(Segmentation)** 을 차례로 구현해 봅니다.

---

## 목차

1. [학습 목표](#1-학습-목표)
2. [환경 준비](#2-환경-준비)
3. [공통 뼈대: 웹캠 루프](#3-공통-뼈대-웹캠-루프)
4. [1교시 — Object Detection](#4-1교시--object-detection)
5. [2교시 — Object Tracking](#5-2교시--object-tracking)
6. [3교시 — Instance Segmentation](#6-3교시--instance-segmentation)
7. [세 가지 작업 비교](#7-세-가지-작업-비교)
8. [실습 중 만난 문제와 해결](#8-실습-중-만난-문제와-해결)
9. [연습 문제](#9-연습-문제)

---

## 1. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

- Ultralytics 라이브러리로 YOLO26 모델을 불러와 웹캠 영상에 적용한다.
- **탐지 / 추적 / 분할** 세 작업의 차이를 설명하고, 결과 객체(`Results`)에서 필요한 값을 꺼낸다.
- OpenCV로 웹캠을 열고, 프레임마다 결과를 그려 화면에 띄운다.
- 실행 중 흔히 생기는 문제(패키지 누락, 웹캠 번호 등)를 스스로 해결한다.

| 파일 | 작업 | 사용 모델 |
| --- | --- | --- |
| `webcam_detect.py` | 객체 탐지 | `yolo26n.pt` |
| `webcam_track.py` | 객체 추적 | `yolo26n.pt` + ByteTrack |
| `webcam_segment.py` | 인스턴스 분할 | `yolo26n-seg.pt` |

---

## 2. 환경 준비

### 2-1. 설치

```bash
pip install -U ultralytics opencv-python
```

- `ultralytics` : YOLO 모델을 불러오고 실행하는 라이브러리 (PyTorch도 함께 설치됨)
- `opencv-python` : 웹캠 입력, 화면 출력, 도형/글자 그리기

> 💡 실습 환경: Windows 11, Python 3.14, ultralytics 8.4.174, OpenCV 5.0

### 2-2. 모델 파일

모델 가중치(`.pt`)는 **처음 실행할 때 자동으로 다운로드**됩니다. 그래서 저장소에는 올리지 않습니다(`.gitignore`에 `*.pt`).

| 크기 | 탐지 모델 | 분할 모델 | 특징 |
| --- | --- | --- | --- |
| n (nano) | `yolo26n.pt` | `yolo26n-seg.pt` | 가장 빠름, 실습 기본값 |
| s (small) | `yolo26s.pt` | `yolo26s-seg.pt` | |
| m (medium) | `yolo26m.pt` | `yolo26m-seg.pt` | |
| l / x | `yolo26l.pt` / `yolo26x.pt` | `-seg` 동일 | 가장 정확, 가장 느림 |

> 📌 **핵심:** 모델 이름 뒤의 접미사가 작업을 결정합니다. 접미사가 없으면 탐지, `-seg`이면 분할입니다.

### 2-3. 실행과 종료

```bash
python webcam_detect.py    # 탐지
python webcam_track.py     # 추적
python webcam_segment.py   # 분할
```

창을 클릭한 뒤 **`q` 키**를 누르면 종료됩니다.

---

## 3. 공통 뼈대: 웹캠 루프

세 파일은 모두 같은 구조입니다. 바뀌는 건 **② 모델 호출**과 **③ 결과 그리기**뿐입니다.

```
모델 불러오기 → 웹캠 열기
     ↓
┌─▶ ① 프레임 읽기 (cap.read)
│   ② 모델 호출   (model(...) 또는 model.track(...))
│   ③ 결과 그리기 (result.plot + FPS 표시)
│   ④ 화면 출력   (cv2.imshow)
└── ⑤ 'q' 키가 아니면 반복
     ↓
자원 정리 (cap.release, destroyAllWindows)
```

```python
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("window", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

### 짚고 넘어갈 포인트

- **`cv2.CAP_DSHOW`** : Windows에서 DirectShow 백엔드를 써서 웹캠을 빨리 엽니다.
- **`results[0]`** : 모델은 여러 장을 한 번에 처리할 수 있어 결과가 리스트로 나옵니다. 프레임을 한 장씩 넣으니 항상 `[0]`입니다.
- **`result.plot()`** : 박스·라벨·마스크를 그린 이미지를 돌려줍니다. 직접 그리지 않아도 됩니다.
- **`try / finally`** : 중간에 오류가 나도 웹캠을 꼭 놓아 주기 위해서입니다. 안 그러면 다음 실행 때 웹캠이 "사용 중"으로 남을 수 있습니다.
- **FPS 계산** : 이전 프레임과의 시간 차이의 역수입니다.

  ```python
  fps = 1.0 / max(now - prev_time, 1e-6)   # 0으로 나누기 방지
  ```

---

## 4. 1교시 — Object Detection

📄 `webcam_detect.py`

### 개념

> **탐지 = "무엇이 어디에 있는가?"**
> 각 객체마다 **박스(위치) + 클래스(이름) + 신뢰도(확률)** 를 출력합니다.

프레임마다 독립적으로 처리하기 때문에, 같은 사람이라도 프레임 사이의 연결 정보는 없습니다.

### 핵심 코드

```python
results = model(frame, conf=CONF_THRESHOLD, verbose=False)
annotated = results[0].plot()
num_objects = len(results[0].boxes)
```

### 결과에서 값 꺼내기

| 속성 | 의미 |
| --- | --- |
| `boxes.xyxy` | 박스 좌상단·우하단 좌표 `(x1, y1, x2, y2)` |
| `boxes.xywh` | 박스 중심과 크기 `(cx, cy, w, h)` |
| `boxes.conf` | 신뢰도 (0~1) |
| `boxes.cls` | 클래스 번호 → `model.names[번호]`로 이름 확인 |

### 파라미터

- **`conf=0.5`** : 신뢰도 50% 미만은 버립니다. 낮추면 더 많이 잡지만 오탐이 늘고, 높이면 확실한 것만 남습니다.
- **`verbose=False`** : 매 프레임 터미널 로그 출력을 끕니다.

---

## 5. 2교시 — Object Tracking

📄 `webcam_track.py`

### 개념

> **추적 = 탐지 + "같은 객체에 같은 번호(ID) 붙이기"**

탐지만 하면 프레임마다 "사람 1명"이라는 사실만 압니다. 추적을 하면 "**3번 사람**이 왼쪽에서 오른쪽으로 이동했다"를 알 수 있습니다. 사람 수 세기, 이동 경로 분석, 영역 출입 감지 등에 쓰입니다.

### 핵심 코드

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                      conf=CONF_THRESHOLD, verbose=False)
```

- **`model.track(...)`** : 탐지 후 트래커가 이전 프레임의 객체와 짝을 지어 ID를 붙입니다.
- **`persist=True`** : ⚠️ **가장 중요한 옵션.** 트래커 상태를 프레임 사이에 유지합니다. 빠뜨리면 매 프레임 ID가 새로 시작됩니다.
- **`tracker=`** : 추적 알고리즘 선택

  | 트래커 | 특징 |
  | --- | --- |
  | `bytetrack.yaml` | 빠르고 가벼움 (실습 기본값) |
  | `botsort.yaml` | 외형 정보까지 활용, 가려졌다 다시 나타나는 객체에 강함, 조금 느림 |

### 이동 궤적 그리기

트랙 ID별로 박스 중심점을 최근 30개까지 저장해 선으로 잇습니다.

```python
track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

if boxes.id is not None:                      # ID가 아직 없을 수 있음!
    ids = boxes.id.int().cpu().tolist()
    centers = boxes.xywh.cpu().tolist()
    for track_id, (x, y, _, _) in zip(ids, centers):
        trail = track_history[track_id]
        trail.append((int(x), int(y)))
        for i in range(1, len(trail)):
            thickness = max(1, int(4 * i / len(trail)))   # 최근일수록 굵게
            cv2.line(annotated, trail[i - 1], trail[i], (0, 255, 255), thickness)
```

### 짚고 넘어갈 포인트

- **`boxes.id is not None` 체크** : 화면에 객체가 없거나 트래커가 아직 ID를 확정하지 않으면 `id`가 `None`입니다. 체크하지 않으면 오류가 납니다.
- **`deque(maxlen=N)`** : 길이가 N을 넘으면 오래된 점이 자동으로 빠집니다. 궤적 길이 관리에 딱 맞는 자료구조입니다.
- **`.cpu().tolist()`** : 결과는 PyTorch 텐서라서, OpenCV에 쓰려면 파이썬 숫자로 바꿔야 합니다.
- **사라진 트랙 정리** : 화면에서 없어진 ID의 기록은 지워서 메모리가 계속 늘지 않게 합니다.

---

## 6. 3교시 — Instance Segmentation

📄 `webcam_segment.py`

### 개념

> **분할 = 탐지 + "객체의 정확한 모양(픽셀 단위 마스크)"**

박스는 사각형이라 배경이 섞입니다. 분할은 객체에 해당하는 **픽셀만** 골라내므로 배경 제거, 면적 계산, 정밀한 위치 파악에 쓰입니다.

> 📌 "인스턴스" 분할이라서 같은 사람이라도 **사람마다 다른 마스크**가 나옵니다.

### 핵심 코드

```python
MODEL_PATH = "yolo26n-seg.pt"   # -seg 모델을 써야 마스크가 나옴

results = model(frame, conf=CONF_THRESHOLD, retina_masks=True, verbose=False)
result = results[0]
annotated = result.plot(boxes=SHOW_BOXES, masks=True)
num_masks = 0 if result.masks is None else len(result.masks)
```

### 짚고 넘어갈 포인트

- **모델만 바꾸면 된다** : 코드 구조는 탐지와 같고, 모델 파일만 `-seg`로 바꾸면 `result.masks`가 채워집니다.
- **`retina_masks=True`** : 마스크를 원본 해상도로 계산해 경계가 매끄럽습니다. 대신 조금 느려집니다.
- **`result.masks is None`** : 화면에 객체가 없으면 `masks`가 `None`이라 `len()`을 바로 쓰면 오류가 납니다.
- **`plot(boxes=False)`** : 박스·라벨을 숨기고 마스크만 볼 수 있습니다.
- 마스크 값 꺼내기
  - `result.masks.data` : 객체별 0/1 마스크 텐서
  - `result.masks.xy` : 객체별 외곽선 좌표(폴리곤)

---

## 7. 세 가지 작업 비교

| | Detection | Tracking | Segmentation |
| --- | --- | --- | --- |
| 질문 | 무엇이 어디에? | 그게 계속 같은 놈인가? | 정확히 어떤 모양인가? |
| 출력 | 박스, 클래스, 신뢰도 | + 트랙 ID | + 픽셀 마스크 |
| 호출 | `model(frame)` | `model.track(frame, persist=True)` | `model(frame)` (`-seg` 모델) |
| 모델 | `yolo26n.pt` | `yolo26n.pt` | `yolo26n-seg.pt` |
| 프레임 간 정보 | 없음 | 있음 | 없음 |
| 속도 | 가장 빠름 | 조금 느림 | 가장 느림 |

> 💡 **응용:** 분할 모델에도 `model.track()`을 쓸 수 있습니다. 그러면 마스크와 ID를 동시에 얻습니다.

---

## 8. 실습 중 만난 문제와 해결

실제로 오늘 실습하면서 겪은 문제들입니다.

### ❶ `ModuleNotFoundError: No module named 'ultralytics'`

- **원인:** 패키지가 설치되지 않음
- **해결:** `pip install -U ultralytics opencv-python`

### ❷ 웹캠을 열 수 없다는 오류 (`웹캠(0)을 열 수 없습니다`)

- **원인:** 웹캠 장치 번호가 **0번이 아니라 1번**으로 잡혀 있었음. 연결된 장치나 드라이버에 따라 번호가 바뀔 수 있음.
- **진단 방법:** 번호와 백엔드를 바꿔 가며 직접 열어 보기

  ```python
  import cv2
  for i in range(3):
      cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
      print(i, cap.isOpened(), cap.read()[0] if cap.isOpened() else False)
      cap.release()
  ```

- **해결:** 원하는 번호가 안 열리면 다른 번호를 차례로 시도하는 함수를 추가 (`webcam_track.py`, `webcam_segment.py`)

  ```python
  def open_camera(preferred_index, max_index=5):
      indices = [preferred_index] + [i for i in range(max_index) if i != preferred_index]
      for index in indices:
          cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
          if cap.isOpened() and cap.read()[0]:   # 열리고 + 실제로 읽혀야 성공
              print(f"웹캠({index}) 연결됨")
              return cap
          cap.release()
      raise RuntimeError("사용 가능한 웹캠을 찾을 수 없습니다.")
  ```

  > 📌 `isOpened()`만으로는 부족합니다. 실습 중 "열리긴 하는데 프레임은 못 읽는" 경우가 있었기 때문에 `read()`까지 확인합니다.

  > ⚠️ `webcam_detect.py`에는 아직 이 함수가 없어 0번만 시도합니다.

### ❸ 추적 첫 실행 때 `lap` 패키지 자동 설치

- **현상:** `requirements: Ultralytics requirement ['lap>=0.5.12'] not found, attempting AutoUpdate...`
- **원인:** ByteTrack/BoT-SORT가 객체 짝짓기에 `lap` 패키지를 사용
- **해결:** Ultralytics가 자동으로 설치해 줌. "Restart runtime" 경고가 떠도 실습에서는 그대로 동작했음. 미리 설치하려면 `pip install lap`

---

## 9. 연습 문제

1. **신뢰도 실험** — `CONF_THRESHOLD`를 0.25, 0.5, 0.8로 바꿔 보고 탐지 개수와 오탐이 어떻게 달라지는지 비교해 보세요.
2. **모델 크기 실험** — `yolo26n.pt`와 `yolo26s.pt`의 FPS와 정확도를 비교해 보세요.
3. **특정 클래스만** — 사람만 탐지하도록 바꿔 보세요.
   <details><summary>힌트</summary>

   `model(frame, classes=[0])` — COCO 기준 0번이 `person`입니다.
   </details>
4. **트래커 비교** — `TRACKER`를 `botsort.yaml`로 바꾸고, 손으로 얼굴을 잠깐 가렸다 뗐을 때 ID가 유지되는지 ByteTrack과 비교해 보세요.
5. **persist 실험** — `persist=False`로 바꾸면 ID가 어떻게 되는지 관찰하고 이유를 설명해 보세요.
6. **분할 + 추적** — `webcam_segment.py`에서 `model(...)`을 `model.track(..., persist=True)`로 바꿔 마스크와 ID를 함께 표시해 보세요.
7. **웹캠 함수 적용** — `webcam_detect.py`에도 `open_camera()`를 적용해 보세요.
