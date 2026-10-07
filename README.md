# 웹 게임에서 실시간 비전까지 — 하루 실습 강의노트

> 강의노트 · 2026-10-07
> 오늘은 세 단계로 실습했습니다.
> **① 브라우저 게임 만들기 → ② 웹캠 + 머신러닝으로 게임 조종하기 → ③ Python + YOLO26으로 실시간 비전 처리하기**

| Part | 주제 | 저장소 | 핵심 기술 |
| --- | --- | --- | --- |
| 1 | 🧱 벽돌깨기 | [pakuyama/brick-breaker](https://github.com/pakuyama/brick-breaker) | HTML Canvas, 게임 루프, 충돌 처리 |
| 2 | 🐍 웹캠 모션 뱀게임 | [pakuyama/teachable-snake-game](https://github.com/pakuyama/teachable-snake-game) | Teachable Machine, TensorFlow.js |
| 3 | 👁️ YOLO26 웹캠 비전 | 이 저장소 (`vision`) | Ultralytics YOLO26, OpenCV |

> 💡 흐름: Part 1에서 **게임 루프**를 익히고, Part 2에서 키보드 대신 **웹캠 + 분류 모델**을 입력으로 씁니다. Part 3에서는 분류를 넘어 **탐지·추적·분할**로 확장합니다.

---

## 목차

- [Part 1. 벽돌깨기 — Canvas 게임의 기본](#part-1-벽돌깨기--canvas-게임의-기본)
- [Part 2. 웹캠 모션 뱀게임 — 머신러닝을 입력장치로](#part-2-웹캠-모션-뱀게임--머신러닝을-입력장치로)
- [Part 3. YOLO26 웹캠 비전 — 탐지·추적·분할](#part-3-yolo26-웹캠-비전--탐지추적분할)
- [마무리: 오늘 배운 것 연결하기](#마무리-오늘-배운-것-연결하기)

---

# Part 1. 벽돌깨기 — Canvas 게임의 기본

📄 [brick-breaker/index.html](https://github.com/pakuyama/brick-breaker/blob/main/index.html) · HTML 파일 하나로 완성

### 학습 목표

- `<canvas>`에 도형을 그리고, **게임 루프**로 화면을 계속 갱신한다.
- 공·벽·패들·벽돌 사이의 **충돌**을 계산한다.
- **상태(state)** 로 게임 흐름(대기 → 플레이 → 클리어/게임오버)을 관리한다.

### 실행

`index.html`을 브라우저로 열면 바로 플레이할 수 있습니다. 외부 라이브러리가 없습니다.

- 마우스 / 터치 / ← → 키로 패들 이동
- 클릭 또는 스페이스바로 공 발사

### 1-1. 게임 루프

게임은 "**상태 갱신 → 그리기**"를 1초에 약 60번 반복합니다.

```javascript
function loop() { update(); draw(); requestAnimationFrame(loop); }
```

- **`requestAnimationFrame`** : 브라우저 화면 갱신 주기(보통 60Hz)에 맞춰 다음 프레임을 호출합니다. `setInterval`보다 부드럽고, 탭이 안 보이면 자동으로 멈춰 배터리를 아낍니다.
- **`update()`와 `draw()`를 나누는 이유** : 계산과 그리기를 분리하면 버그를 찾기 쉽고 코드가 읽기 좋아집니다.

> 📌 Part 3의 Python 웹캠 루프(`while True: 읽기 → 처리 → 그리기 → imshow`)도 **같은 구조**입니다.

### 1-2. 상태 관리

```javascript
let state = "ready"; // ready | play | over | clear
```

| 상태 | 화면 | 클릭하면 |
| --- | --- | --- |
| `ready` | 공이 패들 위에 붙어 있음 | 공 발사 → `play` |
| `play` | 공이 움직임 | — |
| `over` | GAME OVER | 처음부터 다시 → `ready` |
| `clear` | STAGE CLEAR | 다음 스테이지 → `ready` |

하나의 `launch()` 함수가 상태에 따라 다르게 동작합니다. 상태 변수 하나로 흐름을 관리하면 `if`문이 흩어지지 않습니다.

### 1-3. 충돌 처리

**① 벽 반사** — 속도의 부호만 바꿉니다.

```javascript
if (ball.x < ball.r) { ball.x = ball.r; ball.vx *= -1; }
```

> ⚠️ 위치를 벽 안쪽으로 되돌리는(`ball.x = ball.r`) 이유: 안 그러면 공이 벽에 박혀 매 프레임 방향이 뒤집히는 "떨림" 버그가 생깁니다.

**② 패들 반사** — 맞은 위치로 반사 각도를 정합니다.

```javascript
const hit = (ball.x - (paddle.x + paddle.w / 2)) / (paddle.w / 2);  // -1(왼쪽 끝) ~ 1(오른쪽 끝)
const a = -Math.PI / 2 + hit * (Math.PI / 3);                       // 위쪽 기준 ±60°
ball.vx = Math.cos(a) * ball.speed;
ball.vy = Math.sin(a) * ball.speed;
```

패들 가운데에 맞으면 수직으로, 끝에 맞을수록 비스듬히 튑니다. 플레이어가 공 방향을 **조종**할 수 있게 만드는 핵심입니다.

**③ 벽돌 충돌** — 원과 사각형의 충돌

```javascript
const nx = Math.max(b.x, Math.min(ball.x, b.x + BW));   // 사각형에서 공 중심과 가장 가까운 점
const ny = Math.max(b.y, Math.min(ball.y, b.y + BH));
const dx = ball.x - nx, dy = ball.y - ny;
if (dx * dx + dy * dy <= ball.r * ball.r) { ... }       // 그 점까지 거리 ≤ 반지름이면 충돌
```

- 공 중심을 사각형 범위로 **clamp**하면 가장 가까운 점이 나옵니다.
- 루트(`Math.sqrt`) 없이 **거리의 제곱**끼리 비교해 계산을 아낍니다.
- `|dx| > |dy|`이면 옆면에 맞은 것이므로 `vx`를, 아니면 `vy`를 뒤집습니다.

### 1-4. 난이도와 연출

- **스테이지 난이도** : 줄 수 `min(4 + stage, 8)`, 공 속도 `5 + stage × 0.6`, 위쪽 줄은 2번 맞아야 깨지는 단단한 벽돌(`hp = 2`)
- **파티클** : 벽돌이 깨지면 작은 사각형 12개를 흩뿌리고, 중력(`vy += 0.2`)과 수명(`life`)을 줘서 사라지게 합니다.
- **화면 크기 대응** : 캔버스가 CSS로 줄어들어도 `(clientX - rect.left) × (W / rect.width)`로 마우스 좌표를 캔버스 좌표로 바꿉니다.

---

# Part 2. 웹캠 모션 뱀게임 — 머신러닝을 입력장치로

📄 [teachable-snake-game/index.html](https://github.com/pakuyama/teachable-snake-game/blob/main/index.html)

### 학습 목표

- **Teachable Machine**으로 학습한 이미지 분류 모델을 웹 페이지에서 불러온다.
- 웹캠 영상을 매 프레임 분류해 **게임 조작 신호**로 바꾼다.
- **임계값**으로 잘못된 인식을 걸러낸다.

### 실행

웹캠과 외부 모델 파일을 쓰기 때문에 파일을 더블클릭하지 말고 **로컬 서버**로 엽니다.

```bash
python -m http.server 8000
```

브라우저에서 `http://localhost:8000` 접속 → **📷 웹캠 & 모델 시작** → **▶ 게임 시작**

> 💡 브라우저는 보안상 `file://` 주소에서 웹캠이나 외부 리소스 사용을 제한할 수 있습니다. `localhost`는 안전한 주소로 취급됩니다.

### 2-1. Teachable Machine이란?

구글의 웹 도구로, 코딩 없이 웹캠 사진을 찍어 **이미지 분류 모델**을 학습시킬 수 있습니다.

이 게임의 모델은 5개 클래스로 학습했습니다.

| 클래스 | 동작 |
| --- | --- |
| `up` / `down` / `left` / `right` | 뱀 방향 전환 |
| `neutral` | 아무것도 안 함 (현재 방향 유지) |

> 📌 **`neutral` 클래스가 중요한 이유:** 분류 모델은 무조건 클래스 중 하나를 고릅니다. "가만히 있는 상태"를 따로 가르치지 않으면, 아무 동작을 안 해도 네 방향 중 하나로 인식해 버립니다.

### 2-2. 모델 불러오기와 웹캠 연결

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8.5/dist/teachablemachine-image.min.js"></script>
```

```javascript
const MODEL_URL = "https://teachablemachine.withgoogle.com/models/U93WCxIU8/";
model = await tmImage.load(MODEL_URL + "model.json", MODEL_URL + "metadata.json");
labels = model.getClassLabels();

webcam = new tmImage.Webcam(240, 240, true); // true = 좌우 반전(거울 모드)
await webcam.setup();
await webcam.play();
```

- **TensorFlow.js** : 브라우저에서 모델을 돌리는 엔진. 서버 없이 **내 컴퓨터에서** 추론합니다.
- **`model.json` + `metadata.json`** : 모델 구조·가중치 정보와 클래스 이름입니다.
- **거울 모드** : 내가 왼쪽으로 움직이면 화면에서도 왼쪽으로 보이게 해서 조작이 자연스럽습니다.

### 2-3. 예측 루프와 임계값

```javascript
async function loop() {
  webcam.update();
  if (!predicting) {          // 이전 예측이 끝났을 때만 새로 예측
    predicting = true;
    await predict();
    predicting = false;
  }
  window.requestAnimationFrame(loop);
}
```

```javascript
if (top.probability >= Number(thEl.value)) {     // 기본 0.80
  if (DIRS[cls]) { if (running) setDirection(cls); }
  // neutral이면 아무것도 안 함
}
```

- **`predicting` 플래그** : 예측은 비동기라 시간이 걸립니다. 끝나기 전에 또 호출하면 예측이 쌓여 느려지므로, 한 번에 하나만 돌립니다.
- **임계값(threshold)** : 가장 높은 확률이 80% 이상일 때만 방향을 바꿉니다. 애매한 자세에서 뱀이 제멋대로 꺾이는 걸 막습니다. 화면의 슬라이더로 0.5~0.99 사이에서 조절할 수 있습니다.

> 📌 Part 3 YOLO의 `conf=0.5`와 **같은 개념**입니다. 모델 출력을 그대로 믿지 않고, 확신이 있을 때만 사용합니다.

### 2-4. 뱀 게임 로직

```javascript
function step() {
  dir = nextDir;
  const head = { x: snake[0].x + DIRS[dir].x, y: snake[0].y + DIRS[dir].y };
  if (벽에 닿음 || 몸에 닿음) return gameOver();
  snake.unshift(head);                 // 머리 추가
  if (먹이를 먹음) { score++; placeFood(); }
  else snake.pop();                    // 안 먹었으면 꼬리 제거 → 길이 유지
  draw();
  timer = setTimeout(step, Number(speedEl.value));
}
```

- **이동 = 머리 추가 + 꼬리 제거.** 먹이를 먹으면 꼬리를 안 지워서 한 칸 길어집니다.
- **`nextDir`를 따로 두는 이유** : 한 칸 이동 사이에 입력이 여러 번 들어와도 마지막 입력만 반영합니다.
- **반대 방향 금지** : 오른쪽으로 가다가 바로 왼쪽으로 꺾으면 자기 몸에 부딪히므로 막습니다.

  ```javascript
  function isOpposite(a, b) {
    return DIRS[a].x + DIRS[b].x === 0 && DIRS[a].y + DIRS[b].y === 0;
  }
  ```

- **`setTimeout` 게임 루프** : 벽돌깨기(`requestAnimationFrame`, 매 프레임)와 달리 뱀은 칸 단위로 움직이므로 정해진 간격(기본 200ms)마다 한 칸씩 갑니다.
- **최고 점수 저장** : `localStorage`를 `try/catch`로 감싸서, 저장소를 못 쓰는 환경에서도 게임이 멈추지 않습니다.
- **키보드 보조 조작** : 방향키로도 조작할 수 있어 모델 없이 테스트할 수 있습니다.

---

# Part 3. YOLO26 웹캠 비전 — 탐지·추적·분할

> 웹캠 영상 하나로 **탐지(Detection) → 추적(Tracking) → 분할(Segmentation)** 을 차례로 구현해 봅니다.
> Part 2의 Teachable Machine은 "화면 전체가 무엇인가"를 맞히는 **분류**였습니다. 여기서는 "화면 **어디에** 무엇이 있는가"로 나아갑니다.

### 3-1. 학습 목표

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

### 3-2. 환경 준비

#### 3-2-1. 설치

```bash
pip install -U ultralytics opencv-python
```

- `ultralytics` : YOLO 모델을 불러오고 실행하는 라이브러리 (PyTorch도 함께 설치됨)
- `opencv-python` : 웹캠 입력, 화면 출력, 도형/글자 그리기

> 💡 실습 환경: Windows 11, Python 3.14, ultralytics 8.4.174, OpenCV 5.0

#### 3-2-2. 모델 파일

모델 가중치(`.pt`)는 **처음 실행할 때 자동으로 다운로드**됩니다. 그래서 저장소에는 올리지 않습니다(`.gitignore`에 `*.pt`).

| 크기 | 탐지 모델 | 분할 모델 | 특징 |
| --- | --- | --- | --- |
| n (nano) | `yolo26n.pt` | `yolo26n-seg.pt` | 가장 빠름, 실습 기본값 |
| s (small) | `yolo26s.pt` | `yolo26s-seg.pt` | |
| m (medium) | `yolo26m.pt` | `yolo26m-seg.pt` | |
| l / x | `yolo26l.pt` / `yolo26x.pt` | `-seg` 동일 | 가장 정확, 가장 느림 |

> 📌 **핵심:** 모델 이름 뒤의 접미사가 작업을 결정합니다. 접미사가 없으면 탐지, `-seg`이면 분할입니다.

#### 3-2-3. 실행과 종료

```bash
python webcam_detect.py    # 탐지
python webcam_track.py     # 추적
python webcam_segment.py   # 분할
```

창을 클릭한 뒤 **`q` 키**를 누르면 종료됩니다.

---

### 3-3. 공통 뼈대: 웹캠 루프

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

#### 짚고 넘어갈 포인트

- **`cv2.CAP_DSHOW`** : Windows에서 DirectShow 백엔드를 써서 웹캠을 빨리 엽니다.
- **`results[0]`** : 모델은 여러 장을 한 번에 처리할 수 있어 결과가 리스트로 나옵니다. 프레임을 한 장씩 넣으니 항상 `[0]`입니다.
- **`result.plot()`** : 박스·라벨·마스크를 그린 이미지를 돌려줍니다. 직접 그리지 않아도 됩니다.
- **`try / finally`** : 중간에 오류가 나도 웹캠을 꼭 놓아 주기 위해서입니다. 안 그러면 다음 실행 때 웹캠이 "사용 중"으로 남을 수 있습니다.
- **FPS 계산** : 이전 프레임과의 시간 차이의 역수입니다.

  ```python
  fps = 1.0 / max(now - prev_time, 1e-6)   # 0으로 나누기 방지
  ```

---

### 3-4. 1교시 — Object Detection

📄 `webcam_detect.py`

#### 개념

> **탐지 = "무엇이 어디에 있는가?"**
> 각 객체마다 **박스(위치) + 클래스(이름) + 신뢰도(확률)** 를 출력합니다.

프레임마다 독립적으로 처리하기 때문에, 같은 사람이라도 프레임 사이의 연결 정보는 없습니다.

#### 핵심 코드

```python
results = model(frame, conf=CONF_THRESHOLD, verbose=False)
annotated = results[0].plot()
num_objects = len(results[0].boxes)
```

#### 결과에서 값 꺼내기

| 속성 | 의미 |
| --- | --- |
| `boxes.xyxy` | 박스 좌상단·우하단 좌표 `(x1, y1, x2, y2)` |
| `boxes.xywh` | 박스 중심과 크기 `(cx, cy, w, h)` |
| `boxes.conf` | 신뢰도 (0~1) |
| `boxes.cls` | 클래스 번호 → `model.names[번호]`로 이름 확인 |

#### 파라미터

- **`conf=0.5`** : 신뢰도 50% 미만은 버립니다. 낮추면 더 많이 잡지만 오탐이 늘고, 높이면 확실한 것만 남습니다.
- **`verbose=False`** : 매 프레임 터미널 로그 출력을 끕니다.

---

### 3-5. 2교시 — Object Tracking

📄 `webcam_track.py`

#### 개념

> **추적 = 탐지 + "같은 객체에 같은 번호(ID) 붙이기"**

탐지만 하면 프레임마다 "사람 1명"이라는 사실만 압니다. 추적을 하면 "**3번 사람**이 왼쪽에서 오른쪽으로 이동했다"를 알 수 있습니다. 사람 수 세기, 이동 경로 분석, 영역 출입 감지 등에 쓰입니다.

#### 핵심 코드

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

#### 이동 궤적 그리기

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

#### 짚고 넘어갈 포인트

- **`boxes.id is not None` 체크** : 화면에 객체가 없거나 트래커가 아직 ID를 확정하지 않으면 `id`가 `None`입니다. 체크하지 않으면 오류가 납니다.
- **`deque(maxlen=N)`** : 길이가 N을 넘으면 오래된 점이 자동으로 빠집니다. 궤적 길이 관리에 딱 맞는 자료구조입니다.
- **`.cpu().tolist()`** : 결과는 PyTorch 텐서라서, OpenCV에 쓰려면 파이썬 숫자로 바꿔야 합니다.
- **사라진 트랙 정리** : 화면에서 없어진 ID의 기록은 지워서 메모리가 계속 늘지 않게 합니다.

---

### 3-6. 3교시 — Instance Segmentation

📄 `webcam_segment.py`

#### 개념

> **분할 = 탐지 + "객체의 정확한 모양(픽셀 단위 마스크)"**

박스는 사각형이라 배경이 섞입니다. 분할은 객체에 해당하는 **픽셀만** 골라내므로 배경 제거, 면적 계산, 정밀한 위치 파악에 쓰입니다.

> 📌 "인스턴스" 분할이라서 같은 사람이라도 **사람마다 다른 마스크**가 나옵니다.

#### 핵심 코드

```python
MODEL_PATH = "yolo26n-seg.pt"   # -seg 모델을 써야 마스크가 나옴

results = model(frame, conf=CONF_THRESHOLD, retina_masks=True, verbose=False)
result = results[0]
annotated = result.plot(boxes=SHOW_BOXES, masks=True)
num_masks = 0 if result.masks is None else len(result.masks)
```

#### 짚고 넘어갈 포인트

- **모델만 바꾸면 된다** : 코드 구조는 탐지와 같고, 모델 파일만 `-seg`로 바꾸면 `result.masks`가 채워집니다.
- **`retina_masks=True`** : 마스크를 원본 해상도로 계산해 경계가 매끄럽습니다. 대신 조금 느려집니다.
- **`result.masks is None`** : 화면에 객체가 없으면 `masks`가 `None`이라 `len()`을 바로 쓰면 오류가 납니다.
- **`plot(boxes=False)`** : 박스·라벨을 숨기고 마스크만 볼 수 있습니다.
- 마스크 값 꺼내기
  - `result.masks.data` : 객체별 0/1 마스크 텐서
  - `result.masks.xy` : 객체별 외곽선 좌표(폴리곤)

---

### 3-7. 세 가지 작업 비교

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

### 3-8. 실습 중 만난 문제와 해결

실제로 오늘 실습하면서 겪은 문제들입니다.

#### ❶ `ModuleNotFoundError: No module named 'ultralytics'`

- **원인:** 패키지가 설치되지 않음
- **해결:** `pip install -U ultralytics opencv-python`

#### ❷ 웹캠을 열 수 없다는 오류 (`웹캠(0)을 열 수 없습니다`)

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

#### ❸ 추적 첫 실행 때 `lap` 패키지 자동 설치

- **현상:** `requirements: Ultralytics requirement ['lap>=0.5.12'] not found, attempting AutoUpdate...`
- **원인:** ByteTrack/BoT-SORT가 객체 짝짓기에 `lap` 패키지를 사용
- **해결:** Ultralytics가 자동으로 설치해 줌. "Restart runtime" 경고가 떠도 실습에서는 그대로 동작했음. 미리 설치하려면 `pip install lap`

---

### 3-9. 연습 문제

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

---

# 마무리: 오늘 배운 것 연결하기

| | Part 1 벽돌깨기 | Part 2 모션 뱀게임 | Part 3 YOLO26 비전 |
| --- | --- | --- | --- |
| 언어 / 환경 | HTML + JavaScript | HTML + JavaScript | Python |
| 입력 | 마우스·키보드·터치 | **웹캠 + 분류 모델** | **웹캠 + 탐지/추적/분할 모델** |
| 반복 구조 | `requestAnimationFrame` | `setTimeout` (게임) + `requestAnimationFrame` (예측) | `while True` + `cv2.waitKey` |
| 모델 실행 위치 | — | 브라우저 (TensorFlow.js) | 내 컴퓨터 (PyTorch) |
| 신뢰도 거르기 | — | `threshold` 슬라이더 (0.80) | `conf=0.5` |
| 모델이 답하는 질문 | — | 화면 전체가 무엇인가? | 어디에, 무엇이, 어떤 모양으로? |

### 세 가지 공통 개념

1. **루프** — 게임이든 비전이든 "입력 받기 → 처리 → 그리기"를 계속 반복합니다.
2. **상태** — 게임 상태(`ready`/`play`), 뱀의 방향(`nextDir`), 트랙 기록(`track_history`)처럼 프레임 사이에 기억해야 할 값을 관리합니다.
3. **모델 출력은 확률** — 임계값으로 걸러야 실제 서비스에 쓸 수 있습니다.

### 다음 단계 아이디어

- YOLO의 손·사람 위치를 받아 **벽돌깨기 패들**을 몸으로 움직여 보기
- Teachable Machine 대신 **YOLO26 포즈 모델**(`yolo26n-pose.pt`)로 팔 방향을 읽어 뱀 조종하기
- 추적 ID를 이용해 화면을 지나간 **사람 수 세기**
