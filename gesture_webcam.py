"""MediaPipe Gesture Recognizer - 웹캠 실시간 손 제스처 인식

인식 가능한 제스처: None, Closed_Fist, Open_Palm, Pointing_Up,
                    Thumb_Down, Thumb_Up, Victory, ILoveYou

실행: python gesture_webcam.py
종료: q 또는 ESC
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "gesture_recognizer.task"
CAMERA_INDEX = 0
NUM_HANDS = 2

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# 콜백(LIVE_STREAM 모드)에서 받은 최신 결과
latest_result = None


def on_result(result: vision.GestureRecognizerResult, image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def draw_hands(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for c in HAND_CONNECTIONS:
            cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # 손 방향(Left/Right). 화면을 좌우반전했으므로 라벨도 뒤집어 사용자 기준으로 맞춤
        hand = result.handedness[i][0].category_name
        hand = {"Left": "Right", "Right": "Left"}.get(hand, hand)

        gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        text = f"{hand}: {gesture.category_name} ({gesture.score:.2f})" if gesture else hand

        x0 = min(p[0] for p in pts)
        y0 = min(p[1] for p in pts) - 10
        cv2.putText(frame, text, (x0, max(y0, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    prev = time.time()
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 거울 모드
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            if latest_result is not None:
                draw_hands(frame, latest_result)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
