"""3단계: 학습한 커스텀 제스처 실시간 추론

사용법:
    python custom_gesture_webcam.py
    python custom_gesture_webcam.py --threshold 0.8   # 확신도 기준 조절

종료: q 또는 ESC
"""
import argparse
import time

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import (CLASSIFIER_PATH, HAND_MODEL_PATH, draw_landmarks,
                              landmarks_to_features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="이 확률 미만이면 Unknown 으로 표시")
    args = parser.parse_args()

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다. 먼저 train_gestures.py 를 실행하세요: {CLASSIFIER_PATH}")
    model = joblib.load(CLASSIFIER_PATH)
    print("학습된 제스처:", list(model.classes_))

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.camera})을 열 수 없습니다.")

    prev = time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        start = time.monotonic()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                handedness = handed[0].category_name
                pts = draw_landmarks(frame, landmarks)

                feats = landmarks_to_features(landmarks, handedness).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                best = probs.argmax()
                label = model.classes_[best] if probs[best] >= args.threshold else "Unknown"

                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts) - 10
                cv2.putText(frame, f"{label} ({probs[best]:.2f})", (x0, max(y0, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
