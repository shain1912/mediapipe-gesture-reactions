"""1단계: 나만의 제스처 데이터 수집

사용법:
    python collect_gestures.py --label heart
    python collect_gestures.py --label none      # '아무 제스처 아님' 클래스도 꼭 수집

조작키:
    SPACE : 녹화 시작/정지 (녹화 중엔 손이 보이는 매 프레임을 저장)
    q/ESC : 종료

데이터는 data/gestures.csv 에 누적 저장된다 (label, f0..f62).
"""
import argparse
import csv
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_features import (DATA_PATH, HAND_MODEL_PATH, NUM_FEATURES,
                              draw_landmarks, landmarks_to_features)


def count_existing(label: str) -> int:
    if not DATA_PATH.exists():
        return 0
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.reader(f) if row and row[0] == label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="제스처 이름 (영문 권장, 예: heart, ok, none)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.05,
                        help="저장 간격(초). 너무 비슷한 프레임이 몰리지 않게 함")
    args = parser.parse_args()

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DATA_PATH.exists()
    f = open(DATA_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.camera})을 열 수 없습니다.")

    count = count_existing(args.label)
    recording = False
    last_saved = 0.0

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

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_landmarks(frame, landmarks)

                now = time.monotonic()
                if recording and now - last_saved >= args.interval:
                    feats = landmarks_to_features(landmarks, handedness)
                    writer.writerow([args.label] + [f"{v:.6f}" for v in feats])
                    count += 1
                    last_saved = now

            color = (0, 0, 255) if recording else (200, 200, 200)
            status = "REC" if recording else "PAUSED (SPACE to record)"
            cv2.putText(frame, f"[{args.label}] {count} samples", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)

            cv2.imshow("Collect Gestures", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
                f.flush()
            elif key in (ord("q"), 27):
                break

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"'{args.label}' 샘플 총 {count}개 -> {DATA_PATH}")


if __name__ == "__main__":
    main()
