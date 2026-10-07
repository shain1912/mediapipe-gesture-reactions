"""수집/학습/추론 스크립트가 함께 쓰는 공통 모듈

손 랜드마크 21개(x, y, z)를 위치·크기·좌우에 무관한 63차원 특징 벡터로 변환한다.
"""
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATA_PATH = BASE_DIR / "data" / "gestures.csv"
CLASSIFIER_PATH = BASE_DIR / "models" / "custom_gesture.joblib"

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
NUM_FEATURES = 21 * 3


def landmarks_to_features(landmarks, handedness: str) -> np.ndarray:
    """랜드마크 -> 정규화된 특징 벡터 (shape: (63,))

    1) 손목(0번)을 원점으로 이동  -> 화면 내 위치 무관
    2) 손목~가장 먼 점 거리로 나눔 -> 손 크기/카메라 거리 무관
    3) 왼손이면 x 좌표 반전        -> 왼손/오른손 같은 제스처로 취급
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts /= scale
    if handedness == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    return pts
