"""수집한 데이터로 제스처 분류기 학습

gesture_studio.py(UI)에서 train() 을 호출해 사용한다.
단독 실행도 가능: python train_gestures.py

data/gestures.csv 를 읽어 MLP 분류기를 학습하고,
테스트 정확도/혼동행렬을 출력한 뒤 models/custom_gesture.joblib 로 저장한다.
"""
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_features import CLASSIFIER_PATH, DATA_PATH


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("수집된 데이터가 없습니다. 먼저 제스처를 수집하세요.")
    labels, feats = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                labels.append(row[0])
                feats.append([float(v) for v in row[1:]])
    return np.array(feats, dtype=np.float32), np.array(labels)


def train(log=print, test_size=0.2, epochs=500):
    """학습 후 모델을 저장하고 반환한다. 진행 상황은 log(str) 로 전달."""
    X, y = load_data()
    counts = Counter(y)
    log("클래스별 샘플 수:")
    for label, n in sorted(counts.items()):
        log(f"  {label:<15} {n}")

    if len(counts) < 2:
        raise ValueError("클래스가 2개 이상 필요합니다. (예: 원하는 제스처 + none)")
    if min(counts.values()) < 10:
        raise ValueError("샘플이 10개 미만인 클래스가 있습니다. 클래스당 200개 이상을 권장합니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=epochs,
                      early_stopping=True, random_state=42),
    )
    log("\n학습 중...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    classes = model.classes_
    log("\n=== 테스트 결과 ===")
    log(classification_report(y_test, y_pred, digits=3))
    log("혼동행렬 (행=정답, 열=예측):")
    log("  ".join(f"{c[:8]:>8}" for c in [""] + list(classes)))
    for label, row in zip(classes, confusion_matrix(y_test, y_pred, labels=classes)):
        log("  ".join(f"{v:>8}" for v in [label[:8]] + list(row)))

    # 최종 모델은 전체 데이터로 다시 학습
    model.fit(X, y)
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CLASSIFIER_PATH)
    log(f"\n모델 저장 완료: {CLASSIFIER_PATH}")

    import export_web_model
    export_web_model.export(log)
    return model


if __name__ == "__main__":
    train()
