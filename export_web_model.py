"""학습된 분류기(models/custom_gesture.joblib)를 웹에서 쓸 수 있게 web/model.json 으로 내보낸다.

train_gestures.train() 이 끝나면 자동으로 호출된다. 단독 실행도 가능:
    python export_web_model.py
"""
import json
import shutil

import joblib

from gesture_features import BASE_DIR, CLASSIFIER_PATH, HAND_MODEL_PATH

WEB_DIR = BASE_DIR / "web"
WEB_MODEL_PATH = WEB_DIR / "model.json"


def export(log=print):
    model = joblib.load(CLASSIFIER_PATH)
    scaler, mlp = model.steps[0][1], model.steps[1][1]
    data = {
        "classes": [str(c) for c in model.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,
        "layers": [{"W": W.tolist(), "b": b.tolist()}
                   for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }
    WEB_DIR.mkdir(exist_ok=True)
    WEB_MODEL_PATH.write_text(json.dumps(data), encoding="utf-8")

    # 웹 페이지가 손 랜드마크 모델을 같은 폴더에서 불러오도록 복사
    web_hand_model = WEB_DIR / HAND_MODEL_PATH.name
    if not web_hand_model.exists():
        shutil.copy(HAND_MODEL_PATH, web_hand_model)

    log(f"웹 모델 내보내기 완료: {WEB_MODEL_PATH}")


if __name__ == "__main__":
    export()
