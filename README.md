# MediaPipe Gesture Reactions

MediaPipe 손 랜드마크로 나만의 제스처를 수집·학습하고, 웹에서 실시간으로 인식해 반응(로고/이모지)을 띄우는 프로젝트.

**데모:** https://shain1912.github.io/mediapipe-gesture-reactions/

| 제스처 | 반응 |
|---|---|
| `nike` | Nike 로고 |
| `ok` | 👌 |

## 구성

| 파일 | 설명 |
|---|---|
| `gesture_studio.py` | 수집 / 학습 / 인식 데스크톱 UI (tkinter) |
| `train_gestures.py` | 분류기 학습 (scikit-learn MLP), 학습 후 웹 모델 자동 내보내기 |
| `gesture_features.py` | 랜드마크 → 특징 벡터 변환 (공통) |
| `export_web_model.py` | `models/custom_gesture.joblib` → `web/model.json` |
| `serve_web.py` | 웹 버전 로컬 실행 |
| `web/` | 웹 버전 (GitHub Pages 배포 대상) |
| `collect_gestures.py`, `custom_gesture_webcam.py` | 수집 / 인식 CLI 버전 (UI 이전 버전) |
| `hand_webcam.py`, `gesture_webcam.py` | MediaPipe 기본 손 랜드마크 / 제스처 인식 예제 |

## 설치

```
pip install mediapipe opencv-python scikit-learn pillow
```

## 사용법

1. `python gesture_studio.py`
2. **① 수집** 탭: 제스처 이름 입력 → 녹화 시작(SPACE). 제스처당 200~500개, `none` 클래스도 꼭 수집
3. **② 학습** 탭: 학습 시작 → `models/custom_gesture.joblib`, `web/model.json` 생성
4. **③ 인식** 탭에서 확인하거나 `python serve_web.py` 로 웹 버전 실행

웹 반응을 추가하려면 `web/app.js` 의 `REACTIONS` 에 항목을 추가한다.

`main` 브랜치의 `web/` 이 바뀌면 GitHub Actions 가 자동으로 Pages 에 배포한다.
