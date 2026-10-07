// Gesture Reactions - 학습한 커스텀 제스처를 브라우저에서 인식하고 반응(로고/이모지)을 띄운다.
import { FilesetResolver, HandLandmarker } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const HAND_MODEL_URL = "hand_landmarker.task";
const CLASSIFIER_URL = "model.json";

// 최근 N프레임 중 MIN_VOTES 이상 같은 제스처일 때만 반응 (깜빡임 방지)
const HISTORY = 8;
const MIN_VOTES = 5;

const NIKE_SVG = `
<svg viewBox="0 0 100 44" xmlns="http://www.w3.org/2000/svg">
  <path fill="#fff" d="M98 4 L23 37 C11 42 1 38 9 26 C5 33 9 37 19 33 Z"/>
</svg>`;

// 제스처 이름 -> 화면에 띄울 반응. 새 제스처를 학습했다면 여기에 추가하면 된다.
const REACTIONS = {
  nike: { html: NIKE_SVG, caption: "NIKE" },
  ok: { html: `<span class="emoji">👌</span>`, caption: "OK" },
};

const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const reactionEl = $("reaction");

let landmarker = null;
let classifier = null;
let history = [];
let shownLabel = null;
let lastVideoTime = -1;
let fps = 0, lastFrame = performance.now();

// ---------------------------------------------------------------- 분류기
// Python 의 gesture_features.landmarks_to_features 와 동일해야 한다.
function toFeatures(landmarks, handedness) {
  const o = landmarks[0];
  const pts = landmarks.map((p) => [p.x - o.x, p.y - o.y, p.z - o.z]);
  let scale = 0;
  for (const [x, y] of pts) scale = Math.max(scale, Math.hypot(x, y));
  const out = [];
  for (const [x, y, z] of pts) {
    const s = scale > 0 ? scale : 1;
    out.push((handedness === "Left" ? -x : x) / s, y / s, z / s);
  }
  return out;
}

// sklearn StandardScaler + MLPClassifier(relu, softmax) 순전파
function predict(features) {
  const { mean, scale, layers } = classifier;
  let a = features.map((v, i) => (v - mean[i]) / scale[i]);
  layers.forEach(({ W, b }, li) => {
    const out = b.slice();
    for (let i = 0; i < a.length; i++) {
      const ai = a[i], row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
    }
    a = li < layers.length - 1 ? out.map((v) => Math.max(0, v)) : softmax(out);
  });
  return a;
}

function softmax(v) {
  const m = Math.max(...v);
  const e = v.map((x) => Math.exp(x - m));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map((x) => x / s);
}

// ---------------------------------------------------------------- UI
function buildBars() {
  $("bars").innerHTML = classifier.classes.map((c) => `
    <div class="bar-row"><span>${c}</span><div class="bar"><i id="bar-${c}"></i></div><span id="val-${c}">0.00</span></div>
  `).join("");
}

function updateBars(probs) {
  classifier.classes.forEach((c, i) => {
    const p = probs ? probs[i] : 0;
    $(`bar-${c}`).style.width = `${p * 100}%`;
    $(`val-${c}`).textContent = p.toFixed(2);
  });
}

function showReaction(label, anchor) {
  const r = REACTIONS[label];
  if (!r) {
    reactionEl.classList.remove("show");
    shownLabel = null;
    return;
  }
  if (shownLabel !== label) {
    reactionEl.innerHTML = `${r.html}<div class="caption">${r.caption}</div>`;
    reactionEl.classList.remove("show");
    void reactionEl.offsetWidth; // 애니메이션 재시작
    shownLabel = label;
  }
  reactionEl.classList.add("show");
  if (anchor) placeOverCanvas(anchor);
}

// 캔버스 좌표(0~1)를 실제 화면 위치로 변환 (object-fit: contain 고려)
function placeOverCanvas({ x, y }) {
  const stage = $("stage").getBoundingClientRect();
  const ratio = Math.min(stage.width / canvas.width, stage.height / canvas.height);
  const w = canvas.width * ratio, h = canvas.height * ratio;
  const left = (stage.width - w) / 2 + x * w;
  const top = (stage.height - h) / 2 + y * h;
  reactionEl.style.left = `${left}px`;
  reactionEl.style.top = `${Math.max(top - 10, 140)}px`;
}

function drawHand(landmarks) {
  const w = canvas.width, h = canvas.height;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#c6ff3d";
  for (const { start, end } of HAND_CONNECTIONS) {
    const a = landmarks[start], b = landmarks[end];
    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff4d6d";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------------------------------------------------------------- 루프
function loop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // Python 버전과 같게 '좌우반전된 영상'을 모델에 넣는다
    ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const result = landmarker.detectForVideo(canvas, performance.now());
    const thr = parseFloat($("thr").value);

    let frameLabel = "none", anchor = null, firstProbs = null, bestScore = 0;
    result.landmarks.forEach((lms, i) => {
      drawHand(lms);
      const handed = result.handedness[i][0].categoryName;
      const probs = predict(toFeatures(lms, handed));
      if (!firstProbs) firstProbs = probs;
      const k = probs.indexOf(Math.max(...probs));
      const label = probs[k] >= thr ? classifier.classes[k] : "Unknown";
      if (REACTIONS[label] && probs[k] > bestScore) {
        bestScore = probs[k];
        frameLabel = label;
        const xs = lms.map((p) => p.x), ys = lms.map((p) => p.y);
        anchor = { x: (Math.min(...xs) + Math.max(...xs)) / 2, y: Math.min(...ys) };
      } else if (frameLabel === "none" && i === 0) {
        frameLabel = label;
      }
    });

    history.push(frameLabel);
    if (history.length > HISTORY) history.shift();
    const votes = history.filter((l) => l === frameLabel).length;
    const stable = votes >= MIN_VOTES ? frameLabel : (shownLabel && history.includes(shownLabel) ? shownLabel : "none");

    showReaction(stable, anchor);
    $("current").textContent = result.landmarks.length ? stable : "손 없음";
    updateBars(firstProbs);

    const now = performance.now();
    fps = fps * 0.9 + (1000 / (now - lastFrame)) * 0.1;
    lastFrame = now;
    $("status").textContent = `실행 중 · 손 ${result.landmarks.length}개 · ${fps.toFixed(0)} FPS`;
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- 시작
async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  try {
    $("status").textContent = "모델 불러오는 중...";
    classifier = await (await fetch(CLASSIFIER_URL)).json();
    buildBars();
    try {
      landmarker = await createLandmarker("GPU");
    } catch {
      landmarker = await createLandmarker("CPU");
    }

    $("status").textContent = "카메라 연결 중...";
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    $("start").remove();
    loop();
  } catch (err) {
    console.error(err);
    $("status").textContent = `오류: ${err.message}`;
    btn.disabled = false;
  }
}

$("thr").addEventListener("input", (e) => ($("thrVal").textContent = (+e.target.value).toFixed(2)));
$("startBtn").addEventListener("click", start);
