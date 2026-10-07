"""Gesture Studio - 나만의 제스처 수집 / 학습 / 인식을 한 화면에서

실행: python gesture_studio.py

① 수집 탭: 제스처 이름 입력 -> [녹화 시작] (또는 SPACE) -> 목표 개수에 도달하면 자동 정지
② 학습 탭: [학습 시작] -> 결과 리포트 확인
③ 인식 탭: 실시간으로 제스처 인식 결과와 클래스별 확률 표시
"""
import csv
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont, ImageTk

import train_gestures
from gesture_features import (CLASSIFIER_PATH, DATA_PATH, HAND_CONNECTIONS,
                              HAND_MODEL_PATH, NUM_FEATURES, landmarks_to_features)

VIDEO_W, VIDEO_H = 640, 480
SAVE_INTERVAL = 0.05  # 수집 시 저장 간격(초)
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 화면에 한글 라벨 표시용


class CameraWorker(threading.Thread):
    """웹캠 캡처 + 손 랜드마크 검출을 백그라운드에서 수행하고 최신 결과만 보관"""

    def __init__(self, camera_index):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.running = True
        self.lock = threading.Lock()
        self.latest = None  # (frame_rgb, result, seq)
        self.error = None
        self._seq = 0

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error = f"웹캠({self.camera_index})을 열 수 없습니다."
            return
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            start = time.monotonic()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))
                with self.lock:
                    self._seq += 1
                    self.latest = (rgb, result, self._seq)
        cap.release()

    def stop(self):
        self.running = False


class GestureStudio:
    def __init__(self, root):
        self.root = root
        root.title("Gesture Studio")
        root.resizable(False, False)

        self.worker = None
        self.last_seq = 0
        self.last_saved = 0.0
        self.recording = False
        self.counts = self._load_counts()
        self.model = None
        self.prob_bars = {}
        self.log_queue = queue.Queue()
        self.training = False
        self.fps = 0.0
        self.last_frame_time = time.time()
        try:
            self.font = ImageFont.truetype(FONT_PATH, 22)
        except OSError:
            self.font = ImageFont.load_default()

        self._build_ui()
        self._refresh_label_list()
        self._load_model(silent=True)
        self._start_camera()

        root.bind("<space>", self._on_space)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_loop()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # 왼쪽: 영상
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")
        self.video_label = ttk.Label(left)
        self.video_label.pack()
        self.video_label.configure(image=self._blank_image())

        cam_bar = ttk.Frame(left)
        cam_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(cam_bar, text="카메라 번호").pack(side="left")
        self.camera_var = tk.IntVar(value=0)
        ttk.Spinbox(cam_bar, from_=0, to=9, width=4, textvariable=self.camera_var).pack(side="left", padx=4)
        ttk.Button(cam_bar, text="카메라 변경", command=self._start_camera).pack(side="left")
        self.fps_label = ttk.Label(cam_bar, text="")
        self.fps_label.pack(side="right")

        # 오른쪽: 탭
        self.tabs = ttk.Notebook(main, width=380)
        self.tabs.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.tabs.add(self._build_collect_tab(), text=" ① 수집 ")
        self.tabs.add(self._build_train_tab(), text=" ② 학습 ")
        self.tabs.add(self._build_infer_tab(), text=" ③ 인식 ")
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.status_var = tk.StringVar(value="준비")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=(6, 2)).pack(fill="x", side="bottom")

    def _build_collect_tab(self):
        f = ttk.Frame(self.tabs, padding=10)

        ttk.Label(f, text="제스처 이름").grid(row=0, column=0, sticky="w")
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(f, textvariable=self.label_var, width=22)
        self.label_combo.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(f, text="목표 개수").grid(row=1, column=0, sticky="w")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(f, from_=50, to=5000, increment=50, width=8,
                    textvariable=self.target_var).grid(row=1, column=1, sticky="w", pady=2)

        self.rec_button = tk.Button(f, text="● 녹화 시작  (SPACE)", font=("맑은 고딕", 12, "bold"),
                                    bg="#e8e8e8", height=2, command=self._toggle_record)
        self.rec_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 4))

        self.rec_progress = ttk.Progressbar(f, maximum=300)
        self.rec_progress.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.rec_info = ttk.Label(f, text="")
        self.rec_info.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 8))

        ttk.Label(f, text="수집된 데이터").grid(row=5, column=0, columnspan=2, sticky="w")
        self.tree = ttk.Treeview(f, columns=("count",), height=9)
        self.tree.heading("#0", text="제스처")
        self.tree.heading("count", text="샘플 수")
        self.tree.column("#0", width=200)
        self.tree.column("count", width=100, anchor="e")
        self.tree.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        ttk.Button(f, text="선택한 제스처 삭제", command=self._delete_label).grid(
            row=7, column=0, columnspan=2, sticky="e", pady=4)

        tip = ("팁\n"
               "• 제스처당 200~500개 권장\n"
               "• 'none'(평상시 손 모양) 클래스를 꼭 수집하세요\n"
               "• 녹화 중 손을 조금씩 움직이고 기울이고 거리를 바꿔주세요")
        ttk.Label(f, text=tip, foreground="#555", justify="left").grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        f.columnconfigure(1, weight=1)
        return f

    def _build_train_tab(self):
        f = ttk.Frame(self.tabs, padding=10)
        self.train_button = tk.Button(f, text="학습 시작", font=("맑은 고딕", 12, "bold"),
                                      height=2, command=self._start_training)
        self.train_button.pack(fill="x")
        self.train_progress = ttk.Progressbar(f, mode="indeterminate")
        self.train_progress.pack(fill="x", pady=6)
        self.train_log = ScrolledText(f, width=46, height=26, font=("Consolas", 9), state="disabled")
        self.train_log.pack(fill="both", expand=True)
        return f

    def _build_infer_tab(self):
        f = ttk.Frame(self.tabs, padding=10)
        self.model_info = ttk.Label(f, text="", foreground="#555")
        self.model_info.pack(anchor="w")

        ttk.Label(f, text="인식 결과").pack(anchor="w", pady=(10, 0))
        self.pred_var = tk.StringVar(value="-")
        ttk.Label(f, textvariable=self.pred_var, font=("맑은 고딕", 24, "bold"),
                  foreground="#0a6").pack(anchor="w")

        thr = ttk.Frame(f)
        thr.pack(fill="x", pady=(10, 0))
        ttk.Label(thr, text="확신도 기준").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.7)
        self.thr_text = ttk.Label(thr, text="0.70", width=5)
        self.thr_text.pack(side="right")
        ttk.Scale(thr, from_=0.3, to=0.99, variable=self.thr_var,
                  command=lambda v: self.thr_text.configure(text=f"{float(v):.2f}")).pack(
            side="left", fill="x", expand=True, padx=6)
        ttk.Label(f, text="기준 미만이면 Unknown 으로 표시", foreground="#777").pack(anchor="w")

        ttk.Label(f, text="클래스별 확률 (첫 번째 손)").pack(anchor="w", pady=(14, 4))
        self.prob_frame = ttk.Frame(f)
        self.prob_frame.pack(fill="x")

        ttk.Button(f, text="모델 다시 불러오기", command=self._load_model).pack(anchor="e", pady=10)
        return f

    def _blank_image(self):
        self._blank = ImageTk.PhotoImage(Image.new("RGB", (VIDEO_W, VIDEO_H), (30, 30, 30)))
        return self._blank

    # ------------------------------------------------------------ camera
    def _start_camera(self):
        if self.recording:
            self._toggle_record()
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=2)
        self.last_seq = 0
        self.worker = CameraWorker(self.camera_var.get())
        self.worker.start()
        self.status_var.set(f"카메라 {self.camera_var.get()} 연결 중...")

    def _update_loop(self):
        w = self.worker
        if w and w.error:
            self.status_var.set(w.error)
        elif w:
            with w.lock:
                latest = w.latest
            if latest and latest[2] != self.last_seq:
                self.last_seq = latest[2]
                self._process_frame(latest[0], latest[1])
        self._drain_log()
        self.root.after(10, self._update_loop)

    def _process_frame(self, rgb, result):
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now
        self.fps_label.configure(text=f"FPS {self.fps:4.1f}")

        img = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H))
        draw = ImageDraw.Draw(img)
        tab = self.tabs.index(self.tabs.select())

        hands = list(zip(result.hand_landmarks, result.handedness))
        predictions = []
        for i, (landmarks, handed) in enumerate(hands):
            pts = [(lm.x * VIDEO_W, lm.y * VIDEO_H) for lm in landmarks]
            for c in HAND_CONNECTIONS:
                draw.line([pts[c.start], pts[c.end]], fill=(0, 255, 0), width=3)
            for x, y in pts:
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 60))

            feats = landmarks_to_features(landmarks, handed[0].category_name)

            if tab == 0 and i == 0 and self.recording:
                self._save_sample(feats)
            elif tab == 2 and self.model is not None:
                probs = self.model.predict_proba(feats.reshape(1, -1))[0]
                best = probs.argmax()
                label = self.model.classes_[best] if probs[best] >= self.thr_var.get() else "Unknown"
                predictions.append((label, probs))
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                text = f"{label} {probs[best]:.2f}"
                tb = draw.textbbox((x0, y0 - 32), text, font=self.font)
                draw.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=(0, 0, 0))
                draw.text((x0, y0 - 32), text, font=self.font, fill=(255, 255, 0))

        if tab == 0:
            if self.recording:
                draw.ellipse([VIDEO_W - 40, 15, VIDEO_W - 16, 39], fill=(255, 0, 0))
                draw.text((VIDEO_W - 95, 13), "REC", font=self.font, fill=(255, 0, 0))
            if not hands:
                draw.text((10, 10), "손이 보이지 않습니다", font=self.font, fill=(255, 200, 0))
        elif tab == 2:
            self._update_prediction_panel(predictions)

        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)
        if not self.recording and not self.training:
            self.status_var.set(f"카메라 {self.worker.camera_index} 동작 중 · 손 {len(hands)}개 검출")

    # ----------------------------------------------------------- collect
    def _load_counts(self):
        counts = Counter()
        if DATA_PATH.exists():
            with open(DATA_PATH, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        counts[row[0]] += 1
        return counts

    def _refresh_label_list(self):
        self.tree.delete(*self.tree.get_children())
        for label, n in sorted(self.counts.items()):
            self.tree.insert("", "end", iid=label, text=label, values=(n,))
        self.label_combo["values"] = sorted(self.counts)

    def _on_tree_select(self, _):
        sel = self.tree.selection()
        if sel and not self.recording:
            self.label_var.set(sel[0])

    def _on_space(self, event):
        # 입력창에서 스페이스를 치는 경우는 무시
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text)):
            return
        if self.tabs.index(self.tabs.select()) == 0:
            self._toggle_record()

    def _toggle_record(self):
        if self.recording:
            self.recording = False
            self.data_file.close()
            self.rec_button.configure(text="● 녹화 시작  (SPACE)", bg="#e8e8e8", fg="black")
            self.label_combo.configure(state="normal")
            self._refresh_label_list()
            label = self.label_var.get().strip()
            self.status_var.set(f"녹화 정지 · '{label}' {self.counts[label]}개")
            return

        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("제스처 이름", "제스처 이름을 입력하세요. (예: heart, none)")
            return
        self.label_var.set(label)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_file = not DATA_PATH.exists()
        self.data_file = open(DATA_PATH, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.data_file)
        if new_file:
            self.writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        self.recording = True
        self.rec_button.configure(text="■ 녹화 정지  (SPACE)", bg="#d33", fg="white")
        self.label_combo.configure(state="disabled")
        self.root.focus_set()
        self._update_rec_progress()

    def _save_sample(self, feats):
        now = time.monotonic()
        if now - self.last_saved < SAVE_INTERVAL:
            return
        self.last_saved = now
        label = self.label_var.get()
        self.writer.writerow([label] + [f"{v:.6f}" for v in feats])
        self.counts[label] += 1
        self._update_rec_progress()
        self.status_var.set(f"녹화 중 · '{label}' {self.counts[label]}개")
        if self.counts[label] >= self.target_var.get():
            self._toggle_record()
            self.status_var.set(f"목표 도달! '{label}' {self.counts[label]}개 수집 완료")

    def _update_rec_progress(self):
        label = self.label_var.get().strip()
        target = max(self.target_var.get(), 1)
        n = self.counts.get(label, 0)
        self.rec_progress.configure(maximum=target, value=min(n, target))
        self.rec_info.configure(text=f"'{label}' {n} / {target}")
        if self.tree.exists(label):
            self.tree.set(label, "count", n)

    def _delete_label(self):
        sel = self.tree.selection()
        if not sel or self.recording:
            return
        label = sel[0]
        if not messagebox.askyesno("삭제 확인", f"'{label}' 데이터 {self.counts[label]}개를 삭제할까요?\n되돌릴 수 없습니다."):
            return
        with open(DATA_PATH, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.reader(f) if row and row[0] != label]
        with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        del self.counts[label]
        self._refresh_label_list()
        self.status_var.set(f"'{label}' 삭제 완료")

    # ------------------------------------------------------------- train
    def _start_training(self):
        if self.training:
            return
        if self.recording:
            self._toggle_record()
        self.training = True
        self.train_button.configure(state="disabled", text="학습 중...")
        self.train_progress.start(10)
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", "end")
        self.train_log.configure(state="disabled")
        self.status_var.set("학습 중...")

        def job():
            try:
                train_gestures.train(log=self.log_queue.put)
                self.log_queue.put(("__done__", None))
            except Exception as e:
                self.log_queue.put(("__done__", str(e)))

        threading.Thread(target=job, daemon=True).start()

    def _drain_log(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            if isinstance(msg, tuple) and msg[0] == "__done__":
                self._on_training_done(msg[1])
                continue
            self.train_log.configure(state="normal")
            self.train_log.insert("end", msg + "\n")
            self.train_log.see("end")
            self.train_log.configure(state="disabled")

    def _on_training_done(self, error):
        self.training = False
        self.train_progress.stop()
        self.train_button.configure(state="normal", text="학습 시작")
        if error:
            messagebox.showerror("학습 실패", error)
            self.status_var.set("학습 실패")
            return
        self._load_model(silent=True)
        self.status_var.set("학습 완료 · ③ 인식 탭에서 확인하세요")
        if messagebox.askyesno("학습 완료", "학습이 끝났습니다. 인식 탭으로 이동할까요?"):
            self.tabs.select(2)

    # ------------------------------------------------------------- infer
    def _load_model(self, silent=False):
        if not CLASSIFIER_PATH.exists():
            self.model = None
            self.model_info.configure(text="학습된 모델이 없습니다. ② 학습 탭에서 먼저 학습하세요.")
            if not silent:
                messagebox.showinfo("모델 없음", "학습된 모델이 없습니다.")
            self._rebuild_prob_bars([])
            return
        self.model = joblib.load(CLASSIFIER_PATH)
        classes = list(self.model.classes_)
        self.model_info.configure(text=f"모델: {len(classes)}개 제스처 ({', '.join(classes)})")
        self._rebuild_prob_bars(classes)

    def _rebuild_prob_bars(self, classes):
        for child in self.prob_frame.winfo_children():
            child.destroy()
        self.prob_bars = {}
        for r, c in enumerate(classes):
            ttk.Label(self.prob_frame, text=c, width=14).grid(row=r, column=0, sticky="w")
            bar = ttk.Progressbar(self.prob_frame, maximum=1.0, length=200)
            bar.grid(row=r, column=1, pady=2)
            val = ttk.Label(self.prob_frame, text="0.00", width=5)
            val.grid(row=r, column=2)
            self.prob_bars[c] = (bar, val)

    def _update_prediction_panel(self, predictions):
        if not predictions:
            self.pred_var.set("손 없음")
            for bar, val in self.prob_bars.values():
                bar.configure(value=0)
                val.configure(text="0.00")
            return
        self.pred_var.set(" / ".join(p[0] for p in predictions))
        for c, p in zip(self.model.classes_, predictions[0][1]):
            bar, val = self.prob_bars[c]
            bar.configure(value=p)
            val.configure(text=f"{p:.2f}")

    # -------------------------------------------------------------- misc
    def _on_tab_changed(self, _):
        if self.recording and self.tabs.index(self.tabs.select()) != 0:
            self._toggle_record()
        if self.tabs.index(self.tabs.select()) == 2 and self.model is None:
            self._load_model(silent=True)

    def _on_close(self):
        if self.recording:
            self._toggle_record()
        if self.worker:
            self.worker.stop()
            self.worker.join(timeout=2)
        self.root.destroy()


def main():
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"손 랜드마크 모델이 없습니다: {HAND_MODEL_PATH}")
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
