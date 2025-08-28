import os
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "landmarks.csv"
MODEL_PATH = MODELS_DIR / "gesture_hgb.pkl"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"

GESTURES = ["palm", "fist", "thumbs_up", "okay", "peace"]
KEY_TO_LABEL = {ord(str(i)): i for i in range(len(GESTURES))}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

def extract_hand_features(landmarks, w, h):
    coords = np.array([[lm.x*w, lm.y*h, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0] 

    tips = [4, 8, 12, 16, 20]
    dists = [np.linalg.norm(coords[i]-coords[j]) for i in tips for j in tips if i < j]

    fingers = [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),
               (0,13,14,15,16),(0,17,18,19,20)]
    angles = []
    for f in fingers:
        for i in range(1, len(f)-1):
            angles.append(compute_angle(coords[f[i-1]], coords[f[i]], coords[f[i+1]]))

    return np.concatenate([coords.flatten(), dists, angles]).astype(np.float32)

def collect_data_once(min_conf=0.6, cam_index=0):
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        print("✅ Data already exists. Skipping collection.")
        return

    print("\n[COLLECT] Press 0-4 to label gesture, 'q' or ESC to quit.\n")
    active_label = None
    saved = 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=min_conf,
                        min_tracking_confidence=min_conf) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                if active_label is not None:
                    feats = extract_hand_features(hl.landmark, w, h)
                    row = np.concatenate([[active_label], feats])
                    cols = ["label"] + [f"f{i}" for i in range(len(feats))]
                    df = pd.DataFrame([row], columns=cols)
                    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
                    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)
                    saved += 1

            color = (0,255,0) if active_label is not None else (0,0,255)
            status = f"Active: {GESTURES[active_label]}" if active_label is not None else "Active: None"
            cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, "Press 0..4 | 'q' or ESC to quit", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Saved: {saved}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Collect", frame)
            key = cv2.waitKey(10) & 0xFF
            if key in KEY_TO_LABEL: active_label = KEY_TO_LABEL[key]
            elif key == ord('q') or key == 27: break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collection finished. Saved: {saved}")

def train_model():
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        print("❌ No data to train. Collect first.")
        return None

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("❌ Landmarks file is empty. Please collect data again.")
        return None

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=8)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=GESTURES)
    print("\nClassification Report:\n", report)
    joblib.dump({"model": clf, "gestures": GESTURES}, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"✅ Model saved to {MODEL_PATH}")
    print(f"📄 Report saved to {REPORT_PATH}")
    return clf

def realtime(clf, min_conf=0.6, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    last_pred = None
    stable_pred = None
    stable_count = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=min_conf,
                        min_tracking_confidence=min_conf) as hands:

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pred_label = None
            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                feats = extract_hand_features(hl.landmark, w, h).reshape(1,-1)
                pred = int(clf.predict(feats)[0])
                pred_label = GESTURES[pred]
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())

            if pred_label == last_pred and pred_label is not None:
                stable_count += 1
            else:
                stable_count = 0
            last_pred = pred_label
            if stable_count > 2 and pred_label is not None:
                stable_pred = pred_label

            label_to_show = stable_pred if stable_pred else (pred_label or "No hand")
            color = (0,255,0) if stable_pred else (0,200,255)
            cv2.putText(frame, f"Gesture: {label_to_show}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, "Press 'q' or ESC to quit", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow("Realtime", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27: break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    collect_data_once()          
    clf = train_model()          
    if clf: realtime(clf)
