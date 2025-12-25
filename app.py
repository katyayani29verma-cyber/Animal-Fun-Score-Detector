import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "opencv-python-headless==4.8.1.78",
    "ultralytics"
])

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Animal Fun Score Predictor 🐶", layout="wide")
st.title("🐶 Animal Fun Score Predictor")
st.write("Upload pet images and let AI find who is the most playful! 🎉")

model = YOLO("yolov8m.pt")

animal_classes = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "bird"
}

def load_png(path):
    try:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except:
        return None

happy_emoji = load_png("emojis/happy.png")
normal_emoji = load_png("emojis/normal.png")
sleep_emoji = load_png("emojis/sleep.png")
crown = load_png("emojis/crown.png")

def overlay_png(bg, emoji, x, y, w, h):
    if emoji is None:
        return bg
    emoji = cv2.resize(emoji, (w, h))
    try:
        b,g,r,a = cv2.split(emoji)
    except:
        return bg
    mask = cv2.merge((a,a,a))
    emoji_bgr = cv2.merge((b,g,r))
    if y < 0: y = 0
    if x < 0: x = 0
    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        return bg
    roi = bg[y:y+h, x:x+w].astype(float)
    mask = (mask.astype(float) / 255)
    emoji_bgr = emoji_bgr.astype(float)
    fg = cv2.multiply(mask, emoji_bgr)
    bg2 = cv2.multiply(1.0 - mask, roi)
    out = cv2.add(fg, bg2)
    bg[y:y+h, x:x+w] = out
    return bg

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

leaderboard = []
processed_images = {}
image_animals = {}

if st.button("Process Images"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        for file in uploaded_files:
            img = Image.open(file)
            img = np.array(img)
            results = model(img)[0]
            boxes = results.boxes
            total_animals = sum(
                1 for box in boxes
                if model.names[int(box.cls[0])] in animal_classes
            )
            animal_data = []
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                if label not in animal_classes:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                width = x2-x1
                height = y2-y1
                aspect_ratio = height/float(width)
                if aspect_ratio > 1.2:
                    posture = "Standing / Active"
                    posture_score = 20
                elif aspect_ratio < 0.8:
                    posture = "Lying / Sitting"
                    posture_score = 5
                else:
                    posture = "Neutral"
                    posture_score = 10
                conf_score = conf * 70
                crowd_bonus = min(total_animals * 5, 20)
                fun_score = int(conf_score + crowd_bonus + posture_score)
                fun_score = min(fun_score,100)
                animal_data.append({
                    "score": fun_score,
                    "label": label,
                    "posture": posture,
                    "box": (x1,y1,x2,y2)
                })
                leaderboard.append({
                    "animal": label,
                    "score": fun_score,
                    "image": file.name
                })
            if animal_data:
                best = max(animal_data, key=lambda x:x["score"])
                image_animals[file.name] = animal_data
                for a in animal_data:
                    x1,y1,x2,y2 = a["box"]
                    color = (0,0,255) if a == best else (0,255,0)
                    if a["posture"] == "Standing / Active":
                        emoji = happy_emoji
                    elif a["posture"] == "Neutral":
                        emoji = normal_emoji
                    else:
                        emoji = sleep_emoji
                    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
                    cv2.putText(
                        img,
                        f"{a['label']} | Fun:{a['score']} | {a['posture']}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    emoji_size = int((y2-y1)*0.25)
                    emoji_y = y1-emoji_size if y1-emoji_size>0 else y1
                    img = overlay_png(img, emoji, x1, emoji_y, emoji_size, emoji_size)
            processed_images[file.name] = img
        leaderboard.sort(key=lambda x:x["score"], reverse=True)

        st.subheader("🏆 Leaderboard")
        for i,a in enumerate(leaderboard[:5],start=1):
            st.write(f"{i}. **{a['animal']}** — Score: **{a['score']}** — Image: {a['image']}")

        winner = leaderboard[0]
        champ_img = processed_images[winner["image"]]

        for a in image_animals[winner["image"]]:
            if a["score"] == winner["score"]:
                x1,y1,x2,y2 = a["box"]
                crown_w = int((x2-x1)*0.6)
                crown_h = int(crown_w*0.6)
                crown_x = x1 + int((x2-x1)/4)
                crown_y = y1 - crown_h
                champ_img = overlay_png(champ_img, crown, crown_x, crown_y, crown_w, crown_h)

        st.subheader("👑 Global Champion")
        st.image(champ_img, caption="Most Playful Animal", use_container_width=True)

        st.subheader("📸 All Processed Images")
        for name,img in processed_images.items():
            st.image(img, caption=name, use_container_width=True)
