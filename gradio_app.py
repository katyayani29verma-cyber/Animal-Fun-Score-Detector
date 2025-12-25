import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
from PIL import Image

model = YOLO("yolov8m.pt")

animal_classes = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "bird"
}

def load_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

happy_emoji = load_png("emojis/happy.png")
normal_emoji = load_png("emojis/normal.png")
sleep_emoji = load_png("emojis/sleep.png")
crown = load_png("emojis/crown.png")

def overlay_png(bg, emoji, x, y, w, h):
    if emoji is None:
        return bg
    emoji = cv2.resize(emoji, (w, h))
    b,g,r,a = cv2.split(emoji)
    mask = cv2.merge((a,a,a))
    emoji_bgr = cv2.merge((b,g,r))
    if y < 0: y = 0
    if x < 0: x = 0
    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        return bg
    roi = bg[y:y+h, x:x+w].astype(float)
    mask = mask.astype(float)/255
    emoji_bgr = emoji_bgr.astype(float)
    fg = cv2.multiply(mask, emoji_bgr)
    bg2 = cv2.multiply(1.0-mask, roi)
    bg[y:y+h, x:x+w] = cv2.add(fg,bg2)
    return bg

def process_image(image):
    img = np.array(image)
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
            emoji = happy_emoji
        elif aspect_ratio < 0.8:
            posture = "Lying / Sitting"
            posture_score = 5
            emoji = sleep_emoji
        else:
            posture = "Neutral"
            posture_score = 10
            emoji = normal_emoji

        conf_score = conf * 70
        crowd_bonus = min(total_animals * 5, 20)
        fun_score = min(int(conf_score + crowd_bonus + posture_score), 100)

        animal_data.append((fun_score, label, posture, (x1,y1,x2,y2), emoji))

    if not animal_data:
        return img

    best = max(animal_data)

    for score,label,posture,(x1,y1,x2,y2),emoji in animal_data:
        color = (0,0,255) if score==best[0] else (0,255,0)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(img,f"{label} | Fun:{score} | {posture}",
                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        emoji_size = int((y2-y1)*0.25)
        emoji_y = y1-emoji_size if y1-emoji_size>0 else y1
        img = overlay_png(img, emoji, x1, emoji_y, emoji_size, emoji_size)

        if score == best[0]:
            cw = int((x2-x1)*0.6)
            ch = int(cw*0.6)
            cx = x1 + int((x2-x1)/4)
            cy = y1 - ch
            img = overlay_png(img, crown, cx, cy, cw, ch)

    return img

gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(),
    title="🐶 Animal Fun Score Predictor",
    description="Upload animal image → Get Fun Score + Emoji + Crown 👑"
).launch()
