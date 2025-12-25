import cv2
import numpy as np
import os
from ultralytics import YOLO
import gradio as gr

# -----------------------------
# Load Model
# -----------------------------
model = YOLO("yolov8m.pt")

# -----------------------------
# Safe path to emoji images
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

happy_emoji = cv2.imread(os.path.join(BASE_DIR, "emojis/happy.png"), cv2.IMREAD_UNCHANGED)
sad_emoji   = cv2.imread(os.path.join(BASE_DIR, "emojis/sad.png"), cv2.IMREAD_UNCHANGED)
crown       = cv2.imread(os.path.join(BASE_DIR, "emojis/crown.png"), cv2.IMREAD_UNCHANGED)

print("Happy emoji loaded:", happy_emoji is not None)
print("Sad emoji loaded:", sad_emoji is not None)
print("Crown loaded:", crown is not None)


# -----------------------------
# Helper: Overlay PNG With Alpha
# -----------------------------
def overlay_png(background, overlay, x, y, w, h):
    if overlay is None:
        return background

    overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        mask = a / 255.0
        overlay_rgb = cv2.merge((b, g, r))
    else:
        mask = np.ones((overlay.shape[0], overlay.shape[1]))
        overlay_rgb = overlay

    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    if y1 >= y2 or x1 >= x2:
        return background

    overlay_crop = overlay_rgb[:y2-y1, :x2-x1]
    mask = mask[:y2-y1, :x2-x1]

    region = background[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = (overlay_crop * mask[..., None] +
                                region * (1 - mask[..., None])).astype(np.uint8)

    return background


# -----------------------------
# Main Processing Function
# -----------------------------
def process_image(image):
    img = np.array(image)
    results = model(img)[0]

    fun_scores = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if label not in ["dog", "cat", "horse", "sheep", "cow"]:
            continue

        score = np.random.randint(60, 100)
        fun_scores.append(score)

        emoji = happy_emoji if score > 80 else sad_emoji
        emoji_size = int((y2 - y1) * 0.25)
        emoji_y = y1 - emoji_size if y1 - emoji_size > 0 else y1

        img = overlay_png(img, emoji, x1, emoji_y, emoji_size, emoji_size)

        text = f"{label} | Fun:{score}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if fun_scores:
        best = np.argmax(fun_scores)
        x1, y1, x2, y2 = map(int, results.boxes[best].xyxy[0])

        cw = int((x2-x1)*0.6)
        ch = int(cw*0.6)
        cx = x1 + int((x2-x1)/4)
        cy = y1 - ch

        img = overlay_png(img, crown, cx, cy, cw, ch)

    return img


# -----------------------------
# Gradio UI
# -----------------------------
gr.Int
