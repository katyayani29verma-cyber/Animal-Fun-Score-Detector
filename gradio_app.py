import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import os

# ------------ SETTINGS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(BASE_DIR, "emojis")

happy_emoji = cv2.imread(os.path.join(EMOJI_DIR, "happy.png"), cv2.IMREAD_UNCHANGED)
normal_emoji = cv2.imread(os.path.join(EMOJI_DIR, "normal.png"), cv2.IMREAD_UNCHANGED)
sleep_emoji = cv2.imread(os.path.join(EMOJI_DIR, "sleep.png"), cv2.IMREAD_UNCHANGED)
crown = cv2.imread(os.path.join(EMOJI_DIR, "crown.png"), cv2.IMREAD_UNCHANGED)

model = YOLO("yolov8m.pt")

# ---------------- FUNCTIONS -----------------

def overlay_png(img, png, x, y, w, h):
    png = cv2.resize(png, (w, h))
    b, g, r, a = cv2.split(png)
    mask = a / 255.0
    h_img, w_img, _ = img.shape

    if y < 0: y = 0
    if x < 0: x = 0
    if x + w > w_img: w = w_img - x
    if y + h > h_img: h = h_img - y

    roi = img[y:y+h, x:x+w]
    if roi.shape[0] <= 0 or roi.shape[1] <= 0:
        return img

    png = png[:roi.shape[0], :roi.shape[1]]
    mask = mask[:roi.shape[0], :roi.shape[1]]
    inv = 1 - mask

    for c in range(3):
        roi[:, :, c] = (mask * png[:, :, c] + inv * roi[:, :, c])

    img[y:y+h, x:x+w] = roi
    return img


def process_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)[0]

    animals = []
    fun_scores = []

    for box in results.boxes:
        cls = int(box.cls.item())
        name = results.names[cls]

        if name not in ["dog", "cat"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)

        # -------- FUN SCORE LOGIC ----------
        base_score = conf * 70
        crowd_bonus = min(len(results.boxes) * 5, 20)
        posture_score = 20  # simplified

        final_score = int(base_score + crowd_bonus + posture_score)

        animals.append((x1, y1, x2, y2, name, final_score))

    if not animals:
        return image, "No animals detected"

    # -------- DETERMINE WINNER ----------
    animals.sort(key=lambda x: x[5], reverse=True)
    winner = animals[0]

    summaries = []

    for (x1, y1, x2, y2, name, score) in animals:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
        cv2.putText(img, f"{name} | Fun:{score}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Emoji placement
        emoji_size = int((y2-y1)*0.25)
        emoji_y = y1-emoji_size if y1-emoji_size > 0 else y1

        if score >= 80:
            img = overlay_png(img, happy_emoji, x1, emoji_y, emoji_size, emoji_size)
        elif score >= 60:
            img = overlay_png(img, normal_emoji, x1, emoji_y, emoji_size, emoji_size)
        else:
            img = overlay_png(img, sleep_emoji, x1, emoji_y, emoji_size, emoji_size)

        # Save summary text
        summaries.append(f"{name} | Fun Score: {score}")

    # -------- CROWN ON WINNER ----------
    wx1, wy1, wx2, wy2, _, wscore = winner
    cw = int((wx2-wx1)*0.6)
    ch = int(cw*0.6)
    cx = wx1 + int(((wx2-wx1)/4))
    cy = wy1 - ch
    img = overlay_png(img, crown, cx, cy, cw, ch)

    # -------- SUMMARY UI TEXT ----------
    summary_text = "\n".join(summaries)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, summary_text


# ---------------- GRADIO UI -----------------

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Output Image"),
        gr.Textbox(label="Fun Score Report", lines=10)
    ],
    title="🐶 Animal Fun Score Predictor",
    description="Upload animal image → Get Fun Score + Emoji + Crown 👑"
)

demo.launch(share=True)
