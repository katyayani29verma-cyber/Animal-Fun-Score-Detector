import cv2
import numpy as np
import os
import gradio as gr
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolov8m.pt")

happy = cv2.imread(os.path.join(BASE_DIR, "emojis/happy.png"), cv2.IMREAD_UNCHANGED)
normal = cv2.imread(os.path.join(BASE_DIR, "emojis/normal.png"), cv2.IMREAD_UNCHANGED)
sleep = cv2.imread(os.path.join(BASE_DIR, "emojis/sleep.png"), cv2.IMREAD_UNCHANGED)
crown = cv2.imread(os.path.join(BASE_DIR, "emojis/crown.png"), cv2.IMREAD_UNCHANGED)


# ---------------- FIXED PNG OVERLAY ---------------- #
def overlay_png(img, png, x, y, w, h):
    png = cv2.resize(png, (w, h))

    # If PNG has alpha
    if png.shape[2] == 4:
        b, g, r, a = cv2.split(png)
        mask = a / 255.0
        png_rgb = cv2.merge((b, g, r))
    else:
        png_rgb = png
        mask = np.ones((png.shape[0], png.shape[1]), dtype=float)

    h_img, w_img, _ = img.shape

    if y < 0: 
        y = 0
    if x < 0: 
        x = 0
    if x + w > w_img: 
        w = w_img - x
    if y + h > h_img: 
        h = h_img - y

    roi = img[y:y+h, x:x+w]
    if roi.shape[0] <= 0 or roi.shape[1] <= 0:
        return img

    png_rgb = png_rgb[:roi.shape[0], :roi.shape[1]]
    mask = mask[:roi.shape[0], :roi.shape[1]]
    inv = 1 - mask

    for c in range(3):
        roi[:, :, c] = (mask * png_rgb[:, :, c] + inv * roi[:, :, c])

    img[y:y+h, x:x+w] = roi
    return img


# --------------- MAIN PIPELINE ---------------- #
def process_image(image):

    img = np.array(image)
    results = model(img)[0]

    scores = []
    details_text = ""
    best = None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = int(box.cls[0])
        name = results.names[label]

        # Fun Score Logic
        conf_score = round(conf * 70, 2)
        crowd_score = min(len(results.boxes) * 5, 20)
        posture_score = 20
        final_score = int(conf_score + crowd_score + posture_score)

        scores.append((final_score, (x1, y1, x2, y2)))

        img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        img = cv2.putText(img, f"dog | Fun:{final_score}", (x1,y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        details_text += f"""
-----------------------------------
Animal            : {name}
Confidence        : {round(conf,2)}
Total Animals     : {len(results.boxes)}
Posture           : Standing / Active

---- SCORE CONTRIBUTION ----
Confidence Score  : {conf_score} / 70
Crowd Bonus       : {crowd_score} / 20
Posture Score     : 20 / 20

=> FINAL FUN SCORE: {final_score}
"""

        # Emoji Selection
        if final_score > 80:
            emoji = happy
        elif final_score > 60:
            emoji = normal
        else:
            emoji = sleep

        emoji_size = int((y2-y1)*0.25)
        emoji_y = y1-emoji_size if y1-emoji_size>0 else y1
        img = overlay_png(img, emoji, x1, emoji_y, emoji_size, emoji_size)

    if scores:
        best = max(scores, key=lambda x: x[0])
        score, (x1,y1,x2,y2) = best

        cw = int((x2-x1)*0.6)
        ch = int(cw*0.6)
        cx = x1 + int((x2-x1)/4)
        cy = y1 - ch

        img = overlay_png(img, crown, cx, cy, cw, ch)

    return img, details_text


# ------------------- GRADIO UI ---------------- #
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Result"),
        gr.Textbox(label="Fun Score Details", lines=18)
    ],
    title="🐶 Animal Fun Score Predictor",
    description="Upload animal image → Get Fun Score + Emoji + Crown 👑"
)

iface.launch(share=True)
