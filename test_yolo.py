from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8m.pt")
folder_path = "images"

animal_classes = {
    "dog", "cat", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "bird"
}

happy_emoji = cv2.imread("emojis/happy.png", cv2.IMREAD_UNCHANGED)
normal_emoji = cv2.imread("emojis/normal.png", cv2.IMREAD_UNCHANGED)
sleep_emoji = cv2.imread("emojis/sleep.png", cv2.IMREAD_UNCHANGED)
crown = cv2.imread("emojis/crown.png", cv2.IMREAD_UNCHANGED)

def overlay_png(background, emoji, x, y, w, h):
    if emoji is None:
        return background
    emoji = cv2.resize(emoji, (w, h))
    try:
        b,g,r,a = cv2.split(emoji)
    except:
        return background
    mask = cv2.merge((a,a,a))
    emoji_bgr = cv2.merge((b,g,r))
    if y < 0: y = 0
    if x < 0: x = 0
    if y+h > background.shape[0] or x+w > background.shape[1]:
        return background
    roi = background[y:y+h, x:x+w]
    mask = mask.astype(float)/255
    emoji_bgr = emoji_bgr.astype(float)
    roi = roi.astype(float)
    fg = cv2.multiply(mask, emoji_bgr)
    bg = cv2.multiply(1.0 - mask, roi)
    out = cv2.add(fg, bg)
    background[y:y+h, x:x+w] = out
    return background


leaderboard = []
image_records = {}

for file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, file)

    if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_check = cv2.imread(img_path)
    if img_check is None:
        print("Skipping unreadable:", img_path)
        continue

    print(f"\nProcessing: {file}")
    results = model(img_path)

    for r in results:
        img = r.orig_img
        boxes = r.boxes

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

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            aspect_ratio = height / float(width)

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
            fun_score = min(fun_score, 100)

            print("\n----------------------------------")
            print(f"Animal       : {label}")
            print(f"Confidence   : {conf:.2f}")
            print(f"Total Animals: {total_animals}")
            print(f"Posture      : {posture}")
            print("---- SCORE CONTRIBUTION ----")
            print(f"Confidence Score  : {conf_score:.2f} / 70")
            print(f"Crowd Bonus       : {crowd_bonus:.2f} / 20")
            print(f"Posture Score     : {posture_score} / 20")
            print(f"=> FINAL FUN SCORE: {fun_score}")
            print("----------------------------------")

            animal_data.append({
                "score": fun_score,
                "label": label,
                "posture": posture,
                "box": (x1, y1, x2, y2),
                "file": file
            })

            leaderboard.append({
                "animal": label,
                "score": fun_score,
                "image": file,
                "box": (x1, y1, x2, y2)
            })

        if animal_data:
            best = max(animal_data, key=lambda x: x["score"])
            image_records[file] = (img, animal_data)

            for a in animal_data:
                x1, y1, x2, y2 = a["box"]
                color = (0,0,255) if a == best else (0,255,0)

                if a["posture"] == "Standing / Active":
                    emoji_img = happy_emoji
                elif a["posture"] == "Neutral":
                    emoji_img = normal_emoji
                else:
                    emoji_img = sleep_emoji


                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(
                    img,
                    f"{a['label']} | Fun:{a['score']} | {a['posture']}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                emoji_size = int((y2 - y1) * 0.25)
                emoji_x = x1
                emoji_y = y1 - emoji_size if y1 - emoji_size > 0 else y1
                img = overlay_png(img, emoji_img, emoji_x, emoji_y, emoji_size, emoji_size)

        save_name = f"output_{file}"
        cv2.imwrite(save_name, img)
        print("Saved:", save_name)

        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)

print("\n========= LEADERBOARD =========")
for i, a in enumerate(leaderboard[:5], start=1):
    print(f"{i}. {a['animal']} | Score: {a['score']} | Image: {a['image']}")

winner = leaderboard[0]
print("\n👑 MOST PLAYFUL ANIMAL 👑")
print(f"{winner['animal']} | Score: {winner['score']} | Image: {winner['image']}")

champ_img, animals = image_records[winner["image"]]

for a in animals:
    if a["score"] == winner["score"]:
        x1, y1, x2, y2 = a["box"]
        crown_w = int((x2 - x1) * 0.6)
        crown_h = int(crown_w * 0.6)
        crown_x = x1 + int((x2 - x1)/4)
        crown_y = y1 - crown_h
        champ_img = overlay_png(champ_img, crown, crown_x, crown_y, crown_w, crown_h)

cv2.putText(champ_img, "GLOBAL LEADERBOARD", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

y_offset = 80
for i, a in enumerate(leaderboard[:5]):
    text = f"{i+1}. {a['animal']} | Score: {a['score']} | {a['image']}"
    cv2.putText(champ_img, text, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    y_offset += 35

cv2.imshow("GLOBAL CHAMPION", champ_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("GLOBAL_CHAMPION.png", champ_img)
print("Saved GLOBAL_CHAMPION.png")
