🐶 Animal Fun Score Predictor

This project is an AI-powered Animal Fun / Playfulness Score Detector that analyzes pets in images, detects animals using YOLO, evaluates their posture, assigns a Fun Score, overlays emojis, builds a leaderboard, and finally crowns the global champion with a 👑 crown.

It is fun, visually engaging, and a great example of Computer Vision + AI in action.

🚀 Features

✔ Detects multiple animals in images

✔ Calculates Fun / Playfulness Score

✔ Detects posture (Standing / Neutral / Sitting / Lying)

✔ Adds colorful emojis based on behavior 😄 🙂 😴

✔ Builds leaderboard across all images

✔ Highlights most playful animal with 👑 crown

✔ Saves processed output images

✔ Can be converted into a Web App (Streamlit)

🧠 How It Works

1️⃣ YOLO detects animals and provides bounding boxes
2️⃣ Fun Score is calculated using:

Detection confidence

Number of animals in frame (crowd fun bonus)

Posture score

3️⃣ Emojis are assigned based on posture:

Standing / Active → 😄 Happy

Neutral → 🙂

Sitting / Lying → 😴

4️⃣ Leaderboard ranks animals across images
5️⃣ The highest scoring animal becomes Global Champion

🧰 Tech Stack

Python

YOLO (Ultralytics)

OpenCV

NumPy

Pillow

Streamlit (for website version)

▶️ Running the Project Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run App (if using Streamlit)
streamlit run app.py

📸 Output Highlights

Bounding boxes on animals

Fun score text label

Emojis on top of pets

Leaderboard display

Crown on happiest pet 👑
(Visually awesome results!)

🎯 Use Cases

Academic AI Projects

Computer Vision Learning

Fun AI Pet Tools

Portfolio Projects

✨ Developed By

Katyayani Verma
With guidance & code collaboration using AI 🤖
