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

## 🌍 Live Deployment

You can try the hosted version of the project here:

🔗 **Live Demo**  
https://huggingface.co/spaces/Katyayani29/Animal-Fun-Score-Detector

### ⚠️ Note About Deployment
The hosted Space currently shows an **“Error” after uploading images** due to runtime compatibility and server dependency limitations on HuggingFace Spaces.

However:

- ✅ The complete model works perfectly in local environment  
- ✅ It runs successfully via **Gradio locally**  
- ✅ It runs correctly in **Terminal execution**  
- ❌ Only the public hosted deployment is facing an integration/runtime issue

Due to time constraints during submission, I could not fully resolve the Space runtime issue — but the full model logic, YOLO functionality, scoring system, emojis, leaderboard, and UI all work correctly in development.

This demonstrates:
- working Computer Vision pipeline
- functioning scoring engine
- emoji + crown visualization
- leaderboard logic
- Gradio-based UI integration

I will continue improving and fixing the deployment version soon 😊

✨ Developed By

Katyayani Verma
With guidance & code collaboration using AI 🤖
