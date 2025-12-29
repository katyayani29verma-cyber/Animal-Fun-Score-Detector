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
5️⃣ Highest scoring animal becomes Global Champion 👑

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

Fun Score label

Emoji overlay on pets

Leaderboard display

Crown on happiest pet 👑

(Visually awesome results!)

🎯 Use Cases

Academic AI Projects

Computer Vision Learning

Fun AI Pet Tools

Portfolio Projects

🌍 Live Deployment

You can try the hosted version of the project here:

🔗 Live Demo
👉 https://huggingface.co/spaces/Katyayani29/Animal-Fun-Score-Detector

⚠️ Deployment Status

The hosted Space currently shows an “Error after uploading images” due to:

Runtime compatibility issues

YOLO + HuggingFace dependency limitations

However:

✅ Works perfectly on local Gradio

✅ Works through Terminal execution

❌ Only public hosted deployment is affected

Due to time constraints during submission, I couldn’t fully resolve it — but the project:

Implements a working Computer Vision pipeline

Has a functioning scoring engine

Correct emoji + crown visualization

Working leaderboard system

Full Gradio UI integration

I will continue improving and fixing deployment soon 😊

👩‍💻 My Journey So Far

I started my coding journey in May this year, beginning with C programming to build a strong foundation. Gradually, I moved to Python, where my curiosity for AI & Machine Learning really began.

I have developed a strong interest in AIML and plan to learn it deeply this semester. I also plan to work on a dedicated AI/ML project next year and continue improving during my summer vacations.

This project motivated me even more — I genuinely find this field:
✨ exciting
✨ creative
✨ full of possibilities 🚀

🧠 Challenges Faced

The biggest challenge was deploying the application online. While it worked smoothly locally through Gradio and Terminal, deployment introduced multiple unexpected issues.

🛑 Major Challenge

Uploading to Hugging Face Spaces caused failures mainly due to:

YOLO version compatibility issues

Framework dependency conflicts

Through this, I learned:

YOLO has multiple versions and variants

Each has different:

Performance behavior

Compatibility rules

Recognition capability

Some YOLO versions are optimized for speed, while others for accuracy — and choosing the right one really matters.

✅ What I Learned

Through this project, I gained:

✔ Understanding of YOLO & object detection
✔ Experience with AI/ML Python libraries
✔ Knowledge of Git & GitHub workflows
✔ Real-world AI application building experience
✔ Problem-solving resilience

Even though deployment was tough, it taught me that real-world projects come with real challenges — and solving them is what truly makes learning meaningful 😊

✨ Developed By

Katyayani Verma
With guidance & collaborative support using AI 🤖