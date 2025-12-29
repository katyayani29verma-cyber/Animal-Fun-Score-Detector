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

👩‍💻 My Journey So Far

I started my coding journey in May this year, beginning with C programming to build a strong foundation. Gradually, I moved into Python, and that’s where my curiosity for AI and Machine Learning really grew.

Over time, I developed a strong interest in the AIML field, and I plan to start learning it properly this semester. I’m also planning to work on a dedicated AI/ML project next year and continue improving my skills during the summer vacations.

This project played a big role in motivating me to explore more — I genuinely find this field exciting, creative, and full of possibilities 🚀


🧠 Challenges Faced

The biggest challenge I faced in this project was deploying the application online. While my model worked smoothly on local runs through Gradio / Terminal, deployment introduced several unexpected issues that required a lot of debugging and troubleshooting.

One of the major problems I faced was while uploading the project on Hugging Face Spaces. The issue was mainly related to YOLO version compatibility, which caused runtime failures during deployment. Through this, I learned that YOLO has different versions and model variants, and each behaves differently in terms of performance, compatibility, and detection accuracy. Some versions are optimized for speed, while others provide better recognition and precision — and choosing the right one really matters.

Apart from these challenges, everything else was a great learning experience. Through this project, I:

Learned how YOLO models work and how to use them effectively

Worked with multiple Python libraries used in AI/ML

Gained hands-on experience with Git & GitHub workflows

Understood more about building real-world AI applications and handling practical issues beyond just coding

Even though deployment was tough, it taught me that real projects always come with unexpected challenges — but solving them is what truly makes learning meaningful 😊

✨ Developed By

Katyayani Verma
With guidance & code collaboration using AI 🤖

