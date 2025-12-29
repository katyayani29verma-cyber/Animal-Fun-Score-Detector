# 🐶 Animal Fun Score Predictor

This project is an **AI-powered Animal Fun / Playfulness Score Detector** that analyzes pets in images, detects animals using **YOLO**, evaluates their posture, assigns a **Fun Score**, overlays emojis, builds a **leaderboard**, and finally crowns the global champion with a 👑 crown.

It is **fun, visually engaging**, and a great example of **Computer Vision + AI in action**.

---

## 🚀 Features

✔ Detects **multiple animals** in images  
✔ Calculates **Fun / Playfulness Score**  
✔ Detects **posture** (Standing / Neutral / Sitting / Lying)  
✔ Adds colorful emojis 😄 🙂 😴  
✔ Builds leaderboard  
✔ Highlights most playful animal with 👑 crown  
✔ Saves processed output images  
✔ Can be converted to **Streamlit Web App**

---

## 🧠 How It Works

1️⃣ YOLO detects animals and provides **bounding boxes**  
2️⃣ Fun Score is calculated using:
- Detection confidence  
- Number of animals in frame (**crowd fun bonus**)  
- Posture score  

3️⃣ Emojis are assigned based on posture:
- Standing / Active → 😄 Happy  
- Neutral → 🙂  
- Sitting / Lying → 😴  

4️⃣ Leaderboard ranks animals across images  
5️⃣ **Highest scoring animal becomes Global Champion 👑**

---

## 🧰 Tech Stack

- Python  
- YOLO (Ultralytics)  
- OpenCV  
- NumPy  
- Pillow  
- Streamlit (for website version)

---

## ▶️ Running the Project Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run App (if using Streamlit)
```bash
streamlit run app.py
```

---

## 📸 Output Highlights

- Bounding boxes on animals  
- Fun score label  
- Emoji overlay on pets  
- Leaderboard display  
- Crown on happiest pet 👑  

---

## 🎯 Use Cases

- Academic AI Projects  
- Computer Vision Learning  
- Fun AI Pet Tools  
- Portfolio Projects  

---

## 🌍 Live Deployment

You can try the hosted version of the project here:

🔗 **Live Demo**  
https://huggingface.co/spaces/Katyayani29/Animal-Fun-Score-Detector

---

### ⚠️ Deployment Status

The hosted Space currently shows an **“Error after uploading images”** due to:
- Runtime compatibility issues  
- YOLO + HuggingFace dependency limitations  

However:

- ✅ Works perfectly on **local Gradio**
- ✅ Works correctly via **Terminal execution**
- ❌ Only public hosted deployment is affected

Due to time constraints during submission, I couldn’t fully resolve deployment — BUT this project demonstrates:

- Working **Computer Vision pipeline**  
- Functional **Fun Scoring Engine**  
- Correct **Emoji + Crown Visualization**  
- Working **Leaderboard System**  
- Full **Gradio UI Integration**

Deployment fix is planned soon 😊  

---

## 👩‍💻 My Journey So Far

I started my coding journey **in May this year**, beginning with **C programming** to build a foundation. Gradually, I moved into **Python**, where my curiosity for **AI & Machine Learning** truly began.

I developed a strong interest in **AIML** and plan to learn it thoroughly this semester. I also plan to build a dedicated AI/ML project next year and continue learning during my summer vacations.

This project motivated me even more — I genuinely find this field:
- exciting  
- creative  
- full of possibilities 🚀  

---

## 🧠 Challenges Faced

The biggest challenge was **deploying the application online**.

While the project worked smoothly on **Gradio locally** and through **Terminal execution**, deployment on Hugging Face introduced unexpected errors.

A major issue came from **YOLO version compatibility**.  
I learned that:

- YOLO has **multiple versions and model variants**
- Each behaves differently in:
  - Performance  
  - Compatibility  
  - Recognition capability  

Some versions are optimized for speed, while others provide better accuracy — and choosing the right one really matters.

Apart from that, this project helped me:

✔ Understand YOLO and object detection  
✔ Work with AI/ML Python libraries  
✔ Gain hands-on Git & GitHub experience  
✔ Handle real-world project challenges  

Even though deployment was tough, it taught me valuable lessons — real-world projects always come with challenges, and solving them is what truly makes learning meaningful 😊  

---

## ✨ Developed By

**Katyayani Verma**  
With guidance & collaborative support using AI 🤖
