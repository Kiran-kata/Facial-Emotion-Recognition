# 😊 EmoSense — Facial Emotion Recognition

> Detect human emotions from faces in real time using deep learning and computer vision.

EmoSense is a Python-based web application that identifies emotions from live webcam input or uploaded images. It uses a trained TensorFlow Lite model to detect facial expressions and classify them into multiple emotion categories with confidence scores. The project combines deep learning, OpenCV face detection, and a lightweight Flask web interface to provide a simple and interactive emotion recognition experience.

---

## Live Demo

This project is deployed as a Python Flask web service.

- Live App: https://facial-emotion-recognition-f0fl.onrender.com/
- Local Run: `python app.py`
- Local Browser URL: `http://127.0.0.1:5000`

> This is a server-based AI app, so it should be hosted on a Python-capable deployment platform like Render instead of a static hosting site.

---

## Overview

This project demonstrates how a Convolutional Neural Network (CNN) can be trained and deployed to recognize facial emotions from grayscale face crops. It is designed for experimentation, learning, and real-time demos in browser-based applications.

The system can:

- detect faces in an image or webcam stream,
- preprocess them into the required model format,
- classify the dominant emotion,
- display probability scores for each emotion,
- return annotated results in the frontend.

It recognizes the following emotions:

| Emotion | Description |
|---|---|
| 😄 Happiness | Smiling expression with positive visual cues |
| 😢 Sadness | Downturned facial features and low-energy expression |
| 😠 Anger | Tense jaw, intense expression |
| 😲 Surprise | Wide eyes and open-mouth look |
| 😐 Neutral | Calm or neutral expression |
| 😨 Fear | Alert, tense, worried expression |
| 🤢 Disgust | Wrinkled nose or aversive expression |
| 😏 Contempt | Slightly asymmetrical smirk |

---

## Features

### Existing Features
- Real-time webcam emotion recognition
- Upload image-based emotion analysis
- Face detection using OpenCV Haar cascades
- Prediction of the dominant emotion and confidence score
- Display of all emotion probabilities
- Annotated image output with bounding boxes and labels
- Lightweight TensorFlow Lite inference model
- Browser-based UI using Flask, HTML, CSS, and JavaScript

### Functional Workflow
1. Capture an image from webcam or upload an image.
2. Detect one or more faces in the frame.
3. Crop each face and convert it to the required model input format.
4. Run inference using the TensorFlow Lite model.
5. Predict the emotion with the highest probability.
6. Display the emotional result and output image with annotations.

---

## How It Works

The app follows a classic computer vision pipeline:

1. Input image is received from the browser.
2. OpenCV detects the face region.
3. The detected face is resized to 48x48 grayscale.
4. The image is normalized and fed into the model.
5. The model predicts a probability distribution over emotion classes.
6. The class with the highest probability is chosen as the prediction.

The model is a CNN that learns visual patterns such as edges, shapes, and facial features. It combines these cues to infer emotion from expressions like mouth curvature, eye openness, and brow tension.

<p align="center">
  <img src="images/Facial%20Emotion%20Recognition.png" alt="Facial emotion recognition pipeline" width="850"/>
</p>

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Model Deployment | TensorFlow Lite |
| Face Detection | OpenCV |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | NumPy |
| Training Dataset | FER+ |

---

## Project Structure

```bash
Facial Emotion Recognition/
├── app.py                     # Flask web app and inference pipeline
├── models.py                  # CNN model architecture used in training
├── predictions.py             # Prediction and visualization helpers
├── tflite_utils.py            # TensorFlow Lite utility functions
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── model/
│   ├── ferplus_model_pd_best.tflite
│   ├── ferplus_model_pd_best.json
│   ├── fer_model_best.h5
│   └── ...
├── data/
│   ├── __init__.py
│   ├── data.py
│   ├── dataset.py
│   └── outliers_processing.py
├── templates/
│   └── index.html             # Frontend interface
├── static/
│   ├── style.css              # Styling for the app
│   └── script.js              # Javascript for webcam and upload logic
├── images/
│   └── Facial Emotion Recognition.png
└── .gitignore
```

---

## Installation and Run

Make sure Python is installed, then run:

```bash
pip install -r requirements.txt
python app.py
```

Open the browser and visit:

```bash
http://127.0.0.1:5000
```

Then either:
- use the webcam,
- or upload an image,
- and click Analyze to get the emotion prediction.

---

## Use Cases

This technology can be used in a wide range of real-world applications:

### Education
- Detect student engagement and confusion during online classes
- Support adaptive learning systems

### Healthcare
- Monitor emotional states in therapy sessions
- Assist in research related to behavioral and emotional analysis

### Retail and Customer Experience
- Measure reactions to products and advertisements
- Understand customer mood in stores and digital experiences

### Driver Safety
- Detect drowsiness, stress, or distraction in drivers
- Improve safety in vehicles and transport systems

### Gaming and Entertainment
- Adjust difficulty based on player emotion
- Personalize content and recommendations

### Security and Surveillance
- Detect suspicious or distressed behavior in public areas
- Support safety monitoring systems

### Customer Service
- Analyze emotional states during video calls
- Improve customer support interaction quality

---

## Accuracy and Limitations

The model has shown strong performance on benchmark datasets and can achieve high accuracy in controlled conditions. In the current implementation, accuracy is influenced by factors such as:

- lighting quality,
- face orientation,
- camera resolution,
- occlusions such as glasses or masks,
- expression ambiguity.

Emotion recognition is not perfect because emotions are complex and often context-dependent. The model predicts visible facial patterns, not necessarily the full internal emotional state of a person.

The current model has been reported to achieve approximately:

| Metric | Accuracy |
|---|---|
| Probability Distribution (PD) | 87.7% |
| Majority Vote (MV) | 86.9% |

---

## Future Improvements and Features

### Model Improvements
- Train on more diverse datasets with varied age groups and ethnic backgrounds
- Add data augmentation such as rotation, brightness adjustment, and flips
- Use stronger architectures like EfficientNet, MobileNet, or Vision Transformer
- Improve face alignment and landmark detection
- Explore ensemble methods for more robust predictions

### Feature Additions
- Multi-face emotion recognition in a single frame
- Real-time emotion timeline and trend charts
- Video upload support for analysis over time
- Emotion logging and user history dashboard
- Age and gender estimation alongside emotion recognition
- Voice-based emotion analysis using speech data
- Multi-modal AI combining face, voice, and text
- Alerts for distress or abnormal emotional behavior
- Mobile app support for Android and iOS
- Cloud deployment on AWS, Azure, or Google Cloud

### UI/UX Enhancements
- Modern dashboard with charts and visual analytics
- Dark mode and responsive layout
- Better user feedback and error handling
- Downloadable result reports in JSON or CSV format

### Integration Opportunities
- Google MediaPipe Face Mesh
- OpenAI Whisper for audio emotion analysis
- DeepFace and other pre-trained emotion recognition systems
- AWS Rekognition and Azure Face API for cloud-based analysis
- YOLOv8 for real-time detection in crowded scenes

---

## Ethical Considerations

- Always obtain consent before analyzing a person’s face.
- Use the system responsibly and transparently.
- Avoid using emotion detection in ways that could be invasive or discriminatory.
- Treat predictions as supportive tools, not absolute truth.

---

## Conclusion

EmoSense is a practical and educational project that combines computer vision, machine learning, and web development to build a working facial emotion recognition application. It provides a strong foundation for future extensions in AI, human-computer interaction, healthcare, customer analytics, and smart systems.

---

*Built with Python, TensorFlow, OpenCV, and Flask.*
---

## ⚠️ Things to Keep in Mind

- **Privacy matters** — always get consent before analysing someone's face
- **Not perfect** — emotions are complex; this AI reads visible expressions, not inner feelings
- **Lighting affects accuracy** — works best in good, even lighting
- **Use responsibly** — this technology should help people, not surveil or judge them

---

*Built with [Python](https://www.python.org/) · [TensorFlow](https://www.tensorflow.org/) · [Flask](https://flask.palletsprojects.com/) · [OpenCV](https://opencv.org/)*


---

## 🤔 What Does It Actually Do?

Imagine you show the app a photo of your friend, or point your webcam at yourself. Within a second, the app will:

1. **Find faces** in the image automatically
2. **Analyse the expressions** on those faces
3. **Tell you the emotion** — with a confidence score and a colour-coded bar chart

It can detect **8 different emotions**:

| Emotion | What it looks like |
|---|---|
| 😄 Happiness | Smiling, bright eyes |
| 😢 Sadness | Frowning, drooping face |
| 😠 Anger | Furrowed brows, tense jaw |
| 😲 Surprise | Wide eyes, open mouth |
| 😐 Neutral | Relaxed, no strong expression |
| 😨 Fear | Eyes wide, tense |
| 🤢 Disgust | Wrinkled nose, curled lip |
| 😏 Contempt | One-sided smirk |

---

## 🚀 How to Run It

**You only need Python installed.**

```bash
# 1. Install the required packages
pip install -r requirements.txt

# 2. Start the app
python app.py

# 3. Open your browser and go to:
#    http://127.0.0.1:5000
```

That's it! The app opens in your browser. Choose **Webcam** or **Upload an image** and hit Analyse.

---

## 🧠 How Was It Built? (Simple Explanation)

Think of the AI here like a very well-trained student who has studied **35,000+ photos** of human faces and learnt what expressions look like. Here is how it was taught:

### Step 1 — Gather Photos
Researchers collected tens of thousands of face photos and had people label them with emotions. This collection is called the **FER+ dataset**.

### Step 2 — Train the AI Brain (CNN)
The app uses a type of AI called a **Convolutional Neural Network (CNN)** — it works a bit like how your eyes work. It looks at small patches of a face, finds patterns (eyebrows raised? corners of mouth up?), and combines all those clues to make a decision.

### Step 3 — Build the Web App
Once the AI was trained, it was saved into a tiny file (`ferplus_model_pd_best.tflite` — only 692 KB, smaller than most photos!). A simple web server built with **Flask** (a Python tool) loads this file and serves the app in your browser. OpenCV handles face detection using your camera or uploaded images.

### The Tech Stack at a Glance

| What | Tool Used |
|---|---|
| AI / Deep Learning | TensorFlow & Keras |
| Face Detection | OpenCV |
| Web Server | Flask (Python) |
| Frontend | HTML · CSS · JavaScript |
| Model file | TFLite (692 KB) |

---

## 📁 What's in This Project?

```
📦 facial-recognition-model/
├── app.py               ← The web server (brain of the app)
├── models.py            ← CNN model architecture definition
├── tflite_utils.py      ← Helper for running the AI model
├── predictions.py       ← Helper for visualising results
├── fer_model.ipynb      ← Training notebook (for researchers)
├── requirements.txt     ← List of Python packages needed
├── model/               ← Trained AI model files
│   └── ferplus_model_pd_best.tflite
├── templates/
│   └── index.html       ← The web page
├── static/
│   ├── style.css        ← Visual design
│   └── script.js        ← Page behaviour
└── data/                ← Dataset helpers (for training)
```

---

## 🌍 Where Can This Be Used?

This technology has real-world uses everywhere people interact with cameras:

### 🏫 Education
- Detect if students are confused, bored, or engaged during online classes
- Help teachers understand the mood of a classroom in real time

### 🏥 Healthcare & Mental Health
- Monitor patients' emotional states during therapy sessions
- Assist in diagnosing conditions where facial expressions are affected (e.g. autism research)
- Alert caregivers if a patient shows signs of distress or pain

### 🛍️ Retail & Customer Experience
- Measure customer reactions to products, ads, or store layouts
- Personalise shopping experiences based on mood

### 🚗 Driver Safety
- Detect drowsiness, frustration, or distraction in drivers
- Trigger alerts before accidents happen in cars, trucks, or trains

### 🎮 Gaming & Entertainment
- Adapt game difficulty based on whether a player is stressed or bored
- Personalise movie or music recommendations based on your mood

### 🔒 Security & Surveillance
- Detect aggressive or fearful behaviour in public spaces via CCTV
- Flag unusual emotional patterns at airports, banks, or venues

### 📞 Customer Service & Call Centres
- Analyse emotions during video calls to help agents respond better
- Measure satisfaction in real time during conversations

### 🏠 Smart Home
- Adjust lighting, music, or temperature based on detected mood
- Build empathetic AI assistants that respond to how you feel

---

## 🔧 Possible Upgrades & Improvements

Here are ideas to make EmoSense even smarter:

### Better AI Models
| Upgrade | What it means |
|---|---|
| **Transformer models** (like ViT) | Newer, more powerful AI that understands context better |
| **MediaPipe Face Mesh** | Google's tool for detecting 468 face points — much more detail |
| **Multi-modal AI** | Combine face + voice + text for deeper emotion understanding |
| **GPT-4 / Gemini integration** | Use large language models to describe what emotions mean in context |

### More Features
- 🎙️ **Add voice analysis** — detect emotion from tone of voice alongside face
- 📊 **Emotion timeline** — track how emotions change over time in a session
- 👥 **Multiple people** — analyse many faces in one frame simultaneously (group mood)
- 📱 **Mobile app** — package the model into an iOS or Android app
- 🌐 **Cloud deployment** — host on AWS / Google Cloud so anyone can access it online
- 🔔 **Alerts system** — send a notification when a specific emotion (e.g. distress) is detected

### Better Accuracy
- Train on more diverse datasets with different ages, skin tones, and lighting
- Add data augmentation (flipping, brightness changes) during training
- Use ensemble models (combine multiple AIs and vote on the answer)

---

## 📊 How Accurate Is It?

The model was tested on thousands of face images it had never seen before:

| Emotion labels used | Accuracy |
|---|---|
| Probability distribution (PD) | **87.7%** |
| Majority vote (MV) | **86.9%** |

In plain English: **out of every 100 faces, it gets the right emotion about 87 times**. The emotions it's best at are *neutral* and *happiness* (most training examples). It struggles most with *disgust*, *fear*, and *contempt* because those are rarer in the training data.

---

## 🤖 AI Tools That Could Be Integrated

| AI Tool | What It Adds |
|---|---|
| **Google MediaPipe** | Precise face landmark detection (better face tracking) |
| **OpenAI Whisper** | Voice emotion detection alongside face |
| **GPT-4 Vision** | Describe emotions in full sentences from images |
| **DeepFace** | Pre-built emotion, age, and gender recognition |
| **AWS Rekognition** | Cloud-based emotion detection via camera feeds |
| **Azure Face API** | Microsoft's cloud AI for facial analysis |
| **YOLOv8** | Fast real-time face detection for CCTV or crowded scenes |

---

## ⚠️ Things to Keep in Mind

- **Privacy matters** — always get consent before analysing someone's face
- **Not perfect** — emotions are complex; this AI reads visible expressions, not inner feelings
- **Lighting affects accuracy** — works best in good, even lighting
- **Use responsibly** — this technology should help people, not surveil or judge them

---

*Built with Python, TensorFlow, Flask, and OpenCV.*

