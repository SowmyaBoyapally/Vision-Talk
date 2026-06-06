# 👁️🎙️ Vision Talk

### Offline Multimodal AI for Image Question Answering

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Offline AI](https://img.shields.io/badge/Mode-Offline-green)
![CPU Only](https://img.shields.io/badge/Deployment-CPU%20Only-orange)

---

## 🌟 Overview

**Vision Talk** is an offline multimodal AI assistant that enables users to interact with images using natural language. Users can upload an image, ask questions through text or voice, and receive responses in both textual and spoken form.

Unlike cloud-based vision-language systems, Vision Talk operates entirely on local hardware without requiring internet connectivity, paid APIs, or GPU acceleration. The system combines computer vision, speech processing, and language reasoning into a unified pipeline optimized for CPU-only environments.

The project demonstrates how modern vision-language models can be orchestrated efficiently to build practical multimodal AI systems that are accessible, lightweight, and deployable in resource-constrained environments.

---

## ✨ Key Features

* 🖼️ Image-Based Question Answering
* 🎤 Voice Query Support
* ⌨️ Text Query Support
* 🔊 Speech-Based Response Generation
* 🧠 Context-Aware AI Responses
* 💻 Fully Offline Execution
* ⚡ CPU-Optimized Deployment
* 🔗 Modular AI Pipeline Architecture
* 🌐 No Internet Dependency
* 🔒 Privacy-Friendly Processing

---

## 📸 Demo

<p align="center">
  <img src="assets/demo.gif" alt="Vision Talk Demo" width="850"/>
</p>

---

## 🏗️ System Architecture

<img width="1408" height="714" alt="image" src="https://github.com/user-attachments/assets/2f48c896-3ece-4a4a-96c1-c7db5d7f7276" />


---

## 🧠 Models Used

| Model        | Purpose                                    |
| ------------ | ------------------------------------------ |
| BLIP         | Image Understanding and Caption Generation |
| Whisper Base | Speech-to-Text Conversion                  |
| FLAN-T5 Base | Context-Aware Response Generation          |
| gTTS         | Text-to-Speech Synthesis                   |

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Deep Learning & NLP

* PyTorch
* Hugging Face Transformers

### Computer Vision

* BLIP

### Speech Processing

* Whisper
* gTTS

### User Interface

* Gradio

### Supporting Libraries

* Pillow (PIL)
* NumPy

---

## 🔄 Workflow

```text id="4w8q4w"
1. User uploads an image.
2. User enters a text query or records a voice query.
3. BLIP extracts visual context from the image.
4. Whisper converts voice input into text.
5. Visual context and user query are combined.
6. FLAN-T5 generates a context-aware response.
7. The response is displayed as text.
8. gTTS converts the response into speech.
9. User receives both text and audio outputs.
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or above
* Git
* pip package manager

### Clone Repository

```bash id="5lq34s"
git clone https://github.com/your-username/Vision-Talk.git
cd Vision-Talk
```

### Install Dependencies

```bash id="h8vg7j"
pip install -r requirements.txt
```

### Run the Application

```bash id="9eqbwx"
python app.py
```

or

```bash id="wfh1r9"
python main.py
```

*(Use the appropriate entry file based on your project structure.)*

### Launch

After execution, the Gradio interface will automatically open in your browser.

---

## 💡 How to Use

### Step 1

Upload an image through the interface.

### Step 2

Provide a question related to the image using:

* Text Input
* Voice Input

### Step 3

Submit the query.

### Step 4

The system processes the image and query.

### Step 5

Receive:

* Generated Text Response
* Speech Output

---

## 📂 Project Structure

```text id="v5d1o0"
Vision-Talk/
│
├── assets/                    # Screenshots, diagrams, demo files
├── models/                    # Model loading and inference modules
├── outputs/                   # Generated outputs
├── app.py                     # Main application
├── requirements.txt           # Project dependencies
├── README.md
│
└── ...
```

---

## 📷 Screenshots

<img width="873" height="553" alt="image" src="https://github.com/user-attachments/assets/cd5cf98e-a41f-4904-81b3-3786a57a594d" />
<img width="875" height="523" alt="image" src="https://github.com/user-attachments/assets/d3368851-192d-47e8-8370-333fa95f5113" />
<img width="878" height="520" alt="image" src="https://github.com/user-attachments/assets/0f92b5a8-cf5c-4c15-9985-1777c718b62c" />
<img width="883" height="517" alt="image" src="https://github.com/user-attachments/assets/220d4621-c370-436c-a8ba-7353adac6f8a" />

---

## 🎯 Applications

### Accessibility Support

Assist visually impaired users by enabling interaction with visual content through speech and language.

### Education

Help students understand diagrams, images, posters, and visual learning materials.

### Assistive AI Systems

Provide intelligent offline assistance in environments with limited internet connectivity.

### Smart Information Access

Enable natural language interaction with image-based information.

---

## 🚀 Why Vision Talk?

Most modern vision-language systems rely on:

* Cloud APIs
* Internet Connectivity
* GPU Infrastructure
* Proprietary Services

Vision Talk demonstrates that practical multimodal AI can be deployed entirely offline using open-source models while maintaining useful response quality and real-time interaction.

---

## ⚠️ Current Limitations

* Limited numerical reasoning from images
* OCR performance depends on image quality
* Not designed for video understanding
* CPU-only execution may increase inference time
* Performance depends on underlying model capabilities

---

## 🔮 Future Enhancements

* Advanced OCR Integration
* Real-Time Camera Support
* Multilingual Query Processing
* Improved Vision-Language Models
* Mobile Deployment
* Enhanced Scene Understanding
* Faster Local Inference
* Better Reasoning Capabilities

---

## 📊 Project Highlights

* Fully Offline AI System
* No Cloud Dependency
* No Paid APIs
* CPU-Based Deployment
* Vision-Language Integration
* Speech Processing Pipeline
* Open-Source Technology Stack
* Accessibility-Focused Design

---

## 📚 Keywords

Multimodal AI • Image Question Answering • Vision-Language Models • Computer Vision • Natural Language Processing • Speech Recognition • Text-to-Speech • Transformer Models • Offline AI Systems • Accessibility • Assistive Technology • Generative AI • CPU Inference

---

## 👨‍💻 Author

### Sowmya Boyapally

**Vision Talk – Offline Multimodal Image Question Answering System**

Built using Computer Vision, Speech Processing, and Transformer-based Language Models to enable intelligent image interaction without cloud dependencies.
