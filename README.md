# 🚀 30-Day Advanced ML & AI Challenge

### **Day 1: Real-time Face Detection (Computer Vision)**
* **Library:** OpenCV
* **Concept:** Haar Cascade Classifiers.
* **Outcome:** Built a script that accesses the system webcam and draws bounding boxes around human faces in real-time.
* **Skills:** Image preprocessing (Grayscale conversion), coordinate mapping.
---
### **Day 2: Face Verification & Biometrics (Deep Learning)**
* **Library:** DeepFace, OpenCV, Matplotlib
* **Concept:** Facial Encodings and distance-based metric learning (Cosine/Euclidean distance).
* **Outcome:** Built a biometric security script that compares two images and verifies if they belong to the same person using state-of-the-art Deep Learning models.
---
### **Day 3: Emotion Detection (Computer Vision)**
* **Library:** DeepFace
* **Concept:** Facial expression analysis and emotion classification.
* **Outcome:** Built a simple but powerful AI script that analyzes an image and accurately predicts the dominant human emotion (Happy, Sad, Angry, etc.) using a pre-trained deep learning model.
---
### **Day 4: Hand Tracking & Landmark Detection**
* **Library:** MediaPipe, OpenCV
* **Concept:** Skeletal landmark tracking (21 key points per hand).
* **Outcome:** Created a real-time hand tracking application that detects hand movements and maps structural landmarks using Google's MediaPipe framework.........................................
---
### **Day 5: Human Pose Estimation**
* **Library:** MediaPipe, OpenCV
* **Concept:** 33-point body landmark detection and skeletal mapping.
* **Outcome:** Developed a real-time pose tracking system capable of identifying body joints and movements, laying the foundation for AI-based fitness and activity recognition applications.
---
### **Day 6: Gesture-Based Volume Control**
* **Library:** MediaPipe, PyCaw, NumPy
* **Concept:** Mathematical distance calculation between specific landmarks (Thumb tip and Index tip) and mapping values to system hardware parameters.
* **Outcome:** Built a functional hand-gesture controller that adjusts the system volume in real-time, demonstrating Human-Computer Interaction (HCI).
---
### **Day 7: Virtual Paint Application (Week 1 Capstone)**
* **Library:** MediaPipe, OpenCV, NumPy
* **Concept:** Multi-modal interaction (Selection vs. Drawing modes) based on finger-pose logic and bitwise image manipulation.
* **Outcome:** Built a complete air-writing/drawing application that allows users to paint on a digital canvas using hand gestures, select colors, and erase content in real-time.
* **Week 1 Progress:** ✅ Completed Computer Vision & Gesture Control Module.
---
### **Day 8: Deploying AI as a REST API**
* **Library:** FastAPI, Uvicorn, DeepFace
* **Concept:** API Endpoints (GET/POST), Request handling, and Model serving.
* **Outcome:** Converted the Day 3 Emotion Detection model into a production-ready REST API. This allows external applications (Web/Mobile) to consume the AI model via HTTP requests.
---
### **Day 9: Containerization with Docker**
* **Tools:** Docker, FastAPI.
* **Concept:** Dependency management and environment isolation.
* **Outcome:** Successfully containerized the Emotion Detection API. Created a Dockerfile to package the application, OS-level dependencies, and Python libraries into a portable image.
---
### **Day 10: Database Integration (AI Persistence)**
* **Tools:** FastAPI, SQLite3, DeepFace.
* **Concept:** Data persistence, SQL schema design, and integrating ML outputs with structured storage.
* **Outcome:** Built a pipeline that not only predicts emotions but also logs every prediction into a local SQLite database with a timestamp for future auditing and analytics.
---
### **Day 11: Introduction to Deep Learning (Neural Networks)**
* **Library:** TensorFlow, Keras.
* **Concept:** Multi-Layer Perceptrons (MLP), Activation Functions (ReLU, Softmax), and Forward/Backward Propagation.
* **Outcome:** Built and trained a 3-layer neural network from scratch to recognize handwritten digits with >95% accuracy.
---
### **Day 12: Convolutional Neural Networks (CNN)**
* **Library:** TensorFlow, Keras.
* **Concept:** Feature Extraction through Convolution filters, Spatial hierarchy, and Pooling layers.
* **Outcome:** Developed a CNN model to classify RGB color images from the CIFAR-10 dataset. Learned how CNNs are superior for image-related tasks compared to standard Dense networks.
---
### **Day 13: Transfer Learning**
* **Library:** TensorFlow (Keras Applications).
* **Concept:** Leveraging pre-trained weights from MobileNetV2, Feature Extraction vs. Fine-tuning.
* **Outcome:** Built a custom classifier using a high-performance pre-trained architecture. Understand how to reuse existing AI "intelligence" for niche tasks with minimal training time.
---
### **Day 14: Real-time Object Detection (Week 2 Capstone)**
* **Library:** OpenCV (dnn module).
* **Model:** MobileNet-SSD (Single Shot Detector).
* **Concept:** Bounding boxes, Confidence scores, and Non-Maximum Suppression (NMS).
* **Outcome:** Built a high-speed object detection system capable of identifying 80 different object categories in a live video stream.
* **Week 2 Progress:** ✅ Completed Full-Stack AI Integration & Deep Learning Vision Module.
---
### **Day 15: Sequence Modeling with LSTM**
* **Library:** TensorFlow, Keras.
* **Concept:** Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) for handling sequential text data.
* **Outcome:** Built a sentiment classifier using the IMDB movie review dataset. Learned how Embedding layers and LSTMs work together to maintain context in natural language.
---
### **Day 16: Time Series Forecasting with LSTM**
* **Library:** TensorFlow, NumPy, Scikit-learn.
* **Concept:** Windowing/Sequencing data, MinMaxScaler, and Many-to-One LSTM architecture.
* **Outcome:** Developed a predictive model for time-dependent data. Learned how to prepare data sequences and scale them for regression-based forecasting.
---
### **Day 17: Text Generation with RNN/LSTM**
* **Library:** TensorFlow, NumPy.
* **Concept:** Character-level modeling, One-hot encoding for text, and Softmax probability for next-character prediction.
* **Outcome:** Built a generative model that learns the structure of a given text and generates new sequences. Understand the foundation of how LLMs (Large Language Models) work.
---
### **Day 19: Text Summarization Engine**
* **Library:** NLTK (Natural Language Toolkit).
* **Concept:** Extractive Summarization using Word Frequency Scoring.
* **Outcome:** Developed an AI tool that can condense long paragraphs into concise summaries by identifying and ranking key sentences based on importance.
---
### **Day 20: Language Translation with Transformers**
* **Library:** Hugging Face Transformers.
* **Model:** T5-Small (Text-to-Text Transfer Transformer).
* **Concept:** Attention Mechanism and Encoder-Decoder architectures in NLP.
* **Outcome:** Built a functional translation tool that converts English text to other languages using state-of-the-art Transformer models.
---
### **Day 21: Named Entity Recognition (NER)**
* **Library:** spaCy.
* **Concept:** Information Extraction, Part-of-Speech Tagging, and Entity Classification.
* **Outcome:** Built a tool to automatically extract and categorize key information (Names, Locations, Dates, Organizations) from unstructured text data using pre-trained NLP models.
---
### **Day 22: Introduction to Generative AI & LLMs**
* **Library:** Hugging Face Transformers.
* **Model:** Google Flan-T5.
* **Concept:** Text-to-Text Transfer Transformers, Prompting, and Decoding.
* **Outcome:** Successfully deployed a local Large Language Model (LLM) to perform zero-shot tasks like question answering and instruction following.
---
### **Day 23: Advanced Prompt Engineering**
* **Library:** Hugging Face Transformers.
* **Concept:** Few-Shot Prompting, In-Context Learning, and Structured Output Generation (JSON parsing).
* **Outcome:** Built an LLM pipeline that utilizes strategic formatting and examples to force a local text-generation model to output structured metadata instead of conversational prose.
---
### **Day 24: Retrieval-Augmented Generation (RAG)**
* **Library:** Hugging Face Transformers.
* **Concept:** Context injection, Knowledge-base Retrieval mechanisms, and Augmentation flows.
* **Outcome:** Built an in-memory RAG system from scratch. Learned how to query a private knowledge base, extract relevant text metadata, and feed it into a local open-source LLM to produce domain-specific accuracy without model fine-tuning.
---
### **Day 25: Vector Embeddings & Semantic Similarity**
* **Library:** Sentence-Transformers (PyTorch backend).
* **Model:** all-MiniLM-L6-v2.
* **Concept:** Text Vectorization, Dense Embeddings, and Cosine Similarity calculation.
* **Outcome:** Developed an implementation to measure semantic overlap between sentences. Understood how modern production RAG pipelines convert raw knowledge documentation into vector spaces for efficient neural searches.
---
### **Day 26: Vector Databases with ChromaDB**
* **Library:** ChromaDB.
* **Concept:** Vector persistence, Document indexing, Collections management, and Semantic nearest-neighbor retrieval.
* **Outcome:** Configured a local on-disk Vector Database pipeline. Implemented programmatic insertion of text vectors paired with contextual metadata and executed intelligent semantic queries ignoring pure keyword matching.
---
### **Day 27 (Bonus): RAG Web Application with Streamlit**
* **Library:** Streamlit, ChromaDB, Transformers.
* **Concept:** Frontend AI integration, Model Caching (`@st.cache_resource`), and Interactive UI.
* **Outcome:** Wrapped the backend RAG pipeline into a user-friendly web interface, allowing real-time interaction with the Vector DB and LLM through a browser.
---
### **Day 28: Model Quantization & Fine-Tuning Concepts**
* **Library:** NumPy, Core Python Precision Analytics.
* **Concept:** Bit-Depth Reduction, Symmetric Quantization Matrix Scaling, and Parameter-Efficient Fine-Tuning (PEFT/LoRA) fundamentals.
* **Outcome:** Simulated neural network weight scaling from FP32 to INT8 matrix distribution layers. Achieved an absolute 4x optimization architecture compression while evaluating minimal quantization error variance.
---
### **Day 29: Multi-Agent AI Orchestration Systems**
* **Library:** Hugging Face Transformers.
* **Concept:** Specialized AI Agents, Inter-agent Communications, Sequential Task Routing, and Pipeline Orchestration.
* **Outcome:** Engineered a multi-agent cooperative network from scratch. Directed an automated data pipeline where a specialized Research Agent extracts conceptual parameters and hands them off sequentially to a Writer Agent for final deterministic reporting.
---
### **Day 30: Grand Capstone Project - Production Knowledge Chatbot**
* **Library:** Streamlit, ChromaDB, Transformers, PyTorch.
* **Concept:** End-to-End Enterprise RAG Systems, Advanced Session State Tracking, and Vector UI Orchestration.
* **Outcome:** Successfully deployed a fully interactive, production-grade conversational Knowledge Base Assistant. Formally completed the 30-Day Advanced Machine Learning and Artificial Intelligence Challenge!
---
### **Day 31: RAG Evaluation & Hallucination Guardrails**
* **Library:** NumPy, Core NLP Precision Metrics.
* **Concept:** RAG Evaluation (Ragas framework theory), Faithfulness Matrix, and Answer Relevance scoring.
* **Outcome:** Engineered an automated validation pipeline to analyze LLM generation logs against retrieved context vectors. Created guardrails to isolate and alert hallucinated model outputs dynamically.
---
### **Day 32: Graph Neural Networks & Node Classification**
* **Library:** NumPy Network Analytics layer.
* **Concept:** Graph Convolutional Networks (GCN), Adjacency Matrix representations, Message Passing, and Node Aggregation.
* **Outcome:** Simulated a Graph Convolution operation from scratch. Learned how structural network connections act as message passing matrices to diffuse local neighbor node features for downstream classification tasks.
---
### **Day 33: Audio Processing & Acoustic Feature Extraction**
* **Library:** NumPy Digital Signal Processing layer.
* **Concept:** Audio Digital Sampling, Short-Time Fourier Transform (STFT) theory, and Mel-Frequency Cepstral Coefficients (MFCC) matrix mappings.
* **Outcome:** Simulated a complete acoustic processing pipeline from raw wave inputs to 2D feature representations. Understood how complex non-grid auditory signals are structured into mathematical matrices for CNN classification networks.
  ---
### **Day 34: Anomaly Detection & Cybersecurity**
* **Library:** Scikit-Learn (Isolation Forests), Matplotlib.
* **Concept:** Unsupervised Learning, Outlier Detection, and Isolation Forest algorithms.
* **Outcome:** Built an intrusion detection system capable of monitoring synthetic server traffic logs and automatically flagging anomalous request spikes (simulating DDoS attacks) without prior labeled data.
