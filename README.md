# 🛑 Hate Speech Detection

This project implements a **deep learning–based hate speech classifier** using LSTM with an **attention mechanism** and **SMOTE** oversampling to handle class imbalance.  
The full implementation is provided in the Jupyter Notebook:  
`hate_speech_classification.ipynb`

---

## 📌 Project Overview

Automatic hate speech detection is a critical NLP task for moderating online platforms and improving community safety.  
This project explores:

- Text preprocessing and tokenization  
- Handling class imbalance using **SMOTE**
- Training an **LSTM network with attention**
- Evaluation of model performance

---

## 🧠 Methodology

The overall workflow includes:

1. **Data Loading & Exploration**  
   Load raw text data labeled for hate speech vs. non‑hate speech.

2. **Text Preprocessing**  
   - Cleaning and tokenization
   - Padding sequences
   - Building vocabulary

3. **Class Imbalance Handling**  
   - Apply **SMOTE (Synthetic Minority Over‑sampling Technique)** to oversample minority class samples

4. **Model Architecture**
   - **LSTM (Long Short‑Term Memory)** layers to learn sequential text patterns
   - **Attention mechanism** to focus on important tokens for classification

5. **Training & Evaluation**
   - Train the model on augmented training data
   - Evaluate using accuracy, precision, recall, F1 score, and confusion matrix.

---

## 📂 Repository Structure

```
hate-speech-detection/
│
├── hate_speech_classification.ipynb # Main notebook
└── README.md # Project documentation
```

## ⚙️ Technologies & Libraries

- **Python**
- **Jupyter Notebook**
- **TensorFlow / Keras**
- **NLTK / spaCy / other preprocessing libs**
- **Imbalanced‑learn (SMOTE)**
- **Pandas, NumPy**
- **Matplotlib / Seaborn**
