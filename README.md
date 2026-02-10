---
title: AI News Credibility Analyzer
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---


# 🧠 AI News Credibility Analyzer

# Hybrid ML + Explainable AI + Semantic Memory + LLM Reasoning

## 📌 Project Overview

The AI News Credibility Analyzer is a multi-layer hybrid AI system designed to evaluate the credibility of news articles using:

- 📊 Statistical Machine Learning
- 🔍 Explainable Linguistic Signals (XAI)
- 🌐 External Evidence Verification
- 🧬 Semantic Memory Retrieval (Vector DB)
- 🦙 Open-Source LLM Reasoning

**Unlike traditional fake news classifiers that only provide binary predictions, this system generates structured reasoning and contextual explanations.**


# 🚀 Live Features
# Hugging Face Spaces Deployed LINK 
([HF Deployed LINK](https://huggingface.co/spaces/Sowmya135/AI-News-Credibility-Analyzer))

- ✅ Real vs Fake Prediction
- 📈 Credibility Probability Score
- 🧠 Confidence Level (High / Medium / Low)
- 🔍 Linguistic Explainability Signals
- 🌐 Wikipedia-Based Evidence Verification(Wikipediaapi)
- 🧬 Memory-Augmented Retrieval via ChromaDB
- 🦙 LLM-Generated Human-Readable Explanation
- 📊 Performance Metrics Display


## 🏗️ System Architecture

The system follows a 5-layer architecture:

### 1. ML Prediction Layer
* TF-IDF vectorization
* Logistic Regression classifier

### 2. Linguistic Feature Layer
* Emotional intensity
* Capitalization patterns
* Sentence complexity
* Readability proxy

### 3. External Evidence Layer
* Named Entity Recognition (spaCy)
* Wikipedia verification
* Evidence score calculation

### 4. Semantic Memory Layer
* SentenceTransformer embeddings
* ChromaDB vector database
* Similar past case retrieval

### 5. LLM Reasoning Layer
* Qwen 1.8B via HuggingFace
* Context-conditioned explanation generation

## 📂 Project Structure

```
AI-News-Credibility-Analyzer/
│
├── app.py
├── models(pkl)/
│   ├── model.pkl
│   └── tfidf.pkl
│
├── modules/
│   ├── utils.py
│   ├── llm_reasoner.py
│   ├── chroma_memory.py
│   └── external_evidence.py
│
└── README.md
```


## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
```
git clone https://github.com/sowmya13531/AI-News-Credibility-Analyzer.git
cd AI-News-Credibility-Analyzer
```

### 2️⃣ Create Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

or Can Install Manually 

```
pip install gradio scikit-learn scipy numpy transformers sentence-transformers chromadb spacy wikipedia-api
```


### 4️⃣ Download spaCy Model

```
python -m spacy download en_core_web_sm
```


## ▶️ Running the Application Locally

*python app.py*

### Then open:

http://127.0.0.1:7860


## 📊 Performance Metrics

**Achieved Accuracy: 99%**
- Displayed in UI:
- Accuracy
- Precision
- Recall
- F1 Score

#### 🔮 Future Improvements

- Replace Wikipedia with fact-check APIs
- Add SHAP for deeper model interpretability
- Persistent Chroma storage
- Transformer-based classifier upgrade
- Dockerized production deployment


# 🎯 Why This Project is Unique

- Most fake news systems provide only binary predictions.

This system integrates:
* ML prediction
* Interpretable signals
* Memory retrieval
* External verification
* LLM reasoning


***It behaves like an AI analyst rather than a classifier.***


👩‍💻 Author

Sowmya Kanithii
Machine Learning Engineer | AI Systems Builder


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
