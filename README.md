# AI-Assisted-Smart-Recruitment-System
<div align="center">

<!-- Animated Banner -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Smart%20Recruitment%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Where%20Deep%20Learning%20Meets%20Hiring%20Intelligence&descAlignY=60&descSize=18&animation=fadeIn"/>

<br/>

<!-- Badges Row 1 -->
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>

<br/>
<br/>

<!-- Badges Row 2 -->
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-00C7B7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Explainable%20AI-6236FF?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LIME-Interpretability-FF6B6B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>

<br/><br/>

> **🤖 An end-to-end AI-powered recruitment pipeline that classifies resumes, matches candidates to jobs semantically, analyzes skill gaps, predicts hire probability — and explains every decision with XAI.**

</div>

---

## 🧠 Two Approaches, One Goal

This project explores **smart recruitment** through two complementary lenses:

| | 📊 `recruitment-ml.ipynb` | 🔬 `recruitment-dl-xai.ipynb` |
|---|---|---|
| **Core Model** | Random Forest Classifier | Custom Deep Neural Network (PyTorch) |
| **Embeddings** | Sentence Transformers (`all-MiniLM`) | Sentence Transformers + DL Layers |
| **Explainability** | Feature Importance Plots | SHAP + LIME (Full XAI Suite) |
| **Search** | FAISS Vector Index | FAISS Vector Index |
| **Extra** | Skill Gap + Hire Probability | ROC Curves + Multi-class Probability |

---

## ⚙️ How It Works

```
📄 Raw Resume PDFs / CSV
        │
        ▼
🔍 PDF Parsing (pdfplumber) + NLP Cleaning (spaCy, NLTK)
        │
        ▼
🧬 Semantic Embedding (Sentence Transformers → all-MiniLM-L6-v2)
        │
        ├──────────────────────┬──────────────────────────┐
        ▼                      ▼                          ▼
🌲 Resume Classification   📐 Job-Resume Matching     🎯 Skill Gap Analysis
  (Random Forest / DNN)    (FAISS Cosine Similarity)  (spaCy NER Extraction)
        │                      │                          │
        └──────────────────────┴──────────────────────────┘
                                │
                                ▼
                    📊 Hire Probability Score
                    (Semantic Similarity × Skill Coverage)
                                │
                                ▼
                    🔍 XAI Explanations
                    (SHAP Values + LIME Tabular)
```

---

## 🌟 Key Features

### 🗂️ Resume Intelligence
- **PDF & CSV ingestion** with `pdfplumber` for raw resume extraction
- **Multi-category classification** across job domains using fine-tuned sentence embeddings
- **Confusion matrix + ROC curves** for deep model evaluation

### 🔗 Semantic Job Matching
- **FAISS vector search** for lightning-fast cosine similarity retrieval
- Ranks candidates against a job description in milliseconds, at scale

### 📉 Skill Gap Analysis
- **spaCy NER** extracts candidate skills from free-text resumes
- Computes a **skill coverage ratio** against job-required skills
- Visual scatter plot of `Skill Coverage vs Hire Probability`

### 🎯 Hire Probability Engine
- Combines **semantic similarity (65%)** + **skill coverage (35%)**
- Returns a ranked list of top candidates with explainable scores

### 🔍 Explainable AI (XAI) — *What makes this unique*
- **SHAP**: Global + local feature attributions for model transparency
- **LIME**: Instance-level tabular explanations for individual decisions
- Every hiring prediction can be justified — no black-box decisions

---

## 📊 Visualizations Included

| Chart | Description |
|---|---|
| 📦 Resume Category Distribution | Countplot of all resume categories in dataset |
| 🧩 Confusion Matrix | Heatmap of classification accuracy per category |
| 📈 ROC Curve (Multi-class) | One-vs-Rest AUC for each job category |
| 📐 Cosine Similarity Distribution | Distribution of semantic similarity scores |
| 🎯 Skill Coverage vs Hire Probability | Scatter plot linking skills to hire chance |
| 🏆 Candidate Ranking Bar Chart | Top-N candidates ranked by hire probability |
| 🔥 SHAP Summary Plot | Global feature importance from the DL model |
| 🔬 LIME Explanation | Local explanation for a single candidate prediction |
| 🛣️ Pipeline Flowchart | End-to-end recruitment pipeline visualization |

---

## 🛠️ Tech Stack

```python
# Core ML & DL
torch              # Deep Learning (custom MLP classifier)
scikit-learn       # Random Forest, metrics, preprocessing
sentence-transformers  # Semantic embeddings (all-MiniLM-L6-v2)

# Vector Search
faiss-cpu          # Approximate Nearest Neighbour search

# NLP & Parsing
spacy              # Named Entity Recognition (skill extraction)
pdfplumber         # PDF text extraction
nltk               # Stopword removal, text normalization

# Explainability
shap               # SHapley Additive exPlanations
lime               # Local Interpretable Model-agnostic Explanations

# Visualization
matplotlib         # Core plotting
seaborn            # Statistical visualization
```

---

## 🚀 Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/smart-recruitment-ai.git
cd smart-recruitment-ai

pip install pdfplumber sentence-transformers faiss-cpu nltk spacy shap lime torch scikit-learn matplotlib seaborn tqdm
python -m spacy download en_core_web_sm
```

### 2. Dataset
This project uses the **[Resume Dataset on Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)** containing labelled PDF resumes across multiple job categories.

```
/kaggle/input/resume-dataset/
├── Resume/Resume.csv       ← Labels + metadata
└── data/data/              ← PDF resumes by category
```

### 3. Run
Open either notebook on **Kaggle** or locally in Jupyter:

```bash
# Classic ML approach
jupyter notebook recruitment-ml.ipynb

# Deep Learning + XAI approach
jupyter notebook recruitment-dl-xai.ipynb
```

---

## 📁 Project Structure

```
📦 smart-recruitment-ai
 ┣ 📓 recruitment-ml.ipynb          ← ML pipeline (Random Forest + FAISS)
 ┣ 📓 recruitment-dl-xai.ipynb      ← DL pipeline (PyTorch + SHAP + LIME)
 ┗ 📄 README.md
```

---

## 💡 Results at a Glance

- ✅ **Multi-class resume classification** with high accuracy across job domains
- ✅ **Semantic candidate ranking** using FAISS vector search
- ✅ **Interpretable hire scores** backed by SHAP and LIME explanations
- ✅ **Full pipeline visualization** from raw PDF → final hiring decision

---

## 🔮 Future Roadmap

- [ ] 🌐 Streamlit / Gradio web app for real-time resume screening
- [ ] 🤗 Fine-tune domain-specific transformer (e.g., `bert-base-uncased`)
- [ ] 📬 Email notification pipeline for shortlisted candidates
- [ ] 🌍 Multi-language resume support
- [ ] 📊 Interactive dashboard with Plotly / Dash

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a [GitHub Issue](https://github.com/YOUR_USERNAME/smart-recruitment-ai/issues) or submit a pull request.

---

<div align="center">

**Built with ❤️ using Python, PyTorch, and Explainable AI**

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer"/>

</div>
