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
| **Explainability** | Feature Importance Plots | SHAP |
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
                         (SHAP Values)
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
| 🛣️ Pipeline Flowchart | End-to-end recruitment pipeline visualization |

---

## 🛠️ Tech Stack

💻 Programming Language
<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
</p>
🧠 ML & Deep Learning
<p>
  <img src="https://img.shields.io/badge/PyTorch-Custom%20MLP%20Classifier-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Framework-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Random%20Forest%20(n=150)-2E7D32?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-all--MiniLM--L6--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
</p>
🔍 Vector Search
<p>
  <img src="https://img.shields.io/badge/FAISS-Approximate%20Nearest%20Neighbour-00C7B7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Cosine%20Similarity%20Index-0081A7?style=for-the-badge"/>
</p>
📝 NLP & Parsing
<p>
  <img src="https://img.shields.io/badge/spaCy-NER%20Skill%20Extraction-09A3D5?style=for-the-badge&logo=spacy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-en__core__web__sm-09A3D5?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/pdfplumber-PDF%20Text%20Extraction-E53935?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLTK-Stopword%20Removal-154F5B?style=for-the-badge"/>
</p>
🔬 Explainable AI
<p>
  <img src="https://img.shields.io/badge/SHAP-Shapley%20Additive%20Explanations-6236FF?style=for-the-badge"/>
</p>
📦 Libraries
<p>
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/tqdm-Progress%20Bars-FFC107?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/sentence--transformers-Semantic%20Embeddings-FF6F00?style=for-the-badge"/>
</p>
📊 Visualization
<p>
  <img src="https://img.shields.io/badge/Matplotlib-Core%20Plotting-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0?style=for-the-badge"/>
</p>
🧰 Tools & Environment
<p>
  <img src="https://img.shields.io/badge/Kaggle-Notebook%20Environment-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Interactive%20Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

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
 ┣ 📓 recruitment-dl-xai.ipynb      ← DL pipeline (PyTorch + SHAP)
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

- 🧠 LLM + Knowledge GraphsCombine : large language models with knowledge-graph-based candidate profiling for deeper semantic understanding of resumes and job requirements
- ⚖️ Ethical AI & Bias Detection : Integrate fairness-aware learning algorithms and bias-detection methods to promote ethical adoption of AI in hiring workflows
- 🌍 Multilingual & Cross-Domain Matching : Extend the framework to support multilingual resume analysis and cross-domain job matching for international recruitment scenarios
- 🌐 Web Application : Deploy the full recruitment pipeline as an interactive web app — allowing recruiters to upload resumes, enter job descriptions, and receive ranked candidates with XAI explanations in real time
- 📊 Interactive Recruiter Dashboard : Build a web-based dashboard for recruiters to visualize candidate insights and hiring recommendations in real time
- 📂 Larger & More Diverse Datasets : Scale up resume data volume and diversity to improve model generalization and robustness across job categories
- 📡 Real-Time Job Market Integration : Incorporate live job market data and automated job description analysis for more accurate candidate-job matching and smarter hiring decisions

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a [GitHub Issue](https://github.com/YOUR_USERNAME/smart-recruitment-ai/issues) or submit a pull request.

---

<div align="center">

**Built with ❤️ using Python, PyTorch, and Explainable AI**

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer"/>

</div>
