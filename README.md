# 💬 Predicting Customer Emotions From Product Reviews
*Tran Hoai Nam — September 2025*

---

## 📖 Overview  

**Predicting Customer Emotions From Product Reviews** is an applied AI project developed in **Python 3.11.12**, focusing on **Natural Language Processing (NLP)** for Vietnamese text.  
The system implements a complete **end-to-end Machine Learning pipeline**: data ingestion, preprocessing, model training, evaluation, deployment, and real-time prediction.  

The core task is **binary sentiment classification** — distinguishing between **Satisfied** vs. **Unsatisfied** customers.  
By leveraging real-world product reviews from **Tiki.vn**, the project demonstrates how **Python-based AI systems** can transform unstructured text into actionable business insights.  

---

## ✨ Features  

- 🎯 **Automated Sentiment Analysis**: Binary classification with optimized Logistic Regression (**93% accuracy**).  
- 🧹 **Advanced Preprocessing Pipeline in Python**: Regex-based cleaning, Vietnamese word segmentation, stopword & banned-word filtering, text normalization.  
- 🧠 **Machine Learning Engineering**: Model trained with Scikit-learn, vectorized with Bag-of-Words/CountVectorizer, persisted and reloaded using Joblib for reproducibility.  
- ⚡ **Interactive AI Application**: Streamlit web interface enabling real-time single-input prediction and batch processing for CSV/Excel/TXT files.  
- 📈 **Data Logging & Visualization**: Automatic storage of predictions (CSV) and dashboard-style charts (Matplotlib) for analyzing sentiment trends.  

---

## 🛠 Technologies Used  

- 🐍 **Python 3.11.12** — OOP-based modular code, automation scripts, and pipeline reproducibility.  
- 📚 **Pandas, NumPy** — scalable data manipulation and ETL operations.  
- 🤖 **Scikit-learn** — Logistic Regression, evaluation metrics (Accuracy, F1-score, ROC-AUC).  
- 💾 **Joblib** — model/vectorizer serialization & lifecycle management.  
- 🌐 **Streamlit** — user-friendly web app for both real-time and batch predictions.  
- 🔤 **NLP Preprocessing** — custom Vietnamese text pipeline (regex, normalization, tokenization, stopword removal).  
- 📊 **Matplotlib, Seaborn** — sentiment distribution visualization and trend analysis.  

---

## 📊 Steps  

### 1. Data Exploration & Design  
- Analyze review structure: length, word frequency, label distribution.  
- Implemented in Python using **pandas** for reading/writing data, string & list manipulation.  
- Organized reusable code into utility modules (`utils.func`).  

### 2. Text Preprocessing (NLP)  
- Clean text with **regex**: normalization, lowercasing, removing special characters/emojis, diacritic normalization, tokenization.  
- Filter out banned words using **set/list lookups** before feeding into the model.  
- Implemented helper functions in Python such as `clean_data` and `translate_to_vietnamese`.  

### 3. Feature Extraction  
- Transformed text into feature matrices using **CountVectorizer / Bag-of-Words** in scikit-learn.  

### 4. Model Building & Optimization  
- Trained a **Logistic Regression** model with scikit-learn for binary classification (*Satisfied vs. Unsatisfied*).  
- Achieved around **93% accuracy**, ensuring fast and stable inference by controlling seed and feature set consistency.  

### 5. Model Lifecycle Management  
- Saved/loaded model and vectorizer with **joblib**.  
- If no model exists, the system **auto-trains and saves** it (cold-start).  
- Used `os.path` to check files and `try/except` for exception handling, making the workflow robust.  

### 6. AI Application Deployment (Inference UI)  
- Built a multi-page app with **Streamlit**:  
  - **Single prediction page**: input review, banned-word check, real-time prediction.  
  - **Batch prediction page**: upload `.csv/.xlsx/.txt` file → clean → vectorize → predict in bulk → download results.  
  - **History page**: manage `user_comments.csv`, view logs, reset data.  
- The app was developed fully in Python with multi-page structure, state/file management, input validation, and I/O exception handling.  

### 7. Monitoring & Visualization  
- Generated result tables and pie charts to analyze sentiment distribution and trends.  
- Implemented in Python with **matplotlib** and **pandas** (`value_counts`, `groupby`).  
- Exported predictions to CSV with **UTF-8-SIG encoding** for external analysis.  

---

## 🚀 Live Demo
👉 Try the deployed application here:  
🔗 [customer-emotion-prediction.streamlit.app](https://customer-emotion-prediction.streamlit.app/)

---

## ⚙ Installation
To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/HoaiNam032/Predicting_Customer_Emotions_From_Product_Reviews.git

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

```
---
## 📂 Project Structure
```bash
Your-Folder-Name/
├── app.py                      # Main entrypoint for Streamlit app
├── utils/
│   ├── __init__.py
│   └── func.py
└── pages/
    ├── 1_Page_one.py
    └── 2_Page_two.py
    └── 3_Page_three.py

```












