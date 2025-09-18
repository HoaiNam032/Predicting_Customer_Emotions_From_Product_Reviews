# Predicting Customer Emotions From Product Reviews 💬
*Tran Hoai Nam— September 2025*

---

## 📖 Overview  

**Predicting Customer Emotions From Product Reviews** is an academic project completed in **September 2025**.  
The system leverages **Natural Language Processing (NLP)** and **Machine Learning** to automatically interpret customer feedback in **Vietnamese product reviews**.  

The main goal is to perform **binary sentiment classification** — distinguishing between **Satisfied** and **Unsatisfied**.  
This project highlights how data-driven approaches can provide valuable insights into customer satisfaction.  

By analyzing real-world review data from **Tiki.vn**, the application serves both as a **practical tool for businesses** and as a **hands-on case study for applying analytics techniques**.  


---

## ✨ Features
- 🎯 **Emotion Classification**: Binary classification - *Satisfied* or *Unsatisfied*.  
- 📝 **Vietnamese NLP**: Analyze customer feedback in Vietnamese using NLP techniques.  
- 📊 **Real-world Dataset**: Collected from product reviews on *Tiki.vn*.  
- ⚡ **Interactive Web App**: Built with Streamlit for real-time prediction.  

---

## 🛠 Technologies Used
- 🐍 Python **3.11.12**  
- 🤖 Logistic Regression for machine learning (**93% accuracy**)  
- 📚 Pandas, Scikit-learn for data processing and modeling  
- 🌐 Streamlit for building a user-friendly web interface  
- 🔤 Under-the-hood NLP: Custom Vietnamese text preprocessing with normalization and word segmentation  

---

## 🚀 Live Demo
👉 Try the deployed application here:  
🔗 [customer-emotion-prediction.streamlit.app](https://customer-emotion-prediction.streamlit.app/)

---

## ⚙ Installation
To run this project locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/nglhongphuong/Predicting-Consumer-Emotions-from-Product-Reviews.git

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

```






