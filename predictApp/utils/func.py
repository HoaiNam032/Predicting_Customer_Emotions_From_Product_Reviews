# utils/func.py

import re
import joblib
import pandas as pd
from underthesea import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st
from deep_translator import GoogleTranslator

# ====== Biểu thức emoji ======
emoji_pattern = re.compile(
    "["  
    u"\U0001F600-\U0001F64F"  
    u"\U0001F300-\U0001F5FF"  
    u"\U0001F680-\U0001F6FF"  
    u"\U0001F1E0-\U0001F1FF"  
    u"\U00002702-\U000027B0"  
    u"\U0001F926-\U0001F937"  
    u"\U00010000-\U0010FFFF"  
    u"\u200d"  
    u"\u2640-\u2642"  
    u"\u2600-\u2B55"  
    u"\u23cf"  
    u"\u23e9"  
    u"\u231a"  
    u"\u3030"  
    u"\ufe0f"  
    "]+", flags=re.UNICODE
)

# ====== Từ viết tắt ======
short_word_dict = {
    "ko": "không", "kg": "không", "khong": "không", "k": "không", "kh": "không",
    "cx": "cũng", "mik": "mình", "mn": "mọi người", "bt": "bình thường",
    "nv": "nhân viên", "sp": "sản phẩm", "đc": "được", "dc": "được",
    "đk": "điều khoản", "đt": "điện thoại", "j": "gì", "vs": "với",
    "hok": "không", "lun": "luôn", "z": "gì", "zậy": "gì vậy", "thik": "thích",
    "hum": "hôm", "wa": "qua", "m": "mình", "mk": "mình", "bn": "bạn", "ok": "ổn"
}

# ====== Dịch sang Tiếng Việt ======
def translate_to_vietnamese(text):
    try:
        return GoogleTranslator(source='auto', target='vi').translate(text)
    except Exception:
        return text

# ====== Hàm tiền xử lý dữ liệu ======
def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # xóa lặp ký tự
    text = emoji_pattern.sub(r'', text)       # xóa emoji
    text = re.sub(r'[^\w\s]', ' ', text)     # xóa ký tự đặc biệt
    text = text.strip()
    words = text.split()
    words = [short_word_dict.get(word, word) for word in words]
    text = ' '.join(words)
    text = word_tokenize(text, format="text") # tách từ underthesea
    return text

# ====== Load dữ liệu từ Google Sheets ======
@st.cache_resource
def load_data():
    sheet_id = "1EnVA5D0khzVuaD6fn6wtCFLWurUXimusiUuIl-qR7pM"
    gid = "1173905870"
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
    try:
        df = pd.read_csv(csv_url)
        df = df.fillna('')
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ Google Sheets: {e}")
        return pd.DataFrame()  # Trả về rỗng nếu lỗi

# ====== Huấn luyện model ======
def train_model(df):
    vectorizer = CountVectorizer()
    model = LogisticRegression(max_iter=500)

    # Kiểm tra cột cần thiết
    if 'clean_content' not in df.columns or 'title' not in df.columns:
        raise ValueError("DataFrame phải có cột 'clean_content' và 'title'")

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        df['clean_content'],
        df['title'],
        test_size=0.1,
        random_state=42
    )

    train_features = vectorizer.fit_transform(train_sentences)
    model.fit(train_features, train_labels)

    return model, vectorizer

def load_bad_words(file_path="bad_words.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
        return words
    except:
        return []

def contains_bad_words(text, bad_words):
    words = text.split()
    for w in words:
        if w.lower() in bad_words:
            return True
    return False

# ====== Lưu / Load model ======
def save_model(model, vectorizer, model_file='lr_model_2label.pkl', vectorizer_file='count_2label.pkl'):
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

def load_model(model_file='lr_model_2label.pkl', vectorizer_file='count_2label.pkl'):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    return model, vectorizer
