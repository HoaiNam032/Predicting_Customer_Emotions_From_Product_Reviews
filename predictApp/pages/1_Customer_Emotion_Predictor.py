# pages/1_Customer_Emotion_Predictor.py
import pandas as pd

import streamlit as st
import utils.func
import joblib
import os

# ===== Thiết lập giao diện =====
st.set_page_config(page_title="Dự Đoán Cảm Xúc Khách Hàng", page_icon="😊", layout="wide")
st.title("Dự Đoán Cảm Xúc Khách Hàng Từ Bình Luận Sản Phẩm")

# Hướng dẫn
st.markdown("""
### Nhập bình luận của bạn dưới đây và hệ thống sẽ dự đoán cảm xúc khách hàng.
**Mô hình phân loại thành 2 nhãn:**
- **Hài lòng**
- **Không hài lòng**
""")

# ===== Load model và vectorizer =====
if os.path.exists("lr_model_2label.pkl") and os.path.exists("count_2label.pkl"):
    model = joblib.load("lr_model_2label.pkl")
    vectorizer = joblib.load("count_2label.pkl")
else:
    st.error("❌ Model chưa được tạo. Vui lòng chạy trang Page 0 để huấn luyện trước.")

# ===== Load danh sách từ cấm =====
bad_words_file = "bad_words.txt"
if os.path.exists(bad_words_file):
    with open(bad_words_file, "r", encoding="utf-8") as f:
        bad_words = [line.strip().lower() for line in f if line.strip()]
else:
    bad_words = []
    st.warning("⚠️ Không tìm thấy file bad_words.txt. Kiểm tra từ cấm sẽ bị bỏ qua.")

# ===== Nhập bình luận =====
user_input = st.text_area("Nhập bình luận của khách hàng:")

if st.button("Dự đoán cảm xúc"):
    if user_input:
        # Dịch và tiền xử lý
        clean_text = utils.func.clean_data(utils.func.translate_to_vietnamese(user_input))

        # Kiểm tra từ cấm
        tokens = clean_text.lower().split()
        bad_words_set = set(bad_words)
        found_bad_words = [w for w in tokens if w in bad_words_set]

        if found_bad_words:
            st.error(f"🚫 Bình luận không hợp lệ. Phát hiện từ cấm: {', '.join(found_bad_words)}")
        else:
            # Vector hóa và dự đoán
            user_vector = vectorizer.transform([clean_text])
            prediction = model.predict(user_vector)

            # Hiển thị kết quả
            if prediction[0] == "cực kỳ hài lòng":
                st.success("👍 Khách hàng **hài lòng** với sản phẩm.")
            else:
                st.error("👎 Khách hàng **không hài lòng** với sản phẩm.")

            # ===== Lưu bình luận vào CSV =====
            save_file = "user_comments.csv"
            new_data = pd.DataFrame([[user_input, prediction[0]]], columns=["comment", "prediction"])

            if os.path.exists(save_file):
                old_data = pd.read_csv(save_file)
                updated = pd.concat([old_data, new_data], ignore_index=True)
            else:
                updated = new_data

            updated.to_csv(save_file, index=False, encoding="utf-8-sig")
            st.info(f"💾 Bình luận đã được lưu vào `{save_file}`")

    else:
        st.warning("⚠️ Vui lòng nhập bình luận để dự đoán cảm xúc.")
