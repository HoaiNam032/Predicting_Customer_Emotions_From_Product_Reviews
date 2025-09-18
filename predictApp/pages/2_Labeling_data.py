import streamlit as st
import pandas as pd
import joblib
import os
import utils.func

st.set_page_config(page_title="Dự Đoán Cảm Xúc Theo File", page_icon="📄", layout="wide")
st.title("📄 Dự Đoán Cảm Xúc Từ File Bình Luận Khách Hàng")

st.markdown("""
#### 👉 Tải lên file `.csv,.xlsx, .txt` chứa cột `content` (bình luận).
Sau đó, hệ thống sẽ:
- Làm sạch dữ liệu
- Dự đoán cảm xúc theo mô hình đã huấn luyện (Page 1)
- Trả về file `.csv,.xlsx, .txt` đã được gán nhãn để bạn tải về
""")

# Load model & vectorizer
if os.path.exists("lr_model_2label.pkl"):
    model = joblib.load("lr_model_2label.pkl")
    vectorizer = joblib.load("count_2label.pkl")
else:
    st.error("Model chưa được tạo. Vui lòng chạy trang page_1 để huấn luyện trước.")

# Load từ cấm
#bad_words = utils.func.load_bad_words("bad_words.txt")

# Upload file
uploaded_file = st.file_uploader("📤 Tải lên file chứa cột `content` (.csv, .xlsx, .txt)",
                                 type=["csv", "xlsx", "xls", "txt"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("❌ Định dạng file không được hỗ trợ.")
            st.stop()

        if 'content' not in df.columns:
            st.error("❌ File cần có cột tên là `content`.")
        else:
            with st.spinner("🔄 Đang xử lý và dự đoán..."):
                df['clean_content'] = df['content'].apply(
                    lambda x: utils.func.clean_data(utils.func.translate_to_vietnamese(x))
                )

            # Kiểm tra từ cấm
            #df['sentiment'] = df['clean_content'].apply(
                #lambda x: "Bình luận không hợp lệ" if utils.func.contains_bad_words(x, bad_words)
            #              else model.predict(vectorizer.transform([x]))[0]
            #)
                
                # Hiển thị nội dung sau khi làm sạch
                st.subheader("💬 Nội dung sau khi làm sạch:")
                st.dataframe(df[['content', 'clean_content']])

                # Tiến hành dự đoán cảm xúc
                features = vectorizer.transform(df['clean_content'])
                df['sentiment'] = model.predict(features)

                st.success("✅ Xử lý xong! Tải file kết quả bên dưới.")
                st.download_button(
                    label="📥 Tải file kết quả",
                    data=df.to_csv(index=False).encode('utf-8-sig'),
                    file_name="ket_qua_du_doan.csv",
                    mime="text/csv"
                )

                st.dataframe(df[['content', 'sentiment']])

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
