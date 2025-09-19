import streamlit as st
import pandas as pd
import joblib
import os
import utils.func
import matplotlib.pyplot as plt

# ===== Cấu hình giao diện =====
st.set_page_config(page_title="Dự Đoán Cảm Xúc Theo File", page_icon="📄", layout="wide")
st.title("📄 Dự Đoán Cảm Xúc Từ File Bình Luận Khách Hàng")

st.markdown("""
#### 👉 Tải lên file `.csv, .xlsx, .txt` chứa cột `content` (bình luận).
Sau đó, hệ thống sẽ:
- Làm sạch dữ liệu
- Dự đoán cảm xúc theo mô hình đã huấn luyện (Page 1)
- Trả về file `.csv` đã được gán nhãn để bạn tải về
""")

# ===== Load model & vectorizer =====
if os.path.exists("lr_model_2label.pkl") and os.path.exists("count_2label.pkl"):
    model = joblib.load("lr_model_2label.pkl")
    vectorizer = joblib.load("count_2label.pkl")
else:
    st.error("❌ Model chưa được tạo. Vui lòng chạy trang Page 1 để huấn luyện trước.")
    st.stop()

# ===== Upload file =====
uploaded_file = st.file_uploader(
    "📤 Tải lên file chứa cột `content` (.csv, .xlsx, .txt)",
    type=["csv", "xlsx", "xls", "txt"]
)

if uploaded_file is not None:
    try:
        # Đọc file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("❌ Định dạng file không được hỗ trợ.")
            st.stop()

        # Kiểm tra cột bắt buộc
        if "content" not in df.columns:
            st.error("❌ File cần có cột tên là `content`.")
        else:
            with st.spinner("🔄 Đang xử lý và dự đoán..."):
                # Làm sạch dữ liệu
                df["clean_content"] = df["content"].apply(
                    lambda x: utils.func.clean_data(utils.func.translate_to_vietnamese(x))
                )

                # Dự đoán cảm xúc
                features = vectorizer.transform(df["clean_content"])
                df["sentiment"] = model.predict(features)

            # Hiển thị nội dung sau khi làm sạch
            st.subheader("💬 Nội dung sau khi làm sạch:")
            st.dataframe(df[["content", "clean_content"]])

            # Hiển thị kết quả dự đoán
            st.subheader("📄 Kết quả dự đoán cảm xúc:")
            st.dataframe(df[["content", "sentiment"]])

            # Nút tải file kết quả
            st.success("✅ Xử lý xong! Tải file kết quả bên dưới.")
            st.download_button(
                label="📥 Tải file kết quả",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ket_qua_du_doan.csv",
                mime="text/csv"
            )

            # ===== Biểu đồ trực quan =====
            st.subheader("📊 Phân tích cảm xúc khách hàng")

            sentiment_counts = df["sentiment"].value_counts()
            colors = ["#A0C4FF", "#FFADAD"]  # pastel xanh dương và hồng cam

            # Tạo figure
            fig, ax = plt.subplots(figsize=(3.5, 3.5))

            # Thêm số lượng vào nhãn
            labels = [f"{label} ({count})" for label, count in zip(sentiment_counts.index, sentiment_counts.values)]

            # Biểu đồ tròn
            explode = [0.05] * len(sentiment_counts)
            wedges, texts, autotexts = ax.pie(
                sentiment_counts,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                explode=explode,
                textprops={"fontsize": 10}
            )

            # Làm đẹp phần trăm
            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontweight("bold")

            # Hiển thị trên Streamlit
            st.pyplot(fig)


    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
