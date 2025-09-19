import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
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

# ===== Cấu hình nguồn file mẫu =====
# 1) Thử tìm file local (nếu bạn bundle sẵn)
APP_DIR = Path(__file__).parent
LOCAL_SAMPLE = APP_DIR / "data_test_file.csv"

# 2) Nếu không có local, tải từ Google Sheets (link bạn đưa)
# Link gốc: https://docs.google.com/spreadsheets/d/19WSRWUDcjhJjuVx-sE62icv1FWgNDVRLP5PsTpJpUko/edit?gid=1429131216#gid=1429131216
SHEET_ID = "19WSRWUDcjhJjuVx-sE62icv1FWgNDVRLP5PsTpJpUko"
GID = "1429131216"
GSHEETS_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"

def load_sample_df():
    # Ưu tiên local trước để không phụ thuộc mạng
    if LOCAL_SAMPLE.exists():
        try:
            return pd.read_csv(LOCAL_SAMPLE)
        except Exception as e:
            st.warning(f"⚠️ Không đọc được file local data_test_file.csv: {e}")
    # Fallback: đọc trực tiếp từ Google Sheets (public/read-only)
    try:
        return pd.read_csv(GSHEETS_CSV_URL)
    except Exception as e:
        st.warning(f"⚠️ Không tải được file mẫu từ Google Sheets: {e}")
        return None

# ===== Nút tải file mẫu =====
df_sample = load_sample_df()
if df_sample is not None:
    st.info("📥 Bạn có thể tải file mẫu để test ngay:")
    st.download_button(
        label="📄 Tải file mẫu data_test_file.csv",
        data=df_sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_test_file.csv",
        mime="text/csv",
    )
else:
    st.warning("⚠️ Chưa có file mẫu khả dụng (local hoặc Google Sheets).")

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
            # chú ý: cần 'openpyxl' trong requirements nếu đọc .xlsx
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
                explode=explode,
                textprops={"fontsize": 10}
            )

            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontweight("bold")

            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
