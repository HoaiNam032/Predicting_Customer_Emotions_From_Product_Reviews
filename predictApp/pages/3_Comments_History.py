# pages/3_Comments_History.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Lịch Sử Bình Luận", page_icon="📂", layout="wide")
st.title("📂 Lịch Sử Bình Luận Đã Lưu")

save_file = "user_comments.csv"

# Kiểm tra file tồn tại
if os.path.exists(save_file):
    try:
        df = pd.read_csv(save_file)

        if not df.empty:
            st.success(f"✅ Đã tìm thấy {len(df)} bình luận đã lưu.")
            st.dataframe(df, use_container_width=True)

            # Nút tải file CSV
            st.download_button(
                label="⬇️ Tải về file CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="user_comments.csv",
                mime="text/csv",
            )

            # Nút reset (xóa dữ liệu)
            if st.button("🗑️ Reset toàn bộ bình luận"):
                os.remove(save_file)
                st.warning("⚠️ Tất cả bình luận đã được xóa.")
                st.experimental_rerun()

        else:
            st.warning("⚠️ File tồn tại nhưng chưa có bình luận nào.")
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {e}")
else:
    st.info("ℹ️ Chưa có bình luận nào được lưu.")
