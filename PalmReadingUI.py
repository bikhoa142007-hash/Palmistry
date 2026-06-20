import streamlit as st
import pandas as pd
from PIL import Image

class PalmReadingUI:
    def __init__(self, predictor=None):
        st.set_page_config(
            page_title="Dự Đoán Chỉ Tay", 
            layout="centered"
        )
        self.predictor = predictor
        self._init_session_state()

    def _init_session_state(self):
        if "palm_image" not in st.session_state:
            st.session_state.palm_image = None
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

    def run(self):
        self._render_header()
        self._render_image_input()
        self._render_prediction_area()
        if st.session_state.prediction_result is not None:
            self._render_results()

    def _render_header(self):
        st.title("Dự Đoán Chỉ Tay")
        st.markdown("Chụp hoặc tải lên hình ảnh lòng bàn tay để nhận phân tích chi tiết.")
        st.divider()

    def _render_image_input(self):
        tab_cam, tab_upload = st.tabs(["Chụp ảnh trực tiếp", "Tải ảnh từ máy"])
        
        with tab_cam:
            st.caption("Đảm bảo bạn đã cấp quyền Camera")
            cam_image = st.camera_input("Chụp lòng bàn tay", key="cam_input")
            if cam_image:
                st.session_state.palm_image = cam_image
                
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Chọn ảnh lòng bàn tay", 
                type=["jpg", "jpeg", "png"],
                key="file_uploader"
            )
            if uploaded_file:
                st.session_state.palm_image = uploaded_file

    def _render_prediction_area(self):
        if st.session_state.palm_image is None:
            st.info("Vui lòng chụp hoặc tải ảnh để bắt đầu!")
            return

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(st.session_state.palm_image, caption="Ảnh đã chọn", use_container_width=True)
            
        with col2:
            st.markdown("Khu vực điều khiển")
            st.write("Nhấn nút bên dưới để phân tích chỉ tay")
            
            if st.button("PHÂN TÍCH CHỈ TAY", type="primary", use_container_width=True):
                self._handle_prediction()

    def _handle_prediction(self):
        if self.predictor is None:
            st.error("Chưa có predictor")
            return
            
        with st.spinner("Đang phân tích..."):
            try:
                result = self.predictor.predict(st.session_state.palm_image)
                st.session_state.prediction_result = result
                st.success("Hoàn tất!")
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

    def _render_results(self):
        st.divider()
        st.subheader("Kết Quả Phân Tích")
        
        if isinstance(st.session_state.prediction_result, pd.DataFrame):
            st.dataframe(
                st.session_state.prediction_result, 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write(st.session_state.prediction_result)