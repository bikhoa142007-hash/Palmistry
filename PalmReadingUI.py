import streamlit as st
import pandas as pd

DAO_MARKS = {
    "Sinh Đạo": ("生", "Sinh"),
    "Tâm Đạo": ("心", "Tâm"),
    "Trí Đạo": ("智", "Trí"),
    "Vận Mệnh": ("命", "Mệnh"),
}

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Be+Vietnam+Pro:wght@300;400;500;600&display=swap');

:root{
    --ink:#0B0E1A;
    --panel:#13162A;
    --panel-line:#262B45;
    --gold:#C9A24B;
    --gold-dim:#8A713A;
    --parchment:#EFE6D8;
    --parchment-dim:#A9A296;
}

.stApp{
    background:var(--ink);
}
.block-container{
    max-width:560px;
    padding-top:3rem;
    padding-bottom:4rem;
}
html, body, [class*="css"]{
    font-family:'Be Vietnam Pro', sans-serif;
    color:var(--parchment);
}

.pr-eyebrow{
    text-align:center;
    letter-spacing:0.32em;
    font-size:11px;
    color:var(--gold-dim);
    text-transform:uppercase;
    margin-bottom:14px;
}
.pr-title{
    font-family:'Cormorant Garamond', serif;
    font-weight:500;
    font-size:44px;
    text-align:center;
    margin:0 0 6px;
    color:var(--parchment);
}
.pr-title em{
    font-style:normal;
    color:var(--gold);
}
.pr-subtitle{
    text-align:center;
    font-size:14px;
    color:var(--parchment-dim);
    font-weight:300;
    line-height:1.6;
    max-width:340px;
    margin:0 auto 32px;
}

.pr-hero{ display:flex; justify-content:center; margin-bottom:8px; }
.pr-hero svg{ width:170px; height:auto; }
.pr-line{ fill:none; stroke:var(--gold); stroke-width:2.2; stroke-linecap:round;
    filter:drop-shadow(0 0 3px rgba(201,162,75,0.55)); }
.pr-outline{ fill:none; stroke:var(--panel-line); stroke-width:1.4; }

/* tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:0;
    border-bottom:1px solid var(--panel-line);
    background:transparent;
}
.stTabs [data-baseweb="tab"]{
    background:transparent;
    color:var(--parchment-dim);
    font-size:12px;
    letter-spacing:0.12em;
    text-transform:uppercase;
}
.stTabs [aria-selected="true"]{
    color:var(--gold) !important;
    border-bottom:1px solid var(--gold) !important;
}

/* camera / uploader widgets */
[data-testid="stCameraInput"], [data-testid="stFileUploader"]{
    border:1px solid var(--panel-line);
    border-radius:2px;
    padding:14px;
    background:var(--panel);
}
[data-testid="stCameraInput"] button, [data-testid="stFileUploader"] button{
    border:1px solid var(--gold-dim) !important;
    color:var(--gold) !important;
    background:transparent !important;
    letter-spacing:0.08em;
}

/* primary analyze button */
.stButton button[kind="primary"]{
    width:100%;
    background:transparent;
    border:1px solid var(--gold-dim);
    color:var(--gold);
    font-size:12px;
    letter-spacing:0.22em;
    text-transform:uppercase;
    padding:0.8rem 0;
}
.stButton button[kind="primary"]:hover{
    background:var(--gold);
    color:var(--ink);
    border-color:var(--gold);
}

.pr-results-head{ text-align:center; margin:40px 0 24px; }
.pr-results-head h2{
    font-family:'Cormorant Garamond', serif;
    font-weight:500;
    font-size:26px;
    margin:6px 0 0;
    color:var(--parchment);
}

.pr-dao{
    display:grid;
    grid-template-columns:64px 1fr;
    gap:18px;
    padding:22px 0;
    border-top:1px solid var(--panel-line);
}
.pr-dao:last-child{ border-bottom:1px solid var(--panel-line); }
.pr-dao-mark{
    font-family:'Cormorant Garamond', serif;
    font-size:30px;
    color:var(--gold);
    text-align:center;
    line-height:1;
}
.pr-dao-mark sub{
    display:block;
    font-family:'Be Vietnam Pro', sans-serif;
    font-size:9px;
    letter-spacing:0.1em;
    color:var(--parchment-dim);
    margin-top:6px;
    text-transform:uppercase;
}
.pr-dao-name{ font-size:15px; font-weight:500; color:var(--parchment); margin-bottom:6px; }
.pr-dao-state{
    display:inline-block;
    font-size:10px;
    letter-spacing:0.1em;
    text-transform:uppercase;
    color:var(--gold);
    border:1px solid var(--gold-dim);
    padding:2px 8px;
    margin-bottom:10px;
}
.pr-dao-text{ font-size:13.5px; line-height:1.75; color:var(--parchment-dim); font-weight:300; }
</style>
"""

HERO_SVG = """
<div class="pr-hero">
<svg viewBox="0 0 200 260" xmlns="http://www.w3.org/2000/svg">
<path class="pr-outline" d="M70 250 C40 250 30 220 32 190 L34 110 C34 100 44 95 48 105 L52 150 L54 95 C54 84 66 84 66 95 L68 148 L70 88 C70 76 84 76 84 88 L86 148 L90 95 C90 85 102 86 102 97 L100 150 L120 115 C126 106 138 112 134 124 L114 175 C124 178 132 192 128 210 C124 232 102 250 70 250 Z" />
<path class="pr-line" d="M44 180 C70 178 100 172 128 155" />
<path class="pr-line" d="M50 195 C75 190 105 165 118 120" />
<path class="pr-line" d="M58 230 C56 195 56 160 58 130" />
<path class="pr-line" d="M82 240 C80 200 78 165 78 135" />
</svg>
</div>
"""


class PalmReadingUI:
    def __init__(self, predictor=None):
        st.set_page_config(
            page_title="Dự Đoán Chỉ Tay",
            layout="centered",
        )
        st.markdown(THEME_CSS, unsafe_allow_html=True)
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
        st.markdown('<div class="pr-eyebrow">Tướng số học · AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="pr-title">Dự Đoán <em>Chỉ Tay</em></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pr-subtitle">Mỗi đường chỉ trên lòng bàn tay là một câu chuyện. '
            'Hãy để chúng tôi đọc câu chuyện của bạn.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(HERO_SVG, unsafe_allow_html=True)

    def _render_image_input(self):
        tab_cam, tab_upload = st.tabs(["Chụp ảnh trực tiếp", "Tải ảnh từ máy"])

        with tab_cam:
            st.caption("Đảm bảo bạn đã cấp quyền Camera, sau đó chạm nút chụp bên dưới")
            cam_image = st.camera_input("Chụp lòng bàn tay", key="cam_input", label_visibility="collapsed")
            if cam_image:
                st.session_state.palm_image = cam_image

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Chọn ảnh lòng bàn tay",
                type=["jpg", "jpeg", "png"],
                key="file_uploader",
                label_visibility="collapsed",
            )
            if uploaded_file:
                st.session_state.palm_image = uploaded_file

    def _render_prediction_area(self):
        if st.session_state.palm_image is None:
            st.info("Vui lòng chụp hoặc tải ảnh để bắt đầu!")
            return

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.image(st.session_state.palm_image, caption="Ảnh đã chọn", use_container_width=True)

        with col2:
            st.write("Nhấn nút bên dưới để phân tích chỉ tay")
            if st.button("Phân Tích Chỉ Tay", type="primary", use_container_width=True):
                self._handle_prediction()

    def _handle_prediction(self):
        if self.predictor is None:
            st.error("Chưa có predictor")
            return

        with st.spinner("Đang đọc đường chỉ tay…"):
            try:
                result = self.predictor.predict(st.session_state.palm_image)
                st.session_state.prediction_result = result
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

    @staticmethod
    def _derive_state(text):
        lowered = text.lower()
        if "không rõ" in lowered or "không xác định" in lowered:
            return "Không rõ"
        if "mờ nhạt" in lowered:
            return "Mờ nhạt"
        return "Rõ ràng"

    def _render_results(self):
        st.markdown(
            '<div class="pr-results-head">'
            '<div class="pr-eyebrow">Kết quả</div>'
            '<h2>Bốn Đạo Trong Lòng Bàn Tay</h2>'
            '</div>',
            unsafe_allow_html=True,
        )

        result = st.session_state.prediction_result

        if isinstance(result, pd.DataFrame) and {"Chỉ Số", "Kết Quả"}.issubset(result.columns):
            rows = list(result.itertuples(index=False))
        else:
            rows = []
            st.write(result)

        for row in rows:
            name, text = row[0], row[1]
            glyph, sub = DAO_MARKS.get(name, ("·", ""))
            state = self._derive_state(str(text))
            st.markdown(
                f'<div class="pr-dao">'
                f'<div class="pr-dao-mark">{glyph}<sub>{sub}</sub></div>'
                f'<div>'
                f'<div class="pr-dao-name">{name}</div>'
                f'<div class="pr-dao-state">{state}</div>'
                f'<div class="pr-dao-text">{text}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    from dataprocess import PalmPredictor

    predictor = PalmPredictor(model_path="best_palmline_model.keras")
    app = PalmReadingUI(predictor=predictor)
    app.run()
