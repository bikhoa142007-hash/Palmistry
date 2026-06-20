import streamlit as st
from dataprocess import PalmPredictor
from palm_ui import PalmReadingUI

if __name__ == "__main__":
    predictor = PalmPredictor()
    app = PalmReadingUI(predictor)
    app.run()