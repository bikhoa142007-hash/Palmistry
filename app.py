import streamlit as st
from dataprocess import PalmPredictor
from PalmReadingUI import PalmReadingUI


if __name__ == "__main__":
    predictor = PalmPredictor("best_palmline_model.keras")
    app = PalmReadingUI(predictor)
    app.run()