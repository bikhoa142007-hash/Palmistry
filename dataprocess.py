import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


class PalmPredictor:

    def __init__(self, model_path="best_palmline_model.keras"):
        self.model = load_model(model_path)

    def preprocess_for_mobilenetv2_keras(self, image_input):
        if hasattr(image_input, "read"):
            image_input.seek(0)
            image = Image.open(image_input).convert("RGB")
            img_rgb = np.array(image)
        else:
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"khong the doc anh tai duong dan: {image_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed

    def predict_palm_lines(self, image_input):
        img = self.preprocess_for_mobilenetv2_keras(image_input)
        predictions = self.model.predict(img)
        pre_d1 = np.argmax(predictions[0])
        pre_d2 = np.argmax(predictions[1])
        pre_d3 = np.argmax(predictions[2])
        pre_d4 = np.argmax(predictions[3])
        return [pre_d1, pre_d2, pre_d3, pre_d4]

    def map_predictions_to_meanings(self, predictions):
        dict_sinhdao = {
            0: "Sức sống dồi dào, hệ miễn dịch tốt, năng lượng tràn đầy. Người này có khả năng chịu đựng cao và ít khi bị bệnh vặt",
            1: "Sức khỏe ổn định, cuộc sống cân bằng. Khi gặp khó khăn về thể chất, cơ thể có khả năng tự phục hồi tốt",
            2: "Sinh Đạo Không Rõ"
        }
        dict_tamdao = {
            0: "Tâm Đạo Rõ Ràng",
            1: "Tâm Đạo Mờ Nhạt",
            2: "Tâm Đạo Không Rõ"
        }
        dict_tridao = {
            0: "Trí Đạo Rõ Ràng",
            1: "Trí Đạo Mờ Nhạt",
            2: "Trí Đạo Không Rõ"
        }
        dict_vanmenh = {
            0: "Vận Mệnh Rõ Ràng",
            1: "Vận Mệnh Mờ Nhạt",
            2: "Vận Mệnh Không Rõ"
        }
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        result = {
            "Sinh Đạo": dict_sinhdao.get(pre_d1, "Không xác định"),
            "Tâm Đạo": dict_tamdao.get(pre_d2, "Không xác định"),
            "Trí Đạo": dict_tridao.get(pre_d3, "Không xác định"),
            "Vận Mệnh": dict_vanmenh.get(pre_d4, "Không xác định")
        }
        return result

    def predict(self, image_input):
        predictions = self.predict_palm_lines(image_input)
        meanings = self.map_predictions_to_meanings(predictions)
        df = pd.DataFrame(list(meanings.items()), columns=["Chỉ Số", "Kết Quả"])
        return df