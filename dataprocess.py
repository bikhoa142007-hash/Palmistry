import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

model = load_model("best_palmline_model.keras")  

class PalmPredictor:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_for_mobilenetv2_keras(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)
        return img_preprocessed

    def predict_palm_lines(self, image_path):
        img = self.preprocess_for_mobilenetv2_keras(image_path)
        predictions = self.model.predict(img)
        pre_d1 = np.argmax(predictions[0])
        pre_d2 = np.argmax(predictions[1])
        pre_d3 = np.argmax(predictions[2])
        pre_d4 = np.argmax(predictions[3])
        final_result = [pre_d1, pre_d2, pre_d3, pre_d4]
        return final_result

    def map_predictions_to_meanings(predictions):
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