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
            2: "Sức bền thể chất hơi kém, dễ bị mệt mỏi hoặc kiệt sức khi làm việc quá tải. (Lưu ý: Triết học và khoa học hiện đại chứng minh đường này ngắn khum có nghĩa là chết sớm đâu nha nhóm, chỉ là thể lực yếu hơn thôi!)"
        }
        dict_tamdao = {
            0: "Tình cảm dồi dào, lãng mạn nhưng có xu hướng đặt kỳ vọng cao vào đối phương. Người này rất chung thủy nhưng dễ bị tổn thương nếu tình cảm không được đáp lại",
            1: "Kiểm soát cảm xúc tốt, yêu ghét rõ ràng và thực tế trong tình cảm. Họ dùng cả con tim lẫn lý trí khi yêu",
            2: "Người sống lý trí, ít khi để cảm xúc chi phối công việc. Đôi khi họ bị nhận xét là hơi khô khan hoặc khép kín."
        }
        dict_tridao = {
            0: "Tư duy logic, phân tích sâu sắc, trí nhớ tốt và có xu hướng suy nghĩ rất kỹ trước khi hành động. Thích hợp làm các công việc nghiên cứu, lập trình hay chiến lược",
            1: "Trí tuệ thực tế, nhạy bén và có sự cân bằng giữa lý thuyết với thực hành. Người này thích nghi tốt với môi trường mới",
            2: "Tư duy trực diện, thích sự đơn giản, nhanh gọn. Họ giỏi giải quyết các vấn đề thực tế trước mắt hơn là ngồi hoạch định chiến lược dài hạn"
        }
        dict_vanmenh = {
            0: "Sự nghiệp định hình từ rất sớm, có mục tiêu rõ ràng và có nhiều cơ hội thăng tiến lớn trong đời",
            1: "Sự nghiệp có sự thay đổi hoặc chỉ thực sự bứt phá và ổn định ở độ tuổi trung niên (sau 30 tuổi)",
            2: "Người này có xu hướng thích cuộc sống tự do, không thích bị gò bó vào một công việc cố định. Họ thích tự trải nghiệm và thay đổi hướng đi tùy theo hoàn cảnh khách quan của cuộc sống"
        }
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        result = {
            "Sinh Đạo": dict_sinhdao.get(pre_d1, "Không xác định"),
            "Tâm Đạo": dict_tamdao.get(pre_d2, "Không xác định"),
            "Trí Đạo": dict_tridao.get(pre_d3, "Không xác định"),
            "Vận Mệnh": dict_vanmenh.get(pre_d4, "Không xác định")
        }
        return result

    def _get_line_characteristic(self, line_name, prediction_value):
        characteristics = {
            "sinh_dao": {0: "ổn định và rõ nét", 1: "khá rõ nhưng có đoạn mờ", 2: "không rõ hoặc đứt đoạn"},
            "tam_dao": {0: "sâu sắc và chân thành", 1: "tinh tế nhưng đôi khi do dự", 2: "khép kín và khó hiểu"},
            "tri_dao": {0: "logic và thực tế", 1: "sáng tạo và linh hoạt", 2: "thiếu tập trung"},
            "van_menh": {0: "rõ ràng và thuận lợi", 1: "có nhiều trải nghiệm đa dạng", 2: "tự kiến tạo con đường riêng"}
        }
        return characteristics.get(line_name, {}).get(prediction_value, "không xác định")

    def _generate_advice_type1(self, predictions):
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        sinh_dao_desc = self._get_line_characteristic("sinh_dao", pre_d1)
        tri_dao_desc = self._get_line_characteristic("tri_dao", pre_d3)
        tam_dao_desc = self._get_line_characteristic("tam_dao", pre_d2)
        van_menh_desc = self._get_line_characteristic("van_menh", pre_d4)
        advice = (
            f"Xu hướng cân bằng, chín chắn và ổn định"
            f"Bạn là người biết cách duy trì nhịp sống hài hòa, luôn chú trọng đến sức khỏe và sự dẻo dai của bản thân nhờ đường sinh đạo {sinh_dao_desc}. "
            f"Trong tư duy, bạn có lối suy nghĩ {tri_dao_desc}, luôn cân nhắc kỹ lưỡng mọi khía cạnh trước khi đưa ra quyết định dựa trên đặc điểm của đường trí đạo. "
            f"Thế giới cảm xúc của bạn mang tính chất {tam_dao_desc}, biết cách lắng nghe và thấu hiểu người khác qua đường tâm đạo, từ đó tạo nền tảng vững chắc giúp lộ trình công danh và cuộc sống của bạn tiến triển một cách thuận lợi theo định hướng của đường định mệnh {van_menh_desc}.\n\n"
            f"Lời khuyên:\n"
            f"✓ Duy trì lối sống cân bằng và chú trọng sức khỏe\n"
            f"✓ Phát huy tư duy logic và khả năng phân tích\n"
            f"✓ Giữ vững sự chân thành trong các mối quan hệ\n"
            f"✓ Tận dụng sự ổn định để phát triển sự nghiệp bền vững"
        )
        return advice

    def _generate_advice_type2(self, predictions):
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        sinh_dao_desc = self._get_line_characteristic("sinh_dao", pre_d1)
        tri_dao_desc = self._get_line_characteristic("tri_dao", pre_d3)
        tam_dao_desc = self._get_line_characteristic("tam_dao", pre_d2)
        van_menh_desc = self._get_line_characteristic("van_menh", pre_d4)
        advice = (
            f"Xu hướng linh hoạt, thích ứng và hướng ngoại"
            f"Sở hữu một tinh thần lạc quan, năng động và khả năng phục hồi năng lượng thể chất khá tốt qua đường sinh đạo {sinh_dao_desc}, "
            f"bạn còn là người có óc sáng tạo, nhạy bén và thích nghi nhanh chóng với những thay đổi của môi trường xung quanh thể hiện ở đường trí đạo {tri_dao_desc}. "
            f"Bạn luôn sống thiên về tình cảm, {tam_dao_desc} và dễ tạo được thiện cảm lớn với những người xung quanh nhờ đường tâm đạo, "
            f"điều này giúp cuộc đời và sự nghiệp của bạn có nhiều trải nghiệm phong phú, đa dạng và không ngừng dịch chuyển theo hướng tích cực của đường định mệnh {van_menh_desc}.\n\n"
            f"Lời khuyên:\n"
            f"✓ Phát huy khả năng thích ứng và sáng tạo\n"
            f"✓ Tận dụng sự linh hoạt để nắm bắt cơ hội mới\n"
            f"✓ Xây dựng mạng lưới quan hệ rộng rãi\n"
            f"✓ Không ngừng học hỏi và trải nghiệm những điều mới mẻ"
        )
        return advice

    def _generate_advice_type3(self, predictions):
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        sinh_dao_desc = self._get_line_characteristic("sinh_dao", pre_d1)
        tri_dao_desc = self._get_line_characteristic("tri_dao", pre_d3)
        tam_dao_desc = self._get_line_characteristic("tam_dao", pre_d2)
        van_menh_desc = self._get_line_characteristic("van_menh", pre_d4)
        advice = (
            f"Xu hướng độc lập, lý trí và kiên định"
            f"Bản tính bạn là người có sự độc lập cao trong cuộc sống, thích tự chủ và tự rèn luyện thể lực cũng như ý chí kiên cường qua đường sinh đạo {sinh_dao_desc}. "
            f"Bạn sở hữu một tư duy rất thực chiến, trực diện, tập trung cao độ vào những mục tiêu cụ thể và không thích sự dài dòng từ đường trí đạo {tri_dao_desc}. "
            f"Trong các mối quan hệ, bạn luôn giữ được sự tỉnh táo, lý trí và kiểm soát cảm xúc cá nhân rất tốt nhờ đường tâm đạo {tam_dao_desc}, "
            f"chính sự kiên định này là chìa khóa giúp bạn tự kiến tạo nên những bước tiến rõ ràng và gặt hái thành quả xứng đáng trên con đường sự nghiệp của đường định mệnh {van_menh_desc}.\n\n"
            f"Lời khuyên:\n"
            f"✓ Duy trì tính độc lập và tự chủ\n"
            f"✓ Tập trung vào mục tiêu dài hạn\n"
            f"✓ Cân bằng giữa lý trí và cảm xúc\n"
            f"✓ Kiên định với lựa chọn của bản thân"
        )
        return advice

    def generate_advice(self, predictions):
        pre_d1, pre_d2, pre_d3, pre_d4 = predictions
        clarity_score = 0
        balance_score = 0
        for pred in predictions:
            if pred == 0:
                clarity_score += 1
            elif pred == 1:
                balance_score += 1
        if clarity_score >= 3:
            advice = self._generate_advice_type3(predictions)
        elif balance_score >= 2 or (clarity_score <= 1 and balance_score >= 1):
            advice = self._generate_advice_type2(predictions)
        else:
            advice = self._generate_advice_type1(predictions)
        return advice

    def predict(self, image_input):
        predictions = self.predict_palm_lines(image_input)
        meanings = self.map_predictions_to_meanings(predictions)
        advice = self.generate_advice(predictions)
        df = pd.DataFrame(list(meanings.items()), columns=["Chỉ Số", "Kết Quả"])
        return {
            "dataframe": df,
            "advice": advice,
            "predictions": predictions
        }