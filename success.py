import cv2
import numpy as np
import onnxruntime as ort
from paddleocr import PaddleOCR
import time
import os
from PIL import Image, ImageDraw, ImageFont

class ChineseTextRenderer:
    def __init__(self):
        try:
            # 尝试从Windows字体目录加载黑体
            font_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'simhei.ttf')
            if not os.path.exists(font_path):
                font_path = "simhei.ttf"
            self.font = ImageFont.truetype(font_path, 24)
        except Exception:
            # 保底方案：使用默认字体（可能无法显示中文）
            self.font = ImageFont.load_default()
            print("警告：中文字体加载失败，部分中文可能显示异常")

    def put_text(self, img, text, pos, color):
        """稳健的中文文本渲染方法"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            # 兼容新版Pillow
            text_bbox = self.font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # 旧版Pillow
            text_w, text_h = self.font.getsize(text)
        
        # 调整位置防止越界
        x = max(10, min(pos[0], img.shape[1] - text_w - 10))
        y = max(text_h, min(pos[1], img.shape[0] - text_h - 10))
        
        draw.text((x, y), text, font=self.font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class LicensePlateSystem:
    def __init__(self):
        # 初始化OCR（忽略ccache警告）
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang='ch',
            use_gpu=False,
            show_log=False
        )
        
        # 加载YOLOv5模型（修复维度问题）
        model_path = "E:/yoloproject/yolov5/runs/train/exp7/weights/best.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # ONNX Runtime配置
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"模型输入形状: {self.input_shape}")
        
        # 检测参数
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4
        self.target_size = 640  # YOLOv5标准输入尺寸
        
        # 显示工具
        self.text_renderer = ChineseTextRenderer()
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()

    def _preprocess(self, frame):
        """完全兼容的YOLOv5预处理"""
        # 保持纵横比resize
        h, w = frame.shape[:2]
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 转换颜色空间+resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (new_w, new_h))
        
        # 创建填充后的图像
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        padded[:new_h, :new_w] = img.astype(np.float32) / 255.0
        
        # 调整维度顺序：HWC -> NCHW
        padded = np.transpose(padded, (2, 0, 1))  # CHW
        return np.expand_dims(padded, axis=0), scale  # NCHW

    def _postprocess(self, outputs, scale, frame_shape):
        """优化的后处理流程"""
        original_h, original_w = frame_shape[:2]
        
        # 确保输出处理兼容不同版本的YOLOv5
        outputs = np.squeeze(outputs[0])
        if outputs.ndim == 1:
            outputs = outputs.reshape(1, -1)
        
        # 过滤低置信度检测
        scores = outputs[:, 4]
        mask = scores > self.conf_threshold
        outputs = outputs[mask]
        scores = scores[mask]
        
        # 转换坐标到原图空间
        boxes = []
        for i in range(len(outputs)):
            cx, cy, bw, bh = outputs[i, :4]
            # 反算到原始图像坐标
            x1 = int((cx - bw/2) / scale)
            y1 = int((cy - bh/2) / scale)
            x2 = int((cx + bw/2) / scale)
            y2 = int((cy + bh/2) / scale)
            
            # 裁剪到图像范围内
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, float(scores[i])])
        
        # 应用NMS
        if len(boxes) > 0:
            boxes_array = np.array([box[:4] for box in boxes])
            scores_array = np.array([box[4] for box in boxes])
            
            indices = cv2.dnn.NMSBoxes(boxes_array.tolist(),
                                      scores_array.tolist(),
                                      self.conf_threshold,
                                      self.iou_threshold)
            
            if indices is not None:
                if isinstance(indices, np.ndarray):
                    return [boxes[i] for i in indices.flatten()]
                return [boxes[i] for i in indices]  # 兼容不同OpenCV版本
        return []

    def detect_plates(self, frame):
        """完整的车牌检测流程"""
        # 正确的维度预处理
        blob, scale = self._preprocess(frame)
        
        # 模型推理（确保输入维度正确）
        outputs = self.session.run(None, {self.input_name: blob})
        
        # 后处理
        return self._postprocess(outputs, scale, frame.shape)

    def recognize_plate(self, plate_img):
        """稳健的车牌识别方法"""
        try:
            # 图像增强
            plate_img = cv2.resize(plate_img, None, fx=1.8, fy=1.8,
                                 interpolation=cv2.INTER_CUBIC)
            
            # 灰度化+直方图均衡化
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
            gray = cv2.equalizeHist(gray)
            
            # OCR识别
            result = self.ocr.ocr(gray, cls=False)
            
            if result and result[0]:
                # 只保留高置信度结果
                text = ''.join([line[1][0] 
                              for line in result[0] 
                              if line[1][1] > 0.7])
                return text if text else None
        except Exception as e:
            print(f"OCR处理异常: {str(e)}")
        return None

    def run(self):
        """主运行循环"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("摄像头初始化失败")
            return
        
        try:
            # 设置合适的摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("视频帧读取失败")
                    break
                
                # 车牌检测
                detections = self.detect_plates(frame)
                plate_results = []
                
                for (x1, y1, x2, y2, score) in detections:
                    # 根据置信度选择框颜色
                    color = (0, 255, 0) if score > 0.7 else (0, 200, 255)
                    thickness = 2 if score > 0.7 else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # 车牌识别
                    plate_roi = frame[y1:y2, x1:x2]
                    if plate_roi.size > 0:
                        text = self.recognize_plate(plate_roi)
                        if text:
                            # 显示识别结果
                            frame = self.text_renderer.put_text(
                                frame, text, (x1, y1-30), (0, 0, 255)
                            )
                            plate_results.append((text, score))
                
                # 左上角显示汇总信息
                y_offset = 40
                for idx, (text, score) in enumerate(plate_results):
                    line = f"{idx+1}. {text} ({score:.2f})"
                    frame = self.text_renderer.put_text(
                        frame, line, (20, y_offset), (255, 255, 255)
                    )
                    y_offset += 35
                
                # 计算并显示FPS
                self.frame_count += 1
                if self.frame_count % 5 == 0:
                    fps = 5 / (time.time() - self.start_time)
                    self.start_time = time.time()
                    frame = self.text_renderer.put_text(
                        frame, f"FPS: {fps:.1f}", (20, frame.shape[0]-40), (0, 255, 0)
                    )
                
                # 显示结果
                cv2.imshow("车牌识别系统", frame)
                if cv2.waitKey(1) == 27:  # ESC退出
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        print("正在初始化车牌识别系统...")
        system = LicensePlateSystem()
        print("系统初始化完成，开始运行...")
        system.run()
    except Exception as e:
        print(f"系统错误: {str(e)}")
