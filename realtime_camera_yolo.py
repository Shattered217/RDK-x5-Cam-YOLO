#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 一个为“石头、剪刀、布”模型定制的精简版实时检测脚本

import os
import cv2
import numpy as np
import argparse
import logging
from time import time

# --- 依赖检查 ---
try:
    from scipy.special import softmax
except ImportError:
    print("Scipy未安装，正在尝试自动安装...")
    os.system("pip install --user scipy")
    from scipy.special import softmax

try:
    try:
        from hobot_dnn import pyeasy_dnn as dnn
    except ImportError:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
except ImportError:
    print("错误：无法导入hobot_dnn模块。请确保已安装 hobot-dnn-rdkx5。")
    exit(1)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RPS_Detector")

my_class_names = ['Paper', 'Rock', 'Scissors']
# 3个类别自定义颜色 (BGR格式)
my_colors = [(255, 56, 56), (56, 255, 56), (56, 56, 255)]

# --- 绘图函数 ---
def draw_detection(img, bbox, score, class_id) -> None:
    x1, y1, x2, y2 = bbox
    color = my_colors[class_id % len(my_colors)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    label = f"{my_class_names[class_id]}: {score:.2f}"
    
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (x1, label_y - label_height), (x1 + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class RockPaperScissors_Detector:
    def __init__(self, model_path):
        # --- 硬编码模型参数 ---
        self.CLASSES_NUM = 3  # 类别数量
        self.SCORE_THRESHOLD = 0.5  # 置信度阈值 
        self.NMS_THRESHOLD = 0.6    # NMS阈值
        self.REG = 16
        self.strides = [8, 16, 32]
        
        # 加载模型
        try:
            self.quantize_model = dnn.load(model_path)
            logger.info(f"模型 '{model_path}' 加载成功。")
        except Exception as e:
            logger.error(f"❌ 加载模型文件失败: {model_path}\n{e}")
            exit(1)

        # 获取模型输入尺寸
        self.input_H, self.input_W = self.quantize_model[0].inputs[0].properties.shape[2:4]
        self.CONF_THRES_RAW = -np.log(1 / self.SCORE_THRESHOLD - 1)
        self.weights_static = np.array([i for i in range(self.REG)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        
        # 预计算anchors
        self.grids = []
        for stride in self.strides:
            grid_H, grid_W = self.input_H // stride, self.input_W // stride
            self.grids.append(np.stack([
                np.tile(np.linspace(0.5, grid_W - 0.5, grid_W), reps=grid_H),
                np.repeat(np.linspace(0.5, grid_H - 0.5, grid_H), grid_W)
            ], axis=0).transpose(1, 0))
        
        logger.info(f"模型初始化完成, 输入尺寸: {self.input_W}x{self.input_H}")

    def preprocess(self, img):
        self.img_h, self.img_w = img.shape[0:2]
        scale = min(self.input_H / self.img_h, self.input_W / self.img_w)
        self.scale = scale # 保存scale供后处理使用

        new_w, new_h = int(self.img_w * scale), int(self.img_h * scale)
        self.pad_w = (self.input_W - new_w) // 2
        self.pad_h = (self.input_H - new_h) // 2
        
        resized_img = cv2.resize(img, (new_w, new_h))
        padded_img = cv2.copyMakeBorder(resized_img, self.pad_h, self.input_H - new_h - self.pad_h, self.pad_w, self.input_W - new_w - self.pad_w, cv2.BORDER_CONSTANT, value=[127, 127, 127])
        
        # BGR to NV12 for BPU
        area = self.input_H * self.input_W
        yuv420p = cv2.cvtColor(padded_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        return np.concatenate((yuv420p[:area], uv_packed))

    def forward(self, input_tensor):
        outputs = self.quantize_model[0].forward(input_tensor)
        return [dnnTensor.buffer for dnnTensor in outputs]

    def postprocess(self, outputs):
        clses = [outputs[0].reshape(-1, self.CLASSES_NUM), outputs[2].reshape(-1, self.CLASSES_NUM), outputs[4].reshape(-1, self.CLASSES_NUM)]
        bboxes = [outputs[1].reshape(-1, self.REG * 4), outputs[3].reshape(-1, self.REG * 4), outputs[5].reshape(-1, self.REG * 4)]
        
        dbboxes, ids, scores = [], [], []
        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):    
            max_scores = np.max(cls, axis=1)
            selected_indices = np.flatnonzero(max_scores >= self.CONF_THRES_RAW)
            if len(selected_indices) == 0: continue
            
            ids.append(np.argmax(cls[selected_indices, :], axis=1))
            scores.append(1 / (1 + np.exp(-max_scores[selected_indices])))
            
            ltrb = np.sum(softmax(bbox[selected_indices, :].reshape(-1, 4, self.REG), axis=2) * self.weights_static, axis=2)
            grid_selected = grid[selected_indices, :]
            dbboxes.append(np.hstack([(grid_selected - ltrb[:, 0:2]), (grid_selected + ltrb[:, 2:4])]) * stride)

        if not dbboxes: return []
        
        dbboxes = np.concatenate(dbboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)
        
        results = []
        for i in range(self.CLASSES_NUM):
            class_mask = (ids == i)
            if not np.any(class_mask): continue
            
            class_bboxes_ltrb = dbboxes[class_mask, :]
            class_scores = scores[class_mask]

            # LTRB to XYWH for NMS
            xywh = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in class_bboxes_ltrb])
            
            indices = cv2.dnn.NMSBoxes(xywh, class_scores, self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
            
            for idx in indices:
                x1, y1, x2, y2 = class_bboxes_ltrb[idx]
                # 坐标还原
                x1 = int((x1 - self.pad_w) / self.scale)
                y1 = int((y1 - self.pad_h) / self.scale)
                x2 = int((x2 - self.pad_w) / self.scale)
                y2 = int((y2 - self.pad_h) / self.scale)
                
                # 裁剪到图像边界内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.img_w - 1, x2), min(self.img_h - 1, y2)
                
                results.append((i, class_scores[idx], x1, y1, x2, y2))
                
        return results

# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser(description="为“石头、剪刀、布”模型定制的实时检测脚本")
    parser.add_argument('--model-path', type=str, required=True, help="您的 *.bin 模型文件路径。")
    parser.add_argument('--camera-id', type=int, default=0, help="摄像头设备ID号。")
    args = parser.parse_args()

    # 初始化模型处理类
    model = RockPaperScissors_Detector(model_path=args.model_path)
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logger.error(f"错误: 无法打开摄像头 {args.camera_id}。")
        return
    logger.info(f"摄像头 {args.camera_id} 打开成功, 按 'q' 键退出。")

    # 实时检测循环
    while True:
        start_time = time()
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = model.preprocess(frame)
        outputs = model.forward(input_tensor)
        results = model.postprocess(outputs)
        
        for class_id, score, x1, y1, x2, y2 in results:
            draw_detection(frame, (x1, y1, x2, y2), score, class_id)
            
        fps = 1 / (time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Rock-Paper-Scissors Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("程序已退出。")

if __name__ == "__main__":
    main()
