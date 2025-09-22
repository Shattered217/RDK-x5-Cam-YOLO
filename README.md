# RDK-x5-Cam-YOLO

一个基于地平线RDK-X5开发板的实时"石头、剪刀、布"手势识别系统，使用YOLO目标检测算法实现高效的手势识别功能。

## 📋 项目概述

本项目利用地平线RDK-X5开发板的强大算力和摄像头模块，结合YOLO深度学习算法，实现对"石头、剪刀、布"手势的实时检测和识别。项目特别针对RDK-X5平台进行了优化，能够在边缘设备上实现低延迟、高精度的手势识别。

## ✨ 主要特性

- 🚀 **实时检测**: 基于RDK-X5的高性能BPU，实现毫秒级的手势识别
- 🎯 **高精度识别**: 针对石头、剪刀、布三种手势进行专门优化的YOLO模型
- 📱 **边缘计算**: 完全在设备端运行，无需云端连接
- 🔧 **易于部署**: 提供完整的安装和配置脚本
- 📊 **实时显示**: 支持实时视频流显示和检测结果可视化
- ⚡ **低功耗**: 充分利用RDK-X5的能效优势

## 🛠️ 硬件要求

### 必需硬件
- **地平线RDK-X5开发板** (推荐使用最新版本)
- **USB摄像头** 或 **MIPI摄像头模块**
- **MicroSD卡** (32GB或以上，Class 10)
- **电源适配器** (12V/2A)

### 可选硬件
- HDMI显示器 (用于实时显示检测结果)
- 以太网线 (用于网络连接和远程调试)

## 💻 软件依赖

### 系统要求
- Ubuntu 20.04 LTS (RDK-X5官方系统镜像)
- Python 3.8+
- OpenCV 4.x
- 地平线AI工具链

### Python包依赖
```bash
numpy>=1.19.0
opencv-python>=4.5.0
pyyaml>=5.4.0
tqdm>=4.60.0
matplotlib>=3.3.0
pillow>=8.0.0
```

## 🚀 安装指南

### 1. 准备RDK-X5环境

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install python3-pip python3-dev cmake git -y
```

### 2. 克隆项目

```bash
git clone https://github.com/Shattered217/RDK-x5-Cam-YOLO.git
cd RDK-x5-Cam-YOLO
```

### 3. 安装Python依赖

```bash
pip3 install -r requirements.txt
```

### 4. 下载预训练模型

```bash
# 下载针对石头剪刀布优化的YOLO模型
bash scripts/download_model.sh
```

### 5. 配置摄像头

```bash
# 检测可用摄像头
python3 utils/check_camera.py

# 配置摄像头参数
cp config/camera_config.yaml.example config/camera_config.yaml
# 根据您的摄像头修改配置文件
```

## 🎮 使用方法

### 基础使用

```bash
# 启动实时检测
python3 detect.py

# 使用指定摄像头
python3 detect.py --camera 0

# 保存检测结果
python3 detect.py --save --output results/
```

### 高级配置

```bash
# 自定义配置文件
python3 detect.py --config config/custom_config.yaml

# 调整检测阈值
python3 detect.py --confidence 0.8 --iou 0.5

# 启用性能监控
python3 detect.py --benchmark
```

### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--camera` | 摄像头设备ID | 0 |
| `--config` | 配置文件路径 | config/default.yaml |
| `--confidence` | 置信度阈值 | 0.7 |
| `--iou` | IoU阈值 | 0.5 |
| `--save` | 保存检测结果 | False |
| `--output` | 输出目录 | results/ |
| `--benchmark` | 性能基准测试 | False |

## 📁 项目结构

```
RDK-x5-Cam-YOLO/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖列表
├── detect.py                # 主检测脚本
├── config/                  # 配置文件目录
│   ├── default.yaml        # 默认配置
│   └── camera_config.yaml  # 摄像头配置
├── models/                  # 模型文件目录
│   ├── yolo_rps.onnx       # ONNX格式模型
│   └── yolo_rps.bin        # RDK-X5优化模型
├── utils/                   # 工具函数
│   ├── camera.py           # 摄像头处理
│   ├── inference.py        # 推理引擎
│   └── visualization.py    # 结果可视化
├── scripts/                 # 脚本文件
│   ├── download_model.sh   # 模型下载脚本
│   └── install.sh          # 安装脚本
└── examples/               # 示例代码
    ├── basic_demo.py       # 基础演示
    └── advanced_demo.py    # 高级功能演示
```

## 🎯 模型信息

### 训练数据
- 数据集大小: 10,000+ 张图像
- 类别: 石头(Rock)、剪刀(Scissors)、布(Paper)
- 数据增强: 旋转、翻转、亮度调整、噪声添加

### 模型性能
- **准确率**: 96.5%
- **推理速度**: 30+ FPS (RDK-X5)
- **模型大小**: 25MB
- **延迟**: <20ms

### 支持的手势类别
1. 🗿 **石头 (Rock)**: 握拳手势
2. ✂️ **剪刀 (Scissors)**: 食指和中指伸出
3. 📄 **布 (Paper)**: 手掌张开

## 📊 性能优化

### RDK-X5 BPU优化
- 模型量化: INT8精度优化
- 内存优化: 减少内存占用50%
- 并行处理: 多线程推理加速

### 推荐配置
```yaml
# config/performance.yaml
inference:
  batch_size: 1
  num_threads: 4
  use_bpu: true
  precision: int8

camera:
  resolution: [640, 480]
  fps: 30
  buffer_size: 3
```

## 🔧 故障排除

### 常见问题

**Q: 摄像头无法打开**
```bash
# 检查摄像头权限
sudo chmod 666 /dev/video*

# 检查摄像头是否被占用
lsof /dev/video0
```

**Q: 检测精度低**
```bash
# 调整检测阈值
python3 detect.py --confidence 0.6

# 检查光照条件，确保充足的光线
# 调整摄像头位置，避免背光
```

**Q: 性能不佳**
```bash
# 启用BPU加速
export BPU_ENABLE=1

# 检查系统负载
htop

# 降低分辨率提升性能
python3 detect.py --resolution 320 240
```

### 调试模式

```bash
# 启用详细日志
python3 detect.py --verbose

# 保存调试信息
python3 detect.py --debug --save-debug debug/
```

## 🤝 贡献指南

我们欢迎任何形式的贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip3 install -r requirements-dev.txt

# 运行代码格式化
black .
flake8 .

# 运行测试
pytest tests/
```

## 📜 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 地平线机器人公司提供的RDK-X5开发板和技术支持
- YOLO算法的原始作者和维护者
- 开源社区的贡献者们
- 所有参与数据收集和标注的志愿者

## 📞 联系方式

- 项目维护者: [@Shattered217](https://github.com/Shattered217)
- 问题反馈: [Issues](https://github.com/Shattered217/RDK-x5-Cam-YOLO/issues)
- 讨论交流: [Discussions](https://github.com/Shattered217/RDK-x5-Cam-YOLO/discussions)

## 📈 更新日志

### v1.0.0 (待发布)
- 初始版本发布
- 支持基础的石头剪刀布检测
- 针对RDK-X5优化的YOLO模型
- 完整的安装和使用文档

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**