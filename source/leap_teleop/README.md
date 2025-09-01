# Leap Hand Teleoperation

基于 MediaPipe 手势检测和 dex-retargeting 的 Leap Hand 实时遥操作系统。

## 功能特性

- 🎯 实时手势检测和跟踪（使用 MediaPipe）
- 🤖 手势到机器人关节的重定向（使用 dex-retargeting）
- 📊 实时可视化（使用 Rerun）
- 🦾 支持真实 Leap Hand 控制
- 🔧 可配置的参数和设置

## 系统要求

- Python 3.10+
- 网络摄像头
- Leap Hand 机器人（可选，用于真实机器人控制）

## 安装

确保您已经在项目根目录下，然后安装 teleoperation 依赖：

```bash
uv add opencv-python mediapipe loguru --group teleop
```

## 使用方法

### 仅可视化模式（推荐首次使用）

```bash
uv run --group teleop python -m source.leap_teleop.main --no-enable-real-robot
```

这将启动：
- 摄像头输入窗口，显示手势检测结果
- Rerun 可视化服务器（访问 http://127.0.0.1:9876）

### 真实机器人控制模式

```bash
uv run --group teleop python -m source.leap_teleop.main --enable-real-robot --servo-port YOUR_PORT
```

⚠️ **注意**：在控制真实机器人前，请确保：
1. 机器人处于安全位置
2. 有足够的空间进行操作
3. 随时准备紧急停止

## 控制说明

- **Q 键**：退出程序
- **空格键**：开启/关闭机器人控制（在真实机器人模式下）
- 将右手放在摄像头前进行手势控制

## 可视化

程序提供多种可视化选项：

1. **OpenCV 窗口**：显示摄像头画面和手势骨架
2. **Rerun 可视化**：
   - 手部关键点 3D 可视化
   - 机器人关节状态
   - 系统状态信息

访问 Rerun 可视化：
```bash
# 在浏览器中打开
open http://127.0.0.1:9876

# 或使用 Rerun 客户端连接
rerun --connect rerun+http://127.0.0.1:9876/proxy
```

## 命令行参数

```bash
python -m source.leap_teleop.main [OPTIONS]

选项：
  --retargeting-type {vector,position,dexpilot}  重定向算法类型 (默认: vector)
  --hand-type {right,left}                       跟踪的手部 (默认: right)
  --robot-urdf-path STR                          机器人 URDF 文件路径 (可选)
  --enable-real-robot / --no-enable-real-robot   是否连接真实机器人 (默认: False)
  --servo-port STR                               串口设备路径 (默认: /dev/cu.usbserial-FTA2U4SR)
  --baud-rate INT                                串口波特率 (默认: 4000000)
  --camera-id INT                                摄像头设备 ID (默认: 0)
  --fps INT                                      帧率 (默认: 30)
```

## 系统架构

```
摄像头输入 → MediaPipe 手势检测 → dex-retargeting → Leap Hand 控制
     ↓
   OpenCV 显示 ← Rerun 可视化 ← 状态监控
```

### 核心组件

1. **HandDetector**: MediaPipe 手势检测包装器
2. **LeapHandTeleop**: 主要的 teleoperation 类
3. **LeapHandVisualizer**: Rerun 可视化界面
4. **RetargetingConfig**: dex-retargeting 配置

## 配置

系统使用以下默认配置：

- **重定向类型**: Vector retargeting
- **缩放因子**: 1.6 (Leap Hand 比人手大 1.6 倍)
- **低通滤波**: α = 0.2 (平滑运动)
- **目标链接**: thumb_fingertip, fingertip, fingertip_2, fingertip_3

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 尝试不同的 `--camera-id` 值

2. **手势检测不准确**
   - 确保光线充足
   - 保持手部在摄像头视野内
   - 尝试调整手部距离

3. **机器人连接失败**
   - 检查串口路径是否正确
   - 确认波特率设置
   - 检查机器人电源和连接

4. **Rerun 可视化无法连接**
   - 检查端口 9876 是否被占用
   - 尝试刷新浏览器页面

## 开发

### 文件结构

```
source/leap_teleop/
├── __init__.py          # 包初始化
├── main.py             # 主程序入口
├── hand_detector.py    # 手势检测模块
├── visualizer.py       # 可视化模块
└── assets/
    └── leap_right/     # Leap Hand URDF 文件
        ├── leap_hand_right.urdf
        └── *.stl       # 3D 模型文件
```

### 添加新功能

要添加新的重定向算法或修改配置，请编辑 `main.py` 中的 `_setup_retargeting` 方法。

## 安全注意事项

⚠️ **重要安全提醒**：

1. 首次使用时，始终在可视化模式下测试
2. 确保机器人周围有足够的空间
3. 保持紧急停止按钮在手边
4. 不要让机器人接触人体或重要物品
5. 定期检查机器人的机械状态

## 许可证

本项目遵循与主项目相同的许可证。
