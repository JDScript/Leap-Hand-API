# Leap Hand API

用于控制 Leap Hand 机器人的 Python API 和工具集。

## 功能特性

### 核心 API
- 🤖 **Leap Hand 控制API**: 直接控制 Leap Hand 机器人的关节
- ⚙️ **Dynamixel 驱动**: 底层 Dynamixel 舵机通信
- 🔄 **格式转换**: Allegro Hand 和 Leap Hand 格式之间的转换

### Teleoperation (新功能!)
- 🎯 **实时手势控制**: 使用摄像头和 MediaPipe 进行手势检测
- 🤝 **手势重定向**: 将人手动作映射到机器人手部
- 📊 **实时可视化**: 使用 Rerun 进行 3D 可视化
- 🔧 **灵活配置**: 支持多种重定向算法和参数

## 快速开始

### 安装依赖

```bash
# 基础依赖
uv sync

# Teleoperation 依赖
uv sync --group teleop
```

### 基础 API 使用

```python
from source.leap.api import LeapHand
from source.leap.dynamixel.driver import DynamixelDriver

# 初始化驱动
driver = DynamixelDriver(
    servo_ids=list(range(16)),
    port="/dev/cu.usbserial-FTA2U4SR",
    baud_rate=4_000_000,
)

# 创建 Leap Hand 实例
leap_hand = LeapHand(driver)

# 控制机器人
import numpy as np
leap_hand.set_joints_leap(np.zeros(16))  # 移动到零位
```

### Teleoperation 使用

**仅可视化模式（推荐首次使用）：**
```bash
uv run --group teleop python -m source.leap_teleop.main --no-enable-real-robot
```

**连接真实机器人：**
```bash
uv run --group teleop python -m source.leap_teleop.main --enable-real-robot
```

## 项目结构

```
leap-hand-api/
├── source/
│   ├── leap/                   # 核心 API
│   │   ├── api.py             # 主要 Leap Hand API
│   │   ├── utils.py           # 工具函数
│   │   └── dynamixel/         # Dynamixel 驱动
│   └── leap_teleop/           # Teleoperation 系统
│       ├── main.py            # 主程序
│       ├── hand_detector.py   # 手势检测
│       ├── visualizer.py      # 可视化
│       └── README.md          # 详细文档
├── vendor/                     # 第三方库
│   ├── dynamixel-sdk/         # Dynamixel SDK
│   └── leap-urdf/             # Leap Hand URDF 模型
└── third_party/
    └── dex_retargeting/       # 手势重定向库
```

## Teleoperation 系统

### 核心功能

1. **实时手势检测**: 使用 MediaPipe 检测手部关键点
2. **智能重定向**: 使用 dex-retargeting 将人手动作映射到机器人
3. **安全控制**: 支持实时开启/关闭机器人控制
4. **多模式可视化**: OpenCV + Rerun 双重可视化

### 支持的重定向算法

- **Vector Retargeting**: 适用于实时遥操作
- **Position Retargeting**: 适用于离线数据处理  
- **DexPilot**: 带有手指闭合预处理的遥操作

### 控制说明

- **Q 键**: 退出程序
- **空格键**: 开启/关闭机器人控制
- 将右手放在摄像头前进行手势控制

### 可视化

访问 Rerun 可视化界面：http://127.0.0.1:9876

可视化内容包括：
- 手部关键点 3D 位置
- 机器人关节状态
- 系统运行状态

## 安全注意事项

⚠️ **重要**: 在操作真实机器人前，请务必：

1. 首先在可视化模式下测试
2. 确保机器人周围有足够空间
3. 保持紧急停止按钮可及
4. 仔细阅读 `source/leap_teleop/README.md` 中的安全说明

## 示例和教程

### Jupyter Notebook 示例
- `dynamixel_demo.ipynb`: Dynamixel 基础控制示例

### Teleoperation 示例
```bash
# 查看所有可用选项
uv run --group teleop python -m source.leap_teleop.main --help

# 左手控制
uv run --group teleop python -m source.leap_teleop.main --hand-type left

# 调整帧率
uv run --group teleop python -m source.leap_teleop.main --fps 60
```

## 开发

### 添加新功能

1. 核心 API 扩展: 修改 `source/leap/`
2. Teleoperation 功能: 修改 `source/leap_teleop/`
3. 新的重定向算法: 参考 `dex_retargeting` 文档

### 测试

```bash
# 运行基础 API 测试
python -c "from source.leap.api import LeapHand; print('API 导入成功')"

# 测试 teleoperation 导入
uv run --group teleop python -c "from source.leap_teleop import LeapHandTeleop; print('Teleop 导入成功')"
```

## 故障排除

### 常见问题

1. **串口连接失败**: 检查设备路径和权限
2. **摄像头无法打开**: 确认摄像头未被其他程序占用
3. **手势检测不准确**: 确保光线充足，手部完整在视野内
4. **Rerun 无法连接**: 检查端口 9876 是否被占用

更多详细故障排除信息请参考 `source/leap_teleop/README.md`。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

请查看项目根目录下的 LICENSE 文件。
