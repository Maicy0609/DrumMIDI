# DrumMIDI （this shit readme made by ai)

## English

DrumMIDI is a simple tool that converts MIDI files to pulse audio using custom hit samples. It helps you turn MIDI music into rhythmic pulse patterns with your own sounds.

**What it does:**
- Takes a MIDI file and extracts all the notes
- Converts the notes to a simple text format
- Uses your custom hit sample to generate pulse audio
- Creates a log file showing exactly when each hit occurs

**Why use it:**
- Easy to use with a simple command-line interface
- Works with any MIDI file
- Use your own hit samples for personalized sound
- Get precise timing information for every hit

## 中文

DrumMIDI 是一个简单的工具，可以使用自定义打击采样将 MIDI 文件转换为脉冲音频。它可以帮助你将 MIDI 音乐转换为带有自己声音的节奏脉冲模式。

**功能介绍：**
- 读取 MIDI 文件并提取所有音符
- 将音符转换为简单的文本格式
- 使用自定义打击采样生成脉冲音频
- 创建日志文件，显示每个打击的确切时间

**使用理由：**
- 易于使用，具有简单的命令行界面
- 适用于任何 MIDI 文件
- 使用自己的打击采样实现个性化声音
- 获取每个打击的精确时序信息

## Instructions for use

### Install dependencies

```bash
pip install mido numpy scipy
```

### Run program

```bash
python main.py
```

Follow the prompts:
1. Select the MIDI file to convert
2. Choose the percussion sample file (default is hit.wav)
3. Wait for the conversion to complete

## Project Structure

```plaintext
DrumMIDI/
├── main.py              # 主程序入口
├── midi_drum_lib.py     # 核心处理函数
├── hit.wav              # 默认打击采样（可选）
└── README.md            # 说明文档
```
