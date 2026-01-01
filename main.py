import sys
import os
from midi_drum_lib import (
    midi_to_text, 
    txt_to_pulse_synthesizer,
    get_user_input,
    validate_file_path
)

def scan_files(extension: str, script_dir: str):
    """扫描目录下指定扩展名的文件"""
    try:
        files = [f for f in os.listdir(script_dir) if f.lower().endswith(extension)]
        return [os.path.join(script_dir, f) for f in files]
    except:
        return []

def select_file(files: list, file_type: str) -> str:
    """显示文件列表"""
    if not files:
        return None
    
    print(f"\n发现以下{file_type}文件:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    while True:
        choice = input(f"请选择文件编号（或按回车手动输入路径）: ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("无效选择，请重试。")

def main():
    print("=" * 50)
    print("MIDI转HitSound")
    print("=" * 50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 步骤1: MIDI转TXT
    print("\n--- 步骤1: MIDI转TXT ---")
    
    # 自动扫描MIDI文件
    midi_files = scan_files('.mid', script_dir)
    midi_path = select_file(midi_files, "MIDI")
    
    if not midi_path:
        midi_path = get_user_input(
            "MIDI文件路径: ",
            validate_func=validate_file_path
        )
    
    default_txt = os.path.splitext(midi_path)[0] + "_notes.txt"
    txt_path = get_user_input(
        f"输出TXT路径 (默认: {os.path.basename(default_txt)}): ",
        default=default_txt
    )
    
    try:
        result = midi_to_text(midi_path=midi_path, output_path=txt_path)
        print(f"✓ 转换完成: {result['total_notes']} 个音符")
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        sys.exit(1)
    
    # 步骤2: TXT转音频
    print("\n--- 步骤2: TXT转脉冲HitSound ---")
    
    # 自动检测hit.wav
    default_hit = os.path.join(script_dir, 'hit.wav')
    if os.path.exists(default_hit):
        print(f"检测到 hit.wav: {default_hit}")
        hit_path = get_user_input(
            "Hit采样文件 (直接回车使用默认): ",
            default=default_hit,
            validate_func=validate_file_path
        )
    else:
        hit_path = get_user_input(
            "Hit采样文件路径: ",
            validate_func=validate_file_path
        )
    
    default_wav = os.path.splitext(txt_path)[0] + "_pulse.wav"
    wav_path = get_user_input(
        f"输出WAV路径 (默认: {os.path.basename(default_wav)}): ",
        default=default_wav
    )
    
    try:
        result = txt_to_pulse_synthesizer(
            txt_path=txt_path,
            hit_wav_path=hit_path,
            output_path=wav_path
        )
        print(f"✓ 合成完成: {result['notes_count']} 个音符, {result['duration']:.2f}秒")
        print(f"  输出: {os.path.basename(result['output_path'])}")
    except Exception as e:
        print(f"✗ 合成失败: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("全部完成！")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)