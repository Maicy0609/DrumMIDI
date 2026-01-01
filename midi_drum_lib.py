import mido
from mido import MidiFile
import os
import sys
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
from scipy.io import wavfile

DEFAULT_SAMPLE_RATE = 48000
HIT_VOLUME_SCALE = 0.35
TARGET_LEVEL = 0.85

class TempoMap:
    def __init__(self, ticks_per_beat: int):
        self.ticks_per_beat = ticks_per_beat
        self.events: List[Tuple[int, int]] = [(0, 500000)]
    
    def add_event(self, tick: int, tempo: int):
        self.events.append((tick, tempo))
        self.events.sort(key=lambda x: x[0])
    
    def tick_to_us(self, tick: int) -> int:
        if not self.events:
            return 0
        
        total_us = 0
        last_tick = 0
        current_tempo = 500000
        
        for event_tick, tempo in self.events:
            if event_tick > tick:
                break
            delta_ticks = event_tick - last_tick
            total_us += delta_ticks * current_tempo // self.ticks_per_beat
            last_tick = event_tick
            current_tempo = tempo
        
        delta_ticks = tick - last_tick
        total_us += delta_ticks * current_tempo // self.ticks_per_beat
        return total_us

def get_user_input(prompt: str, default=None, validate_func=None):
    while True:
        user_input = input(prompt).strip()
        if not user_input and default is not None:
            return default
        if user_input:
            if validate_func and not validate_func(user_input.strip('"')):
                print("输入无效，请重试。")
                continue
            return user_input.strip('"')
        if default is None:
            print("输入不能为空。")

def validate_file_path(path: str) -> bool:
    return os.path.isfile(path)

def note_number_to_name(number: int) -> str:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (number // 12) - 1
    note = note_names[number % 12]
    return f"{note}{octave}"

def is_drum_track(track) -> bool:
    for msg in track:
        if msg.type == 'track_name':
            name = msg.name.lower()
            if any(keyword in name for keyword in ['drum', 'percussion', '鼓', '打击']):
                return True
        if msg.type == 'channel_prefix' and msg.channel == 9:
            return True
    return False

def midi_to_text(midi_path: str, output_path: str, ticks_per_beat: int = None):
    print("加载MIDI文件...")
    mid = MidiFile(midi_path)
    tpb = ticks_per_beat or mid.ticks_per_beat
    
    non_drum_tracks = []
    for track_idx, track in enumerate(mid.tracks):
        if is_drum_track(track):
            continue
        non_drum_tracks.append((track_idx, track))
    
    if not non_drum_tracks:
        raise ValueError("未找到非鼓音轨")
    
    tempo_map = TempoMap(tpb)
    all_notes: List[Tuple[int, str, int]] = []
    active_notes: Dict[int, Tuple[int, int]] = {}
    
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.add_event(current_tick, msg.tempo)
    
    for track_idx, track in non_drum_tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                start_us = tempo_map.tick_to_us(current_tick)
                active_notes[msg.note] = (current_tick, start_us)
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                note_num = msg.note
                if note_num in active_notes:
                    start_tick, start_us = active_notes[note_num]
                    end_us = tempo_map.tick_to_us(current_tick)
                    duration_us = end_us - start_us
                    
                    if duration_us > 0:
                        note_name = note_number_to_name(note_num)
                        all_notes.append((start_us, note_name, duration_us))
                    
                    del active_notes[note_num]
    
    if all_notes:
        all_notes.sort(key=lambda x: x[0])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for start_us, note_name, duration_us in all_notes:
            f.write(f"{start_us} {note_name} {duration_us}\n")
    
    return {
        'total_notes': len(all_notes),
        'ticks_per_beat': tpb
    }

def parse_txt_notes(txt_path: str):
    notes = []
    max_end_time = 0
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            
            try:
                start_us = int(parts[0])
                note_name = parts[1]
                duration_us = int(parts[2])
                notes.append((start_us, note_name, duration_us))
                max_end_time = max(max_end_time, start_us + duration_us)
            except ValueError:
                continue
    
    if not notes:
        raise ValueError("TXT文件中没有有效音符数据")
    
    notes.sort(key=lambda x: x[0])
    return notes, max_end_time

def load_hit_sample(wav_path: str):
    sample_rate, hit_audio = wavfile.read(wav_path)
    
    if sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"采样率不匹配: 需要{DEFAULT_SAMPLE_RATE}Hz")
    
    if hit_audio.dtype == np.int16:
        hit_audio = hit_audio.astype(np.float32) / 32768.0
    elif hit_audio.dtype == np.int32:
        hit_audio = hit_audio.astype(np.float32) / 8388608.0
    elif hit_audio.dtype != np.float32:
        raise ValueError(f"不支持的位深度: {hit_audio.dtype}")
    
    if len(hit_audio.shape) > 1:
        hit_audio = hit_audio.mean(axis=1)
    
    hit_audio = hit_audio * HIT_VOLUME_SCALE
    return sample_rate, hit_audio

def note_to_frequency(note_name: str) -> float:
    note_map = {
        'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
        'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
    }
    
    note = note_name[:-1]
    octave = int(note_name[-1])
    semitone_offset = note_map[note] + (octave - 4) * 12
    return 440.0 * (2 ** (semitone_offset / 12))

def generate_pulse_wave(hit_audio: np.ndarray, frequency: float, 
                       sample_rate: int, duration_us: int, 
                       start_offset_samples: int, hit_times: list) -> np.ndarray:
    """生成脉冲波并记录hit时间"""
    interval_samples_float = sample_rate / frequency
    hit_length = len(hit_audio)
    total_samples = int(sample_rate * duration_us / 1_000_000)
    
    pulse_wave = np.zeros(total_samples, dtype=np.float64)
    
    pos = 0.0
    while int(pos) < total_samples:
        start_idx = int(pos)
        remaining = min(hit_length, total_samples - start_idx)
        if remaining <= 0:
            break
        
        # 记录hit的绝对时间（纳秒）
        abs_start_sample = start_offset_samples + start_idx
        time_ns = int(abs_start_sample * 1_000_000_000 // sample_rate)
        hit_times.append(time_ns)
        
        pulse_wave[start_idx:start_idx+remaining] += hit_audio[:remaining]
        pos += interval_samples_float
    
    return pulse_wave.astype(np.float32)

def txt_to_pulse_synthesizer(txt_path: str, hit_wav_path: str, output_path: str):
    """TXT转HitSound"""
    sample_rate, hit_audio = load_hit_sample(hit_wav_path)
    
    notes, total_duration_us = parse_txt_notes(txt_path)
    
    total_samples = int(sample_rate * total_duration_us / 1_000_000)
    output_audio = np.zeros(total_samples, dtype=np.float64)
    
    # 存储所有hit时间的列表
    hit_times = []
    
    for start_us, note_name, duration_us in notes:
        start_sample = int(start_us * sample_rate / 1_000_000)
        
        if start_sample >= total_samples:
            continue
        
        freq = note_to_frequency(note_name)
        pulse_wave = generate_pulse_wave(hit_audio, freq, sample_rate, duration_us, 
                                       start_sample, hit_times)
        
        end_sample = min(start_sample + len(pulse_wave), total_samples)
        actual_length = end_sample - start_sample
        
        if actual_length > 0:
            output_audio[start_sample:end_sample] += pulse_wave[:actual_length]
    
    # 写入hit时间日志文件
    log_path = os.path.splitext(output_path)[0] + "_hit_times.txt"
    with open(log_path, 'w') as f:
        for time_ns in hit_times:
            f.write(f"{time_ns}\n")
   
    final_max = np.max(np.abs(output_audio))
    
    if final_max > 2.0:
        pre_scale = 0.5
        output_audio *= pre_scale
        final_max *= pre_scale
    
    if final_max > 0.9:
        mask = np.abs(output_audio) > 0.9
        over = output_audio[mask]
        compressed = 0.9 + (np.abs(over) - 0.9) / 3.0
        output_audio[mask] = np.sign(over) * compressed
    
    if final_max > 0:
        output_audio = output_audio * (TARGET_LEVEL / final_max)
    
    output_audio_16bit = (output_audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, output_audio_16bit)
    
    return {
        'output_path': output_path,
        'duration': total_duration_us / 1_000_000,
        'notes_count': len(notes),
        'hit_count': len(hit_times),
        'log_path': log_path
    }
