import json
import math
import random
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from svg.path import parse_path, Move
from tqdm import tqdm

# 线程锁：确保写文件时的原子性
file_lock = threading.Lock()

def handwrite_normalize_path(path_string, jitter=1.2, drift=0.8, segment_len=5.0):
    """
    核心算法逻辑保持不变
    """
    try:
        path = parse_path(path_string)
    except Exception:
        return path_string
        
    new_path_segments = []
    for segment in path:
        if isinstance(segment, Move):
            new_path_segments.append(f"M {segment.start.real:.1f} {segment.start.imag:.1f}")
            continue
        
        length = segment.length()
        if length == 0: continue
        
        num_steps = max(2, int(length / segment_len))
        points = []
        for i in range(num_steps + 1):
            t = i / num_steps
            curr_pos = segment.point(t)
            x, y = curr_pos.real, curr_pos.imag
            
            # 这里的 math.pi 修正
            dt = 0.01
            t_prev = max(0, t - dt)
            p_prev = segment.point(t_prev)
            dx = x - p_prev.real
            dy = y - p_prev.imag
            angle = math.atan2(dy, dx)
            
            muscle_jitter = math.sin(t * math.pi * 10) * jitter
            random_drift = (random.random() - 0.5) * drift
            total_noise = muscle_jitter + random_drift
            
            nx = x + math.cos(angle + math.pi/2) * total_noise
            ny = y + math.sin(angle + math.pi/2) * total_noise
            points.append((nx, ny))
        
        for idx, (px, py) in enumerate(points):
            if idx == 0 and not new_path_segments:
                new_path_segments.append(f"M {px:.1f} {py:.1f}")
            else:
                new_path_segments.append(f"L {px:.1f} {py:.1f}")
                
    return " ".join(new_path_segments)

def process_and_write(line, f_out, pbar):
    """
    单个线程的任务：处理数据并在加锁状态下写入文件
    """
    line = line.strip()
    if not line:
        return

    try:
        data = json.loads(line)
        # 执行增噪计算
        noisy_strokes = [handwrite_normalize_path(s) for s in data.get('strokes', [])]
        data['strokes'] = noisy_strokes
        result = json.dumps(data, ensure_ascii=False)

        # 并发写保护：使用 Lock 确保写入不冲突
        with file_lock:
            f_out.write(result + "\n")
            pbar.update(1)
            
    except Exception as e:
        with file_lock:
            print(f"处理出错: {e}")

def main_threaded(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    # 统计行数用于进度条
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total = len(lines)

    print(f"开始多线程处理（共 {total} 个字）...")

    # 使用 ThreadPoolExecutor 管理线程池
    # 线程数通常设为 CPU 核心数的 2-4 倍，或者根据 I/O 情况调整
    max_threads = min(32, os.cpu_count() * 2) 
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        with tqdm(total=total, desc="线程处理进度", unit="字") as pbar:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # 提交所有任务
                for line in lines:
                    executor.submit(process_and_write, line, f_out, pbar)

    print(f"\n处理完成！结果保存至：{output_path}")

if __name__ == '__main__':
    input_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans.txt'
    output_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans_handwritten.txt'
    main_threaded(input_file, output_file)