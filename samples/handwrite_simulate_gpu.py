import json
import torch
import math
import os
from svg.path import parse_path, Move, CubicBezier, Line
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor # 使用进程池绕过 GIL

# 硬件探测
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def parse_single_stroke(stroke, segment_len=5.0):
    """
    单个笔画的解析逻辑，将在子进程中运行。
    """
    try:
        path = parse_path(stroke)
        stroke_pts = []
        for segment in path:
            if isinstance(segment, Move): continue
            
            # 优化点：估算点数，避免复杂的 length() 计算
            # 使用起点和终点的欧几里得距离作为基准
            d = math.sqrt((segment.start.real - segment.end.real)**2 + 
                          (segment.start.imag - segment.end.imag)**2)
            num_steps = max(2, int(d / segment_len))
            
            # 线性采样点 (在子进程中完成简单的点生成)
            for i in range(num_steps + 1):
                p = segment.point(i / num_steps)
                stroke_pts.append([p.real, p.imag])
        return stroke_pts
    except:
        return []

def process_char_unit(line):
    """
    单个汉字的所有笔画预处理
    """
    data = json.loads(line)
    char_all_pts = []
    char_boundaries = []
    
    curr_offset = 0
    for stroke in data.get('strokes', []):
        pts = parse_single_stroke(stroke)
        if pts:
            char_all_pts.extend(pts)
            char_boundaries.append((curr_offset, len(pts)))
            curr_offset += len(pts)
            
    return data, char_all_pts, char_boundaries

def apply_batch_jitter(all_points_tensor, jitter=1.2, drift=0.8):
    """
    GPU 并行处理层
    """
    if len(all_points_tensor) < 2: return all_points_tensor
    pts = all_points_tensor.to(device)
    
    # 计算相邻差值向量
    diffs = torch.zeros_like(pts)
    diffs[:-1] = pts[1:] - pts[:-1]
    diffs[-1] = diffs[-2]
    
    # 批量角度计算
    angles = torch.atan2(diffs[:, 1], diffs[:, 0])
    
    # 批量噪声生成
    t = torch.linspace(0, 1, pts.shape[0], device=device)
    noise = (torch.sin(t * math.pi * 10) * jitter) + ((torch.rand(pts.shape[0], device=device) - 0.5) * drift)
    
    # 垂直偏移
    offset_angle = angles + math.pi / 2
    pts[:, 0] += torch.cos(offset_angle) * noise
    pts[:, 1] += torch.sin(offset_angle) * noise
    
    return pts.cpu()

def main_optimized_gpu(input_path, output_path, batch_size=500):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f_out = open(output_path, 'w', encoding='utf-8')
    
    # 使用进程池加速 CPU 解析阶段
    # ProcessPoolExecutor 适合 CPU 密集型解析
    print(f"启动多进程解析器...")
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i : i + batch_size]
            
            # --- 1. CPU 并行解析阶段 ---
            # 这是提速的关键：利用所有 CPU 核心解析 SVG 文本
            batch_results = list(executor.map(process_char_unit, batch_lines))
            
            # --- 2. 数据整合阶段 ---
            flat_coords = []
            batch_meta = [] # 存储原 data 和 boundaries
            current_offset = 0
            
            for data, char_pts, char_bounds in batch_results:
                if not char_pts:
                    batch_meta.append((data, []))
                    continue
                
                # 重新映射全局偏移
                global_bounds = [(b_start + current_offset, b_len) for b_start, b_len in char_bounds]
                flat_coords.extend(char_pts)
                batch_meta.append((data, global_bounds))
                current_offset += len(char_pts)
            
            if not flat_coords: continue

            # --- 3. GPU 并行计算阶段 ---
            batch_tensor = torch.tensor(flat_coords, dtype=torch.float32)
            processed_tensor = apply_batch_jitter(batch_tensor)

            # --- 4. 写回阶段 ---
            for data, global_bounds in batch_meta:
                new_strokes = []
                for start, length in global_bounds:
                    stroke_pts = processed_tensor[start : start + length]
                    s_str = f"M {stroke_pts[0][0]:.1f} {stroke_pts[0][1]:.1f} "
                    s_str += " ".join([f"L {p[0]:.1f} {p[1]:.1f}" for p in stroke_pts[1:]])
                    new_strokes.append(s_str)
                
                data['strokes'] = new_strokes
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            print(f"已完成批次: {i//batch_size + 1}")

    f_out.close()

if __name__ == '__main__':
    input_file = 'F:\\animCJK\\graphicsZhHans.txt'
    output_file = 'F:\\animCJK\\graphicsZhHans_handwritten.txt'
    # 建议根据你的 CPU 核心数调整 batch_size
    main_optimized_gpu(input_file, output_file)

