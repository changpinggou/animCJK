import json
import torch
import math
import os
import multiprocessing
from svg.path import parse_path, Move
from tqdm import tqdm

# 硬件探测
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def parse_single_stroke(stroke, segment_len=5.0):
    """单个笔画的解析逻辑。"""
    try:
        path = parse_path(stroke)
        stroke_pts = []
        for segment in path:
            if isinstance(segment, Move): continue
            # 优化：简单距离估算步数
            d = math.sqrt((segment.start.real - segment.end.real)**2 + 
                          (segment.start.imag - segment.end.imag)**2)
            num_steps = max(2, int(d / segment_len))
            for i in range(num_steps + 1):
                p = segment.point(i / num_steps)
                stroke_pts.append([p.real, p.imag])
        return stroke_pts
    except:
        return []

def process_char_unit(line):
    """单个汉字的预处理函数。"""
    try:
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
    except:
        return None

def apply_batch_jitter(all_points_tensor, jitter=1.2, drift=0.8):
    """GPU 并行处理层。"""
    if len(all_points_tensor) < 2: return all_points_tensor
    pts = all_points_tensor.to(device)
    diffs = torch.zeros_like(pts)
    diffs[:-1] = pts[1:] - pts[:-1]
    diffs[-1] = diffs[-2]
    angles = torch.atan2(diffs[:, 1], diffs[:, 0])
    t = torch.linspace(0, 1, pts.shape[0], device=device)
    noise = (torch.sin(t * math.pi * 10) * jitter) + ((torch.rand(pts.shape[0], device=device) - 0.5) * drift)
    offset_angle = angles + math.pi / 2
    pts[:, 0] += torch.cos(offset_angle) * noise
    pts[:, 1] += torch.sin(offset_angle) * noise
    return pts.cpu()

def main_optimized_gpu(input_path, output_path, batch_size=500):
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 使用 multiprocessing.Pool 代替 Thread/Process Executor 解决 Windows 下的 atexit 问题
    num_cpus = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_cpus} CPU cores for parsing and {device} for jitter calculation.")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        with multiprocessing.Pool(processes=num_cpus) as pool:
            for i in range(0, len(lines), batch_size):
                batch_lines = lines[i : i + batch_size]
                
                # 1. CPU 并行解析 (使用 Pool.map)
                batch_results = pool.map(process_char_unit, batch_lines)
                
                # 2. 数据整合
                flat_coords = []
                batch_meta = []
                current_offset = 0
                for res in batch_results:
                    if res is None: continue
                    data, char_pts, char_bounds = res
                    global_bounds = [(b_s + current_offset, b_l) for b_s, b_l in char_bounds]
                    flat_coords.extend(char_pts)
                    batch_meta.append((data, global_bounds))
                    current_offset += len(char_pts)
                
                if not flat_coords: continue

                # 3. GPU 并行计算
                batch_tensor = torch.tensor(flat_coords, dtype=torch.float32)
                processed_tensor = apply_batch_jitter(batch_tensor)

                # 4. 写回文件
                for data, global_bounds in batch_meta:
                    new_strokes = []
                    for start, length in global_bounds:
                        stroke_pts = processed_tensor[start : start + length]
                        s_str = f"M {stroke_pts[0][0]:.1f} {stroke_pts[0][1]:.1f} "
                        s_str += " ".join([f"L {p[0]:.1f} {p[1]:.1f}" for p in stroke_pts[1:]])
                        new_strokes.append(s_str)
                    data['strokes'] = new_strokes
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                print(f"Processed batch {i//batch_size + 1}/{(len(lines)-1)//batch_size + 1}")

if __name__ == '__main__':
    # 路径确保正确
    input_file = 'F:\\animCJK\\graphicsZhHans.txt'
    output_file = 'F:\\animCJK\\graphicsZhHans_handwritten.txt'
    main_optimized_gpu(input_file, output_file, batch_size=800)