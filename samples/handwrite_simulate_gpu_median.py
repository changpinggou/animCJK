import json
import torch
import math
import os
import multiprocessing
from tqdm import tqdm

# 硬件探测
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def apply_median_jitter_gpu(all_points_tensor, jitter=1.2, drift=0.8):
    """
    对中线骨架点进行并行增噪处理
    """
    if len(all_points_tensor) < 2: return all_points_tensor
    pts = all_points_tensor.to(device)
    num_pts = pts.shape[0]

    # 1. 模拟运笔不稳：低频波动 (Low-frequency drift)
    t_slow = torch.linspace(0, 2 * math.pi, num_pts, device=device)
    # 增加非线性偏移
    slow_drift_x = torch.sin(t_slow * 0.5) * drift
    slow_drift_y = torch.cos(t_slow * 0.3) * drift
    
    # 2. 模拟手部震颤：高频抖动 (High-frequency jitter)
    noise = (torch.rand(num_pts, 2, device=device) - 0.5) * jitter
    
    pts[:, 0] += slow_drift_x + noise[:, 0]
    pts[:, 1] += slow_drift_y + noise[:, 1]
    
    return pts.cpu()

def main_process_medians(input_path, output_path, batch_size=800):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f_out = open(output_path, 'w', encoding='utf-8')
    print(f"开始处理 Medians 骨架数据，加速设备: {device}")

    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i : i + batch_size]
        all_coords = []
        char_meta = []
        current_offset = 0

        for line in batch_lines:
            data = json.loads(line)
            # 获取原生的 medians
            char_medians = data.get('medians', [])
            stroke_bounds = []
            
            flat_char_pts = []
            for stroke in char_medians:
                start_idx = len(flat_char_pts)
                flat_char_pts.extend(stroke)
                stroke_bounds.append((current_offset + start_idx, len(stroke)))
            
            all_coords.extend(flat_char_pts)
            char_meta.append({"data": data, "bounds": stroke_bounds})
            current_offset += len(flat_char_pts)

        if not all_coords: continue

        # GPU 计算并行增噪
        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
        processed_tensor = apply_median_jitter_gpu(coords_tensor)

        # 写回 JSON
        for meta in char_meta:
            new_medians = []
            for start, length in meta["bounds"]:
                stroke_pts = processed_tensor[start : start + length].tolist()
                # 坐标保留一位小数优化体积
                formatted_stroke = [[round(p[0], 1), round(p[1], 1)] for p in stroke_pts]
                new_medians.append(formatted_stroke)
            
            char_obj = meta["data"]
            char_obj['medians'] = new_medians
            f_out.write(json.dumps(char_obj, ensure_ascii=False) + "\n")

    f_out.close()
    print("处理完成！")

if __name__ == '__main__':
    # 替换为你的路径
    input_file = 'F:\\animCJK\\graphicsZhHans.txt'
    output_file = 'F:\\animCJK\\graphicsZhHans_handwritten.txt'
    main_process_medians(input_file, output_file)