import json
import torch
import math
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def apply_running_script_style_gpu(all_points_tensor, jitter=1.5, drift=1.2):
    """
    行书化处理：增加结构倾斜、重心偏移和高频抖动
    """
    if len(all_points_tensor) < 2: return all_points_tensor
    pts = all_points_tensor.to(device)
    num_pts = pts.shape[0]

    # --- 1. 结构化仿射变换 (行书通常向右上方倾斜) ---
    # 剪切变换矩阵: x' = x + k*y
    shear_k = 0.15  # 倾斜系数
    pts[:, 0] = pts[:, 0] + (shear_k * (1024 - pts[:, 1]) / 1024) * 100

    # --- 2. 模拟书写惯性：低频重心漂移 ---
    t = torch.linspace(0, 1, num_pts, device=device)
    # 模拟运笔时的弧度感 (让直线变略微弯曲)
    curve_offset = torch.sin(t * math.pi) * 3.5 
    pts[:, 0] += curve_offset

    # --- 3. 手部震颤：高频噪声 ---
    noise = (torch.rand(num_pts, 2, device=device) - 0.5) * jitter
    pts += noise
    
    return pts.cpu()

def generate_ligatures(medians):
    """
    在 CPU 层面逻辑处理：生成笔画间的牵丝数据
    行书的笔画往往不完全断开
    """
    new_medians = []
    for i in range(len(medians)):
        new_medians.append(medians[i])
        # 如果不是最后一笔，尝试生成到下一笔起点的“牵丝”
        if i < len(medians) - 1:
            start_pt = medians[i][-1] # 当前笔终点
            end_pt = medians[i+1][0]  # 下一笔起点
            
            # 只有当距离不太远时才生成牵丝，模拟轻微提笔
            dist = math.sqrt((start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2)
            if dist < 300:
                # 插入一条只有两个点的轻微弧线作为连带
                ligature = [start_pt, end_pt]
                # 在前端渲染时，我们会给这种“牵丝”设置极低的透明度
                # 这里我们通过特殊的标记（如负数坐标或额外字段）标识，或者直接加入
                new_medians.append(ligature)
    return new_medians

def main_process_running_script(input_path, output_path, batch_size=500):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f_out = open(output_path, 'w', encoding='utf-8')
    
    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i : i + batch_size]
        all_coords = []
        char_meta = []
        current_offset = 0

        for line in batch_lines:
            data = json.loads(line)
            # 1. 首先进行行书连带处理
            char_medians = generate_ligatures(data.get('medians', []))
            
            stroke_bounds = []
            flat_char_pts = []
            for stroke in char_medians:
                start_idx = len(flat_char_pts)
                flat_char_pts.extend(stroke)
                stroke_bounds.append((current_offset + start_idx, len(stroke)))
            
            all_coords.extend(flat_char_pts)
            char_meta.append({"data": data, "bounds": stroke_bounds, "medians_len": len(char_medians)})
            current_offset += len(flat_char_pts)

        if not all_coords: continue

        # 2. GPU 批量变形
        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
        processed_tensor = apply_running_script_style_gpu(coords_tensor)

        # 3. 写回
        for meta in char_meta:
            new_medians = []
            for start, length in meta["bounds"]:
                stroke_pts = processed_tensor[start : start + length].tolist()
                new_medians.append([[round(p[0], 1), round(p[1], 1)] for p in stroke_pts])
            
            char_obj = meta["data"]
            char_obj['medians'] = new_medians
            f_out.write(json.dumps(char_obj, ensure_ascii=False) + "\n")

    f_out.close()

if __name__ == '__main__':
    input_file = 'F:\\animCJK\\graphicsZhHans.txt'
    output_file = 'F:\\animCJK\\graphicsZhHans_handwritten.txt'
    main_process_running_script(input_file, output_file)