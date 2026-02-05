import json
import torch
import math
import os
from svg.path import parse_path, Move
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前使用的加速设备: {device}")

def preprocess_svg_to_tensor(path_string, segment_len=5.0):
    """在 CPU 上预解析 SVG 为坐标点列表"""
    try:
        path = parse_path(path_string)
        all_points = []
        for segment in path:
            if isinstance(segment, Move): continue
            length = segment.length()
            if length == 0: continue
            num_steps = max(2, int(length / segment_len))
            # 采样点
            t = torch.linspace(0, 1, num_steps + 1)
            for val in t:
                p = segment.point(val.item())
                all_points.append([p.real, p.imag])
        return torch.tensor(all_points, dtype=torch.float32)
    except:
        return None

def gpu_handwrite_jitter(points_tensor, jitter=1.2, drift=0.8):
    """在 GPU 上批量并行处理噪声"""
    if points_tensor is None or len(points_tensor) < 2:
        return points_tensor

    points = points_tensor.to(device)
    
    # 1. 计算切线角度 (diff)
    diffs = points[1:] - points[:-1]
    # 补齐最后一个点的角度
    diffs = torch.cat([diffs, diffs[-1:].clone()], dim=0)
    angles = torch.atan2(diffs[:, 1], diffs[:, 0])
    
    # 2. 生成并行噪声
    t = torch.linspace(0, 1, len(points)).to(device)
    muscle_jitter = torch.sin(t * math.pi * 10) * jitter
    random_drift = (torch.rand(len(points)).to(device) - 0.5) * drift
    total_noise = muscle_jitter + random_drift
    
    # 3. 应用垂直偏移 (角度 + 90度)
    offset_angle = angles + math.pi/2
    points[:, 0] += torch.cos(offset_angle) * total_noise
    points[:, 1] += torch.sin(offset_angle) * total_noise
    
    return points.cpu()

def tensor_to_svg_path(points):
    """将处理后的坐标转回 SVG 字符串"""
    if points is None: return ""
    segments = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    for p in points[1:]:
        segments.append(f"L {p[0]:.1f} {p[1]:.1f}")
    return " ".join(segments)

def process_line_gpu(line):
    data = json.loads(line)
    new_strokes = []
    for stroke in data.get('strokes', []):
        # 预处理 -> GPU 运算 -> 转回字符串
        tensor = preprocess_svg_to_tensor(stroke)
        if tensor is not None:
            processed_tensor = gpu_handwrite_jitter(tensor)
            new_strokes.append(tensor_to_svg_path(processed_tensor))
        else:
            new_strokes.append(stroke)
    data['strokes'] = new_strokes
    return json.dumps(data, ensure_ascii=False)

def main_gpu(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"正在通过 GPU 加速处理 {len(lines)} 个汉字...")
    
    # 虽然计算在 GPU，但为了防止解析 SVG 的单线程瓶颈，依然使用线程池分发
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_line_gpu, lines), total=len(lines)))

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(res + "\n")

if __name__ == '__main__':
    input_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans.txt'
    output_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans_handwritten.txt'
    main_gpu(input_file, output_file)