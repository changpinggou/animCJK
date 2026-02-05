import json
import math
import random
import os
from svg.path import parse_path, Move

def handwrite_normalize_path(path_string, jitter=1.2, drift=0.8, segment_len=5.0):
    """
    核心算法：对 SVG 路径字符串进行手写模拟增噪
    """
    try:
        path = parse_path(path_string)
    except Exception:
        return path_string # 如果解析失败，返回原路径
        
    new_path_segments = []
    
    for segment in path:
        if isinstance(segment, Move):
            new_path_segments.append(f"M {segment.start.real:.1f} {segment.start.imag:.1f}")
            continue
        
        length = segment.length()
        if length == 0: continue
        
        # 将路径段切碎
        num_steps = max(2, int(length / segment_len))
        points = []
        
        for i in range(num_steps + 1):
            t = i / num_steps
            curr_pos = segment.point(t)
            x, y = curr_pos.real, curr_pos.imag
            
            # 计算切线方向以应用法线偏移
            dt = 0.01
            t_prev = max(0, t - dt)
            p_prev = segment.point(t_prev)
            dx = x - p_prev.real
            dy = y - p_prev.imag
            angle = math.atan2(dy, dx)
            
            # 噪声：正弦微颤 + 随机漂移
            muscle_jitter = math.sin(t * math.pi * 10) * jitter
            random_drift = (random.random() - 0.5) * drift
            total_noise = muscle_jitter + random_drift
            
            # 垂直偏移
            nx = x + math.cos(angle + math.pi/2) * total_noise
            ny = y + math.sin(angle + math.pi/2) * total_noise
            points.append((nx, ny))
        
        for idx, (px, py) in enumerate(points):
            if idx == 0 and not new_path_segments:
                # 如果是这一笔的第一个点，使用 M 指令
                # 意思是：画笔先落在这个点上
                new_path_segments.append(f"M {px:.1f} {py:.1f}")
            else:
                # 对于后续所有的点，使用 L 指令
                # 意思是：从上一个点画一条直线连到当前这个点
                new_path_segments.append(f"L {px:.1f} {py:.1f}"))
                
    return " ".join(new_path_segments)

def batch_process_graphics(input_path, output_path):
    """
    批量处理整个 graphics 文件
    """
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return

    print(f"开始处理文件：{input_path}")
    count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                # 处理 strokes 数组中的每一个笔画
                noisy_strokes = [
                    handwrite_normalize_path(s) for s in data.get('strokes', [])
                ]
                data['strokes'] = noisy_strokes
                
                # 写入新文件，保持一行一条 JSON 的格式
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                count += 1
                if count % 500 == 0:
                    print(f"已处理 {count} 个汉字...")
                    
            except Exception as e:
                 print(f"处理第 {count+1} 行时出错: {e}")

    print(f"处理完成！共处理 {count} 个汉字。")
    print(f"结果已保存至：{output_path}")

# --- 执行 ---
input_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans.txt'
output_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphicsZhHans_handwritten.txt'

batch_process_graphics(input_file, output_file)