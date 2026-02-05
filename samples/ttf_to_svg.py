import json
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
import os

def ttf_to_all_graphics_json_fixed(ttf_path, output_path):
    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    
    # 获取字体的头信息，用于计算缩放和翻转
    # 通常 TTF 的 unitsPerEm 是 1024 或 2048
    upm = font['head'].unitsPerEm
    
    results = []
    
    # 构造变换矩阵：
    # 1. 缩放：将 upm 缩放到 1024
    # 2. 翻转：Y轴取反
    # 3. 平移：将翻转后的图形移回可见区域 (通常下移 800-900 像素)
    scale_factor = 1024 / upm
    transform = (scale_factor, 0, 0, -scale_factor, 0, 900) 

    for code, glyph_name in cmap.items():
        char = chr(code)
        # 只处理汉字区间
        if not ('\u4e00' <= char <= '\u9fa5'): continue
            
        glyph = glyph_set[glyph_name]
        
        # 使用 TransformPen 进行坐标转换
        svg_pen = SVGPathPen(glyph_set)
        transform_pen = TransformPen(svg_pen, transform)
        glyph.draw(transform_pen)
        
        path_string = svg_pen.getCommands()
        
        if path_string:
            char_data = {
                "character": char,
                "strokes": [path_string],
                "medians": [[[0, 0]]] # 补全占位符防止报错
            }
            results.append(char_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"修正后的数据已导出，共 {len(results)} 字")


if __name__ == '__main__':
    # 配置信息
    font_file = '/Users/zego/Documents/makemeahanzi/animCJK/samples/禹卫书法行书简体(新优化版).ttf' # 替换为你下载的禹卫行书文件名
    output_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphics_custom.txt'
    
    # 你想要提取的汉字列表（可以从原始 graphics.txt 读取所有 character）
    test_chars = "科技改变生活我" 
    
    ttf_to_all_graphics_json_fixed(font_file, output_file)