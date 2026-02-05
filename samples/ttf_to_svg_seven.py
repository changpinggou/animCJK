import json
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
import os

def ttf_to_graphics_json(ttf_path, char_list, output_path):
    """
    将 TTF 字体中的指定汉字提取为 graphics.txt 格式
    """
    if not os.path.exists(ttf_path):
        print(f"找不到字体文件: {ttf_path}")
        return

    font = TTFont(ttf_path)
    cmap = font.getBestCmap()
    glyph_set = font.getGlyphSet()
    
    results = []
    
    print(f"正在从 {os.path.basename(ttf_path)} 提取路径...")
    
    for char in char_list:
        # 获取字符在字体中的编码
        code = ord(char)
        if code not in cmap:
            continue
            
        glyph_name = cmap[code]
        glyph = glyph_set[glyph_name]
        
        # 使用 SVGPathPen 提取路径
        pen = SVGPathPen(glyph_set)
        glyph.draw(pen)
        path_string = pen.getCommands()
        
        if path_string:
            # 构造符合格式的字典
            # 注意：TTF 提取出来通常是一个整体的大路径，我们放入 strokes 数组
            char_data = {
                "character": char,
                "strokes": [path_string], # 字体提取通常是整字路径
                "medians": [[[0, 0]]]            # TTF 不包含中线数据
            }
            results.append(char_data)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"成功提取 {len(results)} 个汉字到: {output_path}")

if __name__ == '__main__':
    # 配置信息
    font_file = '/Users/zego/Documents/makemeahanzi/animCJK/samples/禹卫书法行书简体(新优化版).ttf' # 替换为你下载的禹卫行书文件名
    output_file = '/Users/zego/Documents/makemeahanzi/animCJK/graphics_custom.txt'
    
    # 你想要提取的汉字列表（可以从原始 graphics.txt 读取所有 character）
    test_chars = "科技改变生活我" 
    
    ttf_to_graphics_json(font_file, test_chars, output_file)