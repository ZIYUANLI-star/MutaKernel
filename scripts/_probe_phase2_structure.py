#!/usr/bin/env python3
"""探查 Phase II detail 的真实结构 + 真正的 original_failures 含义"""
import json
from pathlib import Path

f = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details/L1_P89__arith_replace__10.json')
d = json.loads(f.read_text(encoding='utf-8'))

print('=== 顶层 keys ===')
print(list(d.keys()))
print()
print('=== main_track 结构 ===')
mt = d.get('main_track', {})
print(f'  type: {type(mt).__name__}')
if isinstance(mt, dict):
    for k, v in mt.items():
        if isinstance(v, dict):
            print(f'  {k}: dict keys = {list(v.keys())[:10]}')
        elif isinstance(v, list):
            print(f'  {k}: list[{len(v)}]')
            if v and isinstance(v[0], dict):
                print(f'    sample[0] keys: {list(v[0].keys())}')
        else:
            print(f'  {k}: {type(v).__name__} = {str(v)[:80]}')

# 找一个真有 original_failures 的字段
print()
print('=== 全文搜 original_failures ===')
import re
text = f.read_text(encoding='utf-8')
for m in re.finditer(r'"original_failures"\s*:', text):
    start = m.start()
    # 取附近 200 字符
    print(f'  pos {start}: ...{text[max(0,start-100):start+200]}...')
    print('  ---')
