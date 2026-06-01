#!/usr/bin/env python3
"""全面 bug 扫描：检查 Task A / Task C 所有已完成 mutant"""
import json
import glob
import os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

def check_task(name):
    sub = ROOT / name
    files = sorted(sub.glob('details/*.json'))
    print('='*90)
    print(f"### {name}  ({len(files)} files)")
    print('='*90)

    issues = defaultdict(list)
    killed_count = 0
    killed_mutants = []
    
    # 关键不变式
    for f in files:
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception as e:
            issues['JSON_CORRUPT'].append((f.name, str(e)[:80]))
            continue
        
        mid = d.get('mutant_id') or f.stem
        rounds = d.get('rounds') or []
        killed = d.get('killed', False)
        killing_round = d.get('killing_round')
        
        # check 1: 顶层 killed/killing_round 一致性
        if killed and killing_round is None:
            issues['KILLED_BUT_NO_ROUND'].append(mid)
        if not killed and killing_round is not None:
            issues['NOT_KILLED_BUT_HAS_ROUND'].append(mid)
        
        # check 2: 顶层 killed 和 round 内 killed 一致性
        any_round_killed = any(r.get('killed') for r in rounds)
        if killed != any_round_killed:
            issues['KILLED_MISMATCH'].append(f'{mid}: top={killed} round={any_round_killed}')
        
        # check 3: 空 rounds
        if not rounds:
            issues['NO_ROUNDS'].append(mid)
            continue
        
        # check 4: 每个 round 关键字段
        for i, r in enumerate(rounds, 1):
            # round 编号
            if r.get('round') != i:
                issues['ROUND_NUMBER_OFF'].append(f'{mid}: expected {i} got {r.get("round")}')
            
            # error 字段
            err = r.get('error')
            if err:
                if 'timeout' in str(err).lower() or 'timed out' in str(err).lower():
                    issues['BEDROCK_TIMEOUT'].append(f'{mid} round {i}: {str(err)[:80]}')
                else:
                    issues['BEDROCK_ERROR'].append(f'{mid} round {i}: {str(err)[:80]}')
            
            # killable 字段
            killable = r.get('killable')
            if killable is None:
                issues['MISSING_KILLABLE'].append(f'{mid} round {i}')
            
            # reason_category
            cat = r.get('reason_category')
            if cat is None and not err:
                issues['MISSING_CATEGORY'].append(f'{mid} round {i}')
            
            # 若 killable=False，应当早停（即这是最后一轮）
            if killable is False and i != len(rounds) and not killed:
                issues['KILLABLE_FALSE_BUT_CONTINUED'].append(f'{mid} round {i}')
            
            # 若 killable=True 应当有 suggested_code
            if killable is True:
                sc = r.get('suggested_code') or ''
                if len(sc.strip()) < 30:
                    issues['KILLABLE_TRUE_NO_CODE'].append(f'{mid} round {i}: code_len={len(sc)}')
            
            # execution_result 检查
            er = r.get('execution_result')
            if killable is True and not err:
                if er is None:
                    issues['KILLABLE_TRUE_NO_EXEC'].append(f'{mid} round {i}')
                elif isinstance(er, dict):
                    # killed 字段交叉验证
                    er_killed = er.get('killed', False)
                    r_killed = r.get('killed', False)
                    if er_killed != r_killed:
                        issues['EXEC_KILLED_MISMATCH'].append(
                            f'{mid} round {i}: exec={er_killed} round={r_killed}'
                        )
                    
                    # 若 killed=True，应有 diff_summary 或 max_diff
                    if er_killed:
                        diff_summary = er.get('diff_summary') or ''
                        if not diff_summary and er.get('max_abs_diff') is None:
                            issues['KILLED_NO_DIFF_INFO'].append(f'{mid} round {i}')
                    
                    # 若 ref_ok=False，说明 reference 都跑不通
                    if er.get('ref_ok') is False:
                        issues['REF_KERNEL_FAILED'].append(
                            f'{mid} round {i}: {(er.get("error") or "")[:80]}'
                        )
                    if er.get('original_ok') is False:
                        issues['ORIGINAL_KERNEL_FAILED'].append(
                            f'{mid} round {i}: {(er.get("error") or "")[:80]}'
                        )
                    if er.get('mutant_ok') is False and not er_killed:
                        # mutant 编译失败但又没杀死 = 异常
                        issues['MUTANT_COMPILE_FAILED'].append(
                            f'{mid} round {i}: {(er.get("error") or "")[:80]}'
                        )
        
        if killed:
            killed_count += 1
            killed_mutants.append((mid, killing_round, len(rounds)))
    
    # 输出
    print(f"\n📊 总览：")
    print(f"  完成数:    {len(files)}")
    print(f"  已杀死:    {killed_count}")
    print(f"  未杀死:    {len(files) - killed_count}")
    
    print(f"\n🔍 异常检查结果：")
    if not issues:
        print(f"  ✓ 未发现任何异常")
    else:
        for k, v in issues.items():
            tag = '⚠️' if 'TIMEOUT' in k or 'ERROR' in k or 'CORRUPT' in k or 'MISMATCH' in k else '·'
            print(f"  {tag} {k}: {len(v)} 个")
            for item in v[:5]:
                print(f"      - {item}")
            if len(v) > 5:
                print(f"      ... 还有 {len(v)-5} 个")
    
    return killed_mutants, issues

for name in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    print()
    check_task(name)
