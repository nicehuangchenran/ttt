"""
读取 sekai-game-walking.csv，统计 location / scene / crowdDensity / weather / timeOfDay
五个属性的分布，以及条件分布（如 location → weather）。

用法:
    python preprocess/sekai_game_walking/get_metadata.py [--csv path/to/csv]
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "sekai-game-walking.csv")

FIELDS = ["location", "scene", "crowdDensity", "weather", "timeOfDay"]


def print_distribution(field: str, counter: Counter, total: int):
    print(f"\n=== {field} ({len(counter)} unique values) ===")
    for value, count in counter.most_common():
        pct = count / total * 100
        print(f"  {value:<50s} {count:>6d}  ({pct:5.2f}%)")


def print_conditional_distribution(cond_field: str, target_field: str, joint_counter: dict):
    """打印条件分布表格：给定 cond_field，target_field 的分布"""
    print(f"\n{'=' * 100}")
    print(f"Conditional Distribution: {target_field} | {cond_field}")
    print('=' * 100)

    # 获取所有可能的 target 值
    all_targets = set()
    for targets in joint_counter.values():
        all_targets.update(targets.keys())
    all_targets = sorted(all_targets)

    # 计算最长的条件值长度
    max_cond_len = max(len(str(cond_value)) for cond_value in joint_counter.keys())
    max_cond_len = max(max_cond_len, len(cond_field))
    cond_width = max_cond_len + 2

    # 打印表头
    header = f"{cond_field:<{cond_width}}"
    for target in all_targets:
        header += f" | {target:>12}"
    header += f" | {'Total':>8}"
    print(header)
    print("-" * len(header))

    # 打印每个条件值的分布
    for cond_value in sorted(joint_counter.keys()):
        targets = joint_counter[cond_value]
        total = sum(targets.values())
        row = f"{cond_value:<{cond_width}}"
        for target in all_targets:
            count = targets.get(target, 0)
            row += f" | {count:>12}"
        row += f" | {total:>8}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    args = parser.parse_args()

    counters = {field: Counter() for field in FIELDS}
    # 联合分布: joint[cond_field][cond_value][target_field][target_value] -> count
    joint = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

    total = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            for field in FIELDS:
                counters[field][row[field]] += 1

            # 记录联合分布
            location = row["location"]
            joint["location"][location]["weather"][row["weather"]] += 1
            joint["location"][location]["timeOfDay"][row["timeOfDay"]] += 1
            joint["location"][location]["crowdDensity"][row["crowdDensity"]] += 1

            scene = row["scene"]
            joint["scene"][scene]["weather"][row["weather"]] += 1
            joint["scene"][scene]["timeOfDay"][row["timeOfDay"]] += 1

    # 打印全局分布
    print(f"Total samples: {total}")
    for field in FIELDS:
        print_distribution(field, counters[field], total)

    # 打印条件分布
    print("\n\n" + "=" * 80)
    print("CONDITIONAL DISTRIBUTIONS")
    print("=" * 80)

    # location → weather, timeOfDay, crowdDensity
    for target in ["weather", "timeOfDay", "crowdDensity"]:
        target_joint = {loc: joint["location"][loc][target]
                       for loc in joint["location"]}
        print_conditional_distribution("location", target, target_joint)

    # scene → weather, timeOfDay
    for target in ["weather", "timeOfDay"]:
        target_joint = {sc: joint["scene"][sc][target]
                       for sc in joint["scene"]}
        print_conditional_distribution("scene", target, target_joint)


if __name__ == "__main__":
    main()
