import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from math import gcd
from functools import reduce

def lcm(a, b):
    """计算最小公倍数"""
    return a * b // gcd(a, b)

def logistic_map(x, mu=4.0):
    return mu * x * (1 - x)

def tent_map(x, mu=1.999):
    return mu * min(x, 1 - x)

def henon_map(x, y, a=1.4, b=0.3):
    new_x = 1 - a * x * x + y
    new_y = b * x
    return new_x, new_y

def generate_permutation(N, seed, map_type='logistic', warmup=1000):
    """生成置乱表"""
    if map_type == 'logistic':
        x = seed
        for _ in range(warmup):
            x = logistic_map(x)
        sequence = []
        for _ in range(N):
            x = logistic_map(x)
            sequence.append(x)
    elif map_type == 'tent':
        x = seed
        for _ in range(warmup):
            x = tent_map(x)
        sequence = []
        for _ in range(N):
            x = tent_map(x)
            sequence.append(x)
    elif map_type == 'henon':
        x, y = seed, 0.3  # 固定y初始值
        for _ in range(warmup):
            x, y = henon_map(x, y)
        sequence = []
        for _ in range(N):
            x, y = henon_map(x, y)
            sequence.append(x)
    
    # 生成排序后的索引作为置乱表
    sorted_indices = np.argsort(sequence)
    permutation = np.zeros(N, dtype=int)
    for i in range(N):
        permutation[sorted_indices[i]] = i
    return permutation

def analyze_cycles(permutation):
    """分析置乱表的循环特性"""
    N = len(permutation)
    visited = [False] * N
    cycle_info = defaultdict(int)
    
    for i in range(N):
        if not visited[i]:
            cycle_length = 0
            j = i
            current_cycle = []
            while not visited[j]:
                visited[j] = True
                current_cycle.append(j)
                j = permutation[j]
                cycle_length += 1
            cycle_info[cycle_length] += 1
    
    # 计算总阶(所有循环长度的最小公倍数)
    if cycle_info:
        cycle_lengths = list(cycle_info.keys())
        total_order = reduce(lcm, cycle_lengths, 1)
    else:
        total_order = 1
    
    return dict(cycle_info), total_order

def print_cycle_info(cycle_info, total_order):
    """打印循环信息"""
    print(f"总阶(Order): {total_order}")
    print(f"循环圈长度的种类数: {len(cycle_info)}")
    print("每种长度的循环圈数量:")
    for length, count in sorted(cycle_info.items()):
        print(f"  长度 {length}: {count} 个")

def evaluate_single_permutation(N, seed, map_type):
    """评估单个置乱表并打印详细信息"""
    print(f"\n评估 {map_type} 映射, N={N}, seed={seed}")
    permutation = generate_permutation(N, seed, map_type)
    cycle_info, total_order = analyze_cycles(permutation)
    print_cycle_info(cycle_info, total_order)
    return permutation, cycle_info, total_order

# 测试参数
N = 20  # 较小的N便于观察
seeds = [0.123, 0.456, 0.789]  # 3个不同的种子
map_types = ['logistic', 'tent', 'henon']

# 评估并打印详细信息
for map_type in map_types:
    for seed in seeds:
        evaluate_single_permutation(N, seed, map_type)

# 评估更大的N并绘制平均阶曲线
N_values = [50, 100, 200, 300, 400, 500]
seeds_for_avg = [0.1 + i*0.1 for i in range(10)]  # 10个不同的种子

def evaluate_maps(N_values, seeds, map_types):
    """评估不同映射的性能"""
    results = {map_type: {'avg_orders': [], 'cycle_stats': []} for map_type in map_types}
    
    for N in N_values:
        for map_type in map_types:
            orders = []
            all_cycle_stats = []
            
            for seed in seeds:
                permutation = generate_permutation(N, seed, map_type)
                cycle_info, order = analyze_cycles(permutation)
                orders.append(order)
                all_cycle_stats.append(cycle_info)
            
            avg_order = np.mean(orders)
            results[map_type]['avg_orders'].append(avg_order)
            
            # 计算平均循环统计
            merged_stats = defaultdict(list)
            for stat in all_cycle_stats:
                for k, v in stat.items():
                    merged_stats[k].append(v)
            avg_stats = {k: np.mean(v) for k, v in merged_stats.items()}
            results[map_type]['cycle_stats'].append(avg_stats)
    
    return results

results = evaluate_maps(N_values, seeds_for_avg, map_types)

def plot_results(N_values, results):
    """绘制结果"""
    plt.figure(figsize=(12, 6))
    
    for map_type, data in results.items():
        plt.plot(N_values, data['avg_orders'], label=f'{map_type} map', marker='o')
    
    plt.xlabel('N (Permutation Size)')
    plt.ylabel('Average Order')
    plt.title('Average Order vs Permutation Size for Different Chaotic Maps')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(N_values, results)

# 打印某个N值的详细循环统计
sample_N = 200
sample_idx = N_values.index(sample_N)
print(f"\n平均循环统计 for N={sample_N}:")
for map_type in map_types:
    print(f"\n{map_type} map:")
    for length, count in sorted(results[map_type]['cycle_stats'][sample_idx].items()):
        print(f"  长度 {length}: 平均 {count:.2f} 个")
    print(f"  平均阶: {results[map_type]['avg_orders'][sample_idx]:.2f}")