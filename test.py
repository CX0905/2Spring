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
    """生成置乱表的函数（与原始代码一致）"""
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
        x, y = seed, 0.3
        for _ in range(warmup):
            x, y = henon_map(x, y)
        sequence = []
        for _ in range(N):
            x, y = henon_map(x, y)
            sequence.append(x)
    sorted_indices = np.argsort(sequence)
    permutation = np.zeros(N, dtype=int)
    for i in range(N):
        permutation[sorted_indices[i]] = i
    return permutation

def detect_collision(permutation1, permutation2):
    """检测两个置乱表的碰撞情况"""
    N = len(permutation1)
    collisions = sum(1 for i in range(N) if permutation1[i] == permutation2[i])
    collision_ratio = collisions / N
    return collisions, collision_ratio

def test_collision(N, seed1, seed2, map_type='logistic'):
    """生成两个不同种子的置乱表并检测碰撞"""
    perm1 = generate_permutation(N, seed1, map_type)
    perm2 = generate_permutation(N, seed2, map_type)
    collisions, ratio = detect_collision(perm1, perm2)
    print(f"Map Type: {map_type}")
    print(f"Seeds: {seed1} 和 {seed2}")
    print(f"碰撞次数: {collisions}")
    print(f"碰撞比例: {ratio:.4f}")

if __name__ == "__main__":
    N = 100
    seed_a = 0.123
    seed_b = 0.456
    map_type = ['logistic', 'tent', 'henon']
    for cc in map_type:
        test_collision(N, seed_a, seed_b, cc)