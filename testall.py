import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import gcd
from functools import reduce

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False
from math import gcd
from functools import reduce

# 定义最小公倍数函数
def lcm(a, b):
    """计算最小公倍数"""
    return a * b // gcd(a, b)

# 定义 Logistic 映射
def logistic_map(x, mu=4.0):
    return mu * x * (1 - x)

# 定义 Tent 映射
def tent_map(x, mu=1.999):
    return mu * min(x, 1 - x)

# 定义 Henon 映射
def henon_map(x, y, a=1.4, b=0.3):
    new_x = 1 - a * x * x + y
    new_y = b * x
    return new_x, new_y

# 生成置乱表的函数
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

# 检测两个置乱表的碰撞数和比例
def detect_collision(perm1, perm2):
    """检测两个置乱表的碰撞情况"""
    collisions = np.sum(perm1 == perm2)
    ratio = collisions / len(perm1)
    return collisions, ratio

# 打印碰撞信息
def test_collision(N, seed1, seed2, map_type='logistic'):
    """生成两个不同种子的置乱表并检测碰撞"""
    perm1 = generate_permutation(N, seed1, map_type)
    perm2 = generate_permutation(N, seed2, map_type)
    collisions, ratio = detect_collision(perm1, perm2)
    print(f"Map Type: {map_type}")
    print(f"Seeds: {seed1} 和 {seed2}")
    print(f"碰撞次数: {collisions}")
    print(f"碰撞比例: {ratio:.4f}")

# 运行多次测试并统计碰撞比例
def run_multiple_tests(N, seeds, map_type='logistic'):
    """使用一组种子，生成所有置乱表，进行两两比较，统计碰撞比例"""
    permutations = [generate_permutation(N, seed, map_type) for seed in seeds]
    collision_ratios = []

    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            _, ratio = detect_collision(permutations[i], permutations[j])
            collision_ratios.append(ratio)
    
    return collision_ratios

# 主程序：打印碰撞情况并图形化展示
if __name__ == "__main__":
    N = 100  # 置乱表长度
    num_seeds = 50  # 随机种子数量
    seeds = np.random.uniform(0, 1, size=num_seeds)  # 生成随机种子
    map_types = ['logistic', 'tent', 'henon']  # 三种映射类型

    for map_type in map_types:
        print(f"\n=== {map_type.upper()} Map ===")
        collision_ratios = run_multiple_tests(N, seeds, map_type)

        # 打印前两组种子的碰撞信息
        test_collision(N, seeds[0], seeds[1], map_type)

        # 图形化展示碰撞比例分布
        plt.figure(figsize=(8, 5))
        plt.hist(collision_ratios, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'{map_type.capitalize()} Map: 碰撞比例分布（随机种子）')
        plt.xlabel('碰撞比例')
        plt.ylabel('频数')
        plt.grid(True)
        plt.show()