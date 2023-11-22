import numpy as np

# 数据点
data_points = np.array([(1, 3), (2, 5), (4, 8), (7, 9), (9, 12)])

# 初始成员值
membership = np.array([
    [0.8, 0.7, 0.5, 0.3, 0.1],  # 簇 1
    [0.2, 0.3, 0.5, 0.7, 0.9]   # 簇 2
])

# 计算质心的函数
def calculate_centroids(squared_membership, data_points):
    centroids = []
    for i in range(squared_membership.shape[0]):
        numerator = np.sum(squared_membership[i][:, np.newaxis] * data_points, axis=0)
        denominator = np.sum(squared_membership[i])
        centroid = numerator / denominator
        centroids.append(centroid)
    return np.array(centroids)

# 计算距离的函数
def calculate_distances(data_points, centroids):
    distances = []
    for point in data_points:
        distances.append(np.linalg.norm(point - centroids, axis=1))
    return np.array(distances)

# 更新成员值的函数
def update_membership(distances):
    new_membership = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        denominator = np.sum(distances[i, :]**2)
        for j in range(distances.shape[1]):
            new_membership[i, j] = distances[i, 1-j]**2 / denominator
    return new_membership.T

# 打印结果的函数
def print_round_results(round_data):
    print(f"第 {round_data['round']} 轮结果:")
    print("新的质心:")
    for i, centroid in enumerate(round_data['centroids'], start=1):
        print(f"  簇 {i} 质心: {centroid}")

    print("\n每个数据点到质心的距离:")
    for i, point in enumerate(data_points):
        print(f"  数据点 {point}: 到簇 1 质心的距离: {round_data['distances'][i][0]:.4f}, 到簇 2 质心的距离: {round_data['distances'][i][1]:.4f}")

    print("\n更新后的成员值:")
    for i in range(round_data['membership'].shape[0]):
        for j, value in enumerate(round_data['membership'][i]):
            print(f"  数据点 {data_points[j]} 在簇 {i+1} 中的成员值: {value:.4f}")
    print("\n")

# 迭代三轮更新过程
for round_number in range(3):
    # 计算质心
    squared_membership = membership ** 2
    centroids = calculate_centroids(squared_membership, data_points)

    # 计算每个数据点到质心的距离
    distances = calculate_distances(data_points, centroids)

    # 更新成员值
    membership = update_membership(distances)

    # 打印本轮结果
    print_round_results({
        "round": round_number + 1,
        "centroids": centroids,
        "distances": distances,
        "membership": membership
    })


