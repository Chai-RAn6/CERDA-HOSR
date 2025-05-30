import numpy as np
import itertools

# 假设有 4 个 RNA 节点 (r1, r2, r3, r4)，相似性范围为 0 到 1
rna_similarity_matrix = np.array([
    [1.0, 0.8, 0.3, 0.4],
    [0.8, 1.0, 0.5, 0.2],
    [0.3, 0.5, 1.0, 0.7],
    [0.4, 0.2, 0.7, 1.0]
])

# 示例疾病相似性矩阵 (对称矩阵)
# 假设有 3 个疾病节点 (d1, d2, d3)，相似性范围为 0 到 1
disease_similarity_matrix = np.array([
    [1.0, 0.6, 0.4],
    [0.6, 1.0, 0.7],
    [0.4, 0.7, 1.0]
])

# RNA 与疾病的关系矩阵 (二进制矩阵，1 表示相关，0 表示无关)
# 假设有 4 个 RNA 节点和 3 个疾病节点
rna_disease_relation_matrix = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 0]
])

# 计算 RNA 节点相似性的平均值
average_rna_similarity = np.mean(rna_similarity_matrix[np.triu_indices(len(rna_similarity_matrix), k=1)])
# 计算疾病节点相似性的平均值
average_disease_similarity = np.mean(disease_similarity_matrix[np.triu_indices(len(disease_similarity_matrix), k=1)])

# 确定 RNA-RNA-Disease 的高阶结构
higher_order_structures = []

# 遍历所有可能的 RNA 节点对和疾病节点
for (i, j) in itertools.combinations(range(len(rna_similarity_matrix)), 2):
    for k in range(len(disease_similarity_matrix)):
        if (
                rna_similarity_matrix[i, j] >= average_rna_similarity and
                rna_disease_relation_matrix[i, k] == 1 and
                rna_disease_relation_matrix[j, k] == 1
        ):
            higher_order_structures.append((i, j, k))

# 输出平均相似性和高阶结构
print(f"Average RNA Similarity: {average_rna_similarity:.2f}")
print(f"Average Disease Similarity: {average_disease_similarity:.2f}")
print("Higher-Order Structures (RNA-RNA-Disease triples):")
for (i, j, k) in higher_order_structures:
    print(f"RNA {i + 1} - RNA {j + 1} - Disease {k + 1}")


def leaky_relu(x, alpha=0.2):
    return np.maximum(alpha * x, x)

# Xavier (Glorot) 初始化函数
def xavier_init(shape):
    limit = np.sqrt(6 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)

# 初始化权重矩阵 W 和 W0 使用 Xavier 初始化
np.random.seed(42)
W = xavier_init((8, 1))  # 使用 Xavier 初始化，形状为 (8, 1)
W0 = xavier_init((4, 4))  # 使用 Xavier 初始化，形状为 (4, 4)

# 学习率
learning_rate = 0.01
# 迭代次数
num_iterations = 1000


# 计算注意力系数
def calculate_attention_coefficient(h_i, h_j, neighbors):
    h_i_transformed = np.dot(W0, h_i)  # 将节点特征映射到新空间
    h_j_transformed = np.dot(W0, h_j)  # 将节点特征映射到新空间
    concat_feature = np.concatenate((h_i_transformed, h_j_transformed))
    leaky_relu_result = leaky_relu(np.dot(W.T, concat_feature))
    numerator = np.exp(np.clip(leaky_relu_result, -500, 500))  # 防止溢出
    denominator = 0
    for neighbor in neighbors:
        neighbor_transformed = np.dot(W0, neighbor)
        concat_neighbor_feature = np.concatenate((h_i_transformed, neighbor_transformed))
        leaky_relu_neighbor_result = leaky_relu(np.dot(W.T, concat_neighbor_feature))
        denominator += np.exp(np.clip(leaky_relu_neighbor_result, -500, 500))  # 防止溢出
    return numerator / (denominator + 1e-10)  # 防止分母为零

# 梯度下降优化过程
def gradient_descent(W, W0, higher_order_structures):
    global learning_rate
    for iteration in range(num_iterations):
        # 初始化梯度
        grad_W = np.zeros_like(W)
        grad_W0 = np.zeros_like(W0)

        # 遍历所有高阶结构，计算梯度
        for (i, j, k) in higher_order_structures:
            h_i = rna_similarity_matrix[i]
            h_j = rna_similarity_matrix[j]
            neighbors = [rna_similarity_matrix[b] for b in range(len(rna_similarity_matrix)) if b != i]

            # 计算当前的注意力系数
            alpha_ij = calculate_attention_coefficient(h_i, h_j, neighbors)

            # 假设我们想最小化 -(log(alpha_ij))，计算梯度
            h_i_transformed = np.dot(W0, h_i)
            h_j_transformed = np.dot(W0, h_j)
            concat_feature = np.concatenate((h_i_transformed, h_j_transformed))
            grad_W += -(1 - alpha_ij) * concat_feature.reshape(-1, 1)
            grad_W0 += -(1 - alpha_ij) * (np.outer(h_i_transformed, h_i) + np.outer(h_j_transformed, h_j))
        # 更新权重矩阵
        W -= learning_rate * grad_W
        W0 -= learning_rate * grad_W0


# 执行梯度下降优化
gradient_descent(W, W0, higher_order_structures)

# 遍历高阶结构，计算注意力系数并输出所有系数和选择最大的输出
print("Attention Coefficients for Higher-Order Structures:")
for (i, j, k) in higher_order_structures:
    h_i = rna_similarity_matrix[i]
    h_j = rna_similarity_matrix[j]
    h_k = rna_similarity_matrix[k]
    neighbors = [rna_similarity_matrix[b] for b in range(len(rna_similarity_matrix)) if b != i]

    # 计算所有注意力系数
    alpha_ij = calculate_attention_coefficient(h_i, h_j, neighbors)
    alpha_ik = calculate_attention_coefficient(h_i, h_k, neighbors)
    alpha_jk = calculate_attention_coefficient(h_j, h_k, neighbors)

    # 输出所有注意力系数
    print(
        f"RNA {i + 1} - RNA {j + 1} - Disease {k + 1}: alpha_ij = {alpha_ij}, alpha_ik = {alpha_ik}, alpha_jk = {alpha_jk}")

    # 取最大值作为最终的注意力系数
    max_alpha = max(alpha_ij, alpha_ik, alpha_jk)

    print(f"RNA {i + 1} - RNA {j + 1} - Disease {k + 1}: Max Attention Coefficient = {max_alpha}")
