import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.alpha = alpha

        # 初始化权重矩阵 W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 初始化注意力系数矩阵 a
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 激活函数 LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # 线性变换
        e = self._prepare_attentional_mechanism_input(Wh)  # 计算注意力系数

        zero_vec = -9e15 * torch.ones_like(e)  # 将没有连接的部分设置为负无穷，避免 softmax 考虑这些节点
        attention = torch.where(adj > 0, e, zero_vec)  # 仅保留邻接矩阵中有连接的部分
        attention = F.softmax(attention, dim=1)  # 使用 softmax 计算注意力系数
        attention = F.dropout(attention, self.dropout, training=self.training)  # 随机丢弃部分注意力系数，防止过拟合

        h_prime = torch.matmul(attention, Wh)  # 特征加权聚合

        # 使用激活函数，如果需要将输出拼接起来
        if self.concat:
            h_prime = F.elu(h_prime)

        # 返回新特征和注意力系数
        return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # Wh1.shape (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # Wh2.shape (N, 1)
        e = Wh1 + Wh2.T  # 计算注意力系数，e.shape (N, N)
        return self.leakyrelu(e)  # 使用 leaky ReLU 激活

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# 示例使用
if __name__ == "__main__":

    # Number of RNAs and diseases
    num_rnas = 585
    num_diseases = 88
    # 假设有 个节点，特征维度为 ，输出特征维度为
    in_features = 585
    out_features = 128
    num_nodes = 673

    # Replace these with the actual reading logic for CS and DS
    CS = np.loadtxt('./rna_sim_585.txt')
    DS = np.loadtxt('./disease_sim_88.txt')
    R_train = np.loadtxt('association.txt')
    adj=np.loadtxt('./adj.txt')

    # 初始化输入特征矩阵
    h = torch.rand((num_nodes, in_features))

    # 转换为 PyTorch 张量
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    # 初始化 GAT 层
    gat_layer = GraphAttentionLayer(in_features, out_features, dropout=0.5, alpha=0.2, concat=True)

    # 前向传播计算注意力和新特征
    h_prime, attention = gat_layer(h, adj_tensor)
    print("初始的邻接矩阵：\n", adj_tensor)
    print("增强后的特征矩阵：\n", h_prime)
    print("注意力系数矩阵：\n", attention)

    # Add attention coefficients to CS and DS to get enhanced adjacency matrices
    attention = attention.detach().numpy()  # 转换为 NumPy 矩阵以便和原始矩阵相加
    adj_L = CS + attention[:num_rnas, :num_rnas]
    #adj_R = DS + attention[num_rnas:, num_rnas:]

    print("the fused similarity(adj_L):\n", adj_L)
    #print("Enhanced Disease adjacency matrix (adj_R):\n", adj_R)
