import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import torch
import torch.nn as nn

# 双层图卷积模块
class GCN(nn.Module):
    def __init__(self, in_ft, hidden_ft, out_ft, act=nn.LeakyReLU(), bias=True):
        super(GCN, self).__init__()

        # ------------------------------
        # 1) 定义第一层、第二层的 W
        # ------------------------------
        self.W1 = Parameter(torch.FloatTensor(in_ft, hidden_ft), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.W1, nonlinearity='leaky_relu')

        self.W2 = Parameter(torch.FloatTensor(hidden_ft, out_ft), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.W2, nonlinearity='leaky_relu')

        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_ft)
        self.bn2 = nn.BatchNorm1d(out_ft)

        # 激活函数
        self.act = act

        # ------------------------------
        # 3) 动态定义偏置项，并优化初始化
        # ------------------------------
        if bias:
            self.bias1 = nn.Parameter(torch.FloatTensor(hidden_ft))
            nn.init.uniform_(self.bias1, a=-0.01, b=0.01)

            self.bias2 = nn.Parameter(torch.FloatTensor(out_ft))
            nn.init.uniform_(self.bias2, a=-0.01, b=0.01)
        else:
            self.register_parameter('bias1', None)
            self.register_parameter('bias2', None)

    def forward(self, H_0, A_e, A_h):
        """
        前向传播
        :param H_0: 输入特征矩阵 (H^(0))
        :param A_e: 第一层邻接矩阵 A_e (普通邻接矩阵)
        :param A_h: 第二层邻接矩阵 A_h (高阶邻接矩阵)
        :return: 最终嵌入矩阵 H
        """
        # 第一层传播：H^(1) = ReLU(A_e H^(0) W^(1))
        h = torch.mm(A_e, H_0)        # A_e H^(0)
        h = torch.mm(h, self.W1)      # A_e H^(0) W^(1)
        if self.bias1 is not None:
            h = h + self.bias1        # 加偏置项
        h = self.bn1(h)  # 批归一化
        h = self.act(h)  # 激活

        # 第二层传播：H = A_h H^(1) W^(2)
        h = torch.mm(A_h, h)          # A_h H^(1)
        h = torch.mm(h, self.W2)      # A_h H^(1) W^(2)
        if self.bias2 is not None:
            h = h + self.bias2        # 加偏置项
        h = self.bn2(h)  # 批归一化

        return h


############################################
# 2) 多类型节点的 HG 模块
############################################
class HG(nn.Module):
    def __init__(
            self,
            # 四类节点数:
            n_lnc,
            n_mi,
            n_mr,
            n_dis,
            # 其他超参:
            in_dim,
            hidden_dim,
            out_dim,
            # 训练用边:

            train_pos_edges,
            train_neg_edges,

            # 子矩阵(或相似度矩阵)等  # 关联矩阵
            A_lnc, A_mi, A_mr, A_dis,  # lnc-lnc, mi-mi, mr-mr,dis-dis
            R_lnc_mi, R_lnc_mr,R_lnc_dis,
            R_mi_mr,R_mi_dis,
            R_mr_dis,
            A_h_file,
            lncrna_to_idx, mirna_to_idx, mrna_to_idx, disease_to_idx,  # 添加映射表
            seed=42):

        super(HG, self).__init__()
        self.in_dim = in_dim  # 添加 in_dim 属性
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.seed = seed
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 2.1 节点数、正负样本边
        self.n_lnc = n_lnc
        self.n_mi = n_mi
        self.n_mr = n_mr
        self.n_dis = n_dis

        # 映射表
        self.lncrna_to_idx = lncrna_to_idx
        self.mirna_to_idx = mirna_to_idx
        self.mrna_to_idx = mrna_to_idx
        self.disease_to_idx = disease_to_idx

        self.train_pos_edges = train_pos_edges
        self.train_neg_edges = train_neg_edges

        # 2.2 Embedding: 为简单起见，放在一个大 embedding 里
        # nn.init.xavier_normal_(self.E.weight)改为分别定义：
        self.E_lnc = nn.Embedding(n_lnc, in_dim)
        nn.init.xavier_normal_(self.E_lnc.weight)

        self.E_mi = nn.Embedding(n_mi, in_dim)
        nn.init.xavier_normal_(self.E_mi.weight)

        self.E_mr = nn.Embedding(n_mr, in_dim)
        nn.init.xavier_normal_(self.E_mr.weight)

        self.E_dis = nn.Embedding(n_dis, in_dim)
        nn.init.xavier_normal_(self.E_dis.weight)

        # 2.3 构造块状大邻接矩阵
        #    这里假设 A_lnc, A_mi, A_mr, A_dis, R_lnc_dis, R_mi_dis, R_mr_dis
        #    都是 numpy 或 torch 的二维矩阵(0/1 或相似度)
        #    注意实际中还可能有 Lnc-mi, mi-mr, Lnc-mr 等跨关系,

        # 2.1 构造 big_adj (A_e)
        self.big_adj = self.build_big_adj(  # 确保这里正确定义了 self.big_adj
            A_lnc, A_mi, A_mr, A_dis,
            R_lnc_mi, R_lnc_mr, R_lnc_dis,
            R_mi_mr, R_mi_dis, R_mr_dis
        ).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        # 从文件加载 A_h 矩阵
        self.A_h = torch.tensor(np.loadtxt(A_h_file), dtype=torch.float32)

        # 确保 A_h 和 big_adj 尺寸一致
        assert self.big_adj.shape == self.A_h.shape, "A_e 和 A_h 的维度不一致！"

        # 2.4 初始化 GCN
        self.gcn = GCN(in_ft=in_dim, hidden_ft=hidden_dim,
                out_ft=out_dim, act=nn.LeakyReLU())

        # 2.5 下面和原 HG 一样，用 We 做内积打分
        self.We = Parameter(torch.FloatTensor(out_dim, out_dim), requires_grad=True)
        nn.init.xavier_normal_(self.We)

        # print(f"We shape: {self.We.shape}")

    def build_big_adj(
            self,
            A_lnc, A_mi, A_mr, A_dis,
            R_lnc_mi, R_lnc_mr, R_lnc_dis, R_mi_mr,R_mi_dis,
            R_mr_dis):
        """
        构造一个包含 [lnc, mi, mr, disease] 四个块的
        大邻接矩阵 (n_lnc+n_mi+n_mr+n_dis, n_lnc+n_mi+n_mr+n_dis)
        构造大邻接矩阵 A_e (big_adj)
        """
        # ---------- (1) 转为 Tensor ---------- #
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32)
            return x

        A_lnc = to_tensor(A_lnc)
        A_mi = to_tensor(A_mi)
        A_mr = to_tensor(A_mr)
        A_dis = to_tensor(A_dis)

        R_lnc_mi = to_tensor(R_lnc_mi)
        R_lnc_mr = to_tensor(R_lnc_mr)
        R_lnc_dis = to_tensor(R_lnc_dis)
        R_mi_mr = to_tensor(R_mi_mr)
        R_mi_dis = to_tensor(R_mi_dis)
        R_mr_dis = to_tensor(R_mr_dis)

        # ---------- (2) 拼接对角块 ---------- #
        # 各自维度
        n_l, n_m, n_r, n_d = self.n_lnc, self.n_mi, self.n_mr, self.n_dis

        # 先拼接四个对角块
        # block1: [A_lnc,       0,      0,         0      ]
        # block2: [    0,   A_mi,      0,         0      ]
        # block3: [    0,       0,   A_mr,        0      ]
        # block4: [    0,       0,      0,     A_dis     ]
        block_left = torch.cat([A_lnc, torch.zeros(n_l, n_m), torch.zeros(n_l, n_r)], dim=1)
        block_right = torch.cat([torch.zeros(n_l, n_d)], dim=1)
        block1 = torch.cat([block_left, block_right], dim=1)  # [n_l, n_l+n_m+n_r+n_d]

        block2_left = torch.cat([torch.zeros(n_m, n_l), A_mi, torch.zeros(n_m, n_r)], dim=1)
        block2_right = torch.cat([torch.zeros(n_m, n_d)], dim=1)
        block2 = torch.cat([block2_left, block2_right], dim=1)

        block3_left = torch.cat([torch.zeros(n_r, n_l), torch.zeros(n_r, n_m), A_mr], dim=1)
        block3_right = torch.cat([torch.zeros(n_r, n_d)], dim=1)
        block3 = torch.cat([block3_left, block3_right], dim=1)

        block4_left = torch.cat([torch.zeros(n_d, n_l), torch.zeros(n_d, n_m), torch.zeros(n_d, n_r)], dim=1)
        block4_right = torch.cat([A_dis], dim=1)
        block4 = torch.cat([block4_left, block4_right], dim=1)

        # 先得到对角大块
        diag_part = torch.cat([block1, block2, block3, block4], dim=0)

        # 再补充跨块 lnc-dis, mi-dis, mr-dis (示意)
        # 例如 lnc-dis 的位置：位于 [0:n_l, (n_l+n_m+n_r) : (n_l+n_m+n_r+n_d)]
        # 最终的 big_adj 先 copy diag_part
        big_adj = diag_part.clone()  # [n_l+n_m+n_r+n_d, n_l+n_m+n_r+n_d]

        # 注意 index 范围：
        # lnc 索引区间 [0,         n_l)
        # mi  索引区间 [n_l,       n_l + n_m)
        # mr  索引区间 [n_l + n_m, n_l + n_m + n_r)
        # dis 索引区间 [n_l + n_m + n_r, n_l + n_m + n_r + n_d)

        # lnc - mi
        big_adj[0:n_l, (n_l):(n_l + n_m)] = R_lnc_mi
        big_adj[(n_l):(n_l + n_m), 0:n_l] = R_lnc_mi.T

        # lnc - mr
        big_adj[0:n_l, (n_l+n_m):(n_l + n_m + n_r)] = R_lnc_mr
        big_adj[(n_l + n_m):(n_l + n_m + n_r), 0:n_l] = R_lnc_mr.T

        # lnc ~ disease
        big_adj[0:n_l, (n_l + n_m + n_r):(n_l + n_m + n_r + n_d)] = R_lnc_dis
        big_adj[(n_l + n_m + n_r):(n_l + n_m + n_r + n_d), 0:n_l] = R_lnc_dis.T

        # mi - mr
        big_adj[n_l:(n_l + n_m), (n_l + n_m):(n_l + n_m + n_r)] = R_mi_mr
        big_adj[(n_l + n_m):(n_l + n_m + n_r), n_l:(n_l + n_m)] = R_mi_mr.T

        # mi ~ disease
        big_adj[n_l:(n_l + n_m), (n_l + n_m + n_r):(n_l + n_m + n_r + n_d)] = R_mi_dis
        big_adj[(n_l + n_m + n_r):(n_l + n_m + n_r + n_d), n_l:(n_l + n_m)] = R_mi_dis.T

        # mr ~ disease
        big_adj[(n_l + n_m):(n_l + n_m + n_r), (n_l + n_m + n_r):(n_l + n_m + n_r + n_d)] = R_mr_dis
        big_adj[(n_l + n_m + n_r):(n_l + n_m + n_r + n_d), (n_l + n_m):(n_l + n_m + n_r)] = R_mr_dis.T

        return big_adj

    def forward(self):
        """
        返回四类节点的嵌入
        """
        # 拼接 embedding
        h_lnc = self.E_lnc.weight
        h_mi = self.E_mi.weight
        h_mr = self.E_mr.weight
        h_dis = self.E_dis.weight
        H_0 = torch.cat([h_lnc, h_mi, h_mr, h_dis], dim=0)  # [N, in_dim]

        # 执行 GCN
        h_out = self.gcn(H_0, self.big_adj, self.A_h)

        # 拆分输出
        n_l, n_m, n_r, n_d = self.n_lnc, self.n_mi, self.n_mr, self.n_dis
        h_lnc_out = h_out[:n_l]
        h_mi_out = h_out[n_l:n_l + n_m]
        h_mr_out = h_out[n_l + n_m:n_l + n_m + n_r]
        h_disease_out = h_out[n_l + n_m + n_r:]

        return h_lnc_out, h_mi_out, h_mr_out, h_disease_out

    def compute_loss(self, h_lnc_out, h_mi_out, h_mr_out, h_disease_out):
        """
        计算 (lnc, mi, mr, disease) 四元组的打分
        """
        logits = []
        # 正样本
        for (n1, n2, n3, n4) in self.train_pos_edges:
            lnc_emb = h_lnc_out[n1]  # [out_dim]
            mi_emb = h_mi_out[n2]
            mr_emb = h_mr_out[n3]
            dis_emb = h_disease_out[n4]

            # 示例: 三RNA 取平均，再与 disease 内积
            rna_emb = (lnc_emb + mi_emb + mr_emb) / 3.0
            score = rna_emb @ self.We @ dis_emb
            logits.append(score)

        # 负样本
        for (n1, n2, n3, n4) in self.train_neg_edges:
            lnc_emb = h_lnc_out[n1]
            mi_emb = h_mi_out[n2]
            mr_emb = h_mr_out[n3]
            dis_emb = h_disease_out[n4]

            rna_emb = (lnc_emb + mi_emb + mr_emb) / 3.0
            score = rna_emb @ self.We @ dis_emb
            logits.append(score)

        if len(logits) == 0:
            raise ValueError("Logits list is empty. Check your (n1,n2,n3,n4) definitions.")

        logits = torch.stack(logits, dim=0)
        lbl1 = torch.ones(len(self.train_pos_edges), device=logits.device)
        lbl0 = torch.zeros(len(self.train_neg_edges), device=logits.device)
        label = torch.cat((lbl1, lbl0), dim=0)
        return logits, label

    def embed(self, sparse=False):
        """
        返回四类节点的嵌入
        """
        h_lnc = self.E_lnc.weight
        h_mi = self.E_mi.weight
        h_mr = self.E_mr.weight
        h_dis = self.E_dis.weight
        x_all = torch.cat([h_lnc, h_mi, h_mr, h_dis], dim=0)  # 拼接

        # 使用 self.big_adj 替代 adj_all
        h_out = self.gcn(x_all, self.big_adj, self.A_h)

        # 拆分
        n_l, n_m, n_r, n_d = self.n_lnc, self.n_mi, self.n_mr, self.n_dis
        h_lnc_out = h_out[:n_l]
        h_mi_out = h_out[n_l: n_l + n_m]
        h_mr_out = h_out[n_l + n_m: n_l + n_m + n_r]
        h_dis_out = h_out[n_l + n_m + n_r:]

        return h_lnc_out, h_mi_out, h_mr_out, h_dis_out

    #########################################
    # 你也可以在此添加“分类器”来对 (lnc, mi, mr, disease) 四元组打分
    # 或者分开三种 RNA 单独做聚合
    #########################################

class RNADiseaseClassifier(nn.Module):
    """
    分别为 (h_lnc, h_mi, h_mr) 定义独立的权重矩阵，并输出 (0~1) 概率
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        # 为每种 RNA 类型定义独立的权重矩阵
        self.W_lnc = nn.Parameter(torch.FloatTensor(in_dim, in_dim))  # lncRNA 权重矩阵
        self.W_mi = nn.Parameter(torch.FloatTensor(in_dim, in_dim))  # miRNA 权重矩阵
        self.W_mr = nn.Parameter(torch.FloatTensor(in_dim, in_dim))  # mRNA 权重矩阵

        # 为疾病定义权重矩阵
        self.W_dis = nn.Parameter(torch.FloatTensor(in_dim, in_dim))  # Disease 权重矩阵

        # 初始化权重矩阵
        nn.init.xavier_normal_(self.W_lnc)
        nn.init.xavier_normal_(self.W_mi)
        nn.init.xavier_normal_(self.W_mr)
        nn.init.xavier_normal_(self.W_dis)

        # 定义MLP层
        self.triplet_lin = nn.Linear(3 * in_dim, hidden_dim)  # RNA特征融合
        self.dis_lin = nn.Linear(hidden_dim + in_dim, hidden_dim)  # RNA和疾病特征融合
        self.out_lin = nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, h_lnc, h_mi, h_mr, h_dis):
        # --------------------------
        # 1. 各种 RNA 特征加权
        # --------------------------
        h_lnc_weighted = torch.mm(h_lnc, self.W_lnc)  # lncRNA 加权
        h_mi_weighted = torch.mm(h_mi, self.W_mi)  # miRNA 加权
        h_mr_weighted = torch.mm(h_mr, self.W_mr)  # mRNA 加权

        # --------------------------
        # 2. RNA 特征聚合
        # --------------------------
        rna_cat = torch.cat([h_lnc_weighted, h_mi_weighted, h_mr_weighted], dim=-1)
        rna_emb = F.relu(self.triplet_lin(rna_cat))

        # --------------------------
        # 3. 疾病特征加权
        # --------------------------
        h_dis_weighted = torch.mm(h_dis, self.W_dis)  # Disease 加权

        # --------------------------
        # 4. RNA-Disease 特征融合
        # --------------------------
        dis_cat = torch.cat([rna_emb, h_dis_weighted], dim=-1)
        fuse_emb = F.relu(self.dis_lin(dis_cat))
        score = self.out_lin(fuse_emb)

        # --------------------------
        # 5. Sigmoid 输出概率
        # --------------------------
        prob = torch.sigmoid(score)
        return prob


def train_main(pos_file, neg_file, test_pos_file, test_neg_file, n_lnc, n_mi, n_mr, n_dis, in_dim, out_dim,
               A_lnc, A_mi, A_mr, A_dis, R_lnc_mi, R_lnc_mr, R_lnc_dis, R_mi_mr, R_mi_dis, R_mr_dis,
               lr, epochs, lr_decay_step, lr_decay_factor, A_h_file):
    """
    示例：构造 HG 模型(内部用 GCN)，然后对 (三RNA + disease) 正负样本做二分类训练
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1) 读取 A_h 文件 ----
    A_h = torch.tensor(np.loadtxt(A_h_file), dtype=torch.float32)

    # ---- 1) 读入训练和测试集的正负样本 DataFrame ----
    train_pos_df = pd.read_excel(pos_file)
    train_neg_df = pd.read_excel(neg_file)
    test_pos_df = pd.read_excel(test_pos_file)
    test_neg_df = pd.read_excel(test_neg_file)

    # 为正负样本标注标签
    train_pos_df["label"] = 1
    train_neg_df["label"] = 0
    test_pos_df["label"] = 1
    test_neg_df["label"] = 0

    # ---- 2) 创建训练集的节点映射 ----
    lncRNA2idx_train = {}
    miRNA2idx_train = {}
    mRNA2idx_train = {}
    disease2idx_train = {}

    def get_or_assign_train(mapping, key):
        """为训练集的节点分配新索引"""
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    # 仅基于训练集构建索引
    for df in [train_pos_df, train_neg_df]:
        for _, row in df.iterrows():
            get_or_assign_train(lncRNA2idx_train, row['lncRNA'])
            get_or_assign_train(miRNA2idx_train, row['miRNA'])
            get_or_assign_train(mRNA2idx_train, row['mRNA'])
            get_or_assign_train(disease2idx_train, row['Disease'])

    # ---- 3) 创建完整的节点映射（训练集 + 测试集） ----
    lncRNA2idx_all = dict(lncRNA2idx_train)  # 复制训练集的映射
    miRNA2idx_all = dict(miRNA2idx_train)
    mRNA2idx_all = dict(mRNA2idx_train)
    disease2idx_all = dict(disease2idx_train)

    def get_or_assign_all(mapping, key):
        """为所有数据集分配新索引"""
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    # 包括测试集的节点
    for df in [test_pos_df, test_neg_df]:
        for _, row in df.iterrows():
            get_or_assign_all(lncRNA2idx_all, row['lncRNA'])
            get_or_assign_all(miRNA2idx_all, row['miRNA'])
            get_or_assign_all(mRNA2idx_all, row['mRNA'])
            get_or_assign_all(disease2idx_all, row['Disease'])

    # 打印节点数量统计
    # print(f"Training lncRNA count : {len(lncRNA2idx_train)}")
    # print(f"Total lncRNA count    : {len(lncRNA2idx_all)}")
    # print(f"Training miRNA count  : {len(miRNA2idx_train)}")
    # print(f"Total miRNA count     : {len(miRNA2idx_all)}")
    # print(f"Training mRNA count   : {len(mRNA2idx_train)}")
    # print(f"Total mRNA count      : {len(mRNA2idx_all)}")
    # print(f"Training disease count: {len(disease2idx_train)}")
    # print(f"Total disease count   : {len(disease2idx_all)}")

    # ---- 4) 构造正负样本边 ----
    def build_edges(df, lncRNA2idx, miRNA2idx, mRNA2idx, disease2idx):
        """构建边列表"""
        edges = []
        for _, row in df.iterrows():
            n1 = lncRNA2idx[row['lncRNA']]
            n2 = miRNA2idx[row['miRNA']]
            n3 = mRNA2idx[row['mRNA']]
            n4 = disease2idx[row['Disease']]
            edges.append((n1, n2, n3, n4))
        return edges

    train_pos_edges = build_edges(train_pos_df, lncRNA2idx_train, miRNA2idx_train, mRNA2idx_train, disease2idx_train)
    train_neg_edges = build_edges(train_neg_df, lncRNA2idx_train, miRNA2idx_train, mRNA2idx_train, disease2idx_train)

    # ---- 5) 初始化 HG 模型 ----
    model_hg = HG(
        n_lnc=len(lncRNA2idx_train),  # 使用训练集的节点数
        n_mi=len(miRNA2idx_train),
        n_mr=len(mRNA2idx_train),
        n_dis=len(disease2idx_train),
        in_dim=in_dim,
        hidden_dim=256,
        out_dim=out_dim,
        train_pos_edges=train_pos_edges,
        train_neg_edges=train_neg_edges,
        A_lnc=A_lnc,
        A_mi=A_mi,
        A_mr=A_mr,
        A_dis=A_dis,
        R_lnc_mi=R_lnc_mi,
        R_lnc_mr=R_lnc_mr,
        R_lnc_dis=R_lnc_dis,
        R_mi_mr=R_mi_mr,
        R_mi_dis=R_mi_dis,
        R_mr_dis=R_mr_dis,
        lncrna_to_idx=lncRNA2idx_all,  # 测试集仍然使用完整的索引
        mirna_to_idx=miRNA2idx_all,
        mrna_to_idx=mRNA2idx_all,
        disease_to_idx=disease2idx_all,
        A_h_file=A_h_file  # 添加 A_h_file 参数
    ).to(device)

    # ---- 4) 初始化分类器 ----
    model_cls = RNADiseaseClassifier(
        in_dim=out_dim,  # GCN 输出维度
        hidden_dim=out_dim
    ).to(device)

    # ---- 5) 定义优化器与调度器 ----
    optimizer = optim.Adam(
        list(model_hg.parameters()) + list(model_cls.parameters()),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # ---- 6) 训练循环 ----
    print("Starting training...")
    for epoch in range(epochs):
        model_hg.train()
        model_cls.train()
        optimizer.zero_grad()

        # 打印训练进度
        print(f"Epoch {epoch + 1}/{epochs}:")

        # 获取节点嵌入
        h_lnc, h_mi, h_mr, h_dis = model_hg.forward()

        # 计算损失
        logits, labels = model_hg.compute_loss(h_lnc, h_mi, h_mr, h_dis)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 梯度回传与更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_hg.parameters(), max_norm=0.5)  # 梯度裁剪
        optimizer.step()
        scheduler.step()

        # 输出损失和学习率
        print(f"  Loss: {loss.item():.5f}")
        print(f"  Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.5e}")

    print("Training completed.")

    # ---- 7) 保存模型和嵌入 ----
    print("Saving model and embeddings...")
    with torch.no_grad():
        h_lnc, h_mi, h_mr, h_dis = model_hg.embed(sparse=False)

    torch.save({
        "HG_state": model_hg.state_dict(),  # 保存 HG 模型的状态字典
        "cls_state": model_cls.state_dict(),  # 保存分类器的状态字典
        "lncRNA2idx_train": lncRNA2idx_train,  # 保存训练集的 lncRNA 映射
        "lncRNA2idx_all": lncRNA2idx_all,  # 保存完整的 lncRNA 映射（包括测试集）
        "miRNA2idx_train": miRNA2idx_train,  # 保存训练集的 miRNA 映射
        "miRNA2idx_all": miRNA2idx_all,  # 保存完整的 miRNA 映射
        "mRNA2idx_train": mRNA2idx_train,  # 保存训练集的 mRNA 映射
        "mRNA2idx_all": mRNA2idx_all,  # 保存完整的 mRNA 映射
        "disease2idx_train": disease2idx_train,  # 保存训练集的 Disease 映射
        "disease2idx_all": disease2idx_all,  # 保存完整的 Disease 映射
        "h_lnc": h_lnc.cpu(),  # 保存 lncRNA 的嵌入
        "h_mi": h_mi.cpu(),  # 保存 miRNA 的嵌入
        "h_mr": h_mr.cpu(),  # 保存 mRNA 的嵌入
        "h_dis": h_dis.cpu()  # 保存 Disease 的嵌入
    }, "trained_gcn_hg.pt")

    print("Model and embeddings saved to 'trained_gcn_hg.pt'.")

if __name__ == "__main__":
    train_main(
        pos_file="train_pos.xlsx",
        neg_file="train_neg.xlsx",
        test_pos_file = 'test_pos.xlsx',
        test_neg_file = 'test_neg.xlsx',
        A_h_file = 'A_h_dis.txt',
        A_lnc = np.loadtxt('A_lnc.txt'),
        A_mi = np.loadtxt('A_mi.txt'),
        A_mr = np.loadtxt('A_mr.txt'),
        A_dis = np.loadtxt('A_dis.txt'),
        R_lnc_mi=np.loadtxt('R_lnc_mi.txt'),
        R_lnc_mr=np.loadtxt('R_lnc_mr.txt'),
        R_lnc_dis=np.loadtxt('R_lnc_dis.txt'),
        R_mi_mr=np.loadtxt('R_mi_mr.txt'),
        R_mi_dis=np.loadtxt('R_mi_dis.txt'),
        R_mr_dis=np.loadtxt('R_mr_dis.txt'),
        n_lnc=200,
        n_mi=545,
        n_mr=896,
        n_dis=280,
        in_dim=2188,
        out_dim=64,
        lr=0.001,
        epochs=50,
        lr_decay_step=10,
        lr_decay_factor=0.6
    )
