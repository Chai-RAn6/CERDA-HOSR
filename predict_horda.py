import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_recall_curve, auc, roc_curve
)
from sklearn.metrics.pairwise import cosine_similarity
from fr import HG
from utils import generate_neg_edges, read


def plot_roc_pr_curve(y_true, y_scores, prefix="HoRDA"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title(f"{prefix} - ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid(True)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.title(f"{prefix} - PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()


def main(args):
    print("🔍 读取训练图结构...")
    G_train_info = read(args.train_graph, args.rna_num, args.disease_num)
    G_all_info = read(args.full_graph, args.rna_num, args.disease_num)

    G_train = G_train_info[2]
    G_all = G_all_info[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = torch.FloatTensor(G_train_info[0]).to(device)
    train_pos_edges = G_train.edges()
    train_neg_edges = generate_neg_edges(G_all, len(train_pos_edges),
                                         args.rna_num, args.disease_num,
                                         G_train_info[8], args.seed)

    print("📦 加载模型...")
    model = HG(args.rna_num + args.disease_num, args.hidden, 'ReLU',
               args.rna_num, args.disease_num,
               train_pos_edges, train_neg_edges,
               adj, G_train_info[7], args.seed, args).to(device)

    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    print("📐 生成嵌入向量...")
    with torch.no_grad():
        embeds = model.embed(model.E.weight.to(device), adj,
                             G_train_info[5], G_train_info[6], G_train_info[7],
                             sparse=False, msk=None)
    embeddings = embeds.detach().cpu().numpy()

    print("🧪 构造评估样本...")
    test_pos_edges = list(G_train.edges())
    test_neg_edges = generate_neg_edges(G_all, len(test_pos_edges),
                                        args.rna_num, args.disease_num,
                                        G_train_info[8], args.seed)

    y_true = [1] * len(test_pos_edges) + [0] * len(test_neg_edges)
    all_edges = test_pos_edges + test_neg_edges

    y_scores = []
    for i, j in all_edges:
        ii, jj = int(i), int(j)
        score = cosine_similarity(
            embeddings[ii].reshape(1, -1),
            embeddings[jj].reshape(1, -1)
        )[0][0]
        y_scores.append(score)

    y_scores = np.array(y_scores)
    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    y_pred = (y_scores >= 0.5).astype(int)

    auc_roc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall, precision)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📊 模型评估指标：")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"AUC-PR :  {auc_pr:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_roc_pr_curve(y_true, y_scores)

    print(f"\n📌 Top-{args.topk} 推荐结果（未在训练集中出现的 RNA–Disease 对）：")
    recs = []

    existing_edges = set((int(u), int(v)) for u, v in G_train.edges())

    for i in range(args.rna_num):
        for j in range(args.rna_num, args.rna_num + args.disease_num):
            ii, jj = int(i), int(j)
            if (ii, jj) not in existing_edges:
                try:
                    score = cosine_similarity(
                        embeddings[ii].reshape(1, -1),
                        embeddings[jj].reshape(1, -1)
                    )[0][0]
                    recs.append((ii, jj - args.rna_num, score))
                except IndexError:
                    continue

    recs.sort(key=lambda x: x[2], reverse=True)
    for idx, (rna, dis, score) in enumerate(recs[:args.topk]):
        print(f"Top {idx+1}: RNA {rna} - Disease {dis} | Score: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='best.pkl')
    parser.add_argument('--train_graph', type=str, required=True)
    parser.add_argument('--full_graph', type=str, required=True)
    parser.add_argument('--rna_num', type=int, required=True)
    parser.add_argument('--disease_num', type=int, required=True)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    main(args)
