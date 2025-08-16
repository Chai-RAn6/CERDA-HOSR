import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
from embed_train import embedding
from utils import load, split, Prediction
import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', choices=[
        'CircR2Disease.txt'], default='CircR2Disease.txt')
    parser.add_argument('--output', choices=[
        'Default_c.txt'], default='Default_c.txt')
    parser.add_argument('--testingratio', default=0.2, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--weighted', type=bool, default=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int, help='seed value')
    parser.add_argument('--runs', default=2, type=int, help='number of runs for calculating error')

    args = parser.parse_args()

    return args


def main(args):
    """
    主函数：执行RNA-疾病关联预测的完整流程

    Args:
        args: 命令行参数
    """
    start_time = time.time()
    print("=" * 50)
    print(f"开始HoRDA模型运行, 数据集: {args.input}")
    print(
        f"参数: epochs={args.epochs}, hidden={args.hidden}, seed={args.seed}, testing_ratio={args.testingratio}, runs={args.runs}")
    print("=" * 50)

    # 运行次数
    n = args.runs
    AUC_ROC = []
    AUC_PR = []
    ACC = []
    F1 = []

    # 多次运行实验
    for run in range(n):
        print(f"\n运行 {run + 1}/{n}:")

        # 设置不同的随机种子
        current_seed = args.seed + run
        random.seed(current_seed)
        np.random.seed(current_seed)

        # 数据拆分
        print(f"\n第1步 (运行 {run + 1}): 数据拆分")
        G, G_train, testing_pos_edges, train_graph_filename, rna_nodes_num, disease_nodes_num = split(
            args.input, current_seed, args.testingratio, weighted=args.weighted)

        # 嵌入学习
        print(f"\n第2步 (运行 {run + 1}): 嵌入学习")

        # 为每次运行创建不同的输出文件名
        current_output = f"{os.path.splitext(args.output)[0]}_{run + 1}{os.path.splitext(args.output)[1]}"

        n1n2R = embedding(args, train_graph_filename, G, rna_nodes_num, disease_nodes_num, current_output)

        # 加载嵌入
        print(f"\n第3步 (运行 {run + 1}): 加载嵌入向量")
        # 为每次运行创建不同的输出文件名
        current_output = f"{os.path.splitext(args.output)[0]}_{run + 1}{os.path.splitext(args.output)[1]}"
        embeddings = load(current_output)

        # 预测评估
        print(f"\n第4步 (运行 {run + 1}): 预测与评估")
        auc_roc, auc_pr, accuracy, f1 = Prediction(
            embeddings, G, G_train, testing_pos_edges, n1n2R[0], n1n2R[1], n1n2R[2], n1n2R[3], current_seed)

        AUC_ROC.append(auc_roc)
        AUC_PR.append(auc_pr)
        ACC.append(accuracy)
        F1.append(f1)

        print(
            f"运行 {run + 1} 结果: AUC-ROC = {auc_roc:.4f}, AUC-PR = {auc_pr:.4f}, Acc = {accuracy:.4f}, F1 = {f1:.4f}")

    # 转换为NumPy数组以计算统计值
    AUC_ROC = np.array(AUC_ROC)
    AUC_PR = np.array(AUC_PR)
    ACC = np.array(ACC)
    F1 = np.array(F1)

    # 输出结果
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 50)
    print(f"共运行 {n} 次实验的最终结果:")
    print('AUC-ROC= %.4f +- %.4f | AUC-PR= %.4f +- %.4f | Acc= %.4f +- %.4f | F1= %.4f +- %.4f' % (
        AUC_ROC.mean(), AUC_ROC.std(), AUC_PR.mean(), AUC_PR.std(),
        ACC.mean(), ACC.std(), F1.mean(), F1.std()))
    print(f"总运行时间: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")
    print("=" * 50)

    return


def run_main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    main(args)


if __name__ == "__main__":
    run_main()