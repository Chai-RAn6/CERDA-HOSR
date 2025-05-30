import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# 设置随机种子（确保结果可重复）
random.seed(42)

# 读取 Excel 数据
file_path = 'test_pos.xlsx'  # 替换为本地文件路径
data = pd.read_excel(file_path)

# 提取唯一的 lncRNA、miRNA、mRNA 和疾病列表
lncRNAs = data['lncRNA'].unique()
miRNAs = data['miRNA'].unique()
mRNAs = data['mRNA'].unique()
diseases = data['Disease'].unique()

# 构建 ceRNA 三元组
ceRNA_triplets = data[['lncRNA', 'miRNA', 'mRNA']].drop_duplicates()

# 构建正样本集（已知关联）
positive_samples = data[['lncRNA', 'miRNA', 'mRNA', 'Disease']].drop_duplicates()
positive_samples['label'] = 1  # 正样本标签为 1

# 将正样本转换为集合，方便快速查找
positive_set = set(tuple(x) for x in positive_samples[['lncRNA', 'miRNA', 'mRNA', 'Disease']].values)

# 创建疾病全集的集合
all_diseases_set = set(diseases)

# 构建映射：每个 ceRNA 三元组对应的关联疾病集合
print("构建 ceRNA 三元组到疾病的映射...")
ceRNA_to_diseases = {}
for triplet in tqdm(ceRNA_triplets.itertuples(index=False), total=ceRNA_triplets.shape[0], desc="映射 ceRNA 三元组到疾病"):
    lncRNA, miRNA, mRNA = triplet
    # 获取与该 ceRNA 三元组关联的所有疾病
    associated_diseases = set(data[
        (data['lncRNA'] == lncRNA) &
        (data['miRNA'] == miRNA) &
        (data['mRNA'] == mRNA)
    ]['Disease'])
    ceRNA_to_diseases[(lncRNA, miRNA, mRNA)] = associated_diseases

# 初始化负样本列表
negative_samples = []

# 获取所有 ceRNA 三元组
ceRNA_triplet_list = list(ceRNA_to_diseases.keys())

# 需要生成的负样本总数
num_negative_samples_needed = len(positive_samples)  # 2234

print("开始生成负样本...")

# 使用 while 循环，直到生成足够的负样本
attempts = 0  # 尝试次数计数器
max_attempts = num_negative_samples_needed * 10  # 设置一个最大尝试次数，防止无限循环

with tqdm(total=num_negative_samples_needed, desc="生成负样本") as pbar:
    while len(negative_samples) < num_negative_samples_needed and attempts < max_attempts:
        attempts += 1

        # 随机选择一个 ceRNA 三元组
        ceRNA = random.choice(ceRNA_triplet_list)
        lncRNA, miRNA, mRNA = ceRNA

        # 获取当前 ceRNA 三元组关联的疾病
        associated_diseases = ceRNA_to_diseases[ceRNA]

        # 计算可选的疾病（全集 - 关联疾病）
        available_diseases = all_diseases_set - associated_diseases

        # 如果没有可选疾病，跳过
        if not available_diseases:
            continue

        # 随机选择一个疾病
        disease = random.choice(list(available_diseases))

        # 构建负样本元组
        negative_tuple = (lncRNA, miRNA, mRNA, disease)

        # 确保负样本不在正样本集中
        if negative_tuple in positive_set:
            continue  # 如果偶然选中正样本，继续尝试

        # 构建负样本字典
        negative_sample = {
            'lncRNA': lncRNA,
            'miRNA': miRNA,
            'mRNA': mRNA,
            'Disease': disease,
            'label': 0  # 负样本标签为 0
        }

        # 添加负样本
        negative_samples.append(negative_sample)

        # 更新进度条
        pbar.update(1)

if len(negative_samples) < num_negative_samples_needed:
    print(f"警告：在尝试 {attempts} 次后，仅生成了 {len(negative_samples)} 个负样本，少于所需的 {num_negative_samples_needed} 个。")
else:
    print(f"成功生成 {len(negative_samples)} 个负样本，共尝试了 {attempts} 次。")

# 将负样本转换为 DataFrame
negative_samples_df = pd.DataFrame(negative_samples)

# 保存负样本数据集
negative_samples_df.to_excel('test_neg.xlsx', index=False)

print("负采样完成，负样本数据已保存到 'test_neg.xlsx'")
