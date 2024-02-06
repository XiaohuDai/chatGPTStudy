import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 关于 word embedding， 以序列建模为例，考虑 source sentence 和 target sentence
# 构建序列，序列的字符以其在词表中的索引的形式表示

# 批大小，指一次训练迭代中用于更新模型权重的数据样本数量
BATCH_SIZE = 2

# 原序列词表大小，一共8个单词，下标从1-8，下标0让给 padding
MAX_SRC_NUM_SEQ_LEN = 8
# 目标序列词表大小
MAX_TGT_NUM_SEQ_LEN = 8

# 模型特征大小，原论文中是512，这里取8
MODEL_DIM = 8

# 源序列的最大长度
MAX_SRC_SEQ_LEN = 5
# 目标序列的最大长度
MAX_TGR_SEQ_LEN = 5
# 位置的最大值
MAX_POSITION_LEN = 5

# 1. 原序列和目标序列长度
# 形状为 (BATCH_SIZE,) 的一维张量，每个元素都是在 [2, 5) 范围内的随机整数。
# src_len = torch.randint(2, 5, (BATCH_SIZE,))
# tgt_len = torch.randint(2, 5, (BATCH_SIZE,))

# 为了避免每次产生变化，所以固定 src_len 和 tgt_len 的长度
# 原序列，例子：tensor([2, 4], dtype=torch.int32)， 即原序列包含两个样本，第一个样本长度为2， 第二个样本长度为4
src_len = torch.Tensor([2, 4]).to(torch.int32)
# 目标序列，例子：tensor([4, 3], dtype=torch.int32)，即目标序列包含两个样本，第一个样本长度为4， 第二个样本长度为3
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# # 2. 原序列和目标序列都是由单词索引构成的句子
# # 原序列， 例：[tensor([7, 2]), tensor([6, 7, 2, 3])]，原序列包含两个样本，第一个样本（长度为2，其值由词表中下标为7和2的元素组成），第二个样本（长度为4，其值由原序列词表中下标为6、7、2和3的元素组成）
# source_seq = [torch.randint(1, MAX_SRC_NUM_SEQ_LEN, (L,)) for L in src_len]
# # 目标序列， 例：[tensor([1, 6, 2, 6]), tensor([1, 4, 7])]，原序列包含两个样本，第一个样本（长度为4，其值由词表中下标为1、6、2和6的元素组成），第二个样本（长度为3，其值由目标序列词表中下标为1、4和7的元素组成）
# target_seq = [torch.randint(1, MAX_TGT_NUM_SEQ_LEN, (L,)) for L in tgt_len]
# print(source_seq)
# print(target_seq)

# # 为了使原序列和目标序列的张量的维度一致，需要对张量进行填充
# # source_seq = [F.pad(torch.randint(1, MAX_SRC_NUM_SEQ_LEN + 1, (L,)), (0, max(src_len) - L)) for L in src_len]
# target_seq = [F.pad(torch.randint(1, MAX_TGT_NUM_SEQ_LEN + 1, (L,)), (0, max(tgt_len) - L)) for L in tgt_len]

# 单词索引构成源句子和目标句子， 并且做了 padding， 默认值为 0
# 先使用 torch.unsqueeze 来增加张量的维度，使用 torch.cat 拼接张量，使其变成二维张量
source_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, MAX_SRC_NUM_SEQ_LEN + 1, (L,)), (0, MAX_SRC_SEQ_LEN - L)), 0) for L in
     src_len])
target_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, MAX_TGT_NUM_SEQ_LEN + 1, (L,)), (0, MAX_TGR_SEQ_LEN - L)), 0) for L in
     tgt_len])
# print(source_seq)
# print(target_seq)

# 3.构造 embedding，nn.Embedding 是一个预训练好的查找表，用于将离散的单词索引映射到连续的向量空间中，每个单词索引对应于一个向量。
# 使用 MAX_SRC_NUM_SEQ_LEN + 1 作为 num_embeddings 的参数值是为了确保嵌入层能够处理所有可能的单词索引。这里的“+1”是因为索引从0开始，如果最大单词索引是 MAX_SRC_NUM_SEQ_LEN，那么词表的大小应该是 MAX_SRC_NUM_SEQ_LEN + 1以包含索引0。
src_embedding_table = nn.Embedding(MAX_SRC_NUM_SEQ_LEN + 1, MODEL_DIM)
tgt_embedding_table = nn.Embedding(MAX_TGT_NUM_SEQ_LEN + 1, MODEL_DIM)
# print(src_embedding_table.weight)
# print(tgt_embedding_table.weight)

# 返回该索引对应的嵌入向量
src_embedding = src_embedding_table(source_seq)
tgt_embedding = src_embedding_table(target_seq)
# print(src_embedding)
# print(tgt_embedding)

# 4.构造 position embedding
pos_mat = torch.arange(MAX_POSITION_LEN).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / MODEL_DIM)

pe_embedding_table = torch.zeros(MAX_POSITION_LEN, MODEL_DIM)
# 偶数行
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
# 奇数行
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
print(pe_embedding_table)

pe_embedding = nn.Embedding(MAX_POSITION_LEN, MODEL_DIM)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
print(pe_embedding.weight)

# 生成位置索引
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)
print(src_pos)
print(tgt_pos)

# 生成 position embedding
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
print(src_pe_embedding)
print(tgt_pe_embedding)
