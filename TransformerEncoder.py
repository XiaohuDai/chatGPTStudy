import torch
import torch.nn as nn
import torch.nn.functional as F

# 批大小，指一次训练迭代中用于更新模型权重的数据样本数量
BATCH_SIZE = 2

# 源序列词表大小，例：一共8个单词，下标从1-8，下标0让给 padding
MAX_SRC_NUM_SEQ_LEN = 8
# 目标序列词表大小
MAX_TGT_NUM_SEQ_LEN = 8

# 模型特征大小，源论文中是512，这里取8
MODEL_DIM = 8

# 源序列的最大长度
MAX_SRC_SEQ_LEN = 5
# 目标序列的最大长度
MAX_TGR_SEQ_LEN = 5

# 位置的最大值
MAX_POSITION_LEN = 5

# 关于 word embedding， 以序列建模为例，考虑 source sentence 和 target sentence
# 构建序列，序列字符以其在词表中的索引的形式表示

# 1. 构建词向量 word embedding
# 1.1 源序列和目标序列长度
# 源序列，例子：tensor([2, 4], dtype=torch.int32)， 即源序列包含两个样本，第一个样本长度为2， 第二个样本长度为4
src_len = torch.Tensor([2, 4]).to(torch.int32)
print(src_len)
# 目标序列，例子：tensor([4, 3], dtype=torch.int32)，即目标序列包含两个样本，第一个样本长度为4， 第二个样本长度为3
tgt_len = torch.Tensor([4, 3]).to(torch.int32)
print(tgt_len)

# # 1.2 构建源序列和目标序列
# # 源序列和目标序列都是由单词索引构成的句子
# # 源序列， 例：[tensor([7, 2]), tensor([6, 7, 2, 3])]，包含两个样本，第一个样本（长度为2，其值由词表中下标为7和2的元素组成），第二个样本（长度为4，其值由源序列词表中下标为6、7、2和3的元素组成）
# source_seq = [torch.randint(1, MAX_SRC_NUM_SEQ_LEN + 1, (L,)) for L in src_len]
# # 目标序列， 例：[tensor([1, 6, 2, 6]), tensor([1, 4, 7])]，源序列包含两个样本，第一个样本（长度为4，其值由词表中下标为1、6、2和6的元素组成），第二个样本（长度为3，其值由目标序列词表中下标为1、4和7的元素组成）
# target_seq = [torch.randint(1, MAX_TGT_NUM_SEQ_LEN + 1, (L,)) for L in tgt_len]

# # 为了使源序列和目标序列的张量的维度一致，需要对张量进行填充
# # source_seq = [F.pad(torch.randint(1, MAX_SRC_NUM_SEQ_LEN + 1, (L,)), (0, max(src_len) - L)) for L in src_len]
# target_seq = [F.pad(torch.randint(1, MAX_TGT_NUM_SEQ_LEN + 1, (L,)), (0, max(tgt_len) - L)) for L in tgt_len]

# 单词索引构成源句子和目标句子， 并且做了 padding， 默认值为 0
# 先使用 torch.unsqueeze 来增加张量的维度，使用 torch.cat 拼接张量，使其变成二维张量
source_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, MAX_SRC_NUM_SEQ_LEN + 1, (L,)), (0, max(src_len) - L)), 0) for L in
     src_len])

target_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, MAX_TGT_NUM_SEQ_LEN + 1, (L,)), (0, max(tgt_len) - L)), 0) for L in
     tgt_len])
# print(source_seq)
# print(target_seq)

# 1.3 构造 embedding，nn.Embedding 是一个预训练好的查找表，用于将离散的单词索引映射到连续的向量空间中，每个单词索引对应于一个向量。
# 使用 MAX_SRC_NUM_SEQ_LEN + 1 作为 num_embeddings 的参数值是为了确保嵌入层能够处理所有可能的单词索引。
# 这里的“+1”是因为索引从0开始，如果最大单词索引是 MAX_SRC_NUM_SEQ_LEN，那么词表的大小应该是 MAX_SRC_NUM_SEQ_LEN + 1以包含索引0，其中下标0让给 padding。
src_embedding_table = nn.Embedding(MAX_SRC_NUM_SEQ_LEN + 1, MODEL_DIM)
tgt_embedding_table = nn.Embedding(MAX_TGT_NUM_SEQ_LEN + 1, MODEL_DIM)
# print(src_embedding_table.weight)
# print(tgt_embedding_table.weight)

# 返回该索引对应的嵌入向量
src_embedding = src_embedding_table(source_seq)
tgt_embedding = src_embedding_table(target_seq)
# print(src_embedding)
# print(tgt_embedding)

# 2. 构造位置向量 position embedding
pos_mat = torch.arange(MAX_POSITION_LEN).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / MODEL_DIM)

# 2.1 构建 embedding 偶数行和奇数行表达式
pe_embedding_table = torch.zeros(MAX_POSITION_LEN, MODEL_DIM)
# 偶数行
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
# 奇数行
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# print(pe_embedding_table)

pe_embedding = nn.Embedding(MAX_POSITION_LEN, MODEL_DIM)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)
# print(pe_embedding.weight)

# 2.2 生成位置索引
src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len]).to(torch.int32)
# print(src_pos)
# print(tgt_pos)

# 2.3 将生成位置索引映射到 embedding 维度中，生成 position embedding
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)
print("src_pe_embedding")
print(src_pe_embedding)
print("tgt_pe_embedding")
print(tgt_pe_embedding)

# 3. 构造 encoder self-attention mask（一次性输入，不需要遮掩）
# mask 的 shape：[BATCH_SIZE, MAX_SRC_LEN, MAX_SRC_LEN], 值为1或 -inf
valid_encoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len]).to(torch.int32), 2)
print(valid_encoder_pos)
# 有效矩阵
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
# print(valid_encoder_pos_matrix)
# 无效矩阵
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
# print(invalid_encoder_pos_matrix)
# 用 bool 值进行填充
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)
# print(mask_encoder_self_attention)

score = torch.randn(BATCH_SIZE, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, -1)
# print(score)
# print(masked_score)
# print(prob)

# 4. 构造 intra-attention 的 mask
# Q K^T shape：[BATCH_SIZE, tgt_seq_len, src_seq_len]
valid_encoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(src_len) - L)), 0) for L in src_len]).to(torch.int32), 2)
# print("valid_encoder_pos")
# print(valid_encoder_pos)

valid_decoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max(tgt_len) - L)), 0) for L in tgt_len]).to(torch.int32), 2)
# print("valid_decoder_pos")
# print(valid_decoder_pos)

valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
# print(valid_cross_pos_matrix)
valid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = valid_cross_pos_matrix.to(torch.bool)
# print(mask_cross_attention)

# 5. 构造 decoder self-attention 的 mask
valid_decoder_tri_matrix = torch.cat(
    [torch.unsqueeze(F.pad(torch.tril(torch.ones(L, L)), (0, max(tgt_len) - L, 0, max(tgt_len) - L)), 0) for L in
     tgt_len], 0)
print("valid_decoder_tri_matrix")
print(valid_decoder_tri_matrix)
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix;
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
print("invalid_decoder_tri_matrix")
print(invalid_decoder_tri_matrix)

score = torch.randn(BATCH_SIZE, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(masked_score, -1)


# print(prob)

# 6. 构建 scaled self-attention
def scaled_dot_product_attention(Q, K, V, attn_mask):
    # shape of Q,K,V: [BATCH_SIZE * num_head, seq_len, MODEL_DIM / num_head]
    score = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(MODEL_DIM)
    masked_score = score.masked_fill(attn_mask, -1e9)
    prob = F.softmax(masked_score, -1)
    context = torch.bmm(prob, V)
    return context

# # softmax 演示，scaled 的重要性
# alpha1 = 0.1
# alpha2 = 10
# score = torch.randn(5)
# prob1 = F.softmax(score * alpha1, -1)
# prob2 = F.softmax(score * alpha2, -1)
#
# def softmax_func(score):
#     return F.softmax(score)
#
# jaco_mat1 = torch.autograd.functional.jacobian(softmax_func, score * alpha1)
# jaco_mat2 = torch.autograd.functional.jacobian(softmax_func, score * alpha2)
#
# print(jaco_mat1)
# print(jaco_mat2)
