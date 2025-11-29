import torch
import torch.nn.functional as F
from torch import digamma

# 每个解耦空间排除一个视图（被假设为无判别信息的）如果是-1就不排除
def compute_centroids_for_decoupled(y_list, exclude_index):
    """
    y_list: list of [B, D] 视图特征
    exclude_index: int，表示要排除的视图索引（即视为负样本）
    return: [num_valid_views, D] 张量
    """
    valid_views = [y for i, y in enumerate(y_list) if i != exclude_index]
    centroids = torch.stack([y.mean(dim=0) for y in valid_views])
    return centroids

# 每个样本属于质心的概率（用 Student-t 分布）
def student_t_distribution(z, centroids, mu=1.0):
    dist = torch.cdist(z, centroids, p=2).pow(2)
    num = (1 + dist / mu).pow(-(mu + 1) / 2)
    return num / num.sum(dim=1, keepdim=True)

# 计算解耦空间的可信程度
def compute_trustworthiness(probs):
    """
    probs: [B, K] 概率分布
    return: scalar in [0, 1], 表示一个batch平均可信度
    """
    evidence = probs
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    trust = (digamma(alpha) - digamma(S)).sum(dim=1) / alpha.size(1)
    return torch.sigmoid(trust).mean()
