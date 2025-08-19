from kmeans_gpu import kmeans
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, f1_score, \
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment



def eva(X, ture_labels, predict_labels, show_details=False):
    """
    评估聚类或分类性能，适用于时序数据
    Args:
        predict_labels (array-like): 预测标签，与 true_labels 形状相同
        X (array-like): 原始数据特征，形状为 (num_samples, seq_len, num_features)
        average (str): F1 分数的平均方式，[可选：'macro', 'micro'] [默认: 'macro']
        show_details (bool): 是否打印详细的结果
    Returns:
        dict: 包含各个指标的字典
    """
    # 确保标签是 1 维数组
    predict_labels_flat = predict_labels.reshape(-1)
    true_label_flat = ture_labels.reshape(-1)
    X_reshaped = X.reshape(X.shape[0], -1)  # 将 seq_len 和 feature_dim 合并为一个维度

    # 计算聚类评价指标
    acc = calculate_acc(true_label_flat, predict_labels_flat)
    dcv = calculate_dcv(true_label_flat, predict_labels_flat)
    f1 = calculate_f1(true_label_flat, predict_labels_flat)
    pre = calculate_precision(true_label_flat, predict_labels_flat)
    rec = calculate_recall(true_label_flat, predict_labels_flat)
    nmi = cal_nmi(true_label_flat, predict_labels_flat)
    silhouette = silhouette_score(torch.tensor(X_reshaped).cpu(), predict_labels_flat, metric='euclidean')
    davies_bouldin = davies_bouldin_score(torch.tensor(X_reshaped).cpu(), predict_labels_flat)
    ch_index = calinski_harabasz_score(torch.tensor(X_reshaped).cpu(), predict_labels_flat)
    # silhouette = 0
    # davies_bouldin = 0
    # ch_index = 0

    if show_details:
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    return {
        "acc":acc,
        "dcv":dcv,
        "f1":f1,
        "pre":pre,
        "rec":rec,
        "nmi":nmi,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": ch_index
    }

def calculate_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 获取唯一标签
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # 构建混淆矩阵
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # 计算每个类的精确率、召回率和F1值
    f1_scores = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fp = np.sum(cost_matrix[:, col_ind[i]]) - tp  # 预测为该类的假正类（FP）
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp  # 真实为该类的假负类（FN）
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))
    # 聚类的整体 F1 值
    f1 = np.mean(f1_scores)
    return f1


def calculate_precision(y_true, y_pred):
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 获取标签的唯一值
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # 创建混淆矩阵
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # 计算每个簇的精确率
    precisions = []
    for j in range(len(col_ind)):
        tp = cost_matrix[row_ind[j], col_ind[j]]  # 簇中真实正确分类的样本数
        fp = np.sum(cost_matrix[:, col_ind[j]]) - tp  # 簇中真实错误分类的样本数
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
    # 计算聚类的整体精确率（按簇大小加权平均）
    precision = np.mean(precisions)
    return precision


def calculate_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 获取唯一标签
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # 构建混淆矩阵
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # 计算每个类的召回率
    recalls = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp  # 真正类的总数减去TP得到FN
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
    # 聚类的整体召回率
    recall = np.mean(recalls)
    return recall

def calculate_acc(y_true, y_pred):    #这里输入的都是展平后的一维数组
    # 确保输入是 numpy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 获取标签的唯一值
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # 创建混淆矩阵
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # 计算准确率
    matching = cost_matrix[row_ind, col_ind].sum()
    acc = matching / y_true.size
    return acc


def calculate_dcv(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 样本对的总数
    n = len(y_true)
    # 初始化计数器
    N_00 = N_11 = N_01 = N_10 = 0
    # 计算样本对之间的关系
    for i in range(n):
        for j in range(i + 1, n):  # 避免重复计算对
            true_same = (y_true[i] == y_true[j])
            pred_same = (y_pred[i] == y_pred[j])

            if true_same and pred_same:
                N_11 += 1
            elif not true_same and not pred_same:
                N_00 += 1
            elif true_same and not pred_same:
                N_01 += 1
            elif not true_same and pred_same:
                N_10 += 1
    # 计算 DCV
    dcv = (N_00 + N_11) / (N_00 + N_01 + N_10 + N_11)
    return dcv




def cal_nmi(x, y):
    # Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    n = len(x)
    x = x.reshape(1, n)
    y = y.reshape(1, n)

    # Using flatten and min/max on entire arrays
    l = min(x.min(), y.min())
    x = x - l + 1
    y = y - l + 1
    k = max(x.max(), y.max())

    idx = np.arange(n)
    Mx = np.zeros((n, k), dtype=int)
    My = np.zeros((n, k), dtype=int)
    Mx[idx, x.flatten() - 1] = 1
    My[idx, y.flatten() - 1] = 1
    Pxy = (Mx.T @ My) / n
    Pxy = Pxy[Pxy > 0]
    Hxy = -np.sum(Pxy * np.log2(Pxy))

    Px = np.mean(Mx, axis=0)
    Py = np.mean(My, axis=0)
    Px = Px[Px > 0]
    Py = Py[Py > 0]

    Hx = -np.sum(Px * np.log2(Px))
    Hy = -np.sum(Py * np.log2(Py))

    MI = Hx + Hy - Hxy

    # Avoid division by zero
    if Hx > 0 and Hy > 0:
        z = np.sqrt((MI / Hx) * (MI / Hy))
    else:
        z = 0

    return max(0, z)

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(X, true_label, num_cluster):  #cluster_num用肘部法
    # print(X.shape)
    sample_num, seq_len, feature_dim = X.shape
    predict_labels, dis, initial = kmeans(X=X, num_cluster = num_cluster, max_clusters=sample_num, distance="euclidean", device="cuda")
    metrics = eva(X, true_label, predict_labels.numpy(), show_details=False)

    # 提取各个指标
    slt = metrics["silhouette_score"]
    dbi = metrics["davies_bouldin_index"]
    CH = metrics["calinski_harabasz_index"]
    acc = metrics["acc"]
    dcv = metrics["dcv"]
    f1 = metrics["f1"]
    pre = metrics["pre"]
    rec = metrics["rec"]
    nmi = metrics["nmi"]

    return acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, predict_labels, dis