import numpy as np
import torch
from tqdm import tqdm
from fastdtw import fastdtw
from concurrent.futures import ThreadPoolExecutor

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_cluster,
        max_clusters=100,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cuda')
):
    """
    Perform k-means with automatic selection of cluster numbers using the elbow method.

    :param X: (torch.tensor) matrix
    :param max_clusters: (int) maximum number of clusters to consider [default: 10]
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_euclidean_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float and transfer to device
    X = torch.tensor(X).float().to(device)

    # Step 1: Automatically find the optimal number of clusters using the elbow method
    errors = []
    # print("eeee")
    # print("xxxx")

    # Step 2: Run k-means again with the optimal number of clusters
    return kmeans_fixed_clusters(X, num_cluster, pairwise_distance_function, tol, device)


def kmeans_fixed_clusters(X, num_clusters, pairwise_distance_function, tol, device):
    """
    K-means clustering for a fixed number of clusters.
    :param X: (torch.tensor) input data
    :param num_clusters: (int) number of clusters
    :param pairwise_distance_function: (function) distance function for k-means
    :param tol: (float) convergence threshold
    :param device: (torch.device) computation device
    :return: (torch.tensor, torch.tensor, torch.tensor) cluster ids, distance matrix, final cluster centers
    """
    # Initialize best cluster centers based on minimum sum of distances
    dis_min = float('inf')
    initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, num_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        # 确保 dis 是一个 torch.Tensor
        dis = torch.tensor(dis) if isinstance(dis, np.ndarray) else dis
        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
            if selected.numel() > 0:
                selected = torch.index_select(X, 0, selected)
                initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        iteration += 1
        if iteration > 500 or center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), dis.cpu(), initial_state.cpu()


def find_elbow_point(errors):
    """
    Find the elbow point in the error curve to determine the optimal number of clusters.
    :param errors: (list) total distance error for each cluster count
    :return: (int) optimal number of clusters
    """
    errors = np.array(errors)
    second_derivative = np.diff(errors, n=2)
    optimal_clusters = np.argmax(second_derivative) + 2  # +2 because of indexing offset in second derivative
    return optimal_clusters


# def kmeans(
#         X,
#         num_clusters,
#         distance='euclidean',
#         tol=1e-4,
#         device=torch.device('cuda')
# ):
#     """
#     perform kmeans
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
#     :param tol: (float) threshold [default: 0.0001]
#     :param device: (torch.device) device [default: cpu]
#     :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
#     """
#     # print(f'running k-means on {device}..')
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     # convert to float
#     X = X.float()
#
#     # transfer to device
#     X = X.to(device)
#
#     # initialize
#     dis_min = float('inf')
#     initial_state_best = None
#     for i in range(20):
#         initial_state = initialize(X, num_clusters)
#         dis = pairwise_distance_function(X, initial_state).sum()
#         if dis < dis_min:
#             dis_min = dis
#             initial_state_best = initial_state
#
#     initial_state = initial_state_best
#     iteration = 0
#     while True:
#         dis = pairwise_distance_function(X, initial_state)
#
#         choice_cluster = torch.argmin(dis, dim=1)
#
#         initial_state_pre = initial_state.clone()
#
#         for index in range(num_clusters):
#             selected = torch.nonzero(choice_cluster == index).squeeze().to(device)
#
#             selected = torch.index_select(X, 0, selected)
#             initial_state[index] = selected.mean(dim=0)
#
#         center_shift = torch.sum(
#             torch.sqrt(
#                 torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
#             ))
#
#         # increment iteration
#         iteration = iteration + 1
#
#         if iteration > 500:
#             break
#         if center_shift ** 2 < tol:
#             break
#
#     return choice_cluster.cpu(), dis.cpu(), initial_state.cpu()

def pairwise_euclidean_distance(data1, data2, device=torch.device('cuda')):
    # 将数据移动到指定设备上（通常为 GPU）
    data1 = data1.to(device)
    data2 = data2.to(device)

    # 计算两个数据集之间的欧氏距离矩阵
    distance_matrix = torch.cdist(data1.reshape(data1.shape[0], -1),
                                  data2.reshape(data2.shape[0], -1),
                                  p=2)  # p=2 表示欧氏距离

    # 将结果转回 CPU 并转换为 NumPy 数组
    return distance_matrix.cpu().numpy()
# def scalar_euclidean(a, b):
#     return abs(a - b)
#
# def pairwise_distance(data1, data2, device=torch.device('cuda')):
#     # transfer to device
#
#     sample_num = data1.shape[0]
#     num_clusters = data2.shape[0]
#     distance_matrix = np.zeros((sample_num, num_clusters))
#
#     data1_numpy = data1.cpu().numpy()
#     data2_numpy = data2.cpu().numpy()
#
#     # 计算每一对样本的 DTW 距离
#     for i in range(sample_num):
#         for j in range(num_clusters):
#             # 计算两个样本的 DTW 距离，将所有特征维度的序列分别处理后取平均
#             dtw_distance = 0
#             for dim in range(data1_numpy.shape[2]):
#                 # 取每个维度的序列
#                 series1 = data1_numpy[i, :, dim]
#                 series2 = data2_numpy[j, :, dim]
#                 dist, _ = fastdtw(series1, series2, dist=scalar_euclidean)
#                 dtw_distance += dist
#             dtw_distance /= data1.shape[2]  # 对各个特征的 DTW 距离取平均
#             distance_matrix[i, j] = dtw_distance
#
#     return distance_matrix

def scalar_cosine(a, b):
    # 计算两个标量之间的余弦相似度
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    sample_num = data1.shape[0]
    num_clusters = data2.shape[0]
    distance_matrix = np.zeros((sample_num, num_clusters))

    # 计算每一对样本的余弦相似度
    for i in range(sample_num):
        for j in range(num_clusters):
            cosine_similarity = 0
            # 逐特征维度计算余弦相似度并取平均
            for dim in range(data1.shape[2]):
                # 取每个特征维度的时间序列
                series1 = data1[i, :, dim]
                series2 = data2[j, :, dim]
                # 计算余弦相似度
                similarity = scalar_cosine(series1, series2)
                cosine_similarity += similarity
            # 在特征维度上取平均，得到最终余弦距离
            cosine_similarity /= data1.shape[2]
            distance_matrix[i, j] = 1 - cosine_similarity  # 转换为余弦距离

    return distance_matrix




