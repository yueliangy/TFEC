import numpy as np
import torch
from fastdtw import fastdtw
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

def FTAug(X, goal='recon'):
    # 计算 DTW 距离矩阵
    distance_matrix = compute_euclidean_distance_matrix(X)

    # 使用自适应密度邻域关系来获取邻居集合
    neighbors, _ = adaptive_density_neighbors(X, distance_matrix)

    # 获取 X 的样本数、特征维度
    num_samples, _, feature_dim = X.shape
    max_seq_len = X.shape[1]  # 假设填充到原始的最大序列长度

    # 初始化结果集合为张量
    x_l_collection = torch.zeros((num_samples, max_seq_len, feature_dim), device=X.device)
    x_r_collection = torch.zeros((num_samples, max_seq_len, feature_dim), device=X.device)
    cropNum = np.random.uniform(0.9, 1)  # 随机生成裁剪比例
    # 遍历 X 中的每一个样本，获取正确的顺序
    for i in range(num_samples):
        x = X[i].reshape(-1, feature_dim)  # 当前样本，重塑为原始形状
        x_Neighbors = neighbors[i]  # 当前样本的邻居集合（保持原始形状）

        # 将邻居集合转换为 numpy 数组
        x_Neighbors = [x.cpu() if isinstance(x, torch.Tensor) else x for x in x_Neighbors]
        x_Neighbors = np.array(x_Neighbors)

        # 执行 Left_Right_cropping 和 freq_mix_with_neighbors

        x_L, x_R, X_n_R = Left_Right_cropping(x, x_Neighbors, cropNum)
        segment_length = x_R.shape[0]  # 获取实际长度
        m1, m2 = add_mixed_missing_mask(segment_length, feature_dim, missing_rate=0.7)
        x_r = freq_mix_with_neighbors(x_R, X_n_R, goal)

        if goal == 'recon':
            device = m1.device
            x_L = torch.tensor(x_L, device=device) if isinstance(x_L, np.ndarray) else x_L.to(device)
            x_r = x_r.to(device)
            x_l = x_L * m1
        else:
            x_l = x_L

        # 填充较短的序列以适应最大长度
        x_l_padded = F.pad(x_l, (0, 0, 0, max_seq_len - x_l.size(0)))  # 在时间维度填充
        x_r_padded = F.pad(x_r, (0, 0, 0, max_seq_len - x_r.size(0)))

        # 将填充后的结果赋值到预分配的张量中
        x_l_collection[i] = x_l_padded
        x_r_collection[i] = x_r_padded

    return x_l_collection, x_r_collection

# def scalar_euclidean(a, b):
#     return abs(a - b)
#
#
# def compute_dtw_distance_matrix(data):
#     sample_num = data.shape[0]
#     distance_matrix = np.zeros((sample_num, sample_num))
#
#     # 计算每一对样本的 DTW 距离
#     for i in range(sample_num):
#         for j in range(i + 1, sample_num):
#             # 计算两个样本的 DTW 距离，将所有特征维度的序列分别处理后取平均
#             dtw_distance = 0
#             for dim in range(data.shape[2]):
#                 # 取每个维度的序列
#                 series1 = data[i, :, dim]
#                 series2 = data[j, :, dim]
#                 dist, _ = fastdtw(series1.cpu().numpy() if isinstance(series1, torch.Tensor) else series1,
#                                   series2.cpu().numpy() if isinstance(series2, torch.Tensor) else series2,
#                                   dist=scalar_euclidean)
#                 dtw_distance += dist
#             dtw_distance /= data.shape[2]  # 对各个特征的 DTW 距离取平均
#             distance_matrix[i, j] = dtw_distance
#             distance_matrix[j, i] = dtw_distance
#
#     return distance_matrix

def compute_euclidean_distance_matrix(data):
    sample_num = data.shape[0]

    # 确保数据在 CPU 上并转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # 计算欧氏距离矩阵
    distance_matrix = cdist(data.reshape(sample_num, -1), data.reshape(sample_num, -1), metric='euclidean')
    return distance_matrix
def adaptive_density_neighbors(data, distance_matrix):
    density_gaps = []
    neighbors = []

    for i in range(len(distance_matrix)):
        # 按照距离对邻居进行排序
        sorted_distances = np.sort(distance_matrix[i])
        sorted_indices = np.argsort(distance_matrix[i])

        # 寻找密度突变点（自适应选择邻域大小）
        ki = len(distance_matrix)-1
        for q in range(2, len(sorted_distances)):
            if sorted_distances[q] == sorted_distances[q - 1]:
                break
            else:
                # 正常情况，保持原来的分母
                if sorted_distances[q] / q < sorted_distances[q - 1] / (q - 1):
                    # print("111")
                    ki = q - 2
                    break

        if ki==0: ki = 1      #如果没有邻居，则取最近一位作为邻居

        # 存储邻域样本（直接存储数据而不是索引）
        neighbor_samples = data[sorted_indices[1:ki + 1]]  # 提取邻域样本数据
        neighbors.append(neighbor_samples)

        # 计算密度差
        density_gap = sorted_distances[ki] if ki < len(sorted_distances) else float('inf')
        density_gaps.append(density_gap)

    return neighbors, density_gaps

# def NAN(data, min_rounds=3):
#     # 将每个时序样本展平成单个一维特征向量
#     sample_num, seq_len, feature_dim = data.shape
#     reduced_data = data.reshape(sample_num, seq_len * feature_dim)  # 直接展开为 (sample_num, seq_len * feature_dim)
#
#     # 构建 k-d 树
#     T = KDTree(reduced_data)
#     NaN_Neighbors = [[] for _ in range(sample_num)]  # 使用列表来存储每个样本的邻居
#     NaN_Num = [0] * sample_num  # 存储每个数据点的自然邻居数量
#     r = 1
#     flag = False
#
#     while not flag:
#         cnt = 0  # 计数自然邻居数量为0的点
#         rep = 0  # 记录cnt连续出现的次数
#
#         for i in range(sample_num):
#             # 找到第 r 个最近邻
#             knn_r_i = T.query(reduced_data[i], r + 1)[1][-1]  # 找到第 r 个最近邻索引
#             knn_r_neighbor_set = set(T.query(reduced_data[knn_r_i], r + 1)[1])  # 第 r 个邻居的邻居集合
#
#             # 检查对称性：如果两个点互为最近邻，添加到邻居列表
#             if i in knn_r_neighbor_set:
#                 # 将邻居样本重塑为 (seq_len, feature_dim) 形状并存入邻居列表中
#                 neighbor_sample = data[knn_r_i].reshape(seq_len, feature_dim)
#                 if not any(np.array_equal(neighbor_sample, neighbor) for neighbor in NaN_Neighbors[i]):
#                     NaN_Neighbors[i].append(neighbor_sample)
#
#                 # 将自身作为邻居添加到对方的邻居列表中
#                 self_sample = data[i].reshape(seq_len, feature_dim)
#                 if not any(np.array_equal(self_sample, neighbor) for neighbor in NaN_Neighbors[knn_r_i]):
#                     NaN_Neighbors[knn_r_i].append(self_sample)
#
#                 NaN_Num[i] = len(NaN_Neighbors[i])
#                 NaN_Num[knn_r_i] = len(NaN_Neighbors[knn_r_i])
#
#         # 更新cnt：统计 NaN_Num 中为 0 的点
#         cnt = NaN_Num.count(0)
#
#         # 检查cnt是否连续不变（即更新rep）
#         if cnt == 0:
#             flag = True
#         elif cnt == rep:
#             rep += 1
#         else:
#             rep = 1
#
#         # 修改稳定条件，至少运行 min_rounds 次
#         if r >= min_rounds and (all(num > 0 for num in NaN_Num) and rep >= np.sqrt(r) - rep):
#             flag = True
#
#         # 增加轮次
#         r += 1
#
#     # 输出结果
#     λ = r - 1  # 自然邻居特征值
#     return NaN_Neighbors, NaN_Num, T


# def dynamic_k_selection(x, xi, max_k=20, alpha=0.1):
#     """
#     使用 CM-kNN 的稀疏重构方法为每个样本动态选择 k 值。
#
#     参数：
#     - x: numpy数组，形状为 (sample_NUM, sequence_length, feature_dim)，所有时间序列样本
#     - xi: numpy数组，形状为 (sequence_length, feature_dim)，目标时间序列样本
#     - max_k: int，最大 k 值
#     - alpha: float，稀疏正则化项的系数，用于控制稀疏性
#
#     返回：
#     - int，动态选择的最佳 k 值
#     """
#     x_flat = x.reshape(x.shape[0], -1)  # 展平训练样本
#     xi_flat = xi.reshape(1, -1)  # 展平目标样本
#
#     # 使用稀疏编码（Lasso 回归）来构建重构权重
#     lasso = Lasso(alpha=alpha, max_iter=1000)
#     lasso.fit(x_flat.T, xi_flat.flatten())
#     weights = lasso.coef_
#
#     # 根据权重的稀疏性选择最佳 k 值，选择非零权重的数量作为 k
#     dynamic_k = np.count_nonzero(weights)
#     dynamic_k = min(dynamic_k, max_k)  # k 不超过 max_k
#
#     print("动态选择的 k 值:", dynamic_k)
#     return max(1, dynamic_k)
#
# def knn_select_neighbors(x, xi):
#     """
#     使用 KNN 找到距离样本 xi 最近的 k 个时间序列。
#
#     参数：
#     - x: numpy数组，形状为 (sample_NUM, sequence_length, feature_dim)，所有时间序列样本
#     - xi: numpy数组，形状为 (sequence_length, feature_dim)，目标时间序列样本
#     - k: int，最近邻的数量
#
#     返回：
#     - numpy数组，形状为 (k, sequence_length, feature_dim)，距离 xi 最近的 k 个时间序列
#     """
#     k = dynamic_k_selection(x, xi)
#     x_flat = x.reshape(x.shape[0], -1)  # 展开成 2D 数组，以便 KNN 计算
#     xi_flat = xi.reshape(1, -1)
#
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_flat)
#     distances, indices = nbrs.kneighbors(xi_flat)
#     return x[indices[0]]


def Left_Right_cropping(xi, X, cropNum):
    """
    对单个时间序列样本 xi 进行裁剪，生成两个等长的随机区间，并允许重叠。
    同时对邻居集合 X 的每个样本进行相同长度的裁剪。

    参数：
    - xi: numpy数组，形状为 (sequence_length, feature_dim)，目标时间序列样本
    - X: numpy数组，形状为 (k, sequence_length, feature_dim)，目标样本的邻居集合

    返回：
    - tuple，包含三个裁剪后的数据
      - segment1: xi 在区间 [a1, b1] 的裁剪片段
      - segment2: xi 在区间 [a2, b2] 的裁剪片段
      - neighbors_segments: X 中每个邻居在区间 [a2, b2] 的裁剪片段
    """
    T, feature_dim = xi.shape  # 序列的总长度和特征维度
    # cropNum = np.random.uniform(0.1, 0.5)  # 随机生成裁剪比例
    segment_length = int(T * cropNum)  # 计算裁剪片段的长度

    # 随机生成第一个区间的起始点和终止点
    a1 = np.random.randint(0, T - segment_length + 1)
    b1 = a1 + segment_length

    # 随机生成第二个区间的起始点和终止点，允许区间重叠
    a2 = np.random.randint(0, T - segment_length + 1)
    b2 = a2 + segment_length

    # 裁剪 xi 的两个区间的数据
    segment1 = xi[a1:b1]  # [a1, b1] 的片段
    segment2 = xi[a2:b2]  # [a2, b2] 的片段

    # 对邻居集合 X 的每个样本进行裁剪 [a2, b2]
    neighbors_segments = X[:, a2:b2, :]  # 裁剪 X 的每个邻居片段，得到形状 (k, segment_length, feature_dim)

    return segment1, segment2, neighbors_segments


# def freq_mix_with_neighbors(xi, x_neighbors, m1, m2, goal="recon", mix_ratio_range=(0.1, 0.3)):
#     """
#     将 xi 与邻居的时间序列片段在频域中混合。
#
#     参数：
#     - xi: torch张量，形状为 (segment_length, feature_dim)，目标时间序列片段
#     - x_neighbors: list，包含 k 个邻居的时间序列片段，形状为 (k, segment_length, feature_dim)
#     - goal: str，目标类型 ("reconstruction" 使用自掩码策略，否则直接混合)
#     - mix_ratio_range: tuple，频域替换的比例范围（默认(0.1, 0.3)）
#
#     返回：
#     - torch张量，形状为 (segment_length, feature_dim)，混合后的时间序列
#     """
#     # 将 xi 转换为 torch.Tensor，如果它是 numpy.ndarray
#     if isinstance(xi, np.ndarray):
#         xi = torch.tensor(xi, dtype=torch.complex64)
#     else:
#         xi = xi.to(dtype=torch.complex64)
#
#     # 将 x_neighbors 中的每个元素都转换为 torch.Tensor
#     x_neighbors = [torch.tensor(neigh, dtype=torch.complex64) if isinstance(neigh, np.ndarray) else neigh.to(dtype=torch.complex64)
#                    for neigh in x_neighbors]
#
#
#     xi_f = torch.fft.fft(xi, dim=0)  # 对每个特征维度进行 FFT
#     segment_length, feature_dim = xi.shape
#     # 如果目标是 "reconstruction"，使用自掩码策略
#     if goal == "recon":
#         # 初始化频率混合后的结果
#         mixed_real = m1 * xi_f.real.clone()
#         mixed_imag = m1 * xi_f.imag.clone()
#     else:
#         # 不使用掩码策略，直接进行频率混合
#         replace_ratio = np.random.uniform(*mix_ratio_range)
#         num_replace = int(segment_length * replace_ratio)
#         indices = np.random.choice(segment_length, num_replace, replace=False)
#         mask = torch.zeros((segment_length, feature_dim), dtype=torch.bool)
#         mask[indices, :] = True
#         mixed_real = xi_f.real.clone()
#         mixed_imag = xi_f.imag.clone()
#
#     # 对每个邻居的频率成分进行混合
#     for x_neighbor in x_neighbors:
#         x_neighbor_f = torch.fft.fft(x_neighbor, dim=0)
#         if goal == "recon":
#             # 使用自掩码策略
#             mixed_real += m2 * x_neighbor_f.real
#             mixed_imag += m2 * x_neighbor_f.imag
#         else:
#             # 使用简单掩码
#             mixed_real = torch.where(mask, x_neighbor_f.real, mixed_real)
#             mixed_imag = torch.where(mask, x_neighbor_f.imag, mixed_imag)
#
#     # 将混合后的实部和虚部合成为复数频谱
#     mixed_f = torch.complex(mixed_real, mixed_imag)
#
#     # 对混合后的频域信号进行逆 FFT 转换回时域
#     mixed_x = torch.fft.ifft(mixed_f, dim=0)
#     if goal == "recon":
#         mixed_x = mixed_x * m2
#     return mixed_x.real  # 返回实数部分


# 添加混合缺失的函数--自掩码策略（连续随机时序掩码+随机离散掩码）


def freq_mix_with_neighbors(xi, x_neighbors, goal="recon", mix_ratio_range=(0.1, 0.3)):
    # 将 xi 转换为 torch.Tensor，如果它是 numpy.ndarray
    if isinstance(xi, np.ndarray):
        xi = torch.tensor(xi, dtype=torch.float32)
    else:
        xi = xi.to(dtype=torch.float32)

    # 将 x_neighbors 中的每个元素都转换为 torch.Tensor
    x_neighbors = [torch.tensor(neigh, dtype=torch.float32) if isinstance(neigh, np.ndarray) else neigh.to(dtype=torch.float32)
                   for neigh in x_neighbors]

    device = xi.device  # 获取 xi 所在的设备，假设 xi 在正确的设备上
    distances = [
        torch.mean(torch.norm(xi.to(device) - x_neighbor.to(device), dim=1))
        for x_neighbor in x_neighbors
    ]
    distances = torch.tensor(distances)

    # 将距离转换为权重（距离越小，权重越大）
    inverse_distances = 1 / (distances + 1e-8)
    weights = inverse_distances / inverse_distances.sum()

    # 随机选择一个频率区间的比例（10%到30%之间），并计算总的替换频率段长度
    replace_ratio = np.random.uniform(*mix_ratio_range)
    total_replace_length = int(xi.shape[0] * replace_ratio)

    # 计算每个邻居的替换频率段长度（根据权重分配）
    replace_lengths = (weights * total_replace_length).int().tolist()

    # 将 xi 转换到频域
    xi_f = torch.fft.fft(xi.to(dtype=torch.complex64), dim=0)
    segment_length, feature_dim = xi.shape
    m1, _ = add_mixed_missing_mask(segment_length, feature_dim, missing_rate=0.7)

    device = xi_f.real.device  # 假设 xi_f 已经在正确的设备上

    # 将 m1 和其他张量移动到 device
    m1 = m1.to(device)

    # 初始化频率混合后的结果
    if goal == "recon":
        mixed_real = m1 * xi_f.real.clone()
        mixed_imag = m1 * xi_f.imag.clone()
    else:
        mixed_real = xi_f.real.clone()
        mixed_imag = xi_f.imag.clone()

    # 记录已经替换过的频率索引，避免重复
    used_indices = set()

    # 对每个邻居的频率成分进行替换
    for i, (x_neighbor, replace_length) in enumerate(zip(x_neighbors, replace_lengths)):
        x_neighbor_f = torch.fft.fft(x_neighbor.to(dtype=torch.complex64), dim=0)

        # 确保选择的频率段不与之前的邻居重叠
        available_indices = list(set(range(segment_length)) - used_indices)
        if replace_length > len(available_indices):
            replace_length = len(available_indices)

        # 随机选择要替换的频率段索引
        indices = np.random.choice(available_indices, replace_length, replace=False)
        used_indices.update(indices)

        # 创建替换掩码，针对每个邻居的替换长度和特征维度生成适配的 m2 掩码
        m2, _ = add_mixed_missing_mask(replace_length, feature_dim, missing_rate=0.7)

        device = mixed_real.device  # 获取 mixed_real 的设备

        # 确保 m2 和 x_neighbor_f 在同一设备上
        m2 = m2.to(device)
        x_neighbor_f = x_neighbor_f.to(device)

        # 替换频率段
        if goal == "recon":
            mixed_real[indices] += m2 * x_neighbor_f.real[indices]
            mixed_imag[indices] += m2 * x_neighbor_f.imag[indices]
        else:
            mask = torch.zeros((segment_length, feature_dim), dtype=torch.bool)
            mask = mask.to(device)
            mask[indices, :] = True
            mixed_real = torch.where(mask, x_neighbor_f.real, mixed_real)
            mixed_imag = torch.where(mask, x_neighbor_f.imag, mixed_imag)

    # 将混合后的实部和虚部合成为复数频谱
    mixed_f = torch.complex(mixed_real, mixed_imag)

    # 对混合后的频域信号进行逆 FFT 转换回时域
    mixed_x = torch.fft.ifft(mixed_f, dim=0)
    if goal == "recon":
        mixed_x = mixed_x * m1
    return mixed_x.real



def add_mixed_missing_mask(seq_len, feature_dim, missing_rate=0.7, max_continuous_length=3):
    """
    创建两个包含连续缺失和随机散点缺失的自掩码 m1 和 m2。

    参数：
    - seq_len: int，序列长度
    - feature_dim: int，特征维度
    - missing_rate: float，总缺失率
    - max_continuous_length: int，最大连续缺失的时间步数

    返回：
    - m1: 掩码，包含连续缺失和随机散点缺失
    - m2: 另一个类似的掩码，包含独立的连续缺失和随机散点缺失
    """

    def create_mask():
        # 计算需要缺失的总元素数
        total_missing_elements = int(seq_len * feature_dim * missing_rate)
        continuous_missing_elements = total_missing_elements // 2
        scattered_missing_elements = total_missing_elements - continuous_missing_elements

        # 初始化掩码，全为 True
        mask = torch.ones((seq_len, feature_dim), dtype=torch.bool)

        # 确定所有可选的起始时间步
        available_start_indices = list(range(seq_len))

        # 分布连续缺失
        while continuous_missing_elements > 0 and available_start_indices:
            # 每次掩盖一个随机长度的时间步
            continuous_length = min(max_continuous_length, continuous_missing_elements // feature_dim)

            if continuous_length <= 0:
                break

            # 随机选择非重复的起始时间步
            start_time_idx = np.random.choice(available_start_indices[:seq_len - continuous_length + 1], replace=False)
            mask[start_time_idx:start_time_idx + continuous_length, :] = False

            # 移除已使用的时间步以避免重复
            for idx in range(start_time_idx, start_time_idx + continuous_length):
                if idx in available_start_indices:
                    available_start_indices.remove(idx)

            # 更新连续缺失剩余元素数量
            continuous_missing_elements -= continuous_length * feature_dim

        # 添加散点缺失
        scattered_indices = np.random.choice(seq_len * feature_dim, scattered_missing_elements, replace=False)
        for idx in scattered_indices:
            i = idx // feature_dim
            j = idx % feature_dim
            mask[i, j] = False

        return mask

    # 创建 m1 和 m2 掩码
    m1 = create_mask()
    m2 = create_mask()

    return m1, m2


# # 示例用法
# X = np.random.randn(3, 4, 10)  # 假设有100个样本 ，每个邻居的长度也是 100，每个时间步有 10 个特征
#
# # 调用函数
# x_l, x_r = FTAug(X,  goal='recon')

# print(x_l)
# print(x_r)
