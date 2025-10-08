import numpy as np
import torch
from fastdtw import fastdtw
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

def FTAug(X, goal='recon'):

    distance_matrix = compute_euclidean_distance_matrix(X)


    neighbors, _ = adaptive_density_neighbors(X, distance_matrix)


    num_samples, _, feature_dim = X.shape
    max_seq_len = X.shape[1]


    x_l_collection = torch.zeros((num_samples, max_seq_len, feature_dim), device=X.device)
    x_r_collection = torch.zeros((num_samples, max_seq_len, feature_dim), device=X.device)
    cropNum = np.random.uniform(0.9, 1)

    for i in range(num_samples):
        x = X[i].reshape(-1, feature_dim)
        x_Neighbors = neighbors[i]


        x_Neighbors = [x.cpu() if isinstance(x, torch.Tensor) else x for x in x_Neighbors]
        x_Neighbors = np.array(x_Neighbors)



        x_L, x_R, X_n_R = Left_Right_cropping(x, x_Neighbors, cropNum)
        segment_length = x_R.shape[0]
        m1, m2 = add_mixed_missing_mask(segment_length, feature_dim, missing_rate=0.7)
        x_r = freq_mix_with_neighbors(x_R, X_n_R, goal)

        if goal == 'recon':
            device = m1.device
            x_L = torch.tensor(x_L, device=device) if isinstance(x_L, np.ndarray) else x_L.to(device)
            x_r = x_r.to(device)
            x_l = x_L * m1
        else:
            x_l = x_L


        x_l_padded = F.pad(x_l, (0, 0, 0, max_seq_len - x_l.size(0)))  # 在时间维度填充
        x_r_padded = F.pad(x_r, (0, 0, 0, max_seq_len - x_r.size(0)))


        x_l_collection[i] = x_l_padded
        x_r_collection[i] = x_r_padded

    return x_l_collection, x_r_collection



def compute_euclidean_distance_matrix(data):
    sample_num = data.shape[0]


    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()


    distance_matrix = cdist(data.reshape(sample_num, -1), data.reshape(sample_num, -1), metric='euclidean')
    return distance_matrix

def adaptive_density_neighbors(data, distance_matrix):
    density_gaps = []
    neighbors = []

    for i in range(len(distance_matrix)):

        sorted_distances = np.sort(distance_matrix[i])
        sorted_indices = np.argsort(distance_matrix[i])


        ki = len(distance_matrix)-1
        for q in range(2, len(sorted_distances)):
            if sorted_distances[q] == sorted_distances[q - 1]:
                break
            else:

                if sorted_distances[q] / q < sorted_distances[q - 1] / (q - 1):

                    ki = q - 2
                    break

        if ki==0: ki = 1


        neighbor_samples = data[sorted_indices[1:ki + 1]]
        neighbors.append(neighbor_samples)

        density_gap = sorted_distances[ki] if ki < len(sorted_distances) else float('inf')
        density_gaps.append(density_gap)

    return neighbors, density_gaps




def Left_Right_cropping(xi, X, cropNum):

    T, feature_dim = xi.shape

    segment_length = int(T * cropNum)


    a1 = np.random.randint(0, T - segment_length + 1)
    b1 = a1 + segment_length


    a2 = np.random.randint(0, T - segment_length + 1)
    b2 = a2 + segment_length


    segment1 = xi[a1:b1]
    segment2 = xi[a2:b2]


    neighbors_segments = X[:, a2:b2, :]

    return segment1, segment2, neighbors_segments



def freq_mix_with_neighbors(xi, x_neighbors, goal="recon", mix_ratio_range=(0.1, 0.3)):

    if isinstance(xi, np.ndarray):
        xi = torch.tensor(xi, dtype=torch.float32)
    else:
        xi = xi.to(dtype=torch.float32)


    x_neighbors = [torch.tensor(neigh, dtype=torch.float32) if isinstance(neigh, np.ndarray) else neigh.to(dtype=torch.float32)
                   for neigh in x_neighbors]

    device = xi.device
    distances = [
        torch.mean(torch.norm(xi.to(device) - x_neighbor.to(device), dim=1))
        for x_neighbor in x_neighbors
    ]
    distances = torch.tensor(distances)


    inverse_distances = 1 / (distances + 1e-8)
    weights = inverse_distances / inverse_distances.sum()


    replace_ratio = np.random.uniform(*mix_ratio_range)
    total_replace_length = int(xi.shape[0] * replace_ratio)


    replace_lengths = (weights * total_replace_length).int().tolist()


    xi_f = torch.fft.fft(xi.to(dtype=torch.complex64), dim=0)
    segment_length, feature_dim = xi.shape
    m1, _ = add_mixed_missing_mask(segment_length, feature_dim, missing_rate=0.7)

    device = xi_f.real.device


    m1 = m1.to(device)


    if goal == "recon":
        mixed_real = m1 * xi_f.real.clone()
        mixed_imag = m1 * xi_f.imag.clone()
    else:
        mixed_real = xi_f.real.clone()
        mixed_imag = xi_f.imag.clone()


    used_indices = set()


    for i, (x_neighbor, replace_length) in enumerate(zip(x_neighbors, replace_lengths)):
        x_neighbor_f = torch.fft.fft(x_neighbor.to(dtype=torch.complex64), dim=0)


        available_indices = list(set(range(segment_length)) - used_indices)
        if replace_length > len(available_indices):
            replace_length = len(available_indices)


        indices = np.random.choice(available_indices, replace_length, replace=False)
        used_indices.update(indices)


        m2, _ = add_mixed_missing_mask(replace_length, feature_dim, missing_rate=0.7)

        device = mixed_real.device


        m2 = m2.to(device)
        x_neighbor_f = x_neighbor_f.to(device)


        if goal == "recon":
            mixed_real[indices] += m2 * x_neighbor_f.real[indices]
            mixed_imag[indices] += m2 * x_neighbor_f.imag[indices]
        else:
            mask = torch.zeros((segment_length, feature_dim), dtype=torch.bool)
            mask = mask.to(device)
            mask[indices, :] = True
            mixed_real = torch.where(mask, x_neighbor_f.real, mixed_real)
            mixed_imag = torch.where(mask, x_neighbor_f.imag, mixed_imag)


    mixed_f = torch.complex(mixed_real, mixed_imag)


    mixed_x = torch.fft.ifft(mixed_f, dim=0)
    if goal == "recon":
        mixed_x = mixed_x * m1
    return mixed_x.real



def add_mixed_missing_mask(seq_len, feature_dim, missing_rate=0.7, max_continuous_length=3):


    def create_mask():

        total_missing_elements = int(seq_len * feature_dim * missing_rate)
        continuous_missing_elements = total_missing_elements // 2
        scattered_missing_elements = total_missing_elements - continuous_missing_elements


        mask = torch.ones((seq_len, feature_dim), dtype=torch.bool)


        available_start_indices = list(range(seq_len))


        while continuous_missing_elements > 0 and available_start_indices:

            continuous_length = min(max_continuous_length, continuous_missing_elements // feature_dim)

            if continuous_length <= 0:
                break


            start_time_idx = np.random.choice(available_start_indices[:seq_len - continuous_length + 1], replace=False)
            mask[start_time_idx:start_time_idx + continuous_length, :] = False


            for idx in range(start_time_idx, start_time_idx + continuous_length):
                if idx in available_start_indices:
                    available_start_indices.remove(idx)


            continuous_missing_elements -= continuous_length * feature_dim


        scattered_indices = np.random.choice(seq_len * feature_dim, scattered_missing_elements, replace=False)
        for idx in scattered_indices:
            i = idx // feature_dim
            j = idx % feature_dim
            mask[i, j] = False

        return mask


    m1 = create_mask()
    m2 = create_mask()

    return m1, m2



