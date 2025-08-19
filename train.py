import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
import numpy as np
from utils import *
from tqdm import tqdm
from torch import optim
from model import TSRL
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import optuna
from random import uniform
from Enhancement import *


def compute_reconstruction_loss(outputs, encoder_output, mask, loss_type='mae'):
    """
    计算重构损失和插补损失。

    Args:
        outputs: 解码器重构后的输出，形状为 (sample_num, seq_len, feature_dim)
        encoder_output: 原始输入序列，形状为 (sample_num, seq_len, feature_dim)，包含缺失值
        mask: 掩码矩阵，形状为 (sample_num, seq_len)，非缺失位置为 1，缺失位置为 0
        loss_type: 损失类型，可选 'mse' 或 'mae'，默认为 'mse'

    Returns:
        reconstruction_loss: 非缺失位置的重构损失
        imputation_loss: 缺失位置的插补损失
        total_loss: 总损失（重构损失和插补损失之和）
    """
    # 扩展 mask，使其与 outputs 和 encoder_output 对齐
    # feature_dim = outputs.size(-1)
    # mask_expanded = mask.unsqueeze(-1)
    # print(mask.shape)
    # print(outputs.shape)
    # print(encoder_output.shape)

    outputs = outputs.to(dtype=torch.float32, device=mask.device)
    encoder_output = encoder_output.to(dtype=torch.float32, device=mask.device)

    # 非缺失位置重构损失
    if loss_type == 'mse':
        reconstruction_loss = F.mse_loss(outputs * mask, encoder_output * mask, reduction='sum')
    elif loss_type == 'mae':
        reconstruction_loss = F.l1_loss(outputs * mask, encoder_output * mask, reduction='sum')
    else:
        raise ValueError("loss_type must be 'mse' or 'mae'")

    # 缺失位置插补损失
    imputation_mask = (1 - mask)  # 缺失位置掩码
    if loss_type == 'mse':
        imputation_loss = F.mse_loss(outputs * imputation_mask, encoder_output * imputation_mask, reduction='sum')
    elif loss_type == 'mae':
        imputation_loss = F.l1_loss(outputs * imputation_mask, encoder_output * imputation_mask, reduction='sum')

    # 计算有效元素数量
    valid_elements_recon = mask.sum()
    valid_elements_imp = imputation_mask.sum()

    # 平均化损失，防止有效元素为 0
    reconstruction_loss = reconstruction_loss / (valid_elements_recon + 1e-10)
    imputation_loss = imputation_loss / (valid_elements_imp + 1e-10)

    # 总损失
    total_loss = reconstruction_loss + imputation_loss

    return total_loss



acc_list = []
dcv_list = []
f1_list = []
pre_list = []
nmi_list = []
slt_list = []
dbi_list = []
CH_list = []
Rec_list = []

dataname = "Adiac"

data = np.load(f'datasets2/data/{dataname}_mu_feature_X_test.npy', allow_pickle=True)
label = np.load(f'datasets2/labels/{dataname}_mu_feature_y_test.npy', allow_pickle=True).astype(int)

# data = np.load(f'datasets2/data/{dataname}_multi_feature_X_test.npy', allow_pickle=True)
# label = np.load(f'datasets2/labels/{dataname}_multi_feature_y_test.npy', allow_pickle=True).astype(int)


# data = np.load('./datasets/data/gwp_features.npy', allow_pickle=True)
# label = np.load('./datasets/labels/gwp_labels_reassigned.npy', allow_pickle=True)
# num_cluster = len(np.unique(label))
print(label)
print(label.shape)
print(data.shape)

# 数据形状为 (sample_num, seq_len, feature_dim)
sample_num, seq_len, feature_dim = data.shape

# 将 data 重塑为 (sample_num * seq_len, feature_dim)，以便对每个特征进行统一归一化
data_reshaped = data.reshape(-1, feature_dim)

# 使用 MinMaxScaler 进行归一化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_reshaped)

# 恢复到原始形状 (sample_num, seq_len, feature_dim)
data = data_normalized.reshape(sample_num, seq_len, feature_dim)

train_data = data
train_label = label

# num_samples = sample_num
# subset_size = int(0.5 * num_samples)  # 选择 50% 的样本数
#
# # 随机选择 50% 的样本的索引
# indices = torch.randperm(num_samples)[:subset_size]
#
# # 根据索引选择样本和标签
# train_data = train_data[indices]
# train_label = train_label[indices]
# num_cluster = len(np.unique(train_label))
num_cluster = len(np.unique(train_label))
print(train_label.shape)
print(train_data.shape)

unique_labels, counts = np.unique(train_label, return_counts=True)
proportions = counts / counts.sum()  # 计算每种 label 的比例

# 打印每种 label 和对应的比例
for label, proportion in zip(unique_labels, proportions):
    print(f"Label {label}: {proportion * 100:.2f}%")


lr = 0.001
device = 'cuda'
epoch_num = 100



for seed in range(3,4):

    setup_seed(seed)
    alpha = uniform(0.1, 1)

    # init
    best_acc, best_dcv, best_f1, best_pre, best_rec, best_nmi, best_slt, best_dbi, best_CH, predict_labels, dis = \
        clustering(train_data, train_label, num_cluster)

    #初始化模型
    model = TSRL(input_dims=feature_dim, output_dims=10, hidden_dims=64, depth=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # GPU
    model.to(device)
    train_data = torch.tensor(train_data).to(device)
    sample_size = train_data.shape[0]
    # x1, x2 = FTAug(train_data, goal='recon')  # 有缺失值
    # X1, X2 = FTAug(train_data, goal='con')  # 无缺失值
    x1 = train_data.float()
    x2 = train_data.float()
    X1 = train_data.float()
    X2 = train_data.float()
    # target = torch.eye(train_data.shape[0]*train_data.shape[1]).to(device)   #相似度匹配

    for epoch in range(epoch_num + 1):
        model.train()
        # 模型前向传播（保留梯度）
        r1, r2, x1, x2, m1, m2, o1, o2 = model(x1, x2, X1, X2, train_data)
        recon1 = compute_reconstruction_loss(x1, o1, m1)
        recon2 = compute_reconstruction_loss(x2, o2, m2)
        recon_loss = sample_num * (recon1 + recon2) / 2  # 保留梯度

        print(f"recon_loss: {recon_loss}")

        # if epoch > 60:
        #     high_confidence = torch.min(dis, dim=1).values.to(device)
        #     threshold = torch.sort(high_confidence).values[int(len(high_confidence) * 0.5)]
        #     high_confidence_idx = torch.where(high_confidence < threshold)[0]  # 用torch.where替代numpy索引
        #
        #     # pos samples（全程使用张量操作，移除cpu()和numpy()）
        #     index = torch.tensor(range(train_data.shape[0]), device=device)[high_confidence_idx]
        #     y_sam = torch.tensor(predict_labels, device=device)[high_confidence_idx]
        #     index = index[torch.argsort(y_sam)]
        #     class_num = {}
        #
        #     for label in torch.sort(y_sam).values:
        #         label = label.item()
        #         class_num[label] = class_num.get(label, 0) + 1
        #     key = sorted(class_num.keys())
        #     if len(class_num) < 2:
        #         continue
        #
        #     pos_contrastive = 0.0  # 初始化为浮点张量
        #     centers_1 = torch.tensor([], device=device)
        #     centers_2 = torch.tensor([], device=device)
        #
        #     for i in range(len(key[:-1])):
        #         class_num[key[i + 1]] += class_num[key[i]]
        #         now = index[class_num[key[i]]:class_num[key[i + 1]]]
        #
        #         # 直接索引张量，保留梯度（移除detach()和numpy()）
        #         pos_embed_1 = r1[torch.randint(now.shape[0], (int(now.shape[0] * 0.8),), device=device)]
        #         pos_embed_2 = r1[torch.randint(now.shape[0], (int(now.shape[0] * 0.8),), device=device)]
        #
        #         # 用PyTorch归一化替代numpy（保留梯度）
        #         pos_embed_1 = F.normalize(pos_embed_1, dim=1, p=2)
        #         pos_embed_2 = F.normalize(pos_embed_2, dim=1, p=2)
        #
        #         pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
        #         centers_1 = torch.cat([centers_1, torch.mean(r1[now], dim=0).unsqueeze(0)], dim=0)
        #         centers_2 = torch.cat([centers_2, torch.mean(r1[now], dim=0).unsqueeze(0)], dim=0)
        #
        #     pos_contrastive = pos_contrastive / num_cluster
        #
        #     if len(class_num) < 2:
        #         con_loss = pos_contrastive
        #         loss = con_loss + recon_loss
        #     else:
        #         centers_1 = F.normalize(centers_1, dim=-1, p=2)
        #         centers_2 = F.normalize(centers_2, dim=-1, p=2)
        #         sample_size = centers_1.shape[0]
        #
        #         # 移除detach()和numpy()，直接对张量操作
        #         centers_1_flat = r1.reshape(-1, 10)
        #         centers_2_flat = r2.reshape(-1, 10)
        #         centers_1_flat = F.normalize(centers_1_flat, dim=1, p=2)
        #         centers_2_flat = F.normalize(centers_2_flat, dim=1, p=2)
        #
        #         total_neg_contrastive_loss = 0.0  # 初始化为浮点张量
        #         chunk_size = 100
        #         for i in range(0, sample_size, chunk_size):
        #             for j in range(0, sample_size, chunk_size):
        #                 c1_chunk = centers_1_flat[i:i + chunk_size]
        #                 c2_chunk = centers_2_flat[j:j + chunk_size]
        #                 S_chunk = c1_chunk @ c2_chunk.T
        #
        #                 if i == j:
        #                     S_chunk = S_chunk - torch.diag_embed(torch.diag(S_chunk))
        #
        #                 # 累加张量损失（移除.item()）
        #                 neg_contrastive_chunk = F.mse_loss(S_chunk, torch.zeros_like(S_chunk))
        #                 total_neg_contrastive_loss += neg_contrastive_chunk
        #
        #         neg_contrastive = total_neg_contrastive_loss
        #         con_loss = pos_contrastive + alpha * neg_contrastive
        #         loss = con_loss + recon_loss  # 最终loss为带梯度的张量
        #         print(f"neg_contrastive: {alpha * neg_contrastive}")
        #         print(f"pos_contrastive: {pos_contrastive}")
        #         print(f"Loss: {loss}")
        # else:
        #     # 移除detach()和numpy()，直接对张量操作
        #     r1_flat = r1.reshape(-1, 10)
        #     r2_flat = r2.reshape(-1, 10)
        #
        #     # 用PyTorch归一化替代numpy（保留梯度）
        #     r1_flat = F.normalize(r1_flat, dim=1, p=2)
        #     r2_flat = F.normalize(r2_flat, dim=1, p=2)
        #
        #     chunk_size = 1000
        #     total_loss = 0.0  # 初始化为浮点张量
        #
        #     for i in range(0, r1_flat.shape[0], chunk_size):
        #         for j in range(0, r2_flat.shape[0], chunk_size):
        #             r1_chunk = r1_flat[i:i + chunk_size]
        #             r2_chunk = r2_flat[j:j + chunk_size]
        #             S_chunk = r1_chunk @ r2_chunk.T
        #             target_chunk = torch.eye(S_chunk.size(0), S_chunk.size(1), device=device)
        #
        #             loss_chunk = F.mse_loss(S_chunk, target_chunk)
        #             total_loss += loss_chunk
        #
        #     con_loss = total_loss
        loss = recon_loss
        print(f"Loss: {loss}")

        # # 验证梯度是否正常（测试用，可保留）
        # print("loss.requires_grad:", loss.requires_grad)  # 应输出True
        # print("loss.grad_fn:", loss.grad_fn)  # 应输出非None

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()  # 建议添加梯度清零

        print(f"epoch-{epoch}")

        if epoch % 10 == 0:
            model.eval()
            R = model(x1,x2,X1,X2,train_data)
            print("evaluating")

            # acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, predict_labels, dis = \
                clustering(R, train_label, num_cluster)
            print("-----------------------")
            print(f"acc:{acc:.4f} dcv:{dcv:.4f} f1:{f1:.4f} pre:{pre:.4f} rec:{rec:.4f}  nmi:{nmi:.4f}" )
            print(f"SC:{slt} DBI:{dbi} CH:{CH}")
            print("-----------------------")

            if acc >= best_acc:
                best_acc = acc
                best_dcv = dcv
                best_f1 = f1
                best_pre = pre
                best_rec = rec
                best_nmi = nmi
                best_slt = slt
                best_dbi = dbi
                best_CH = CH

            # acc_list.append(acc)
            # dcv_list.append(dcv)
            # f1_list.append(f1)
            # pre_list.append(pre)
            # Rec_list.append(rec)
            # nmi_list.append(nmi)
            # slt_list.append(slt)
            # dbi_list.append(dbi)
            # CH_list.append(CH)

    acc_list.append(best_acc)
    dcv_list.append(best_dcv)
    f1_list.append(best_f1)
    pre_list.append(best_pre)
    Rec_list.append(best_rec)
    nmi_list.append(best_nmi)
    slt_list.append(best_slt)
    dbi_list.append(best_dbi)
    CH_list.append(best_CH)

acc_list = np.array(acc_list)
dcv_list = np.array(dcv_list)
f1_list = np.array(f1_list)
pre_list = np.array(pre_list)
Rec_list = np.array(Rec_list)
nmi_list = np.array(nmi_list)
slt_list = np.array(slt_list)
dbi_list = np.array(dbi_list)
CH_list = np.array(CH_list)
print("acc:",acc_list.mean(), "±", acc_list.std())
print("dcv:",dcv_list.mean(), "±", dcv_list.std())
print("f1:",f1_list.mean(), "±", f1_list.std())
print("pre:",pre_list.mean(), "±", pre_list.std())
print("rec:",Rec_list.mean(), "±", Rec_list.std())
print("nmi:",nmi_list.mean(), "±", nmi_list.std())
print("slt:",slt_list.mean(), "±", slt_list.std())
print("dbi:",dbi_list.mean(), "±", dbi_list.std())
print("CH:",CH_list.mean(), "±", CH_list.std())


