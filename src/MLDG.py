import os
import torch
from torch.func import functional_call
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MEDGNet import Model
import copy
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.metrics import f1_score
from typing import List, Tuple
import math
from MyNewDataset import NormalDataset, TargetDataset
import config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='随机种子')
args = parser.parse_args()

# 如果指定了seed，就设置
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

log_messages = []

def log_msg(msg):
    log_messages.append(msg)
    print(msg)

def named_params_dict(module):
    """获取模型所有参数的字典"""
    return {k: v for k, v in module.named_parameters()}

# ---------------------------
# 元学习前向传播函数 (适配 MLDG)
# ---------------------------
def meta_fwd_cls(params, model, x, y):
    """
    专门用于元学习的前向传播函数（只计算分类损失）
    params: 模型的参数字典
    """
    # 剥离前缀以匹配 model 内部的成员变量名
    model_p = {k.replace("model.", ""): v for k, v in params.items() if k.startswith("model.")}
    
    # 使用 functional_call 进行前向计算，MLDG 不需要 alpha 对抗系数，设为 0.0 即可
    outputs = functional_call(model, model_p, (x, 0.0))
    y_logits = outputs[0]  # 第一个返回值是分类 logits
    
    return F.cross_entropy(y_logits, y)

# ---------------------------
# 全局域映射函数
# ---------------------------
def build_global_domain_map(*datasets):
    all_domains = []
    for ds in datasets:
        all_domains.extend(ds.domains)
    uniq = sorted(set(all_domains))
    global_domain_to_id = {dom: i for i, dom in enumerate(uniq)}
    return global_domain_to_id

# ---------------------------
# MLDG 训练函数
# ---------------------------
def train_mldg(
    source_ds, target_ds, val_ds,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-4,
    inner_lr: float = 1e-3,  # MLDG 内环学习率
    weight_beta: float = 1.0, # MLDG 外环损失权重
    device: str = "cuda",
    save_name: str = f"mldg_task{config.TASK}",
    batch_size = 128
):
    # MLDG 是域泛化方法，训练时只使用源域数据
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    num_domains = len(source_ds.domain_to_id)
    model = Model(in_channels=config.channels, feat_dim=128, num_classes=num_classes, num_domains=num_domains).to(device)

    # 对于 MLDG，我们主要优化特征提取器 F 和分类器 C
    optimizer = torch.optim.Adam([
        {"params": model.F.parameters(),  "lr": lr},
        {"params": model.C.parameters(),  "lr": lr},
        # 如果你的模型结构强依赖以下头部，可以保留优化，但 MLDG 算法本身不需要它们
        {"params": model.DC.parameters(), "lr": lr}, 
        {"params": model.D.parameters(),  "lr": lr},
        {"params": model.R.parameters(),  "lr": lr},
    ], weight_decay=1e-4)

    steps_per_epoch = len(source_loader)
    best_acc = 0.0 
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        
        for step, (x_s, y_s, did_s) in enumerate(source_loader):
            x_s, y_s, did_s = x_s.to(device), y_s.to(device), did_s.to(device)

            # ===== 1. 构建元学习任务 (Meta-Task Construction) =====
            unique_domains = torch.unique(did_s).tolist()
            
            if len(unique_domains) < 2:
                # 域数量不足以划分时，随机对半切分
                split_idx = x_s.size(0) // 2
                x_tr, y_tr = x_s[:split_idx], y_s[:split_idx]
                x_te, y_te = x_s[split_idx:], y_s[split_idx:]
            else:
                # MLDG 划分：随机选择 1 个或多个域作为元测试集 (meta-test)
                num_meta_test = 1 if len(unique_domains) == 2 else 2
                test_domains = random.sample(unique_domains, num_meta_test)
                test_domains = torch.tensor(test_domains).to(did_s.device)
                
                test_mask = torch.isin(did_s, test_domains)
                train_mask = ~test_mask  # 其余作为元训练集
                
                x_tr, y_tr = x_s[train_mask], y_s[train_mask]
                x_te, y_te = x_s[test_mask], y_s[test_mask]

                # 防止某些 batch 划分后出现空集
                if x_tr.size(0) == 0 or x_te.size(0) == 0:
                    split_idx = x_s.size(0) // 2
                    x_tr, y_tr = x_s[:split_idx], y_s[:split_idx]
                    x_te, y_te = x_s[split_idx:], y_s[split_idx:]

            # ===== 2. MLDG 内环 (Meta-Train: Inner Loop) =====
            current_params = {f"model.{k}": v for k, v in named_params_dict(model).items()}
            
            # 计算元训练集上的损失
            loss_tr = meta_fwd_cls(current_params, model, x_tr, y_tr)
            
            # 针对参数计算梯度
            grads = torch.autograd.grad(
                loss_tr, current_params.values(), 
                create_graph=True, allow_unused=True
            )
            
            # 模拟梯度下降，得到 fast_params
            fast_params = {}
            for (name, p), g in zip(current_params.items(), grads):
                if g is not None:
                    fast_params[name] = p - inner_lr * g
                else:
                    fast_params[name] = p

            # ===== 3. MLDG 外环 (Meta-Test: Outer Loop) =====
            # 使用 fast_params 在元测试集上计算损失
            loss_te = meta_fwd_cls(fast_params, model, x_te, y_te)

            # ===== 4. 计算总损失并更新 (MLDG Update) =====
            # 标准 MLDG 损失公式: Loss = L_tr + beta * L_te
            total_loss = loss_tr + weight_beta * loss_te

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # --- 验证逻辑 ---
        _, src_acc, _ = eval_cls(model, source_loader, device)
        _, tgt_acc, _ = eval_cls(model, val_loader, device)
        
        print(f"Epoch {epoch:03d} | "
              f"MLDG Meta-Train Loss: {loss_tr.item():.4f} | "
              f"MLDG Meta-Test Loss: {loss_te.item():.4f} | "
              f"Src Acc: {src_acc*100:.2f}% | "
              f"Tgt Acc: {tgt_acc*100:.2f}%")
              
        if tgt_acc > best_acc:
            best_acc = tgt_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'global_map': global_map,  
            }
            save_path = config.MODELS_DIR / f"{save_name}_{args.seed}.pt"
            torch.save(state, save_path)
            print(f" >>> Best model saved with Tgt_Acc: {best_acc*100:.2f}%")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n✓ Loaded best model with Tgt_Acc: {best_acc*100:.2f}%")
    else:
        print("Warning: No best model found during training!")
    
    return model

# ---------------------------
# 评估与测试函数 (与你的原版保持一致，确保特征提取不断裂)
# ---------------------------
@torch.no_grad()
def eval_cls1(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_dom, correct_dom = 0, 0
    loss_sum = 0
    all_z, all_d, all_labels, all_domains = [], [], [], []
    all_preds = []
    all_labels_fat = []

    for x, y, d in loader:
        x, y, d = x.to(device), y.to(device), d.to(device)
        logits, _, logits_dom, _, _, _, _ = model(x, alpha=0.0)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels_fat.extend(y.cpu().numpy())
        correct += (pred == y).sum().item()

        pred_dom = logits_dom.argmax(1)
        correct_dom += (pred_dom == d).sum().item()

        total += y.size(0)
        total_dom += d.size(0)
        loss_sum += loss.item() * y.size(0)

        all_z.append(logits.detach().cpu().numpy()) 
        all_d.append(logits_dom.detach().cpu().numpy()) 
        all_labels.append(y.detach().cpu().numpy()) 
        all_domains.append(d.detach().cpu().numpy())

    all_z = np.concatenate(all_z, axis=0)
    all_d = np.concatenate(all_d, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.concatenate(all_domains, axis=0)
    
    if len(all_labels_fat) > 0:
        macro_f1 = f1_score(all_labels_fat, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels_fat, all_preds, average='weighted', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0

    return loss_sum / total, correct / total, correct_dom / total_dom, all_z, all_d, all_labels, all_domains, macro_f1, weighted_f1

@torch.no_grad()
def eval_cls(model, loader, device):
    model.eval()
    total, correct = 0, 0
    total_dom, correct_dom = 0, 0
    loss_sum = 0
    for x, y, d in loader:
        x, y, d = x.to(device), y.to(device), d.to(device)
        logits, _, logits_dom, _, _, _, _ = model(x, alpha=0.0)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()

        pred_dom = logits_dom.argmax(1)
        correct_dom += (pred_dom == d).sum().item()

        total += y.size(0)
        total_dom += d.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total, correct_dom/total_dom

def test(model, target_ds, batch_size=64, device='cuda', save_path=f'mldg_tsne_results_{args.seed}.pdf'):
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=False)
    loss, acc, dom_acc, all_z, all_d, all_labels, all_domain_labels, macro_f1, weighted_f1 = eval_cls1(model, target_loader, device)
    
    log_msg(f"MLDG 测试完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg(f"测试结果: 准确率={acc*100:.4f}% | Loss={loss:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f}")
    log_msg("=" * 30)
    
    with open(config.LOGS_DIR / 'MLDG_training.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n')
        
    print(f"Test Loss: {loss:.4f}, Target Domain Accuracy: {acc * 100:.2f}%")
    print(f"Domain Classification Accuracy: {dom_acc * 100:.2f}%")
    
    plot_tsne(all_z, all_d, labels=all_labels, domain_labels=all_domain_labels, save_path=save_path)
    print(f"t-SNE plots saved to {save_path}")

def plot_tsne(z_features, d_features, labels, domain_labels, save_path="tsne_output.pdf"):
    tsne = TSNE(n_components=2, random_state=42)
    z_embedded = tsne.fit_transform(z_features)
    d_embedded = tsne.fit_transform(d_features)

    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    axs[0].scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels, cmap='jet', s=10)
    axs[0].set_title('t-SNE of z Features (with Faults)')
    
    axs[1].scatter(z_embedded[:, 0], z_embedded[:, 1], c=domain_labels, cmap='jet', s=10)
    axs[1].set_title('t-SNE of z Features (with Domains)')
    
    axs[2].scatter(d_embedded[:, 0], d_embedded[:, 1], c=domain_labels, cmap='jet', s=10)
    axs[2].set_title('t-SNE of d Features (with Domains)')

    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()

# ---------------------------
# Main 
# ---------------------------
if __name__ == "__main__":
    train_x = config.DIRG_DATA_DIR / "train_x.npy"
    train_y = config.DIRG_DATA_DIR / "train_y.npy"
    train_info = config.DIRG_DATA_DIR / "train_info.npy"

    valid_x = config.DIRG_DATA_DIR / "val_x.npy"
    valid_y = config.DIRG_DATA_DIR / "val_y.npy"
    valid_info = config.DIRG_DATA_DIR / "val_info.npy"

    test_x = config.DIRG_DATA_DIR / "test_x.npy"
    test_y = config.DIRG_DATA_DIR / "test_y.npy"
    test_info = config.DIRG_DATA_DIR / "test_info.npy"

    filter_domains_src = config.DIRG_task_src
    filter_domains_tgt = config.DIRG_task_tgt

    source_ds = NormalDataset(x_path=train_x, y_path=train_y, info_path=train_info, transform=None, filter_domains=filter_domains_src, mmap_mode="r")
    target_ds = TargetDataset(x_path=train_x, info_path=train_info, transform=None, filter_domains=filter_domains_tgt, mmap_mode="r")
    val_ds = NormalDataset(x_path=valid_x, y_path=valid_y, info_path=valid_info, transform=None, filter_domains=filter_domains_tgt, mmap_mode="r")
    test_ds = NormalDataset(x_path=test_x, y_path=test_y, info_path=test_info, transform=None, filter_domains=filter_domains_tgt, mmap_mode="r")

    global_map = build_global_domain_map(source_ds, target_ds, val_ds)
    source_ds.apply_global_map(global_map)
    target_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)
    test_ds.apply_global_map(global_map)

    log_msg("=" * 30)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK}, MLDG Baseline")
    log_msg("." * 30)
    
    # 增加的超参数 MLDG 相关
    mldg_inner_lr = 0.001
    mldg_beta = 1.0
    log_msg(f"参数: num_classes={config.num_classes}, batch_size={config.batch_size}, lr={config.lr}, epochs={config.epochs}, inner_lr={mldg_inner_lr}, beta={mldg_beta}")

    model = train_mldg(
        source_ds=source_ds, 
        target_ds=target_ds, 
        val_ds=val_ds, 
        num_classes=config.num_classes,
        epochs=config.epochs,
        lr=config.lr,
        inner_lr=mldg_inner_lr,
        weight_beta=mldg_beta,
        save_name="mldg_baseline",
        batch_size=config.batch_size
    )
    
    test(model, test_ds, 64)