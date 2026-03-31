import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CDDGNet import CDDGNet
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
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

if args.seed is not None:
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

log_messages = []

def log_msg(msg):
    log_messages.append(msg)
    print(msg)

def causal_aggregation_loss(z, labels):
    """
    Minimizes distance between same class/domain, maximizes distance between different.
    (Aligned with author's source code: uses L2 normalized cosine similarity)
    """
    B = z.size(0)
    D = z.size(1)
    if B < 2:
        return torch.tensor(0.0, device=z.device)
    
    # 按照作者源码：在 dim=1 上做 L2 标准化
    z = F.normalize(z, p=2, dim=1)
    
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    # 相似度矩阵 / D
    sim = torch.matmul(z, z.T) / D
    
    sum_pos = torch.sum(mask)
    sum_neg = torch.sum(1 - mask)
    
    loss_pos = -(mask * sim).sum() / (sum_pos + 1e-8) if sum_pos > 0 else torch.tensor(0.0, device=z.device)
    loss_neg = ((1 - mask) * sim).sum() / (sum_neg + 1e-8) if sum_neg > 0 else torch.tensor(0.0, device=z.device)
    
    return loss_pos + loss_neg

def redundancy_reduction_loss(fm_vec, fh_vec):
    """
    Barlow Twins inspired loss
    (Aligned with author's source code)
    """
    B = fm_vec.size(0)
    D = fm_vec.size(1)
    if B < 2:
        return torch.tensor(0.0, device=fm_vec.device)
    
    # 按照作者源码及注释：在 dim=0 上做 L2 标准化，而不是普通的 standard scaling
    fm_vec = F.normalize(fm_vec, p=2, dim=0)
    fh_vec = F.normalize(fh_vec, p=2, dim=0)
    
    # Cross-correlation matrices
    sim_fm = torch.matmul(fm_vec.T, fm_vec)
    sim_fh = torch.matmul(fh_vec.T, fh_vec)
    
    E = torch.eye(D, device=fm_vec.device)
    
    # Author's normalization for diagonal penalty
    loss_fm = ((1 - E) * sim_fm).pow(2).sum() / torch.sum(1 - E)
    loss_fh = ((1 - E) * sim_fh).pow(2).sum() / torch.sum(1 - E)
    
    loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()
    
    return loss_fm + loss_fh + loss_fmh

def build_global_domain_map(*datasets):
    all_domains = []
    for ds in datasets:
        all_domains.extend(ds.domains)
    uniq = sorted(set(all_domains))
    global_domain_to_id = {dom: i for i, dom in enumerate(uniq)}
    return global_domain_to_id

@torch.no_grad()
def eval_cls(model, loader, device):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0
    all_preds = []
    all_labels = []
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        y_pred, _, _, _ = model(x)
        loss = F.cross_entropy(y_pred, y)
        
        pred = y_pred.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    if len(all_labels) > 0:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0
        
    return loss_sum / total, correct / total, macro_f1, weighted_f1

def train(
    source_ds, val_ds,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda",
    save_name: str = f"cddg_task{config.TASK}",
    batch_size: int = 64
):
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = CDDGNet(in_channels=config.channels, feat_dim=128, num_classes=num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Set step_size to a reasonable value for epochs, e.g., 50 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_acc = 0.0
    best_state_dict = None
    
    alpha = 1.0
    beta = 1.0
    gamma = 0.1
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_sum = 0
        
        for step, (x_s, y_s, did_s) in enumerate(source_loader):
            x_s, y_s, did_s = x_s.to(device), y_s.to(device), did_s.to(device)
            
            y_pred, x_recon, z_c, z_d = model(x_s)
            
            # Classification Loss with Dynamic Domain Weighting
            ce_values = F.cross_entropy(y_pred, y_s, reduction='none')
            
            unique_domains = torch.unique(did_s)
            domain_weights = torch.ones_like(did_s, dtype=torch.float32)
            ce_value_sum = 0
            domain_means = {}
            
            for dom in unique_domains:
                mask = (did_s == dom)
                mean_ce = ce_values[mask].mean()
                domain_means[dom.item()] = mean_ce
                ce_value_sum += mean_ce
                
            for dom in unique_domains:
                mask = (did_s == dom)
                weight = 1 + (domain_means[dom.item()] / (ce_value_sum + 1e-8))
                domain_weights[mask] = weight
                
            L_cl = (ce_values * domain_weights).mean()
            
            # Causal Aggregation Loss
            L_ca_c = causal_aggregation_loss(z_c, y_s)
            L_ca_d = causal_aggregation_loss(z_d, did_s)
            L_ca = L_ca_c + L_ca_d
            
            # Reconstruction Loss
            L_rc = F.mse_loss(x_recon, x_s)
            
            # Redundancy Reduction Loss
            L_rr = redundancy_reduction_loss(z_c, z_d)
            
            # Total Loss
            total_loss = L_cl + alpha * L_ca + beta * L_rc + gamma * L_rr
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_loss_sum += total_loss.item()
            
        scheduler.step()
        
        # Validation
        _, src_acc, _, _ = eval_cls(model, source_loader, device)
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = eval_cls(model, val_loader, device)
        
        log_msg(f"Epoch {epoch:03d} | Loss: {total_loss_sum/len(source_loader):.4f} | LR: {scheduler.get_last_lr()[0]} | "
              f"Src Acc: {src_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            save_path = config.MODELS_DIR / f"{save_name}_{args.seed}.pt"
            torch.save(state, save_path)
            log_msg(f" >>> Best model saved with Val Acc: {best_acc*100:.2f}%")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log_msg(f"\n✓ Loaded best model with Val Acc: {best_acc*100:.2f}%")
    
    return model

def test(model, test_ds, device='cuda'):
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    loss, acc, macro_f1, weighted_f1 = eval_cls(model, test_loader, device)
    
    log_msg(f"Training Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg(f"Test Results: Accuracy={acc*100:.4f}% | Loss={loss:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f}")
    log_msg("=" * 30)
    
    with open(config.LOGS_DIR / 'CDDG_training.log', 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_messages) + '\n')

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

    source_ds = NormalDataset(
        x_path=train_x, y_path=train_y, info_path=train_info,
        transform=None, filter_domains=filter_domains_src, mmap_mode="r"
    )

    val_ds = NormalDataset(
        x_path=valid_x, y_path=valid_y, info_path=valid_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    test_ds = NormalDataset(
        x_path=test_x, y_path=test_y, info_path=test_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    global_map = build_global_domain_map(source_ds, val_ds, test_ds)
    source_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)
    test_ds.apply_global_map(global_map)

    log_msg("=" * 30)
    log_msg(f"CDDG Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Task: {config.TASK}, seed: {args.seed}")
    
    # As per paper
    lr = 0.0001
    batch_size = 64
    epochs = 100

    model = train(
        source_ds=source_ds,
        val_ds=val_ds,
        num_classes=config.num_classes,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        save_name="cddg_task" + str(config.TASK)
    )
    
    test(model, test_ds)
