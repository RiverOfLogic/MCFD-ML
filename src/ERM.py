import torch
from MEDGNet import FeatureEncoder
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from MyNewDataset import NormalDataset, TargetDataset
import config
from torch.utils.data import DataLoader

class ERMModel(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int, num_classes: int):
        super().__init__()
        self.F = FeatureEncoder(input_channel=in_channels, feature_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        z = self.F(x)  # 提取特征
        y_logits = self.classifier(z)  # 分类头
        return y_logits

def train_erm(Source_loader,val_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    for (x_s, y_s, _), (x_t, y_t, _) in zip(Source_loader, val_loader):
        x_s = x_s.to(device)
        y_s = y_s.to(device)
        x_t = x_t.to(device)
        y_t = y_t.to(device)

        # 计算源域损失
        y_logits_s, _ = model(x_s)
        loss_s = F.cross_entropy(y_logits_s, y_s)

        # 计算目标域损失
        y_logits_t, _ = model(x_t)
        loss_t = F.cross_entropy(y_logits_t, y_t)

        # 总损失
        loss = loss_s + loss_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(Source_loader)
    return avg_loss

def els(loader, model, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            y_logits = model(x)
            pred = y_logits.argmax(1)
            loss = F.cross_entropy(y_logits, y)
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        avg_loss = loss_sum / max(1, total)
        accuracy = correct / max(1, total)

    if len(all_labels) > 0:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        #micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0
    return avg_loss, accuracy, macro_f1, weighted_f1

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

    # source 训练域（有标签）
    filter_domains_src = config.DIRG_task_src
    # target 训练域（无标签）
    filter_domains_tgt = config.DIRG_task_tgt
    # datasets
    source_ds = NormalDataset(
        x_path=train_x, y_path=train_y, info_path=train_info,
        transform=None, filter_domains=filter_domains_src, mmap_mode="r"
    )
    # 目标域验证集
    val_ds = NormalDataset(
        x_path=valid_x, y_path=valid_y, info_path=valid_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    test_ds = NormalDataset(
        x_path=test_x, y_path=test_y, info_path=test_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )
    batch_size = config.batch_size
    num_workers = 8

    source_loader = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.DANN_batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )

    model = ERMModel(in_channels=6, feat_dim=128, num_classes=config.ERM_num_classes).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ERM_lr)
    epochs = config.ERM_epochs
    for epoch in range(epochs):
        train_loss = train_erm(source_loader, val_loader, model, optimizer, device="cuda")
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = els(val_loader, model, device="cuda")
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Weighted F1: {val_weighted_f1:.4f}")
    test_loss, test_acc, test_macro_f1, test_weighted_f1 = els(test_loader, model, device="cuda")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Macro F1: {test_macro_f1:.4f} | Test Weighted F1: {test_weighted_f1:.4f}")