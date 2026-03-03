import config
from datetime import datetime
from torch.utils.data import DataLoader
from MEDGNet import FeatureEncoder, StrongDiscriminator
from torch import nn
import torch
from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
from MyNewDataset import NormalDataset, TargetDataset, build_global_domain_map
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='随机种子')
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}") 

def log_msg(msg):
    log_messages.append(msg)

class LabelClassifier(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, f):
        return self.net(f)

class CDAN_solver():
    def __init__(self, in_channels, feature_dim, num_classes, num_domains):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = FeatureEncoder(in_channels, feature_dim).to(self.device)
        self.classifier = LabelClassifier(feature_dim, num_classes).to(self.device)
        
        # 判别器输入维度 = 特征维度 * 类别数
        self.domain_discriminator = StrongDiscriminator(feature_dim * num_classes, num_domains).to(self.device)
        
        self.domain_adv = ConditionalDomainAdversarialLoss(
            self.domain_discriminator, 
            entropy_conditioning=config.get('CDAN_entropy', False) 
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.domain_adv.parameters()), 
            lr=config.CDAN_lr
        )
        
        self.trade_off = config.get('CDAN_trade_off', 1.0)

    def train(self, epochs, source_loader, target_loader, val_loader):
        best_val_acc = 0.0
        best_feature_extractor_state = None
        best_classifier_state = None

        for epoch in range(epochs):
            self.feature_extractor.train()
            self.classifier.train()
            self.domain_adv.train()
            
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            len_dataloader = min(len(source_loader), len(target_loader))

            for i in range(len_dataloader):
                try:
                    x_s, y_s, _ = next(source_iter)
                    x_t, _ = next(target_iter) # 假设 TargetDataset 返回
                except StopIteration:
                    break

                x_s = x_s.to(self.device)
                y_s = y_s.to(self.device)
                x_t = x_t.to(self.device)

                f_s = self.feature_extractor(x_s)
                f_t = self.feature_extractor(x_t)
                
                y_s_pred = self.classifier(f_s)
                y_t_pred = self.classifier(f_t) 

                cls_loss = nn.CrossEntropyLoss()(y_s_pred, y_s)
                
                # CDAN Loss: 传入 logits 和 features
                transfer_loss = self.domain_adv(y_s_pred, f_s, y_t_pred, f_t)
                
                loss = cls_loss + self.trade_off * transfer_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 验证阶段
            tgtval_loss, tgtval_acc, val_macro_f1, val_weighted_f1 = eval_cls(self, val_loader, device=self.device)
            print(f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {loss.item():.4f} | " # 打印 item() 而非 tensor
                f"Val Loss: {tgtval_loss:.4f} | "
                f"Val Acc: {tgtval_acc*100:.2f}%")
            
            if tgtval_acc > best_val_acc:
                best_val_acc = tgtval_acc
                best_feature_extractor_state = self.feature_extractor.state_dict()
                best_classifier_state = self.classifier.state_dict()

        # 加载最佳模型
        if best_feature_extractor_state is not None and best_classifier_state is not None:
            self.feature_extractor.load_state_dict(best_feature_extractor_state)
            self.classifier.load_state_dict(best_classifier_state)
        return self

@torch.no_grad()
def eval_cls(model, loader, device):
    """
    评估分类准确率
    """
    model.feature_extractor.eval()
    model.classifier.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    all_preds = []
    all_labels = []

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        f = model.feature_extractor(x)
        y_logits = model.classifier(f)
        loss = F.cross_entropy(y_logits, y)
        pred = y_logits.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item() * y.size(0)

    avg_loss = loss_sum / max(1, total)
    accuracy = correct / max(1, total)

    if len(all_labels) > 0:
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    else:
        macro_f1 = weighted_f1 = 0.0

    return avg_loss, accuracy, macro_f1, weighted_f1

            
if __name__ == "__main__":
    log_messages = []
    
    # 路径设置
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
    target_ds = TargetDataset(
        x_path=train_x, info_path=train_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )
    val_ds = NormalDataset(
        x_path=valid_x, y_path=valid_y, info_path=valid_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )
    test_ds = NormalDataset(
        x_path=test_x, y_path=test_y, info_path=test_info,
        transform=None, filter_domains=filter_domains_tgt, mmap_mode="r"
    )

    global_map = build_global_domain_map(source_ds, target_ds, val_ds)
    source_ds.apply_global_map(global_map)
    target_ds.apply_global_map(global_map)
    val_ds.apply_global_map(global_map)

    num_classes = config.CDAN_num_classes

    log_msg("=" * 80)
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK}, seed: {args.seed}")
    log_msg("-" * 80)
    batch_size = config.CDAN_batch_size
    log_msg(f"参数: num_classes={num_classes}, batch_size={batch_size}, lr={config.CDAN_lr}, epochs={config.CDAN_epochs}")

    src_loader = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=False, drop_last=True 
    )

    tgt_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=True, 
        num_workers=8, pin_memory=False, drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False, drop_last=False
    )

    solver = CDAN_solver(
        in_channels=6,
        feature_dim=128,
        num_classes=num_classes,
        num_domains=len(global_map)
    )

    solver = solver.train(
        epochs=config.CDAN_epochs,
        source_loader=src_loader,
        target_loader=tgt_loader,
        val_loader=val_loader
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=False, drop_last=False
    )
    
    log_msg(f"训练完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    avg_loss, accuracy, macro_f1, weighted_f1 = eval_cls(solver, test_loader, device=solver.device)
    
    log_msg(f"测试结果: 准确率={accuracy*100:.4f}% | Loss={avg_loss:.4f} | Macro F1={macro_f1:.4f} | Weighted F1={weighted_f1:.4f}")
    log_msg("=" * 80)
    
    with open(config.LOGS_DIR / 'CDAN_training.log', 'a') as f:
        f.write('\n'.join(log_messages) + '\n')
