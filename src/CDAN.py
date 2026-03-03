import config
from datetime import datetime
from torch.utils.data import DataLoader
from MEDGNet import FeatureEncoder
from torch import nn
import torch
from MyNewDataset import NormalDataset, TargetDataset
from MEDG import build_global_domain_map
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score
import torch.nn.functional as F

# --- 手动实现 CDAN 核心组件 (替代 tllib) ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * (-ctx.lambd), None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class ConditionalDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator, entropy_conditioning=False):
        super().__init__()
        self.domain_discriminator = domain_discriminator
        self.entropy_conditioning = entropy_conditioning

    def forward(self, y_s, f_s, y_t, f_t):
        softmax = nn.Softmax(dim=1)
        g_s = softmax(y_s)
        g_t = softmax(y_t)

        # Multilinear Map: f * g
        multilinear_s = torch.bmm(f_s.unsqueeze(2), g_s.unsqueeze(1)).view(f_s.size(0), -1)
        multilinear_t = torch.bmm(f_t.unsqueeze(2), g_t.unsqueeze(1)).view(f_t.size(0), -1)

        input_disc = torch.cat((multilinear_s, multilinear_t), dim=0)
        
        batch_size = y_s.size(0)
        domain_label = torch.zeros(batch_size * 2).to(y_s.device)
        domain_label[batch_size:] = 1
        
        # Entropy Conditioning (Optional)
        if self.entropy_conditioning:
            entropy_s = -torch.sum(g_s * torch.log(g_s + 1e-5), dim=1)
            entropy_t = -torch.sum(g_t * torch.log(g_t + 1e-5), dim=1)
            weight_s = 1.0 + torch.exp(-entropy_s)
            weight_t = 1.0 + torch.exp(-entropy_t)
            weight = torch.cat([weight_s, weight_t], dim=0)
            weight = weight / torch.mean(weight)
        else:
            weight = None

        reverse_input = grad_reverse(input_disc, lambd=1.0)
        domain_pred = self.domain_discriminator(reverse_input)
        
        if domain_pred.size(1) == 1:
            loss = F.binary_cross_entropy_with_logits(domain_pred.squeeze(), domain_label, weight=weight)
        else:
            loss = F.cross_entropy(domain_pred, domain_label.long(), reduction='none')
            if weight is not None: loss = (loss * weight).mean()
            else: loss = loss.mean()

        return loss

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

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, num_domains: int):
        """
        CDAN 判别器
        input_dim: feature_dim * num_classes
        num_domains: 域的数量 (如果是二分类则通常为 1 或 2)
        """
        super(Discriminator, self).__init__()
        # 定义隐藏层维度
        hidden_dim = 1024
        
        self.net = nn.Sequential(
            # 第一层：高维映射
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 第二层：深度特征提取
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 输出层
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        return self.net(x)

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
        self.domain_discriminator = Discriminator(feature_dim * num_classes, num_domains).to(self.device)
        
        self.domain_adv = ConditionalDomainAdversarialLoss(
            self.domain_discriminator, 
            entropy_conditioning=config.CDAN_entropy
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.domain_discriminator.parameters()), 
            lr=config.CDAN_lr
        )
        
        self.trade_off = config.CDAN_trade_off

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
    log_msg(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, 任务: {config.TASK}, seed: {args.seed}, weight_domain: {config.CDAN_trade_off}")
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
