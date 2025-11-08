import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_info_nce_loss(feat_syn, label_syn, c, temperature=0.07):
    """
    Compute the InfoNCE loss for contrastive learning.

    Args:
        feat_syn: Tensor of shape (total_samples, channels, height, width) containing feature vectors.
        label_syn: Tensor of shape (total_samples,) containing labels for each feature vector.
        c: The class label considered as the positive sample.
        temperature: Scaling factor for the similarity scores.

    Returns:
        loss: The InfoNCE loss.
    """
    torch.cuda.synchronize()  # 同步以捕捉可能的CUDA错误

    # 获取类别为 c 的样本特征和负样本特征
    pos_indices = torch.arange(c * 10, (c + 1) * 10).cuda()  # shape: (num_pos_samples,)
    neg_indices = torch.cat([torch.arange(0, c * 10), torch.arange((c + 1) * 10, 100)]).cuda()  # shape: (num_neg_samples,)


    feat_syn_pos = feat_syn[pos_indices]  # shape: (num_pos_samples, 128, 4, 4)
    feat_syn_neg = feat_syn[neg_indices]  # shape: (num_neg_samples, 128, 4, 4)


    # 调整特征张量形状
    feat_syn_pos = feat_syn_pos.view(feat_syn_pos.size(0), -1)  # shape: (num_pos_samples, 128*4*4)
    feat_syn_neg = feat_syn_neg.view(feat_syn_neg.size(0), -1)  # shape: (num_neg_samples, 128*4*4)



    torch.cuda.synchronize()  # 同步以捕捉可能的CUDA错误

    # 计算正样本相似度得分
    pos_similarity = torch.mm(feat_syn_pos, feat_syn_pos.T)  # shape: (num_pos_samples, num_pos_samples)
    pos_similarity = pos_similarity / temperature  # shape: (num_pos_samples, num_pos_samples)

    # 计算负样本相似度得分
    neg_similarity = torch.mm(feat_syn_pos, feat_syn_neg.T)  # shape: (num_pos_samples, num_neg_samples)
    neg_similarity = neg_similarity / temperature  # shape: (num_pos_samples, num_neg_samples)


    # 将正样本和负样本的相似度得分拼接在一起
    logits = torch.cat((pos_similarity, neg_similarity), dim=1)  # shape: (num_pos_samples, num_pos_samples + num_neg_samples)
    labels = torch.arange(pos_similarity.size(0)).cuda()  # shape: (num_pos_samples,)


    torch.cuda.synchronize()  # 同步以捕捉可能的CUDA错误

    # 计算InfoNCE损失
    loss = F.cross_entropy(logits, labels)

    return loss

    
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    # check the shape of the sample, convert it into (bs, d)
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return MMD(input, target, kernel="rbf")


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        input = input / input.norm(dim=1)[:, None]
        target = target / target.norm(dim=1)[:, None]
        return 1 - torch.mean(torch.sum(input * target, dim=1))




import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, feat_syn, label_syn, c):
        """
        Compute the Supervised Contrastive Loss.

        Args:
            feat_syn: Tensor of shape (total_samples, channels, height, width) containing feature vectors.
            label_syn: Tensor of shape (total_samples,) containing labels for each feature vector.
            c: The class label considered as the positive sample.
            temperature: Scaling factor for the similarity scores.

        Returns:
            loss: The Supervised Contrastive Loss.
        """
        device = torch.device('cuda' if feat_syn.is_cuda else 'cpu')

        # Reshape features to (total_samples, channels * height * width)
        feat_syn = feat_syn.view(feat_syn.size(0), -1)  # shape: (total_samples, channels*height*width)

        batch_size = feat_syn.shape[0]

        # Generate mask
        mask = torch.eq(label_syn.unsqueeze(1), label_syn.unsqueeze(0)).float().to(device)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(feat_syn, feat_syn.T),
            self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

# Example usage:
# sup_con_loss = SupConLoss(temperature=0.07, contrast_mode='all')
# loss = sup_con_loss(feat_syn, label_syn, c)


def likelihood(mu, log_var, tgt_embed, eps=1e-5):
    return ((mu - tgt_embed) ** 2 / (log_var.exp() + eps) - log_var).mean()


def club(mu, log_var, tgt_embed, eps=1e-5):
    random_idx = torch.randperm(mu.size(0)).long().to(mu.device)
    positive = - (mu - tgt_embed) ** 2 / (log_var.exp() + eps)
    negative = - (mu - tgt_embed[random_idx]) ** 2 / (log_var.exp() + eps)
    return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2.
