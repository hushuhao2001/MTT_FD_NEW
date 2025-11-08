import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import (
    get_dataset, get_network, get_eval_pool, evaluate_synset, get_time,
    DiffAugment, ParamDiffAug
)
import copy
import random
from reparam_module import ReparamModule
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------
# Utilities
# -------------------------
def tv_loss(x):
    """Total Variation loss for smoothing images."""
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

def mixup_batch(x, y, alpha: float, num_classes: int):
    """Standard mixup with hard-label CE (lambda*CE(y) + (1-lambda)*CE(y_perm))."""
    if alpha <= 0:
        return x, y, None  # no mixup
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_perm = y[perm]
    return x_mix, y, (y_perm, lam)

def clip_grad_tensor(g, max_norm: float):
    """Gradient clipping for a single tensor gradient."""
    if max_norm is None or max_norm <= 0:
        return g
    with torch.no_grad():
        n = torch.linalg.vector_norm(g)
        if n > max_norm:
            g = g * (max_norm / (n + 1e-12))
    return g

class DummyWandb:
    def __init__(self): self.run = type('', (), {'name':'no_wandb'})()
    def log(self, *_ , **__): pass
    def finish(self): pass

# -------------------------
# Main
# -------------------------
def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset / eval pool
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = \
        get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.im_size = im_size

    # diff aug param
    if args.dsa:
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param
    zca_trans = args.zca_trans if args.zca else None

    # --- wandb (best-effort) ---
    use_wandb = True
    try:
        import wandb
        wandb.init(sync_tensorboard=False,
                   project="DatasetDistillation",
                   job_type="distill_pre",
                   config=args)
        # overwrite args with wandb (to respect sweep)
        _tmp = type('', (), {})()
        for k, v in wandb.config._items.items():
            setattr(_tmp, k, v)
        args = _tmp
        args.dsa_param = dsa_params
        args.zca_trans = zca_trans
        wb = wandb
    except Exception as e:
        print(f"[WARN] wandb init failed ({e}), continue without wandb logging.")
        wb = DummyWandb()
        use_wandb = False

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    # --------- organize real dataset ----------
    images_all = []
    labels_all = []
    indices_class = [[] for _ in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    # --------- initialize synthetic data ----------
    label_syn = torch.tensor(
        [np.ones(args.ipc, dtype=np.int_) * i for i in range(num_classes)],
        dtype=torch.long, requires_grad=False, device=args.device
    ).view(-1)  # [0..0,1..1,...]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    # init learning-rate param
    if args.log_lr:
        # unconstrained param -> sigmoid -> [min_lr, max_lr]
        syn_lr_raw = torch.tensor(np.log(args.lr_teacher + 1e-8), dtype=torch.float32, device=args.device, requires_grad=True)
        get_syn_lr = lambda: args.min_lr + torch.sigmoid(syn_lr_raw)*(args.max_lr - args.min_lr)
    else:
        syn_lr = torch.tensor(args.lr_teacher, dtype=torch.float32, device=args.device, requires_grad=True)
        get_syn_lr = lambda: syn_lr

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                                       j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for _ in range(args.ipc)]
                        )
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    # training params / optimizers
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    if args.log_lr:
        optimizer_lr = torch.optim.SGD([syn_lr_raw], lr=args.lr_lr, momentum=0.5)
    else:
        syn_lr = get_syn_lr().detach().requires_grad_(True)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_img.zero_grad()

    # EMA buffers (for image + lr)
    if args.ema:
        ema_img = image_syn.detach().clone()
        ema_lr_val = float(get_syn_lr().detach().cpu())

    # CE with smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.ce_smooth).to(args.device)

    print('%s training begins' % get_time())

    # --------- load expert buffer ----------
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            buffer = buffer + torch.load(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if n == 0:
            raise AssertionError(f"No buffers detected at {expert_dir}")
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, f"replay_buffer_{n}.pt")):
            expert_files.append(os.path.join(expert_dir, f"replay_buffer_{n}.pt"))
            n += 1
        if n == 0:
            raise AssertionError(f"No buffers detected at {expert_dir}")
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    acc_window = []  # sliding window tracking

    # ---- image lr schedule (warmup + cosine) ----
    img_lr_base = args.lr_img
    def adjust_img_lr(it):
        if args.img_warmup and it < args.img_warmup:
            lr = img_lr_base * it / max(1, args.img_warmup)
        elif args.img_cosine:
            progress = (it - args.img_warmup) / max(1, args.Iteration - args.img_warmup)
            progress = min(max(progress, 0.0), 1.0)
            lr = 0.5 * img_lr_base * (1 + np.cos(np.pi * progress))
        else:
            lr = img_lr_base
        for g in optimizer_img.param_groups:
            g['lr'] = lr
        return lr

    # ------------------ main loop ------------------
    for it in range(0, args.Iteration + 1):
        # adjust image lr
        cur_lr_img = adjust_img_lr(it)

        # -------- evaluation --------
        save_this_it = False
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'
                      % (args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []

                # choose eval tensors (EMA or current)
                with torch.no_grad():
                    if args.ema:
                        image_eval = ema_img.detach().clone()
                        lr_eval = ema_lr_val
                    else:
                        image_eval = image_syn.detach().clone()
                        lr_eval = float(get_syn_lr().detach().cpu())

                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # random init
                    eval_labs = label_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_eval), copy.deepcopy(eval_labs.detach())
                    args.lr_net = lr_eval
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)

                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                # sliding-window average to avoid noisy "best"
                acc_window.append(acc_test_mean)
                if len(acc_window) > args.save_window:
                    acc_window.pop(0)
                avg_recent = float(np.mean(acc_window))

                if avg_recent > best_acc[model_eval]:
                    best_acc[model_eval] = avg_recent
                    best_std[model_eval] = acc_test_std
                    save_this_it = True

                print('Evaluate %d random %s, mean = %.4f std = %.4f (window avg=%.4f)\n-------------------------'
                      % (len(accs_test), model_eval, acc_test_mean, acc_test_std, avg_recent))
                if use_wandb:
                    wb.log({'Accuracy/{}'.format(model_eval): acc_test_mean,
                            'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval],
                            'Std/{}'.format(model_eval): acc_test_std,
                            'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        # ---- save periodic / best ----
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = ema_img.cuda() if args.ema else image_syn.cuda()
                save_dir = os.path.join(".", "logged_files", args.dataset, "distill_pre")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))
                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt"))

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    # (no wandb image logging if disabled)

                    if args.zca:
                        image_rec = image_save.to(args.device)
                        image_rec = args.zca_trans.inverse_transform(image_rec)
                        image_rec = image_rec.cpu()
                        torch.save(image_rec, os.path.join(save_dir, "images_zca_{}.pt".format(it)))

        if use_wandb:
            wb.log({"Synthetic_LR": float(get_syn_lr().detach().cpu()),
                    "Img_LR": cur_lr_img}, step=it)

        # --------- pick an expert trajectory (w/ bias late) ----------
        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        # sample start_epoch
        if args.bias_late_start:
            # sample more late epochs: k controls skewness
            # sample u~U[0,1], t = u^k -> bias towards 1.0
            u = np.random.rand()
            t = u ** args.late_k
            start_epoch = int(t * (args.max_start_epoch - 1))
        else:
            start_epoch = np.random.randint(0, args.max_start_epoch)

        starting_params_list = expert_trajectory[start_epoch]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params_list], 0)

        # multi-target: average of k successive endpoints (robust target)
        targets = []
        k = max(1, int(args.multi_target_k))
        for j in range(k):
            end_list = expert_trajectory[min(start_epoch + args.expert_epochs + j, len(expert_trajectory)-1)]
            end_vec = torch.cat([p.data.to(args.device).reshape(-1) for p in end_list], 0)
            targets.append(end_vec)
        target_params = torch.stack(targets, dim=0).mean(dim=0)

        # --------- build student net (reparam) ----------
        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        student_net = ReparamModule(student_net)
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        # start param vector (leaf)
        current_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params_list], 0).requires_grad_(True)
        student_params = [current_params]

        syn_images = image_syn
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        # --------- inner synthetic steps ----------
        for step in range(args.syn_steps):

            # stratified or random batch indices
            if not indices_chunks:
                if args.stratified_batch:
                    all_indices = []
                    for c in range(num_classes):
                        cls_idx = list(range(c * args.ipc, (c + 1) * args.ipc))
                        random.shuffle(cls_idx)
                        all_indices += cls_idx
                    indices = torch.tensor(all_indices, device=args.device)
                else:
                    indices = torch.randperm(len(syn_images), device=args.device)
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            # texture cropping
            if args.texture:
                x = torch.cat([
                    torch.stack([
                        torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)),
                                        torch.randint(im_size[1]*args.canvas_size, (1,))), (1, 2))[:, :im_size[0], :im_size[1]]
                        for im in x
                    ])
                    for _ in range(args.canvas_samples)
                ])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            # DSA after warmup
            if args.dsa and (not args.no_aug):
                if it >= args.aug_warmup:
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            # Mixup (optional)
            mixup_meta = None
            if args.mixup_alpha > 0:
                x, this_y, mixup_meta = mixup_batch(x, this_y, args.mixup_alpha, num_classes)

            # forward with reparam params
            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]

            logits = student_net(x, flat_param=forward_params)

            # CE (with smoothing), support mixup by two CEs
            if mixup_meta is None:
                ce_loss = criterion(logits, this_y)
            else:
                y_perm, lam = mixup_meta
                ce_loss = lam * criterion(logits, this_y) + (1 - lam) * criterion(logits, y_perm)

            # inner regularization on images (tv/pixel)
            reg_tv = args.lambda_tv * tv_loss(image_syn)
            reg_pix = args.lambda_pix * torch.mean(image_syn ** 2)

            inner_loss = ce_loss  # param matching is outer

            # grad of inner wrt student param vector
            g = torch.autograd.grad(inner_loss, student_params[-1], create_graph=True)[0]
            g = clip_grad_tensor(g, args.clip_grad)

            # step with learned synthetic lr
            cur_syn_lr = get_syn_lr()
            student_params.append(student_params[-1] - cur_syn_lr * g)

        # --------- outer param matching ----------
        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        # normalize
        param_loss /= num_params
        param_dist /= num_params
        param_loss /= (param_dist + 1e-12)

        grand_loss = param_loss + reg_tv + reg_pix

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()
        optimizer_img.step()
        optimizer_lr.step()

        # pixel clamp around mean±k*std
        if args.clip_pixel and args.clip_pixel > 0:
            with torch.no_grad():
                m = torch.mean(image_syn)
                s = torch.std(image_syn)
                lo = m - args.clip_pixel * s
                hi = m + args.clip_pixel * s
                image_syn.data.clamp_(lo, hi)

        # EMA update
        if args.ema:
            with torch.no_grad():
                ema_img.mul_(args.ema_decay).add_(image_syn.detach(), alpha=1 - args.ema_decay)
                ema_lr_val = args.ema_decay * ema_lr_val + (1 - args.ema_decay) * float(get_syn_lr().detach().cpu())

        # free graph tensors
        for _t in student_params:
            del _t

        if it % 10 == 0:
            print('%s iter = %04d, grand_loss = %.6f, param_loss_norm = %.6f, tv=%.6f, pix=%.6f, start_epoch=%d, img_lr=%.4f, syn_lr=%.5f' %
                  (get_time(), it, float(grand_loss.detach().cpu()),
                   float(param_loss.detach().cpu()), float(reg_tv.detach().cpu()),
                   float(reg_pix.detach().cpu()), start_epoch, cur_lr_img, float(get_syn_lr().detach().cpu())))
        if use_wandb:
            wb.log({
                "Progress": it,
                "Grand_Loss": float(grand_loss.detach().cpu()),
                "Param_Loss_Norm": float(param_loss.detach().cpu()),
                "TV": float(reg_tv.detach().cpu()),
                "PIX": float(reg_pix.detach().cpu()),
                "Start_Epoch": start_epoch,
                "cur_img_lr": cur_lr_img,
                "syn_lr_effective": float(get_syn_lr().detach().cpu()),
            }, step=it)

    if use_wandb:
        wb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # -------- base args (原版) --------
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    # -------- 已实现 trick 开关 --------
    parser.add_argument('--ema', action='store_true', help='EMA tracking/saving for image & lr')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay')

    parser.add_argument('--bias-late-start', action='store_true', help='Bias sampling start_epoch to later epochs')
    parser.add_argument('--late-k', type=float, default=2.8, help='Exponent for late-start bias (larger -> later)')
    parser.add_argument('--multi-target-k', type=int, default=1, help='Average targets from k successive endpoints')

    parser.add_argument('--log-lr', action='store_true', help='Use bounded lr via sigmoid(min,max)')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Min synthetic lr bound')
    parser.add_argument('--max-lr', type=float, default=5e-2, help='Max synthetic lr bound')

    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='mixup alpha; 0 disables')
    parser.add_argument('--clip-grad', type=float, default=0.0, help='clip norm for student param gradient (inner loop)')

    parser.add_argument('--lambda-tv', type=float, default=0.0, help='TV regularization weight')
    parser.add_argument('--lambda-pix', type=float, default=0.0, help='Pixel L2 regularization weight')

    # -------- 新补全的增强开关 --------
    parser.add_argument('--aug-warmup', type=int, default=500, help='iters before enabling DSA')
    parser.add_argument('--ce-smooth', type=float, default=0.05, help='label smoothing for CE loss')
    parser.add_argument('--img-warmup', type=int, default=500, help='warmup iters for image lr')
    parser.add_argument('--img-cosine', action='store_true', help='use cosine decay for image lr after warmup')

    parser.add_argument('--stratified-batch', action='store_true', help='class-balanced synthetic batching')
    parser.add_argument('--save-window', type=int, default=3, help='sliding window size for best-save trigger')
    parser.add_argument('--clip-pixel', type=float, default=3.0, help='clamp pixels to mean±k*std after step')

    args = parser.parse_args()
    main(args)