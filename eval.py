import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug,FlushFile,evaluate_synset_w_feature,TensorDataset,epoch
# import wandb
import copy
import random
from functools import partial
from reparam_module import ReparamModule
from feature_metric import MMDLoss, CosineLoss, SupConLoss
import os.path as osp
import warnings
from torchvision.utils import save_image
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 


def manual_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    # manual_seed()
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    
    feature_strategy = args.use_feature
    print("feature_strategy: ", feature_strategy)
    feat_size = [128, 4, 4]
    if args.layer_idx:
        if args.layer_idx == 1:
            feat_size = [128, 16, 16]
        elif args.layer_idx == 2:
            feat_size = [128, 8, 8]
    if args.pooling:
        feat_size = [128, 1, 1]

    if args.feat_metric == "MSE":
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == "MMD":
        feature_criterion = MMDLoss()
    elif args.feat_metric == "cosine":
        feature_criterion = CosineLoss()
    if args.contrast:
        adversarial_criterion = SupConLoss()

    #

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
        
    # define pooling function
    pooling_function = None
    if args.pooling:
        if args.pooling == "avg":
            # avgpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_avg_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "max":
            # maxpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_max_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "sum":
            # sumpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_sum_pool2d, output_size=(1, 1)
            )
        else:
            raise NotImplementedError("Pooling method not implemented")
    #
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    image_syn_pretrained = torch.load('/home/wangshaobo/data/yyt/FD/distilled_data/MTT_ours_c100/IPC10/images_best.pt').to(
        args.device
    )
    image_syn = (
        image_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
    )
    print('image shape',image_syn.shape)
    del image_syn_pretrained
    feature_syn = []
    for _ in range(args.n_feat):
        pretrained_net = get_network(args.model, channel, num_classes, im_size).to(
            args.device
        )  # get a random model
        pretrained_net.train()
        image_syn_train, label_syn_train = copy.deepcopy(
            image_syn.detach()
        ), copy.deepcopy(label_syn.detach())
        image_train, label_train = copy.deepcopy(
            images_all.detach()
        ), copy.deepcopy(
            labels_all.detach()
        )  # avoid any unaware modification
        dst_syn_train = TensorDataset(image_train, label_train)
        trainloader = torch.utils.data.DataLoader(
            dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
        )
        optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=0.01)
        optimizer_pretrained.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        epoch(
            "train",
            trainloader,
            pretrained_net,
            optimizer_pretrained,
            criterion,
            args,
            aug=False
        )
        if isinstance(pretrained_net, torch.nn.DataParallel):
            feature_syn_single = (
                pretrained_net.module.get_features(image_syn_train, args.layer_idx)
                .detach()
                .clone()
            )
        else:
            feature_syn_single = (
                pretrained_net.get_features(image_syn_train, args.layer_idx)
                .detach()
                .clone()
            )
        # copy feature_syn into n_feat in the first dimension
        feature_syn.append(feature_syn_single)
        del pretrained_net
    feature_syn = torch.stack(feature_syn, dim=0).to(args.device)
    feature_syn.requires_grad_()
    print('feature_syn shape',feature_syn.shape)



    feature_criterion = feature_criterion.to(args.device)

    #

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    
    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}


    image_syn_eval = torch.load('/home/wangshaobo/data/yyt/FD/distilled_data/MTT_ours_c100/IPC10/images_best.pt')
    label_syn_eval = torch.load('/home/wangshaobo/data/yyt/FD/distilled_data/MTT_ours_c100/IPC10/labels_best.pt')
    
    print('label_syn_eval shape',label_syn_eval.shape)
    
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s '%(args.model, model_eval ))

        accs_test = []
        accs_train = []
        feat_loss = None
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            feature_syn_eval = copy.deepcopy(feature_syn.detach())
            # if it_eval == 0:
            #     print("Evaluate without feature!")
            #     _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
            # else:
            args.lr_net = syn_lr.item()
            _, acc_train, acc_test = evaluate_synset_w_feature(it_eval, net_eval, image_syn_eval, label_syn_eval,feature_syn_eval, testloader, args,feature_criterion, texture=args.texture,pooling_function=pooling_function,
                    feat_loss=feat_loss,
                    feature_strategy=feature_strategy,)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        if acc_test_mean > best_acc[model_eval]:
            best_acc[model_eval] = acc_test_mean
            best_std[model_eval] = acc_test_std
            save_this_it = True
        print('Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std,feature_strategy))


    # wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode, check utils.py for more info')

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

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='dataset', help='dataset path')
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
    parser.add_argument("--lbd", type=float, default=0.01, help="scale of MSE")
    parser.add_argument(
        "--lr_feat",
        type=float,
        default=0.1,
        help="learning rate for updating synthetic features",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=None, help="layer index for feature"
    )
    parser.add_argument(
        "--pooling", type=str, default=None, help="feature pooling method"
    )
    parser.add_argument("--feat-lbd", type=float, default=0.02, help="scale of CE")
    parser.add_argument("--feat-opt", type=str, default="SGD", help="featureoptimizer")
    parser.add_argument("--n-feat", type=int, default=1, help="number of features")
    parser.add_argument("--eval-freq", type=int, default=4, help="evaluate frequency")
    parser.add_argument(
        "--use-feature", type=str, default="mean", help="the way to use the feature"
    )
    parser.add_argument(
        "--feat-norm", action="store_true", default=False, help="normalize the feature"
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        default=False,
        help="use the contrastive loss",
    )
    parser.add_argument(
        "--lbd-contrast", type=float, default=0.1, help="scale of contrastive loss"
    )
    parser.add_argument(
        "--feat-metric", type=str, default="MSE", help="feature criterion"
    )
    parser.add_argument(
        "--img-method", default="MTT", help="image distillation method"
    )
    parser.add_argument("--img-path", default="../distilled_data", help="image path")
    parser.add_argument("--res-path", default="eval", help="result path")

    args = parser.parse_args()
    
    if args.dataset == 'ImageNet':
        model_args = f'{args.dataset}_{args.subset}_{args.ipc}ipc_[{args.feat_metric}{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_lr-feat{args.lr_feat}_{get_time()}'
    else:
        model_args = f'{args.dataset}_{args.ipc}ipc_[{args.feat_metric}{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_lr-feat{args.lr_feat}_{get_time()}'
    res_path = args.res_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    log_file = os.path.join(res_path, f'{model_args}.txt')
    log_file_handle = open(log_file, 'w')
    sys.stdout = FlushFile(log_file_handle)

    main(args)


