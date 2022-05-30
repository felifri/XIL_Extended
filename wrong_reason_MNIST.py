"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, decoy_mnist_both
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss, MixLoss1, MixLoss2
import util
import explainer
import matplotlib.pyplot as plt
import argparse
import os


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['Vanilla','RRR','RRR-G','HINT','CDEP','CE','RBR', 'Mix1', 'Mix2'],
                    help='Which XIL method to test?')
parser.add_argument('--rrr', default=10, type=int)
parser.add_argument('--rbr', default=100000, type=int)
parser.add_argument('--rrrg', default=1, type=int)
parser.add_argument('--hint', default=100, type=float)
parser.add_argument('--dataset', default='Mnist', type=str, choices=['Mnist','FMnist'],
                    help='Which dataset to use?')
parser.add_argument('--method', default='GradCAM IG LIME', type=str, choices=['GradCAM','IG','LIME'], nargs='+', 
                    help='Which explainer to use?')
parser.add_argument('--run', default=0, type=int,
                    help='Which seed?')


args = parser.parse_args()
# -

# Get cpu or gpu device for training.
DEVICE = "cuda"
SEED = [1, 10, 100, 1000, 10000]
SHUFFLE = True
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 50
SAVE_BEST = True
VERBOSE_AFTER_N_EPOCHS = 2

print("\nUsing {} device".format(DEVICE))

# +
############# Initalize dataset and dataloader
if args.dataset == 'Mnist':
    train_dataloader, test_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode = 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        # args.reg = 10
        args.reg = args.rrr
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        # args.reg = 100000
        args.reg = args.rbr
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        # args.reg = 1
        args.reg = args.rrrg
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        args.reg = 100
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
    elif args.mode == 'CE':
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        args.reg = 1000000  
        loss_fn = CDEPLoss(args.reg)
    elif args.mode == 'Mix1':
        # Loss function combination of RRR, RBR, and RRRG
        args.reg = None
        loss_fn = MixLoss1(regrate_rrr=args.rrr, regrate_rbr=args.rbr, regrate_rrrg=args.rrrg)
    elif args.mode == 'Mix2':
        # Loss function combination of RRRG + HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss2(regrate_rrrg=args.rrrg, regrate_hint=args.hint)
        
elif args.dataset == 'FMnist':
    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode == 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        args.reg = 10
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        args.reg = 1000000
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        args.reg = 10
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        args.reg = 0.00001
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
    elif args.mode == 'CE':
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        args.reg = 2000000  
        loss_fn = CDEPLoss(args.reg)
    elif args.mode == 'Mix1':
        # Loss function combination of RRR, RBR, and RRRG
        args.reg = None
        loss_fn = MixLoss1(regrate_rrr=args.rrr, regrate_rbr=args.rbr, regrate_rrrg=args.rrrg)
    elif args.mode == 'Mix2':
        # Loss function combination of RRRG + HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss2(regrate_rrrg=args.rrrg, regrate_hint=args.hint)


# +
avg1, avg2, avg3 = [], [], []
# i = args.run
# util.seed_all(SEED[i])
# model = dnns.SimpleConvNet().to(DEVICE)
# if args.mode == 'Mix1':
#     MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.rbr},{args.rrrg}--seed={SEED[i]}--run={i}--rrcliping=None'
# elif args.mode == 'Mix2':
#     pass
# else:
#     MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.reg}--seed={SEED[i]}--run={i}'
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# learner = Learner(model, loss_fn, optimizer, DEVICE, MODELNAME, load=True)
#
# if 'GradCAM' in args.method:
#     os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_grad/', exist_ok=True)
#     explainer.explain_with_captum('grad_cam', learner.model, test_dataloader, range(len(test_dataloader)), \
#      next_to_each_other=False, save_name=f'{args.dataset}-expl/{args.mode}_grad/{args.dataset}-{args.mode}-test-wp-grad')
#     thresh = explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean', name=f'{args.mode}-grad', \
#         threshold=None, flags=False, device=DEVICE)
#     avg1.append(explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean', name=f'{args.mode}-grad', \
#         threshold=thresh, flags=False, device=DEVICE))
#
# if 'IG' in args.method:
#     os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_ig/', exist_ok=True)
#     explainer.explain_with_ig(learner.model, test_dataloader, range(len(test_dataloader)), \
#      next_to_each_other=False, save_name=f'{args.dataset}-expl/{args.mode}_ig/{args.dataset}-{args.mode}-test-wp-ig')
#     thresh = explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean', name=f'{args.mode}-ig', \
#         threshold=None, flags=False, device=DEVICE)
#     avg2.append(explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean', name=f'{args.mode}-ig', \
#         threshold=thresh, flags=False, device=DEVICE))
#
# if 'LIME' in args.method:
#     os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_lime/', exist_ok=True)
#     explainer.explain_with_lime(learner.model, test_dataloader, range(len(test_dataloader)), \
#      next_to_each_other=False, save_name=f'{args.dataset}-expl/{args.mode}_lime/{args.dataset}-{args.mode}-test-wp-lime')
#     thresh = explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean', name=f'{args.mode}-lime', \
#         threshold=None, save_raw_attr=True, num_samples=1000, flags=False, gray_images=True)
#     avg3.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{args.mode}-lime', \
#         threshold=thresh, device=DEVICE, batch_size=BATCH_SIZE))

for i in range(5):
    util.seed_all(SEED[i])
    model = dnns.SimpleConvNet().to(DEVICE)
    if args.mode == 'Mix1':
        MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.rbr},{args.rrrg}--seed={SEED[i]}--run={i}'
    elif args.mode == 'Mix2':
        MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrrg},{args.hint}--seed={SEED[i]}--run={i}'
    else:
        MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.reg}--seed={SEED[i]}--run={i}'
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    learner = Learner(model, loss_fn, optimizer, DEVICE, MODELNAME, load=True)

    if 'GradCAM' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_grad/', exist_ok=True)
        explainer.explain_with_captum('grad_cam', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{args.dataset}-expl/{args.mode}_grad/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-grad', \
                                                 threshold=None, flags=False, device=DEVICE)
        avg1.append(explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-grad', \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'IG' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_ig/', exist_ok=True)
        explainer.explain_with_ig(learner.model, test_dataloader, range(len(test_dataloader)), \
                                  next_to_each_other=False,
                                  save_name=f'{args.dataset}-expl/{args.mode}_ig/{args.dataset}-{args.mode}-test-wp-ig')
        thresh = explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-ig', \
                                                 threshold=None, flags=False, device=DEVICE)
        avg2.append(explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-ig', \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'LIME' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_lime/', exist_ok=True)
        explainer.explain_with_lime(learner.model, test_dataloader, range(len(test_dataloader)), \
                                    next_to_each_other=False,
                                    save_name=f'{args.dataset}-expl/{args.mode}_lime/{args.dataset}-{args.mode}-test-wp-lime')
        thresh = explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean',
                                                      name=f'{args.mode}-lime', \
                                                      threshold=None, save_raw_attr=True, num_samples=1000, flags=False,
                                                      gray_images=True)
        avg3.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{args.mode}-lime', \
                                                                 threshold=thresh, device=DEVICE,
                                                                 batch_size=BATCH_SIZE))

if args.mode == 'Mix1':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.rbr}-{args.rrrg}.txt", "w")
elif args.mode == 'Mix2':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrrg}-{int(args.hint)}.txt", "w")
else :
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}.txt", "w")
f.write(f'Grad P: mean:{np.mean(avg1)}, std:{np.std(avg1)}\n '
        f'IG P: mean:{np.mean(avg2)}, std:{np.std(avg2)}\n '
        f'LIME P: mean:{np.mean(avg3)}, std:{np.std(avg3)}\n ')
f.close()
