"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import isic_2019, isic_2019_hint
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import explainer
import matplotlib.pyplot as plt


# Get cpu or gpu device for training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 0.001
BATCH_SIZE = 16
SEED = 100
TRAIN_SHUFFLE = True
SAVE_LAST = True
SCHEDULER = True # ReduceLROnPlateau patience=8
EPOCHS = 50
VERBOSE_AFTER_N_EPOCHS = 1

print("\nUsing {} device".format(DEVICE))

############# Initalize dataset and dataloader
util.seed_all(SEED)
dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=True, \
    number_nc=800, number_c=100)
train_dataloader, test_dataloader, test_no_patches = dataloaders["train"], dataloaders["test"],\
    dataloaders["test_no_patches"]


########### initalize model, loss and optimizer
model = dnns.VGG16_pretrained_isic().to(DEVICE)
base_criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))
loss = RRRLoss(10, base_criterion=base_criterion, weight=loss_weights)
#loss = RRRGradCamLoss(1, base_criterion=base_criterion, weight=loss_weights)
#loss = CDEPLoss(10, base_criterion=base_criterion, weight=loss_weights, model_type='vgg', rr_clipping=10.)
# # loss = RRRLoss(100, rawr=True, rawr_threshold=2.)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

#### Load pretrained model from model_store
learner = Learner(model, loss, optimizer, DEVICE, 'ISIC19-CE-seed=100', base_criterion=base_criterion, \
    load=True)

#### To generate heatmaps 
# make sure subfolders exists in the output_images folder
explainer.explain_with_captum_one_by_one('grad_cam', learner.model, test_dataloader, \
     next_to_each_other=True, save_name='isic-expl/ce_gradcam/isic-ce-test-wp-gradcam', device=DEVICE)
#### To generate IG heatmaps
explainer.explain_with_ig_one_by_one(learner.model, test_dataloader, \
      next_to_each_other=True, save_name='isic-expl/cdep_ig/isic-cdep-test-wp-ig', device=DEVICE)
#### To generate LIME heatmaps
explainer.explain_with_lime_one_by_one(learner.model, test_dataloader, \
     next_to_each_other=True, save_name='isic-expl/ce_lime/isic-ce-test-wp-lime')

#### CALCULATE WRONG REASON activation
# IG
# first run with threshold=None to calculate mean (median) threshold and then insert threshold and run again
explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, DEVICE, "name", \
    threshold=None, mode='mean', flags=True)

# GRADCAM
explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, DEVICE, "name", \
    threshold=None, mode='mean', flags=True)

# LIME
explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean', name="RRR-lime", \
    threshold=None, save_raw_attr=True, num_samples=1000)
explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name="RRR-lime", \
    threshold=0.17508005321213746, device=DEVICE, batch_size=BATCH_SIZE)