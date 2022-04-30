"""Main routine for training with ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data
import logging

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, isic_2019, isic_2019_hint
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util


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
MODELNAME = "ISIC19-RRRGradCAM-RPEXP-new-reg=0-1-seed=100-clip=1"

print("\nUsing {} device".format(DEVICE))

logging.basicConfig(filename='isic_runs.log', level=logging.INFO, filemode='a', \
    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.info(f"\n\n--------RPEXP Train VGG16 (pretrained), XIL=RRRGradCAM(0.1, clip=1.), base_criterion=CEL, dataset=ISIC2019")
logging.info(f"-HYPERPARAMS epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=SGD(momentum=0.9), save_last={SAVE_LAST}, seeds={SEED}, scheduler={SCHEDULER}, shuffle={TRAIN_SHUFFLE}")

############# Initalize dataset and dataloader
util.seed_all(SEED)
# for CE set ce_augment=True in isic_2019() and train with cross entropy loss
# for HINT use isic_2019_hint() instead of isic_2019() 
dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE)
train_dataloader, test_dataloader, test_no_patches = dataloaders["train"], dataloaders["test"],\
    dataloaders["test_no_patches"]

########### initalize model, loss and optimizer
model = dnns.VGG16_pretrained_isic().to(DEVICE)
base_criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))
#loss = RRRLoss(100, base_criterion=base_criterion, weight=loss_weights)
loss = RRRGradCamLoss(0.1, base_criterion=base_criterion, reduction='mean', weight=loss_weights.to(DEVICE), rr_clipping=1.)
#loss = RBRLoss(100, base_criterion=base_criterion, rr_clipping=10.0, weight=loss_weights)
#loss = CDEPLoss(10, base_criterion=base_criterion, weight=loss_weights, model_type='vgg')
#loss = HINTLoss(1., base_criterion=base_criterion, reduction='none', weight=loss_weights.to(DEVICE))
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)


############ Learn routine
learner = Learner(model, loss, optimizer, DEVICE, MODELNAME, base_criterion=base_criterion)
# for training VANILLA and CE
learner.fit_isic(train_dataloader, test_dataloader, EPOCHS, alternative_dataloader=test_no_patches,\
     scheduler_=SCHEDULER, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, save_last=SAVE_LAST)
# for training RRR, RRR-G, HINT, CDEP
#learner.fit_n_expl_shuffled_dataloader(train_dataloader, test_dataloader, EPOCHS, \
#    alternative_dataloader=test_no_patches, save_last=SAVE_LAST,\
#    verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, scheduler_=SCHEDULER)

logging.info("Training DONE")

############# Evaluate on test-P and test-NP set
print("TEST with patches: ")
learner.validation_statistics(test_dataloader, savename="-STATS-test-with-patches")
logging.info("Test set only patches DONE (see file in logfolder)")
print("Test no patches: ")
learner.validation_statistics(test_no_patches, savename="-STATS-test-no-patches")
logging.info("Test set no patches DONE (see file in logfolder)")