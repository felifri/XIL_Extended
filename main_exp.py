#!/usr/bin/env python3
"""
Bunch of helper functions to run the main cross validation runs on 
DecoyMNIST and DecoyFMNIST
"""
import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import explainer

# Hyperparameters
EPOCHS = 64
BATCH_SIZE = 256
LR = 0.001
SAVE_LAST = True
TRAIN_SHUFFLE = True
VERBOSE_AFTER_N_EPOCHS = 2
SEEDS = [1, 10, 100, 1000, 10000]
N_RUNS = len(SEEDS)
DISABLE_XIL_LOSS_FIRST_N_EPOCHS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Using {DEVICE} device]")

logging.basicConfig(filename='main_exp_small_terminal.log', level=logging.INFO, filemode='a', \
    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def cross_validate_helper(n_runs, seeds, modelname, loss_, model_, dataset, n_expl=None, feedback=None):
    """
    To ensure reproducible runs we  have to reset seed every run and also
    reload dataset, init new model and loss every run.

    params:
        loss_: tuple of form ('rrr', reg_rate) in case of cross_entropy only string
        model_: String either 'mlp' or 'cnn'
        dataset: string either 'decoy_mnist', 'fashion_mnist' etc.
    """
    print(f"\n\n--------CROSS VALIDATION different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}")
    print(modelname)
    logging.info(f"--------CROSS VALIDATION different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}, n_expl={str(n_expl)}, feedback={str(feedback)}")
    logging.info(f"-HYPERPARAMS (for all runs): epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=ADAM, save_last={SAVE_LAST}, seeds={SEEDS}")

    train_accs, train_losses, test_accs, test_losses, run_times = [], [], [], [], []
    rr_clipping_rbr = 1.0

    for i in range(n_runs):
        print(f"\n>>>>>> RUN {i}\n")
        seed = seeds[i]
        util.seed_all(seed)

        # Load loss
        if loss_ == 'cross_entropy':
            loss = nn.CrossEntropyLoss()
        
        elif loss_ == 'ce_loss':
            loss = nn.CrossEntropyLoss()


        elif loss_[0] == 'rrr':
            loss = RRRLoss(loss_[1], rr_clipping=rr_clipping_rbr)

        elif loss_[0] == 'rrr_grad_cam':
            loss = RRRGradCamLoss(loss_[1], reduction='mean')
        
        elif loss_[0] == 'rbr':
            loss = RBRLoss(loss_[1], rr_clipping=rr_clipping_rbr)

        elif loss_[0] == 'cdep':
            loss = CDEPLoss(loss_[1])
        
        elif loss_[0] == 'hint':
            loss = HINTLoss(loss_[1], last_conv_specified=loss_[2], upsample=loss_[3], reduction='mean')

        # Load dataset

        if type(dataset) is tuple:

            if dataset[0] == 'decoy_mnist':
                train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, \
                    batch_size=BATCH_SIZE, n_instances=n_expl, \
                        n_counterexamples_per_instance=dataset[2], ce_strategy=dataset[3], \
                            train_shuffle=TRAIN_SHUFFLE, feedback=feedback)
            
            elif dataset[0] == 'decoy_fmnist':
                train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, \
                    fmnist=True, batch_size=BATCH_SIZE, n_instances=dataset[1], \
                        n_counterexamples_per_instance=dataset[2], ce_strategy=dataset[3], \
                            train_shuffle=TRAIN_SHUFFLE, feedback=feedback)
        
        elif dataset == 'mnist':
            train_loader, test_loader = decoy_mnist(no_decoy=True, device=DEVICE, \
                batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, \
                    feedback=feedback)

        elif dataset == 'decoy_mnist':
            train_loader, test_loader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, \
                train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, feedback=feedback)
        
        elif dataset == 'decoy_mnist_hint':
            train_loader, test_loader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, \
                hint_expl=True, train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, feedback=feedback)
        
        elif dataset == 'fmnist':
            train_loader, test_loader = decoy_mnist(no_decoy=True, fmnist=True, \
                device=DEVICE, batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, \
                    feedback=feedback)

        elif dataset == 'decoy_fmnist':
            train_loader, test_loader = decoy_mnist(fmnist=True, device=DEVICE, \
                batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, \
                    feedback=feedback)

        elif dataset == 'decoy_fmnist_hint':
            train_loader, test_loader = decoy_mnist(fmnist=True, device=DEVICE, \
                batch_size=BATCH_SIZE, hint_expl=True, train_shuffle=TRAIN_SHUFFLE, n_expl=n_expl, 
                feedback=feedback)

        # Load model
        if model_ == 'mlp':
            # init model, optimizer and loss
            model = dnns.SimpleMlp().to(DEVICE)

        elif model_ == 'cnn':
            model = dnns.SimpleConvNet().to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        try:
            reg_rate = loss.regularizer_rate
        except:
            reg_rate = 'None'

        if n_expl is not None:
            if isinstance(loss, RBRLoss):
                extras = "--rr_clip=" + str(rr_clipping_rbr)
                modelname_cur = modelname + extras +'--n_expl='+ str(n_expl) + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)
            else:
                modelname_cur = modelname + '--n_expl='+ str(n_expl) + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)

        elif feedback is not None:
            modelname_cur = modelname + '--fb='+ str(feedback) + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)

        elif DISABLE_XIL_LOSS_FIRST_N_EPOCHS is not None:
            modelname_cur = modelname + '--so='+ str(DISABLE_XIL_LOSS_FIRST_N_EPOCHS) + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)
        else:    
            modelname_cur = modelname + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)

        # INIT LEARNER
        learner = Learner(model, loss, optimizer, DEVICE, modelname_cur)
        print(f"Learner config: {learner.config_to_string()}")
        if n_expl is not None and loss_!= "ce_loss":
            run_train_acc, run_train_loss, elapsed_time = learner.fit_n_expl_shuffled_dataloader(train_loader, test_loader, \
            epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)

        elif DISABLE_XIL_LOSS_FIRST_N_EPOCHS is not None:
            run_train_acc, run_train_loss, elapsed_time, bs_store = learner.fit(train_loader, test_loader, \
                epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, \
                    disable_xil_loss_first_n_epochs=DISABLE_XIL_LOSS_FIRST_N_EPOCHS)

        else:
            run_train_acc, run_train_loss, elapsed_time, _ = learner.fit(train_loader, test_loader, \
                epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)

                    
        run_test_acc, run_test_loss = learner.score(test_loader, F.cross_entropy, verbose=False)

        if DISABLE_XIL_LOSS_FIRST_N_EPOCHS is not None:
            logging.info(f"Run {str(i)} done! Model config: {modelname_cur} |", end='')
            logging.info(f"bs_store= {str(bs_store)}")
        else:
            logging.info(f"Run {str(i)} done! Model config: {modelname_cur}")

        # update accuracies and losses
        train_accs.append(run_train_acc)
        train_losses.append(run_train_loss)
        test_accs.append(run_test_acc)
        test_losses.append(run_test_loss)
        run_times.append(elapsed_time)

    # printing and logging
    logging.info(f"Train accs: {train_accs}")
    logging.info(f"Train losses: {train_losses}")
    logging.info(f"Test accs: {test_accs}")
    logging.info(f"Test losses: {test_losses}")
    logging.info(f"Run times: {run_times}")

    # calc mean
    mean_train_acc = sum(train_accs) / n_runs
    mean_train_loss = sum(train_losses) / n_runs
    mean_test_acc = sum(test_accs) / n_runs
    mean_test_loss = sum(test_losses) / n_runs
    mean_run_time = sum(run_times) / n_runs
    print("\n####### MEAN RESULTS #######")
    logging.info(f"--> Model cross validated with n_expl={str(n_expl)}!")
    print(f"-->Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    print(f"-->Mean run time= {mean_run_time}")
    logging.info(f"-->Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    logging.info(f"-->STD train acc= {np.std(np.array(train_accs)):>0.2f}, STD test acc= {np.std(np.array(test_accs)):>0.2f}")
    logging.info(f"-->Mean run time= {mean_run_time:>0.2f}")
    print("\n--------FINISHED CROSS VALIDATION with different seeds\n\n")
    return mean_train_acc, mean_train_loss, mean_test_acc, mean_test_loss, mean_run_time


def cross_validate_ce_switch_on(n_runs, seeds, modelname, loss_, model_, dataset):
    print(f"\n--------CROSS VALIDATION CE different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}")
    print(modelname)
    logging.info(f"--------CROSS VALIDATION CE different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}")
    logging.info(f"-HYPERPARAMS (for all runs): epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=ADAM, save_last={SAVE_LAST}, seeds={SEEDS}, disable_xil_for_n_epochs={DISABLE_XIL_LOSS_FIRST_N_EPOCHS}")

    train_accs, train_losses, test_accs, test_losses, run_times = [], [], [], [], []

    for i in range(n_runs):
        print(f"\n>>>>>> RUN {i}\n")
        seed = seeds[i]
        util.seed_all(seed)

        # Load loss
        loss = nn.CrossEntropyLoss()
        
        model = dnns.SimpleConvNet().to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        modelname_cur = modelname + '--so=50' + '--reg=CE60k' + "--seed=" + str(seed) + '--run=' + str(i)

        # INIT LEARNER
        learner = Learner(model, loss, optimizer, DEVICE, modelname_cur)
        print(f"Learner config: {learner.config_to_string()}")

        train_loader, test_loader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE)

        run_train_acc_bs, run_train_loss_bs, _, _ = learner.fit(train_loader, test_loader, \
            epochs=50, save_last=False, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
        run_test_acc_bs, run_test_loss_bs = learner.score(test_loader, F.cross_entropy, verbose=False)
        bs_store = (50, run_train_acc_bs, run_train_loss_bs, run_test_acc_bs, run_test_loss_bs)


        train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, \
                    batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, n_instances=60000, \
                        n_counterexamples_per_instance=1, ce_strategy='random')

           
        run_train_acc, run_train_loss, elapsed_time, _ = learner.fit(train_loader, test_loader, \
            epochs=50, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)

        run_test_acc, run_test_loss = learner.score(test_loader, F.cross_entropy, verbose=False)

        logging.info(f"Run {str(i)} done! Model config: {modelname_cur} | bs_store= {str(bs_store)}")

        # update accuracies and losses
        train_accs.append(run_train_acc)
        train_losses.append(run_train_loss)
        test_accs.append(run_test_acc)
        test_losses.append(run_test_loss)
        run_times.append(elapsed_time)

    # printing and logging
    logging.info(f"Train accs: {train_accs}")
    logging.info(f"Train losses: {train_losses}")
    logging.info(f"Test accs: {test_accs}")
    logging.info(f"Test losses: {test_losses}")
    logging.info(f"Run times: {run_times}")

    # calc mean
    mean_train_acc = sum(train_accs) / n_runs
    mean_train_loss = sum(train_losses) / n_runs
    mean_test_acc = sum(test_accs) / n_runs
    mean_test_loss = sum(test_losses) / n_runs
    mean_run_time = sum(run_times) / n_runs
    print("\n####### MEAN RESULTS #######")
    print(f"Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    print(f"Mean run time= {mean_run_time}")
    logging.info(f"Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    logging.info(f"STD train acc= {np.std(np.array(train_accs)):>0.2f}, STD test acc= {np.std(np.array(test_accs)):>0.2f}")
    logging.info(f"Mean run time= {mean_run_time:>0.2f}")
    print("\n--------FINISHED CROSS VALIDATION with different seeds\n\n")

def cross_validate_helper_rbr(n_runs, seeds, modelname, loss_, model_, dataset):
    """
    To ensure reproducible runs we  have to reset seed every run and also
    reload dataset, init new model and loss every run.

    params:
        loss_: tuple of form ('rrr', reg_rate) in case of cross_entropy only string
        model_: String either 'mlp' or 'cnn'
        dataset: string either 'decoy_mnist', 'fashion_mnist' etc.
    """
    print(f"\n--------CROSS VALIDATION different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}")
    print(modelname)
    logging.info(f"--------CROSS VALIDATION different seeds on model={model_}, loss={str(loss_)}, dataset={dataset}")
    logging.info(f"-HYPERPARAMS (for all runs): epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=ADAM, save_last={SAVE_LAST}, seeds={SEEDS}, disable_xil_for_n_epochs={DISABLE_XIL_LOSS_FIRST_N_EPOCHS}")

    train_accs, train_losses, test_accs, test_losses, run_times = [], [], [], [], []

    for i in range(n_runs):
        print(f"\n>>>>>> RUN {i}\n")
        seed = seeds[i]
        util.seed_all(seed)

        # Load loss
        if loss_ == 'cross_entropy':
            loss = nn.CrossEntropyLoss()
        
        elif loss_[0] == 'rbr':
            loss = RBRLoss(loss_[1])
        # Load dataset

        if type(dataset) is tuple:

            if dataset[0] == 'decoy_mnist':
                train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, \
                    batch_size=BATCH_SIZE, n_instances=dataset[1], \
                        n_counterexamples_per_instance=dataset[2], ce_strategy=dataset[3])
            
            elif dataset[0] == 'decoy_fmnist':
                train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, \
                    fmnist=True, batch_size=BATCH_SIZE, n_instances=dataset[1], \
                        n_counterexamples_per_instance=dataset[2], ce_strategy=dataset[3])

        elif dataset == 'mnist':
            train_loader, test_loader = decoy_mnist(no_decoy=True, device=DEVICE, \
                batch_size=BATCH_SIZE)

        elif dataset == 'decoy_mnist':
            train_loader, test_loader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)
        
        elif dataset == 'decoy_mnist_hint':
            train_loader, test_loader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, \
                hint_expl=True)
        
        elif dataset == 'fmnist':
            train_loader, test_loader = decoy_mnist(no_decoy=True, fmnist=True, \
                device=DEVICE, batch_size=BATCH_SIZE)

        elif dataset == 'decoy_fmnist':
            train_loader, test_loader = decoy_mnist(fmnist=True, device=DEVICE, \
                batch_size=BATCH_SIZE)

        elif dataset == 'decoy_fmnist_hint':
            train_loader, test_loader = decoy_mnist(fmnist=True, device=DEVICE, \
                batch_size=BATCH_SIZE, hint_expl=True)

        # Load model
        if model_ == 'mlp':
            # init model, optimizer and loss
            model = dnns.SimpleMlp().to(DEVICE)

        elif model_ == 'cnn':
            model = dnns.SimpleConvNet().to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        try:
            reg_rate = loss.regularizer_rate
        except:
            reg_rate = 'None'
        modelname_cur = modelname + '--soAfter=50' + '--reg=' + str(reg_rate) + "--seed=" + str(seed) + '--run=' + str(i)

        # INIT LEARNER
        learner = Learner(model, loss, optimizer, DEVICE, modelname_cur)
        print(f"Learner config: {learner.config_to_string()}")
        run_train_acc, run_train_loss, elapsed_time, _ = learner.fit(train_loader, test_loader, \
            epochs=50, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, \
                disable_xil_loss_first_n_epochs=DISABLE_XIL_LOSS_FIRST_N_EPOCHS)

        # INIT LEARNER
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        learner = Learner(model, loss, optimizer, DEVICE, modelname_cur, load=True)
        run_train_acc, run_train_loss, elapsed_time, _ = learner.fit(train_loader, test_loader, \
            epochs=50, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)

        run_test_acc, run_test_loss = learner.score(test_loader, F.cross_entropy, verbose=False)

        logging.info(f"Run {str(i)} done! Model config: {modelname_cur}")

        # update accuracies and losses
        train_accs.append(run_train_acc)
        train_losses.append(run_train_loss)
        test_accs.append(run_test_acc)
        test_losses.append(run_test_loss)
        run_times.append(elapsed_time)

    # printing and logging
    logging.info(f"Train accs: {train_accs}")
    logging.info(f"Train losses: {train_losses}")
    logging.info(f"Test accs: {test_accs}")
    logging.info(f"Test losses: {test_losses}")
    logging.info(f"Run times: {run_times}")

    # calc mean
    mean_train_acc = sum(train_accs) / n_runs
    mean_train_loss = sum(train_losses) / n_runs
    mean_test_acc = sum(test_accs) / n_runs
    mean_test_loss = sum(test_losses) / n_runs
    mean_run_time = sum(run_times) / n_runs
    print("\n####### MEAN RESULTS #######")
    print(f"Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    print(f"Mean run time= {mean_run_time}")
    logging.info(f"Mean train acc= {mean_train_acc:>0.1f}, Mean train loss= {mean_train_loss:>8f}, Mean test acc= {mean_test_acc:>0.1f}, Mean test loss= {mean_test_loss:>8f}")
    logging.info(f"STD train acc= {np.std(np.array(train_accs)):>0.2f}, STD test acc= {np.std(np.array(test_accs)):>0.2f}")
    logging.info(f"Mean run time= {mean_run_time:>0.2f}")
    print("\n--------FINISHED CROSS VALIDATION with different seeds\n\n")

def interactions_runs(modelname, loss_, model_, dataset):

    N_EXPLS = [25, 50, 100, 200, 400, 800, 1600, 5000, 10000, 20000]
    N_EXPLS = [25, 100, 200, 400, 800, 1600, 5000, 10000, 20000]
    train_accs, train_losses, test_accs, test_losses, run_times = [], [], [], [], []

    logging.info(f"\n\n\n******DIFFERENT N_EXPL on model={model_}, loss={str(loss_)}, dataset={dataset}")
    logging.info(f"------HYPERPARAMS (for all runs): epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=ADAM, save_last={SAVE_LAST}, seeds={SEEDS}")
    logging.info(f"------N_EXPL = {N_EXPLS}")

    for n_expl in N_EXPLS:
        print(f"\n*********TRAIN MODEL n_expl={n_expl}: {modelname}")
        # cross validate 
        mean_train_acc, mean_train_loss, mean_test_acc, mean_test_loss, mean_run_time = \
            cross_validate_helper(N_RUNS, SEEDS, modelname, loss_, model_, dataset, n_expl=n_expl)

        # update accuracies and losses
        train_accs.append(mean_train_acc)
        train_losses.append(mean_train_loss)
        test_accs.append(mean_test_acc)
        test_losses.append(mean_test_loss)
        run_times.append(mean_run_time)

    logging.info(f":-----------RESULTS-----------:")
    logging.info(f"train_accs = {train_accs}")
    logging.info(f"train_loss = {train_losses}")
    logging.info(f"test_acc   = {test_accs}")
    logging.info(f"test_loss  = {test_losses}")
    logging.info(f"Run times  = {run_times}")
    logging.info(f":-----------------------------:")




# model_config = (modelname, loss, modeltype, dataset)
    # Loss := the following options for loss tuple (name, regularizer_rate): 
    #### --> loss = {cross_entropy, ('rrr', reg), ('rrr_grad_cam', reg), ('cdep', reg), ('cdep', reg), ('hint', reg)}
    # modeltype := {'mlp', 'cnn'}
    # dataset := {'mnist', 'decoy_mnist', 'decoy_mnist_hint', 'fmnist', 'decoy_fmnist', 'decoy_fmnist_hint'}
    ### -> for CE := {(dataset, n_instances, n_counterexamples, ce_strategy='random')}
    # Notes: HINt can only be applied on model=cnn



def cross_validate_cnn_decoyMNIST():

    # Notes: HINt can only be applied on model=cnn
    no_decoy_config = ("Mnist-CNN-CEL", 'cross_entropy', 'cnn', 'mnist')
    cel_config = ("DecoyMnist-CNN-CEL", 'cross_entropy', 'cnn', 'decoy_mnist')
    rrr_config = ("DecoyMnist-CNN-RRR", ('rrr', 10), 'cnn', 'decoy_mnist')
    rrr_grad_cam_config = ("DecoyMnist-CNN-RRRGradCAM", ('rrr_grad_cam', 1), 'cnn', 'decoy_mnist')
    cdep_config = ("DecoyMnist-CNN-CDEP", ('cdep', 1000), 'cnn', 'decoy_mnist')
    rbr_config = ("DecoyMnist-CNN-RBR", ('rbr', 1000000), 'cnn', 'decoy_mnist')
    hint_config = ("DecoyMnist-CNN-HINT", ('hint', 100, True, False), 'cnn', 'decoy_mnist_hint')
    ce_config = ("DecoyMnist-CNN-CE", 'cross_entropy', 'cnn', ('decoy_mnist', 60000, 1, 'random'))
    # run 5 train and validation runs with different seeds for all XIL Losses
    configs = [no_decoy_config, cel_config, rrr_config, rrr_grad_cam_config, cdep_config, rbr_config, hint_config, ce_config]

    for modelname, loss, model, dataset in configs:
        cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_cnn_decoyFashionMNIST():

    # Notes: HINt can only be applied on model=cnn
    #no_decoy_config = ("FMnist-CNN-CEL", 'cross_entropy', 'cnn', 'fmnist')
    #cel_config = ("DecoyFMnist-CNN-CEL", 'cross_entropy', 'cnn', 'decoy_fmnist')
    #rrr_config = ("DecoyFMnist-CNN-RRR", ('rrr', 10), 'cnn', 'decoy_fmnist')
    rrr_grad_cam_config = ("DecoyFMnist-CNN-RRRGradCAM", ('rrr_grad_cam', 1), 'cnn', 'decoy_fmnist')
    #cdep_config = ("DecoyFMnist-CNN-CDEP", ('cdep', 1000), 'cnn', 'decoy_fmnist')
    #rbr_config = ("DecoyFMnist-CNN-RBR", ('rbr', 1000000), 'cnn', 'decoy_fmnist')
    hint_config = ("DecoyFMnist-CNN-HINT", ('hint', 100, True, False), 'cnn', 'decoy_fmnist_hint')
    #ce_config = ("DecoyFMnist-CNN-CE", 'cross_entropy', 'cnn', ('decoy_fmnist', 60000, 1, 'random'))
    # run 5 train and validation runs with different seeds for all XIL Losses
    #configs = [no_decoy_config, cel_config, rrr_config, rrr_grad_cam_config, cdep_config, rbr_config, hint_config, ce_config]
    configs = [rrr_grad_cam_config, hint_config]

    for modelname, loss, model, dataset in configs:
        cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_cnn_decoyMNIST_switch_on():

    rrr_config = ("DecoyMnist-CNN-RRR-SO", ('rrr', 100), 'cnn', 'decoy_mnist')
    rbr_config = ("DecoyMnist-CNN-RBR-SO", ('rbr', 1000000), 'cnn', 'decoy_mnist')
    rrr_grad_cam_config = ("DecoyMnist-CNN-RRRGradCAM-SO", ('rrr_grad_cam', 1), 'cnn', 'decoy_mnist')
    cdep_config = ("DecoyMnist-CNN-CDEP-SO", ('cdep', 1000000), 'cnn', 'decoy_mnist')
    hint_config = ("DecoyMnist-CNN-HINT-SO", ('hint', 1000, True, False), 'cnn', 'decoy_mnist_hint')
    ce_config = ("DecoyMnist-CNN-CE-S0", 'cross_entropy', 'cnn', ('decoy_mnist', 60000, 1, 'random'))
    # run 5 train and validation runs with different seeds for all XIL Losses
    configs = [rrr_config, rbr_config, rrr_grad_cam_config, cdep_config, hint_config, ce_config]

    for modelname, loss, model, dataset in configs:
        if modelname == "DecoyMnist-CNN-CE-S0":
            cross_validate_ce_switch_on(N_RUNS, SEEDS, modelname, loss, model, dataset)
        else:
            cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_cnn_decoyMNIST_switch_on_rbr():

    rbr_config = ("DecoyMnist-CNN-RBR", ('rbr', 1000), 'mlp', 'decoy_mnist')
    
    # run 5 train and validation runs with different seeds for all XIL Losses
    configs = [rbr_config]

    for modelname, loss, model, dataset in configs:
        cross_validate_helper_rbr(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_reg_rates_decoyMNIST_mlp():

    reg_rates = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

    for rate in reg_rates:
        rrr_config = ("DecoyMnist-MLP-RRR", ('rrr', rate), 'mlp', 'decoy_mnist')
        config = [rrr_config]
        for modelname, loss, model, dataset in config:
            cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_reg_rates_decoyMNIST_cnn():

    reg_rates = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]

    for rate in reg_rates:
        config = ("DecoyMnist-CNN-RRR", ('rrr', rate), 'cnn', 'decoy_mnist')
        #config = ("DecoyMnist-CNN-RBR", ('rbr', rate), 'cnn', 'decoy_mnist')
        #config = ("DecoyMnist-CNN-RRRGradCAM", ('rrr_grad_cam', rate), 'cnn', 'decoy_mnist')
        #config = ("DecoyMnist-CNN-CDEP", ('cdep', rate), 'cnn', 'decoy_mnist')
        #config = ("DecoyMnist-CNN-HINT", ('hint', rate, True, False), 'cnn', 'decoy_mnist_hint')
        config = [config]
        for modelname, loss, model, dataset in config:
            cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset)

def cross_validate_mlp_decoyMNIST():
    #no_decoy_config = ("Mnist-MLP-CEL", 'cross_entropy', 'mlp', 'mnist')
    #cel_config = ("DecoyMnist-MLP-CEL", 'cross_entropy', 'mlp', 'decoy_mnist')
    #rrr_config = ("DecoyMnist-MLP-RRR", ('rrr', 10), 'mlp', 'decoy_mnist')
    #dep_config = ("DecoyMnist-MLP-CDEP", ('cdep', 1000), 'mlp', 'decoy_mnist')
    #rbr_config = ("DecoyMnist-MLP-RBR", ('rbr', 1000000), 'mlp', 'decoy_mnist')
    ce_config = ("DecoyMnist-MLP-CE", 'cross_entropy', 'mlp', ('decoy_mnist', 60000, 1, 'random'))
    # run 5 train and validation runs with different seeds for all XIL Losses
    #configs = [no_decoy_config, cel_config, rrr_config, cdep_config, rbr_config, ce_config]
    configs = [ce_config]

def cross_validate_cnn_feedback_decoyMNIST():
    rrr_config = ("DecoyMnist-CNN-RRR-FQ", ('rrr', 100), 'cnn', 'decoy_mnist')
    hint = ("DecoyMnist-CNN-HINT-FQ", ('hint', 1000, True, False), 'cnn', 'decoy_mnist_hint')
    rrr_grad_cam = ("DecoyMnist-CNN-RRRGradCAM-FQ", ('rrr_grad_cam', 1), 'cnn', 'decoy_mnist')
    #rbr_config = ("DecoyMnist-MLP-RBR-FQ", ('rbr', 1000000), 'mlp', 'decoy_mnist')
    #ce_config = ("DecoyMnist-MLP-CE-FQ", 'cross_entropy', 'mlp', ('decoy_mnist', 60000, 1, 'random'))
    # run 5 train and validation runs with different seeds
    #configs = [no_decoy_config, cel_config, rrr_config, cdep_config, rbr_config, ce_config]
    configs = [rrr_config, hint, rrr_grad_cam]
    
    for modelname, loss, model, dataset in configs:
        cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset, feedback='random')
        #cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset, feedback='wrong')
        #cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset, feedback='incomplete')
        cross_validate_helper(N_RUNS, SEEDS, modelname, loss, model, dataset, feedback='adversarial')

#util.empty_log_run_model_store_folders()
#cross_validate_cnn_decoyMNIST_switch_on()
#cross_validate_cnn_decoyMNIST_switch_on()
#cross_validate_ce_switch_on(N_RUNS, SEEDS, 'DecoyFMnist-CNN-CE', 'cross_entropy', 'cnn', 'decoy_fmnist')
cross_validate_cnn_decoyFashionMNIST()
c#ross_validate_mlp_feedback_decoyMNIST()