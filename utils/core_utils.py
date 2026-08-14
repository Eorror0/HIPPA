from argparse import Namespace
from collections import OrderedDict
import os
import pickle

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch_geometric

from datasets.dataset_generic import save_splits
from models.gtnet import LGI_SUPER_Surv
from utils.utils import *

import time

class Monitor_CIndex:

    def __init__(self):
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):
        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        torch.save(model.state_dict(), ckpt_name)

def train(datasets: tuple, cur: int, args: Namespace):

    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    reg_fn = None

    print('\nInit Model...', end=' ')
    if args.model_type == 'lgi_super':
        model = LGI_SUPER_Surv(
                gconv_dim=args.gconv_dim,
                tlayer_dim=args.tlayer_dim,
                dataset_name=args.dataset_name,
                beta_value=args.beta,
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                gconv_ffn_dropout=args.gconv_ffn_dropout,
                gconv_type=args.gconv_type,
                num_layers=args.num_layers,
                middle_layer_type=args.middle_layer_type,
                readout=args.readout,
                )
    else:
        raise NotImplementedError

    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    params_list = [
        {'params': model.predict_head.parameters(), 'lr': args.lr, 'weight_decay': args.reg}
    ]
    other_params = [param for name, param in model.named_parameters() if "predict_head" not in name]
    params_list.append({'params': other_params, 'lr': args.lr})
    optimizer = optim.Adam(params_list)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing,
                                    weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = Early_Stopping(warmup=0, patience=args.patience, stop_epoch=args.stop_epoch, verbose = True)
    else:
        early_stopping = None
    print('Done!')

    scheduler = None

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            stop = train_val_loop_survival(args.exp_code,args.model_type,args.cancer_type,cur, epoch, model, train_loader, val_loader, optimizer, scheduler, args.n_classes, early_stopping, writer, loss_fn, reg_fn, lambda_reg=args.lambda_reg, mygc=args.gc, results_dir=args.results_dir, scheduler_factor=args.scheduler_factor, scheduler_threshold=args.scheduler_threshold, scheduler_patience=args.scheduler_patience )
            if stop:
                break

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    writer.close()
    return results_val_dict, val_cindex

def train_val_loop_survival(exp_code,model_type,cancer_type,cur, epoch, model, train_loader, val_loader, optimizer, scheduler, n_classes,
                            early_stopping=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., mygc=16, results_dir=None,
                            scheduler_factor=0, scheduler_threshold=0, scheduler_patience=0):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_start_time = time.time()
    model.train()
    train_loss_surv, train_loss = 0., 0.

    all_risk_scores = np.zeros((len(train_loader)))
    all_censorships = np.zeros((len(train_loader)))
    all_event_times = np.zeros((len(train_loader)))

    for batch_idx, (data_WSI, label, event_time, c, slide_ids) in enumerate(train_loader):
        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)
        if data_WSI.x.shape[0] != data_WSI.centroid.shape[0]:
            continue

        hazards, S, Y_hat, _, _, _, _ = model(x_path=data_WSI)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))

        loss = loss / mygc + loss_reg
        loss.backward()

        if (batch_idx + 1) % mygc == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    train_loss_surv /= len(train_loader)
    train_loss /= len(train_loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))
    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

    val_start_time = time.time()
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(val_loader)))
    all_label = np.zeros((len(val_loader)))
    all_censorships = np.zeros((len(val_loader)))
    all_event_times = np.zeros((len(val_loader)))
    all_hazards = []

    for batch_idx, (data_WSI, label, event_time, c, slide_ids) in enumerate(val_loader):
        if isinstance(data_WSI, torch_geometric.data.Batch):
            if data_WSI.x.shape[0] > 100_000:
                continue

        data_WSI = data_WSI.to(device)
        label = label.to(device)
        c = c.to(device)

        hazards, S, Y_hat, _, _, _, _ = model(x_path=data_WSI)

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        all_hazards.append(hazards)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time
        all_label[batch_idx] = label

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_end_time = time.time()
    val_duration = val_end_time - val_start_time
    val_loss_surv /= len(val_loader)
    val_loss /= len(val_loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss_surv, val_loss, c_index))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

class Early_Stopping:

    def __init__(self, warmup=50, patience=15, stop_epoch=20, verbose=False):
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score_c_index = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, c_index, model, ckpt_name = 'checkpoint.pt'):
        score = -val_loss
        score_c_index = c_index

        if epoch < self.warmup:
            pass
        elif self.best_score is None or self.best_score_c_index is None:
            self.best_score = score
            self.best_score_c_index = score_c_index
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score and score_c_index < self.best_score_c_index:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch < self.stop_epoch:
                self.early_stop = True
        elif score >= self.best_score or c_index >= self.best_score_c_index:
            self.best_score = score
            self.best_score_c_index = c_index
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    patient_results = {}

    for batch_idx, (data_WSI, label, event_time, c, slide_ids) in enumerate(loader):
        if isinstance(data_WSI, torch_geometric.data.Batch):
            pass

        data_WSI = data_WSI.to(device)
        label = label.to(device)

        slide_id = slide_ids[0][0]

        with torch.no_grad():
            hazards, survival, Y_hat, _, _, _, _ = model(x_path=data_WSI)

        risk = np.ndarray.item(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.ndarray.item(event_time.numpy())
        c = np.ndarray.item(c.numpy())
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return patient_results, c_index
