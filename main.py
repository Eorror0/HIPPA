from __future__ import print_function
import os
import argparse
import sys
from timeit import default_timer as timer
import numpy as np
import pandas as pd

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

import torch

def main(args):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)

    for i in folds:
        start_time = timer()
        seed_torch(args.seed)
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(results_pkl_path):
            print("Skipping Split %d" % i)
            continue

        train_dataset, val_dataset = dataset.return_splits(from_id=False,
                    csv_path='{}/splits_{}.csv'.format(args.split_dir, i))

        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        if args.task_type == 'survival':
            val_latest, cindex_latest = train(datasets, i, args)
            latest_val_cindex.append(cindex_latest)

        save_pkl(results_pkl_path, val_latest)
        end_time = timer()
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

    if args.task_type == 'survival':
        results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
parser.add_argument('--cancer_type', type=str, default='gbmlgg', help='cancer type')
parser.add_argument('--data_root_dir', type=str, default=' ', help='data directory')
parser.add_argument('--mode', type=str, default='', help='Specifies which modality to use')
parser.add_argument('--model_type', type=str, default='lgi_super', help='Type of model')
parser.add_argument('--beta', type=float, default=1, help='Parameter Code')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--k', type=int, default=5, help='Number of folds')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold')
parser.add_argument('--k_end', type=int, default=-1, help='End fold')
parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
parser.add_argument('--which_splits', type=str, default='5foldnew', help='Which splits folder')
parser.add_argument('--split_dir', type=str, default='tcga_{}'.format('gbmlgg'), help='Which cancer type within splits')
parser.add_argument('--log_data', action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite', action='store_true', default=False, help='Whether to overwrite')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--num_gcn_layers', type=int, default=4, help='# of GCN layers')
parser.add_argument('--edge_agg', type=str, default='spatial', help="Edge relationship for aggregation")
parser.add_argument('--resample', type=float, default=0.00, help='Dropping out random patches')
parser.add_argument('--drop_out', action='store_true', default=True, help='Enable dropout')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--gc', type=int, default=10, help='Gradient Accumulation Step')
parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of epochs')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='loss function')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels')
parser.add_argument('--bag_weight', type=float, default=0.7, help='weight coefficient for bag-level loss')
parser.add_argument('--reg', type=float, default=1e-5, help='L2-regularization weight decay')
parser.add_argument('--alpha_surv', type=float, default=0.05, help='Weight for uncensored patients')
parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'], default='None', help='L1-Regularization submodules')
parser.add_argument('--lambda_reg', type=float, default=1e-4, help='L1-Regularization Strength')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--patience', type=int, default=30, help='patience')
parser.add_argument('--stop_epoch', type=int, default=100, help='stop_epoch')
parser.add_argument('--scheduler_factor', default=1, type=float, help='scheduler_factor')
parser.add_argument('--scheduler_patience', default=30, type=int, help='scheduler_patience')
parser.add_argument('--scheduler_threshold', default=0, type=float, help='scheduler_threshold')
parser.add_argument('--dataset_name', type=str, default='tcga')
parser.add_argument('--num_rw_steps', type=int, default=None)
parser.add_argument('--dim_pe', type=int, default=None)
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--out_dim', type=int, default=4)
parser.add_argument('--node_num_types', type=int, default=None)
parser.add_argument('--edge_num_types', type=int, default=None)
parser.add_argument('--scheduler', type=str, default='none', choices=['linear', 'cosine', 'none'])
parser.add_argument('--warmup', type=int, default=40)
parser.add_argument('--gconv_dim', type=int, default=256)
parser.add_argument('--tlayer_dim', type=int, default=256)
parser.add_argument('--gconv_attn_dropout', type=float, default=0.25)
parser.add_argument('--gconv_ffn_dropout', type=float, default=0.25)
parser.add_argument('--tlayer_attn_dropout', type=float, default=0.25)
parser.add_argument('--tlayer_ffn_dropout', type=float, default=0.25)
parser.add_argument('--tlayer_ffn_hidden_times', type=int, default=1)
parser.add_argument('--gconv_type', type=str, default='gcn', choices=['gin', 'gcn', 'gen', 'eela'])
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--middle_layer_type', type=str, default='mlp', choices=['none', 'mlp', 'residual'])
parser.add_argument('--skip_connection', type=str, default='short', choices=['none', 'long', 'short'])
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add', 'cls'])
parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn'])
parser.add_argument('--out_layer', type=int, default=3)
parser.add_argument('--out_hidden_times', type=int, default=4)
parser.add_argument('--save_state', action='store_true')
parser.add_argument('--seeds', type=int, default=0)
parser.add_argument('--need_pos', default=True, help='need_pos')
parser.add_argument('--need_edge_attr', default=True, help='need_edge_attr')
parser.add_argument('--pos_type', default='sine', help='pos_type')
parser.add_argument('--pos_weight', default=0.0001, help='pos_weight')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

from datetime import datetime
settings = {'num_splits': args.k, 'k_start': args.k_start, 'k_end': args.k_end, 'task': args.task,
            'max_epochs': args.max_epochs, 'results_dir': args.results_dir, 'lr': args.lr,
            'experiment': args.exp_code, 'reg': args.reg, 'label_frac': args.label_frac,
            'bag_loss': args.bag_loss, 'bag_weight': args.bag_weight, 'seed': args.seed,
            'model_type': args.model_type, 'weighted_sample': args.weighted_sample, 'gc': args.gc,
            'opt': args.opt, 'split_dir': args.split_dir, 'data_root_dir': args.data_root_dir,
            'cancer_type': args.cancer_type, 'gconv_type': args.gconv_type, 'need_pos': args.need_pos,
            'need_edge_attr': args.need_edge_attr, 'pos_type': args.pos_type, 'patience': args.patience,
            'stop_epoch': args.stop_epoch, 'scheduler_patience': args.scheduler_patience,
            'scheduler_factor': args.scheduler_factor, 'scheduler_threshold': args.scheduler_threshold,
            'early_stopping': args.early_stopping, 'alpha_surv': args.alpha_surv,
            'gconv_attn_dropout': args.gconv_attn_dropout, 'gconv_ffn_dropout': args.tlayer_ffn_dropout,
            'tlayer_attn_dropout': args.tlayer_attn_dropout, 'tlayer_ffn_dropout': args.tlayer_ffn_dropout,
            'mode': args.mode, 'date_time': str(datetime.now()), 'tlayer_dim': args.tlayer_dim,
            'gconv_dim': args.gconv_dim}

if 'survival' in args.task:
    args.n_classes = 4
    dataset = Generic_MIL_Survival_Dataset(
        csv_path='dataset_csv/tcga_{}_all_clean.csv'.format(args.cancer_type),
        mode=args.mode, cancer_type=args.cancer_type, data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True, patient_strat=False, n_bins=4,
        label_col='survival_months', ignore=[])
else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    sys.exit()

args.split_dir = os.path.join('splits', args.which_splits, args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
