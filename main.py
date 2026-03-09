import numpy as np
import pandas as pd
import torch
import scanpy as sc
import os
from SIGHT import SIGHT
from sklearn.metrics import *
from utils import clustering

def compute_ARI(adata, gt_key, pred_key):
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_NMI(adata, gt_key, pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_COM(adata, gt_key, pred_key):
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])

def do_GraphST(datafile, cluster_num, alpha, gamma):
    adata = sc.read_h5ad(datafile)

    # define model
    model = SIGHT(adata, device=device, alpha=alpha, gamma=gamma)

    adata = model.train()
    #
    # # set radius to specify the number of neighbors considered during refinement
    radius = 50

    tool = 'mclust'  # mclust, leiden, and louvain

    # clustering
    if tool == 'mclust':
        clustering(adata, cluster_num, radius=radius, method=tool,
                   refinement=True)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, cluster_num, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

    return adata


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training model device on ..., ', device)

    datadir = '.../data/DLPFC/'  # dataset path
    filename_list = ['151507', '151508', '151509', '151510', '151669',
                     '151670', '151671', '151672', '151673', '151674',
                     '151675', '151676']
    clusters_num_list = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

    alpha_list = [9.0,3.0,7.0,6.0,9.0,9.0,7.0,4.0,4.0,5.0,9.0,6.0]
    gamma_list = [6.0,5.0,6.0,5.0,10.0,9.0,6.0,9.0,5.0,1.0,0.0,0.0]

    out_path = "results/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    run_all_re = []
    for fname, cluster_num, alpha, gamma in zip(filename_list, clusters_num_list, alpha_list, gamma_list):
        datafile = f'{datadir}/{fname}.h5ad'
        adata = do_GraphST(f'{datadir}/{fname}.h5ad', cluster_num, alpha, gamma)
        # 保存结果
        # adata.obs['pred_{}'.format(i + 1)] = adata.obs['domain']
        adata.write(f'{out_path}/{fname}.h5ad')
        # 保存结果
        adata.obs['pred'] = adata.obs['domain']
        adata = adata[
            np.logical_not(adata.obs['ground_truth'].isna())]  # remove NAN
        # compute ari
        ARI = compute_ARI(adata, 'ground_truth', f'pred')
        # compute nmi
        NMI = compute_NMI(adata, 'ground_truth', f'pred')
        # compute COM
        COM = compute_COM(adata, f'ground_truth', f'pred')

        print(fname, ARI, NMI,COM)


