import torch
from .preprocess import *

import numpy as np
from .model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

    
class SIGHT():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=600, 
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        gamma=1.0
        ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)

        if 'adj' not in adata.obsm.keys():
            construct_interaction(self.adata)

        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)

        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj'] # (spot, spot) 对称矩阵
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        self.coord = torch.FloatTensor(self.adata.obsm['spatial'].copy()).to(self.device)

        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)

    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Begin to train SRT data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()

            self.features_a = permutation(self.features)

            (self.hiden_feat, self.emb,
             mid_fea, node_feats_recon,
             ret, ret_a,
             g, g_a,
             emb, emb_a) = self.model(self.features, self.features_a, self.adj, self.coord)

            self.loss_feat = F.mse_loss(self.features, self.emb)
            self.loss_feat_1 = F.mse_loss(self.features, node_feats_recon)

            loss_graph_diff = F.cosine_similarity(g, g_a, dim=1).mean()
            loss_positive_negative_diff = F.cosine_similarity(emb, emb_a, dim=1).mean()

            loss = (self.alpha * (self.loss_feat + self.loss_feat_1)+
                    self.gamma * (loss_graph_diff + loss_positive_negative_diff))

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        
        print("Optimization finished for SRT data!")
        with torch.no_grad():
            self.model.eval()
            h = self.model(self.features, self.features_a, self.adj, self.coord)[1]
            node_feats_recon = self.model(self.features, self.features_a, self.adj, self.coord)[3]
            average_recon = (h + node_feats_recon) / 2
            self.emb_rec = average_recon.detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec

            return self.adata

