import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mp.models import SparseCIN
from mp.graph_models import GIN0
from torch import digamma


class TMDCProjectionGroups(nn.Module):
    def __init__(self, input_dim, output_dim, num_views):
        super().__init__()
        self.num_views = num_views  # e.g., 4 = [GIN, CWN-6, CWN-9, CWN-12]

        # Common projection group: one projection head per view (shared contrastive space)
        self.common_proj = nn.ModuleDict({
            f'view{i}': self._make_proj(input_dim, output_dim) for i in range(num_views)
        })

        # 3 decoupled contrastive spaces (e.g., excluding CWN-6, CWN-9, CWN-12)
        self.decoupled_proj_groups = nn.ModuleList([
            nn.ModuleDict({
                f'view{i}': self._make_proj(input_dim, output_dim) for i in range(num_views)
            }) for _ in range(num_views - 1)  # 3 decoupled spaces
        ])

    def _make_proj(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward_common(self, features_dict):
        """
        features_dict: { 'view0': tensor[B,D], 'view1': ..., 'view2': ..., 'view3': ... }
        Return: projected features in common space
        """
        return {
            name: self.common_proj[name](features_dict[name])
            for name in features_dict
        }

    def forward_decoupled(self, k, features_dict):
        """
        k: which view is assumed as negative (0-based index: 0→CWN-6, 1→CWN-9, 2→CWN-12)
        features_dict: same as above
        Return: dict of projected features in space k
        """
        return {
            name: self.decoupled_proj_groups[k][name](features_dict[name])
            for name in features_dict
        }



class SimCLRWithTMDC(nn.Module):
    def __init__(self, dataset, args):
        super(SimCLRWithTMDC, self).__init__()
        use_coboundaries = args.use_coboundaries.lower() == 'true'
        readout_dims = tuple(sorted(args.readout_dims))
        num_features = dataset.graph_list[0].num_features

        self.encoder = SparseCIN(
            num_features,
            dataset.num_classes,
            args.num_layers,
            args.emb_dim,
            dropout_rate=args.drop_rate,
            max_dim=dataset.max_dim,
            jump_mode=args.jump_mode,
            nonlinearity=args.nonlinearity,
            readout=args.readout,
            final_readout=args.final_readout,
            apply_dropout_before=args.drop_position,
            use_coboundaries=use_coboundaries,
            graph_norm=args.graph_norm,
            readout_dims=readout_dims
        )

        self.gin_encoder = GIN0(
            num_features,
            args.num_layers,
            args.emb_dim,
            dropout_rate=args.drop_rate,
            nonlinearity=args.nonlinearity,
            readout=args.readout,
        )

        self.embedding_dim = args.max_dim * args.emb_dim
        self.num_views = 4  # GIN + 3 CWN views

        self.proj_groups = TMDCProjectionGroups(
            input_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            num_views=self.num_views
        )

    def forward_encodings(self, data_batch, complex_batch_list):
        """
        Return raw features: dict with keys 'view0', 'view1', 'view2', 'view3'
        """
        h_gin = self.gin_encoder(data_batch)
        h_cwn_list = [self.encoder(cb) for cb in complex_batch_list]

        feats = {'view0': h_gin}
        for i, h_cwn in enumerate(h_cwn_list):
            feats[f'view{i+1}'] = h_cwn
        return feats

    def forward_common_projected(self, raw_feats):
        """
        Return projected features: dict with keys 'view0', 'view1', 'view2', 'view3'(Common Space)
        """
        return self.proj_groups.forward_common(raw_feats)

    def forward_decoupled_projected(self, space_id, raw_feats):
        """
        Return projected features in space k
        """
        return self.proj_groups.forward_decoupled(space_id, raw_feats)

    def simclr_loss_full_batch(self, anchor, positive, all_views, temperature=0.2):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        all_views = [F.normalize(v, dim=1) for v in all_views]

        B = anchor.size(0)
        pos_sim = torch.sum(anchor * positive, dim=1)
        all_negatives = torch.cat(all_views, dim=0)
        neg_sim = torch.matmul(anchor, all_negatives.T)

        for i in range(B):
            for v in range(len(all_views)):
                idx = i + v * B
                neg_sim[i, idx] = -1e9

        pos_exp = torch.exp(pos_sim / temperature)
        neg_exp = torch.exp(neg_sim / temperature).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        return loss

    # 修改之后的版本。
    def simclr_loss_full_batch_exclude(self, anchor, positive, all_views, temperature=0.2, exclude_view=None):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        all_views = [F.normalize(v, dim=1) for v in all_views]

        B = anchor.size(0)
        pos_sim = torch.sum(anchor * positive, dim=1)
        all_negatives = torch.cat(all_views, dim=0)
        neg_sim = torch.matmul(anchor, all_negatives.T)

        for i in range(B):
            for v in range(len(all_views)):
                if v != exclude_view:
                    idx = i + v * B
                    neg_sim[i, idx] = -1e9

        pos_exp = torch.exp(pos_sim / temperature)
        neg_exp = torch.exp(neg_sim / temperature).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
        return loss

    def student_t_distribution(self, z, centroids, mu=1.0):
        dist = torch.cdist(z, centroids, p=2).pow(2)
        num = (1 + dist / mu).pow(-(mu + 1) / 2)
        return num / num.sum(dim=1, keepdim=True)

    def compute_trustworthiness(self, probs):
        evidence = probs
        alpha = evidence + 1
        S = alpha.sum(dim=1, keepdim=True)
        trust = (digamma(alpha) - digamma(S)).sum(dim=1) / alpha.size(1)
        return torch.sigmoid(trust).mean()

    def compute_centroids_for_decoupled(self, features_dict, exclude_index):
        valid_keys = [f'view{i}' for i in range(4) if i != exclude_index]
        valid_feats = [features_dict[k] for k in valid_keys]
        centroids = torch.stack([f.mean(dim=0) for f in valid_feats])
        return centroids

    def compute_common_loss(self, raw_feats):
        feats = self.forward_common_projected(raw_feats)
        views = [feats[f'view{i}'] for i in range(self.num_views)]
        total_loss = 0.0
        count = 0
        for i in range(self.num_views):
            for j in range(self.num_views):
                if i != j:
                    anchor = views[i]
                    positive = views[j]
                    negatives = [v for k, v in enumerate(views) if k != i and k != j]
                    total_loss += self.simclr_loss_full_batch(anchor, positive, negatives)
                    count += 1
        return total_loss / count if count > 0 else 0.0

    def compute_tmdc_loss(self, raw_feats):
        total_loss = 0.0
        for k in range(3):
            proj_feats = self.forward_decoupled_projected(k, raw_feats)
            z_anchor = proj_feats['view0']
            z_pos_list = [proj_feats[f'view{i}'] for i in range(1, 4) if i != k + 1]
            z_neg = proj_feats[f'view{k + 1}']

            all_negatives = z_pos_list + [z_neg]

            loss_pos = 0.0
            for z_pos in z_pos_list:
                loss_pos += self.simclr_loss_full_batch(z_anchor, z_pos, all_negatives)
            loss_k = loss_pos / len(z_pos_list)

            centroids = self.compute_centroids_for_decoupled(proj_feats, exclude_index=k + 1)
            probs = self.student_t_distribution(z_neg, centroids)
            trust_k = self.compute_trustworthiness(probs)

            total_loss += trust_k * loss_k
        return total_loss

    def compute_total_loss(self, raw_feats, alpha=0.5, beta=1.0):
        l_common = self.compute_common_loss(raw_feats)
        l_decoupled = self.compute_tmdc_loss(raw_feats)
        return alpha * l_common + beta * l_decoupled

    def get_embeddings(self, dataloader, device, fusion='concat'):
        self.eval()
        all_z = []
        all_y = []
        for data_batch, complex_batch_list in dataloader:
            data_batch = data_batch.to(device)
            complex_batch_list = [cb.to(device) for cb in complex_batch_list]
            with torch.no_grad():
                feats = self.forward_encodings(data_batch, complex_batch_list)
                z_list = [feats[f'view{i}'] for i in range(self.num_views)]
                if fusion == 'mean':
                    z = torch.stack(z_list).mean(dim=0)
                elif fusion == 'concat':
                    z = torch.cat(z_list, dim=1)
                else:
                    raise ValueError(f'Unsupported fusion method: {fusion}')
            all_z.append(z.cpu().numpy())
            all_y.append(data_batch.y.cpu().numpy())
        z_all = np.concatenate(all_z, 0)
        y_all = np.concatenate(all_y, 0)
        return z_all, y_all