import torch
import torch.nn as nn
import torch as th


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()
        
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters
        
        # Initialize dummy parameters to match state_dict
        init_sc = (1 / torch.sqrt(torch.tensor(feature_size, dtype=torch.float32)))
        clusters = cluster_size + ghost_clusters
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        
        # Load the correct state dict will overwrite these
        self.batch_norm.running_mean.zero_()
        self.batch_norm.running_var.fill_(1.0)
        self.batch_norm.weight.data.fill_(1.0)
        self.batch_norm.bias.data.fill_(0.0)

    def forward(self, x, mask=None):
        """Forward pass with numerical stability"""
        
        # Ensure input is float32 for numerical stability
        x = x.float()
        
        max_sample = x.size()[1]
        x_flat = x.view(-1, self.feature_size)

        # Use float32 for all computations
        clusters_fp32 = self.clusters.float()
        assignment = x_flat @ clusters_fp32
        
        # Exact batch norm calculation in float32
        running_mean = self.batch_norm.running_mean.float()
        running_var = self.batch_norm.running_var.float()
        weight = self.batch_norm.weight.float()
        bias = self.batch_norm.bias.float()
        eps = self.batch_norm.eps
        
        inv_std = torch.rsqrt(running_var + eps)
        assignment = weight * inv_std * (assignment - running_mean) + bias

        # Numerically stable softmax
        assignment = assignment.softmax(dim=1)
        assignment = assignment[:, :self.cluster_size].reshape(-1, max_sample, self.cluster_size)

        clusters2_fp32 = self.clusters2.float()
        a_sum = assignment.sum(dim=1, keepdim=True)
        a = a_sum * clusters2_fp32

        assignment_transposed = assignment.transpose(1, 2)
        x_reshaped = x_flat.reshape(-1, max_sample, self.feature_size)

        vlad = assignment_transposed @ x_reshaped
        vlad = vlad.transpose(1, 2) - a

        # L2 norm calculations with epsilon for stability
        norm_eps = 1e-12
        
        # Intra normalization
        vlad_norm = vlad.norm(dim=-1, keepdim=True).clamp(min=norm_eps)
        vlad = vlad / vlad_norm
        
        # Flatten and final L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad_final_norm = vlad.norm(dim=-1, keepdim=True).clamp(min=norm_eps)
        vlad = vlad / vlad_final_norm

        return vlad