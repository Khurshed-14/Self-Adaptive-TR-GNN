import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_target_incoming_mask(A: torch.Tensor, target_idx: int) -> torch.Tensor:
    """
    Restrict A so only self-loops and covariate -> target edges remain.

    With agg_i = sum_j A[i,j] h_j, row ``target_idx`` is what drives the target node.
    Column ``target_idx`` (demand -> others) is zeroed off-diagonal.
    """
    N = A.size(-1)
    eye = torch.eye(N, device=A.device, dtype=A.dtype)
    out = A * eye
    out[target_idx, :] = A[target_idx, :]
    return out


class TemporalGraphLearning(nn.Module):
    """
    Implementation of Graph Learning Module precisely matching Section 2.2 
    of the GLFN-TC paper.
    """
    def __init__(self, d_model, N):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        # Eq (4): v_i \in R^d - Feature vector for each factor 
        self.node_embed = nn.Parameter(torch.randn(N, d_model))
        
        # Eq (5): W \in R^{d \times d} - Trainable weight matrix
        self.W = nn.Parameter(torch.randn(d_model, d_model))
        nn.init.xavier_uniform_(self.W)

    def forward(self, H=None):
        # Eq (5) inner component: v_i^T W v_j for all nodes i, j
        # node_embed is (N, d), W is (d, d), node_embed.T is (d, N) -> Result: (N, N)
        scores = torch.matmul(torch.matmul(self.node_embed, self.W), self.node_embed.transpose(0, 1))
        
        # Eq (5): \sigma function to control the correlation strictly in the range [0, 1]
        A = torch.sigmoid(scores)
        
        # Eq (5) conditions: A_{i,j} = 1 if i == j
        identity = torch.eye(self.N, device=A.device)
        A = A * (1.0 - identity) + identity
        
        # Note: Row normalization (\tilde{D}^{-1} A) is done in the GCN step per Eq (10).
        return A

class TemporalConv(nn.Module):
    def __init__(self, N, T_in, hidden_dim=64, kernel_size=7, dilation=3, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dilation = dilation
        
        # Calculate padding to maintain temporal length (Causal Convolution)
        padding = (kernel_size - 1) * dilation
        
        self.convs = nn.ModuleList([
            nn.Conv1d(1, hidden_dim, kernel_size, dilation=dilation, padding=padding)
            for _ in range(N)
        ])
        
    def forward(self, x):  # x: (batch, N, T_in)
        outs = []
        for i, conv in enumerate(self.convs):
            h = conv(x[:, i:i+1, :])          # (B, hidden_dim, T_padded)
            h = h[..., :x.size(2)]            # Trim to original length (causal)
            h = F.relu(h)
            h = self.dropout(h)
            h = h[..., -1]                    # Temporal pooling → (B, hidden_dim)
            outs.append(h)
        return torch.stack(outs, dim=1)       # (B, N, hidden_dim)
    
class DenselyResidualGCN(nn.Module):
    """
    Implementation of Densely Connected Residual Convolution Module matching Section 2.4.
    """
    def __init__(self, in_dim, hidden_dim, layers=5, dropout=0.3):
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        
        # Project initial concatenated input g_i to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        ])

    def forward(self, g, A): 
        # g: (B, N, h+d) [Concatenated representation from Eq (9)]
        H_0 = self.input_proj(g) 
        
        # Eq (10): \tilde{A} = \tilde{D}^{-1} A 
        row_sums = A.sum(dim=-1, keepdim=True) + 1e-6
        A_tilde = A / row_sums
        
        # Expand for batched multiplication: (1, N, N)
        A_tilde = A_tilde.unsqueeze(0)
        
        H_list = [H_0]
        Out_H = H_0
        
        for l in range(self.layers):
            # Eq (10): \sum_{i=1}^{l-1} H^{(i)} (Summation of all previous layers)
            H_sum = sum(H_list)
            
            # \tilde{A} @ (\sum H) -> Graph convolution over the dense sum
            agg = torch.matmul(A_tilde, H_sum) 
            
            H_l = F.relu(self.gcn_layers[l](agg))
            H_l = self.dropout(H_l)
            
            # Eq (11): t_H^{(l)} = \sigma(Out_H^{(l-1)} + H^{(l)}) (Residual accumulation)
            Out_H = F.relu(Out_H + H_l)
            
            H_list.append(H_l)
            
        return Out_H
    
class LoadForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.3):
        super().__init__()
        self.trend_linear = nn.Linear(T_in, T_out)
        self.flatten_dim = N * d_model
        
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(dropout),
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        trend = self.trend_linear(x_raw[:, 0, :]) 
        residual = self.net(h_gcn)
        return trend + residual

class MultiScaleForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.2):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.trend_linear = nn.Linear(T_in, T_out)
        self.flatten_dim = N * d_model
        
        self.residual_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, T_out) 
        )

    def forward(self, x_raw, h_gcn):
        x_target = x_raw[:, 0, :] 
        trend = self.trend_linear(x_target) 
        residual = self.residual_mlp(h_gcn) 
        return trend + residual
    
class AttentionForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.3):
        super().__init__()
        self.trend_linear = nn.Linear(T_in, T_out)
        self.query = nn.Linear(d_model, 1) 
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        trend = self.trend_linear(x_raw[:, 0, :])
        scores = self.query(h_gcn)              
        attn_weights = F.softmax(scores, dim=1) 
        context = torch.sum(attn_weights * h_gcn, dim=1) 
        residual = self.proj(context)
        return trend + residual

class GlobalLocalForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.2):
        super().__init__()
        self.trend_linear = nn.Linear(T_in, T_out)
        self.global_fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, T_out)
        )
        self.local_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        x_target = x_raw[:, 0, :] 
        trend = self.trend_linear(x_target) 
        
        g_feat = torch.mean(h_gcn, dim=1)   
        global_pred = self.global_fc(g_feat) 
        
        l_feat = h_gcn - g_feat.unsqueeze(1) 
        local_out = self.local_fc(l_feat)   
        local_pred = torch.mean(local_out, dim=1) 

        return trend + global_pred + local_pred


# ==============================================================================
# WRAPPERS (Eq 9 Logic updated here for integration)
# ==============================================================================

class TR_GNN_Linear(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64, 
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, 
                 GCN_Layer=5, dilation=3, kernel_size=7, graph_temperature=3.0):
        super().__init__()
        # Use d_model = d as defined in the paper
        self.graph_learn = TemporalGraphLearning(d_model=d, N=N)
        
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim, 
                                          dilation=dilation, dropout=dropout_temporal, kernel_size=kernel_size)
        
        # Eq (9) states input to GCN is g_i = o_i \oplus v_i, therefore in_dim = hidden_dim + d
        self.dense_gcn = DenselyResidualGCN(in_dim=hidden_dim + d, hidden_dim=hidden_dim, 
                                            dropout=dropout_gcn, layers=GCN_Layer)
        
        self.forecaster = LoadForecasting(
            N=N, T_in=T_in, T_out=T_out, 
            d_model=hidden_dim, 
            dropout=dropout_forecast
        )

    def forward(self, X): 
        H = self.temporal_conv(X)               # Temporal features: o_i (B, N, h)
        A = self.graph_learn()                  # Adjacency matrix: (N, N)
        
        # Eq (9): g_i = o_i \oplus v_i
        V = self.graph_learn.node_embed         # Node embeddings: v_i (N, d)
        V_expanded = V.unsqueeze(0).expand(H.size(0), -1, -1) # Broadcast to batch: (B, N, d)
        g = torch.cat([H, V_expanded], dim=-1)  # Concatenate to form g: (B, N, h+d)
        
        H_out = self.dense_gcn(g, A)            # Pass concatenated input and adjacency matrix
        Y_hat = self.forecaster(X, H_out)
        return Y_hat, A

class TR_GNN_Attention(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0):          
        super().__init__()
        self.graph_learn = TemporalGraphLearning(d_model=d, N=N)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size, dilation=dilation, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(in_dim=hidden_dim + d, hidden_dim=hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = AttentionForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn()
        
        V = self.graph_learn.node_embed.unsqueeze(0).expand(H.size(0), -1, -1)
        g = torch.cat([H, V], dim=-1)
        
        H_out = self.dense_gcn(g, A)
        Y_hat = self.forecaster(X, H_out)
        return Y_hat, A

class TR_GNN_MultiScale(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0,
                 target_idx=None, graph_structure='full'):          
        super().__init__()
        if graph_structure not in ('full', 'target_in'):
            raise ValueError("graph_structure must be 'full' or 'target_in'")
        if graph_structure == 'target_in' and target_idx is None:
            raise ValueError("target_idx is required when graph_structure='target_in'")
        self.target_idx = target_idx
        self.graph_structure = graph_structure
        self.graph_learn = TemporalGraphLearning(d_model=d, N=N)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size, dilation=dilation, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(in_dim=hidden_dim + d, hidden_dim=hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = MultiScaleForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn()
        if self.graph_structure == 'target_in':
            A = apply_target_incoming_mask(A, self.target_idx)
        
        V = self.graph_learn.node_embed.unsqueeze(0).expand(H.size(0), -1, -1)
        g = torch.cat([H, V], dim=-1)
        
        H_out = self.dense_gcn(g, A)
        Y_hat = self.forecaster(X, H_out)
        return Y_hat, A

class TR_GNN_GlobalLocal(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0):          
        super().__init__()
        self.graph_learn = TemporalGraphLearning(d_model=d, N=N)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size, dilation=dilation, dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(in_dim=hidden_dim + d, hidden_dim=hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = GlobalLocalForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn()
        
        V = self.graph_learn.node_embed.unsqueeze(0).expand(H.size(0), -1, -1)
        g = torch.cat([H, V], dim=-1)
        
        H_out = self.dense_gcn(g, A)
        Y_hat = self.forecaster(X, H_out)
        return Y_hat, A