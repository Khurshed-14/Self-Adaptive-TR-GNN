import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGraphLearning(nn.Module):
    def __init__(self, d_model, dropout=0.1, alpha=0.2, temperature=3.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature  # Moderate temperature: 3.0 keeps gradients alive and preserves structure
        
        # Main query/key projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        
        # Node-specific embeddings to capture per-node uniqueness
        # These ensure each node has a distinct "identity" for different connection patterns
        self.node_bias_q = nn.Parameter(torch.randn(d_model) * 0.1)
        self.node_bias_k = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Xavier initialization for main weights
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, H):
        # H: (B, N, d_model)
        Q = self.W_q(H)  # (B, N, d_model)
        K = self.W_k(H)  # (B, N, d_model)
        
        # Add node-specific biases to ensure per-node uniqueness
        Q = Q + self.node_bias_q.unsqueeze(0).unsqueeze(0)  # (B, N, d_model)
        K = K + self.node_bias_k.unsqueeze(0).unsqueeze(0)  # (B, N, d_model)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, N, N)
        
        # Sigmoid with bias to keep values bounded away from 0/1
        # A = sigmoid(T*s) * scale + bias keeps values in maintained range
        # This prevents natural collapse to zero while allowing learned variation
        A = torch.sigmoid(scores * self.temperature) * 0.6 + 0.2  # Maps to [0.2, 0.8]
        
        # Add strong diagonal for self-connections
        I = torch.eye(A.size(1), device=A.device).unsqueeze(0)  # (1, N, N)
        A = A + I * 1.0  # Diagonal: [1.2, 1.8], off-diagonal: [0.2, 0.8]
        
        A = self.dropout(A)
        return A
    
class TemporalConv(nn.Module):
    def __init__(self, N, T_in, hidden_dim=64, kernel_size=7, dilation=3, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dilation = dilation
        
        # Calculate padding to maintain temporal length
        # For causal conv: padding = (kernel_size - 1) * dilation
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
            h = h[..., -1]                                                                                                                                                                                       # Temporal pooling → (B, hidden_dim)
            outs.append(h)
        return torch.stack(outs, dim=1)       # (B, N, hidden_dim)
    
class DenselyResidualGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers=5, dropout=0.3):
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        ])
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, X, A): 
        # X: (B, N, hidden_dim)
        H = self.input_proj(X)
        
        for layer in self.gcn_layers:
            # Standard GCN Step
            agg = torch.matmul(A, H) 
            out = F.relu(layer(agg))
            out = self.dropout(out)
            
            # FIX: Standard Residual Connection (Add only previous)
            H = H + out 
            
        return H

class LoadForecasting(nn.Module):
    """
    Trend + Residual MLP. 
    Flattens all feature nodes to preserve distinct feature info (Temp != Load).
    """
    def __init__(self, N, T_in, T_out, d_model, dropout=0.3):
        super().__init__()
        
        # --- Path A: Linear Trend (Baselines) ---
        self.trend_linear = nn.Linear(T_in, T_out)
        
        # --- Path B: Residual MLP (Deep Learning) ---
        # We flatten (N * d_model) so the MLP sees "Temp" and "Load" separately
        self.flatten_dim = N * d_model
        
        self.net = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(dropout),
            nn.Linear(self.flatten_dim, 128), # Increased width slightly
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        # 1. Trend: Project raw load history
        trend = self.trend_linear(x_raw[:, 0, :]) 
        
        # 2. Residual: Predict fluctuations from GCN state
        residual = self.net(h_gcn)
        
        return trend + residual

class MultiScaleForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.2):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        
        # --- 1. Trend Component (Linear) ---
        # This takes the raw Load history and projects the baseline trend.
        # We assume index 0 is the Target Load.
        self.trend_linear = nn.Linear(T_in, T_out)

        # --- 2. Residual Component (Deep Learning) ---
        # This takes the GCN output and predicts the fluctuations.
        
        # We flatten N features * d_model to get a global context vector
        self.flatten_dim = N * d_model
        
        self.residual_mlp = nn.Sequential(
            nn.Flatten(), # (B, N, d) -> (B, N*d)
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, T_out) # Output is just the target length
        )

    def forward(self, x_raw, h_gcn):
        """
        x_raw: (B, N, T_in) - The original raw input sequences
        h_gcn: (B, N, d_model) - The output from your DenseGCN
        """
        
        # --- Path A: Trend (The Baseline) ---
        # We take only the target node (assumed index 0) or mean of relevant nodes
        # Here we assume the first feature (index 0) is the Load Demand
        x_target = x_raw[:, 0, :] # (B, T_in)
        trend = self.trend_linear(x_target) # (B, T_out)
        
        # --- Path B: Residual (The Details) ---
        # The GCN output contains the complex spatial-temporal interactions
        residual = self.residual_mlp(h_gcn) # (B, T_out)
        
        # --- Final Forecast ---
        return trend + residual
    
class AttentionForecasting(nn.Module):
    """
    Trend + Node Attention.
    Calculates which features (Nodes) matter most, then projects the result.
    """
    def __init__(self, N, T_in, T_out, d_model, dropout=0.3):
        super().__init__()
        
        # --- Path A: Linear Trend ---
        self.trend_linear = nn.Linear(T_in, T_out)
        
        # --- Path B: Attention Residual ---
        # Score each node: "How important is this feature right now?"
        self.query = nn.Linear(d_model, 1) 
        
        self.proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        # 1. Trend
        trend = self.trend_linear(x_raw[:, 0, :])
        
        # 2. Attention (Node Importance)
        # h_gcn: (B, N, d_model)
        scores = self.query(h_gcn)              # (B, N, 1)
        attn_weights = F.softmax(scores, dim=1) # (B, N, 1) - Sums to 1 across N
        
        # Weighted sum of nodes based on importance
        context = torch.sum(attn_weights * h_gcn, dim=1) # (B, d_model)
        
        # 3. Residual Projection
        residual = self.proj(context)
        
        return trend + residual

class GlobalLocalForecasting(nn.Module):
    def __init__(self, N, T_in, T_out, d_model, dropout=0.2):
        super().__init__()
        
        # --- 1. Trend Component (Linear Baseline) ---
        # Projects raw history directly to future. Critical for 240-step accuracy.
        self.trend_linear = nn.Linear(T_in, T_out)
        
        # --- 2. Global Component (The "Average" System State) ---
        self.global_fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, T_out)
        )
        
        # --- 3. Local Component (The Node-Specific Deviations) ---
        self.local_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, T_out)
        )

    def forward(self, x_raw, h_gcn):
        """
        x_raw: (B, N, T_in)   - Raw input history
        h_gcn: (B, N, d_model) - Latent features from GCN
        """
        
        # --- A. Trend Prediction ---
        # We use the Target Node (Index 0) for the baseline trend
        x_target = x_raw[:, 0, :] # (B, T_in)
        trend = self.trend_linear(x_target) # (B, T_out)
        
        # --- B. Global Context Prediction ---
        # "What is the general state of the grid?"
        g_feat = torch.mean(h_gcn, dim=1)   # Pool over nodes -> (B, d_model)
        global_pred = self.global_fc(g_feat) # (B, T_out)
        
        # --- C. Local Deviation Prediction ---
        # "How does each node differ from the average?"
        # h_gcn: (B, N, d) - g_feat: (B, 1, d)
        l_feat = h_gcn - g_feat.unsqueeze(1) 
        
        # Process every node's deviation independently
        local_out = self.local_fc(l_feat)   # (B, N, T_out)
        
        # Aggregate Local Impacts
        # We assume the sum of local interactions contributes to the target
        local_pred = torch.mean(local_out, dim=1) # (B, T_out)

        # --- Combine ---
        return trend + global_pred + local_pred
    

class TR_GNN_Linear(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64, 
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3, GCN_Layer=5, dilation=3, kernel_size=7, graph_temperature=3.0):
        super().__init__()
        self.graph_learn = TemporalGraphLearning(hidden_dim, dropout=dropout_gcn, temperature=graph_temperature)
        
        # Use the FIXED TemporalConv (from previous step)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim, 
                                          dilation=dilation, dropout=dropout_temporal, kernel_size=kernel_size)
        
        # Use the FIXED DenselyResidualGCN (from previous step)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim, 
                                            dropout=dropout_gcn, layers=GCN_Layer)
        
        # Update Init Args
        self.forecaster = LoadForecasting(
            N=N,  # <--- Added N
            T_in=T_in, # <--- Added T_in
            T_out=T_out, 
            d_model=hidden_dim, 
            dropout=dropout_forecast
        )

    def forward(self, X): 
        H = self.temporal_conv(X)
        A = self.graph_learn(H)
        H = self.dense_gcn(H, A)
        # Update Forward Pass
        Y_hat = self.forecaster(X, H) # Pass Raw X + Hidden H
        return Y_hat, A


class TR_GNN_Attention(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0):          
        super().__init__()
        self.graph_learn = TemporalGraphLearning(hidden_dim, dropout=dropout_gcn, temperature=graph_temperature)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size, 
                                          dilation=dilation,        
                                          dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = AttentionForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn(H)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(X, H)
        return Y_hat, A


class TR_GNN_MultiScale(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0):          
        super().__init__()
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size,  
                                          dilation=dilation,        
                                          dropout=dropout_temporal)
        self.graph_learn = TemporalGraphLearning(hidden_dim, dropout=dropout_gcn, temperature=graph_temperature)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = MultiScaleForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn(H)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(X, H)
        return Y_hat, A


class TR_GNN_GlobalLocal(nn.Module):
    def __init__(self, N, T_in, T_out, d=32, hidden_dim=64,
                 dropout_temporal=0.2, dropout_gcn=0.3, dropout_forecast=0.3,
                 GCN_Layer=5, kernel_size=7, dilation=3, graph_temperature=3.0):          
        super().__init__()
        self.graph_learn = TemporalGraphLearning(hidden_dim, dropout=dropout_gcn, temperature=graph_temperature)
        self.temporal_conv = TemporalConv(N, T_in, hidden_dim,
                                          kernel_size=kernel_size,  
                                          dilation=dilation,        
                                          dropout=dropout_temporal)
        self.dense_gcn = DenselyResidualGCN(hidden_dim, hidden_dim,
                                            dropout=dropout_gcn, layers=GCN_Layer)
        self.forecaster = GlobalLocalForecasting(
            N=N, T_in=T_in, T_out=T_out,
            d_model=hidden_dim, dropout=dropout_forecast
        )

    def forward(self, X):
        H = self.temporal_conv(X)
        A = self.graph_learn(H)
        H = self.dense_gcn(H, A)
        Y_hat = self.forecaster(X, H)
        return Y_hat, A
