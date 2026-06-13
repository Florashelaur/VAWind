import torch
from torch import nn
from einops import rearrange


class VariableWiseConv(nn.Module):
    def __init__(self, n_vars, seq_len, feature_len):
        super(VariableWiseConv, self).__init__()
        kernel_size = 3
        stride = (seq_len - kernel_size) / (feature_len - 1)

        self.conv = nn.Conv1d(
            in_channels=n_vars,
            out_channels=n_vars,
            kernel_size=kernel_size,
            stride=int(stride),
            groups=n_vars
        )

    def forward(self, x):
        return self.conv(x)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_vars, feature_len, horizon_len, d_model,
                 n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_vars = n_vars
        self.feature_len = feature_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.key_value_proj = nn.Linear(1, d_model)
        self.queries = nn.Parameter(torch.randn(feature_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(feature_len * d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm_ffn = nn.LayerNorm(d_model)
        self.lth = nn.Linear(d_model, horizon_len)

    def forward(self, conv_feat, value):
        batch_size = conv_feat.size(0)
        key = rearrange(conv_feat, 'b l n -> b (n l) 1')
        value = rearrange(value, 'b l n -> b (n l) 1')
        key = self.key_value_proj(key)
        value = self.key_value_proj(value)
        query = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        query = rearrange(query, 'b l (h d) -> b h l d', h=self.n_heads)
        key = rearrange(key, 'b l (h d) -> b h l d',h=self.n_heads)
        value = rearrange(value, 'b l (h d) -> b h l d',h=self.n_heads)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        context = torch.matmul(attn_weights, value)
        context = rearrange(context, 'b h l d -> b l (h d)')
        context = self.dropout(self.layer_norm(context))
        context = rearrange(context, 'b l d -> b 1 (l d)')
        context = self.dropout(self.layer_norm_ffn(self.ffn(context)))
        output = rearrange(self.lth(context), 'b n h -> b h n')
        return output, attn_weights


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.lookback_len = args.seq_len
        self.horizon_len = args.pred_len
        self.feedback_len = args.label_len
        self.feature_len = self.feedback_len // 3
        self.n_vars = args.num_variables
        self.max_imf = args.num_components
        self.use_noise = bool(args.use_noise)

        self.LSTM = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.LLinear = nn.Linear(
            64, self.feedback_len + self.horizon_len
        )

        if self.use_noise:
            self.conv = VariableWiseConv(
                n_vars=self.n_vars,
                seq_len=self.feedback_len,
                feature_len=self.feature_len
            )
            self.ALinear = nn.Linear(
                self.feedback_len, self.feature_len
            )
            self.BLinear = nn.Linear(1, self.n_vars)
            self.Mattention = MultiHeadAttentionLayer(
                n_vars=self.n_vars,
                feature_len=self.feature_len,
                horizon_len=self.horizon_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                dropout=args.dropout
            )

    def forward(self, batch_x):
        batch_size = batch_x.size(0)
        imfdata = batch_x[:, :, -self.max_imf:]
        output = torch.zeros(
            batch_size,
            self.feedback_len + self.horizon_len,
            1,
            dtype=batch_x.dtype,
            device=batch_x.device
        )

        for imf_idx in range(self.max_imf):
            imf_data = imfdata[:, :, imf_idx].unsqueeze(-1)
            lstm_out, _ = self.LSTM(imf_data)
            imf_pred = self.LLinear(lstm_out[:, -1, :])
            output += imf_pred.unsqueeze(-1)

        initial_pred = output[:, -self.horizon_len:, :]
        if not self.use_noise:
            return initial_pred, None

        target_history = batch_x[
            :, -self.feedback_len:, self.n_vars - 1
        ].unsqueeze(-1)
        noise = output[:, :self.feedback_len, :] - target_history
        x_conv = batch_x[:, -self.feedback_len:, :self.n_vars]
        x_conv = rearrange(x_conv, 'b l n -> b n l')
        noise = rearrange(noise, 'b l n -> b n l')

        value = self.ALinear(noise)
        value = rearrange(value, 'b n l -> b l n')
        value = self.BLinear(value)
        conv_feat = self.conv(x_conv)
        conv_feat = rearrange(conv_feat, 'b n l -> b l n')
        y_horizon_noise, attn_weights = self.Mattention(
            conv_feat, value
        )
        pred = y_horizon_noise + initial_pred
        return pred, attn_weights
