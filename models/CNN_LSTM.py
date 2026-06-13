import torch
from torch import nn
from einops import rearrange


class CNNLSTMBlock(nn.Module):
    def __init__(self, in_channels, filters=4, kernel_size=2,
                 lstm_units1=16, lstm_units2=16, dropout=0.1):
        super(CNNLSTMBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            filters,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_units1,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_units1,
            hidden_size=lstm_units2,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units2, 4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = out.permute(0, 2, 1)
        out, _ = self.lstm1(out)
        out = self.dropout(out[:, -1, :])
        out, _ = self.lstm2(out.unsqueeze(1))
        out = self.dropout(out[:, -1, :])
        return self.tanh(self.fc(out))


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
        key = rearrange(key, 'b l (h d) -> b h l d', h=self.n_heads)
        value = rearrange(value, 'b l (h d) -> b h l d',h=self.n_heads)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = self.dropout(
            torch.softmax(attn_scores, dim=-1)
        )
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
        self.individual = bool(args.individual)
        self.use_noise = bool(args.use_noise)

        if self.individual:
            self.cnn_lstm_blocks = nn.ModuleList()
            for _ in range(self.max_imf):
                self.cnn_lstm_blocks.append(
                    CNNLSTMBlock(in_channels=1)
                )
        else:
            self.cnn_lstm_block = CNNLSTMBlock(
                in_channels=self.max_imf
            )

        self.final_linear = nn.Linear(
            4, self.feedback_len + self.horizon_len
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

        if self.individual:
            for imf_idx in range(self.max_imf):
                imf_data = imfdata[
                    :, :, imf_idx:imf_idx + 1
                ].permute(0, 2, 1)
                imf_feature = self.cnn_lstm_blocks[imf_idx](
                    imf_data
                )
                imf_pred = self.final_linear(imf_feature)
                output += imf_pred.unsqueeze(-1)
        else:
            x_cnn_lstm = imfdata.permute(0, 2, 1)
            feature = self.cnn_lstm_block(x_cnn_lstm)
            output += self.final_linear(feature).unsqueeze(-1)

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
