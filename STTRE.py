
from torch.utils.data import Dataset
import numpy as np
import torch
from wind_forecasting.preprocessing.data_reader import DataReader

from torch.utils.data import Dataset
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError

NUM_EPOCHS = 1 #10000
TEST_SPLIT = 0.5

class WindFarm(Dataset):
    def __init__(self, data_dir, seq_len=1200):
        self.seq_len = seq_len
        # data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype=None)
        data = DataReader().get_results_data([data_dir])["lut_LUT"].to_pandas()
        
        wind_dir_cols = sorted([col for col in data.columns if "TurbineWindDir_" in col])
        wind_mag_cols = sorted([col for col in data.columns if "TurbineWindMag_" in col])
        turbine_yaw_cols = sorted([col for col in data.columns if "TurbineYawAngle_" in col]) # TODO: Change to yaw_offset
        turbine_power_cols = sorted([col for col in data.columns if "TurbinePower_" in col])

        self.X = self.normalize(data.loc[:, wind_dir_cols + wind_mag_cols + turbine_yaw_cols + turbine_power_cols].to_numpy()) # 5 variables
        self.y = data.loc[:, turbine_power_cols].to_numpy()
        self.num_labels = len(np.unique(self.y))
        self.len = len(self.y)

    def __len__(self):
        return (self.len - self.seq_len)-1

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx:idx+self.seq_len])
        label = self.y[idx+self.seq_len+1]

        return torch.tensor(x, dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def normalize(self, X):
        X = np.transpose(X)
        X_norm = []
        for x in X:
            x = (x-np.min(x)) / (np.max(x)-np.min(x))
            X_norm.append(x)
        return np.transpose(X_norm)
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb, mask_flag):

        super(SelfAttention, self).__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu" # TODO: Remove when moving to GPU
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        self.mask_flag = mask_flag

        modules = ['spatial', 'temporal', 'spatiotemporal', 'decoder']
        assert (modules.__contains__(module)), "Invalid module"

        if module == 'spatial' or module == 'temporal':
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))
        # TODO does this work for decoder
        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

    def forward(self, x):
        N, _, _ = x.shape

        #non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)

        #shared weights between heads for spatio-temporal module
        else:
            values, keys, queries = x, x, x
            values = values.reshape(N, self.seq_len, self.heads, self.head_dim)
            keys = keys.reshape(N, self.seq_len, self.heads, self.head_dim)
            queries = queries.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(values)
            keys = self.keys(keys)
            queries = self.queries(queries)

        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1,2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # compute scores

            # An upper left triangle mask is applied to 𝑄𝐾𝑇 to prevent the queries from attending to keys that occur later in the sequence.
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device), 1) # TODO why do we need mask for encoder ...
            # if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3) + S

        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # compute scores

            # An upper left triangle mask is applied to 𝑄𝐾𝑇 to prevent the queries from attending to keys that occur later in the sequence.
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device), 1)
            # if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1/2)), dim=3)

        #attention(N x Heads x Q_Len x K_len)
        #values(N x V_len x Heads x Head_dim)
        #z(N x Q_len x Heads*Head_dim)

        if self.module == 'spatial' or self.module == 'temporal':
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len * self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len, self.heads * self.head_dim)

        z = self.fc_out(z)

        return z


    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = nn.functional.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(EncoderLayer, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        if module == 'spatial' or module == 'temporal':
            self.norm1 = nn.BatchNorm1d(seq_len * heads)
            self.norm2 = nn.BatchNorm1d(seq_len * heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )


    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        
        return out
    
"""
class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for layer in self.layers: # equivalent to looping through each EncoderLayer above
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)
        return x, attns
"""

# TODO
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)
        self.cross_attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb)

        self.norm1 = nn.BatchNorm1d(seq_len)
        self.norm2 = nn.BatchNorm1d(seq_len)
        self.norm3 = nn.BatchNorm1d(seq_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

    def forward(self, x, cross):
        x = self.self_attention(x) + x
        x = self.norm1(x)

        x = self.cross_attention(x, cross) + x
        y = x = self.norm2(x)

        y = self.feed_forward(x)
        out = self.norm3(y + x)
        
        return out
    

class Encoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module, rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             EncoderLayer(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb=rel_emb)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)

        out = self.fc_out(out)
        return out
    
# TODO
class Decoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module, rel_emb=True):
        super(Decoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             DecoderLayer(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, rel_emb=rel_emb)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)

        out = self.fc_out(out)
        return out
    
# class Decoder(nn.Module):
#     def __init__(self, layers, norm_layer=None):
#         super(Decoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer

#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         for layer in self.layers:
#             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

#         if self.norm is not None:
#             x = self.norm(x)

#         return x

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 +
                (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl

class BBBLinear(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else ("cpu" if torch.backends.mps.is_available() else "cpu"))

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = torch.log1p(torch.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = nn.functional.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + nn.functional.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = torch.sqrt(act_var)

        if self.training or sample:
            eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = calculate_kl(self.prior_mu, self.prior_sigma,
                          self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += calculate_kl(self.prior_mu, self.prior_sigma,
                               self.bias_mu, self.bias_sigma)
        return kl

class Transformer(nn.Module):
    def __init__(self, input_shape, output_size,
                 embed_size, num_layers, forward_expansion, heads):

        super(Transformer, self).__init__()
        self.batch_size, self.num_var, self.seq_len = input_shape
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_elements = self.seq_len * self.num_var
        self.embed_size = embed_size
        self.element_embedding = nn.Linear(self.seq_len, embed_size * self.seq_len)
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        # seq_len*heads in BlockNorm!d = self.seq_len * self.num_var
        # in SelfAttention, shape of self.E is (self.heads, self.head_dim, self.embed_size) = (self.num_var, self.seq_len, embed_size)
        self.temporal = Encoder(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers,
                                heads=self.num_var,
                                device=self.device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        # seq_len*heads in BlockNorm!d = self.num_var * self.seq_len
        # in SelfAttention, shape of self.E is (self.heads, self.head_dim, self.embed_size) = (self.seq_len, self.num_var, embed_size)
        self.spatial = Encoder(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=self.seq_len,
                               device=self.device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        # seq_len*heads in BlockNorm!d = self.seq_len * self.num_var
        # in SelfAttention, shape of self.E is (1, self.seq_len, self.head_dim) = (1, self.seq_len * self.num_var, embed_size // heads)
        self.spatiotemporal = Encoder(seq_len=self.seq_len * self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_layers,
                                      heads=heads,
                                      device=self.device,
                                      forward_expansion=forward_expansion,
                                      module = 'spatiotemporal',
                                      rel_emb=True)
        # TODO
        self.decoder = Decoder(seq_len=self.seq_len * self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_dec_layers,
                                      heads=heads,
                                      device=self.device,
                                      forward_expansion=forward_expansion,
                                      module = 'decoder',
                                      rel_emb=True)
        
        # class DecoderLayer(nn.Module):
        #     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
        #                 dropout=0.1, activation="relu"):
        #         super(DecoderLayer, self).__init__()
                    # self_attention = AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                    #             d_model, n_heads, mix=mix),
                #   cross_attention = AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                #                 d_model, n_heads, mix=False),

        #         d_ff = d_ff or 4*d_model
        #         self.self_attention = self_attention
        #         self.cross_attention = cross_attention
        #         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        #         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        #         self.norm1 = nn.LayerNorm(d_model)
        #         self.norm2 = nn.LayerNorm(d_model)
        #         self.norm3 = nn.LayerNorm(d_model)
        #         self.dropout = nn.Dropout(dropout)
        #         self.activation = F.relu if activation == "relu" else F.gelu

        #     def forward(self, x, cross, x_mask=None, cross_mask=None):
        #         x = x + self.dropout(self.self_attention(
        #             x, x, x,
        #             attn_mask=x_mask
        #         )[0])
        #         x = self.norm1(x)

        #         x = x + self.dropout(self.cross_attention(
        #             x, cross, cross,
        #             attn_mask=cross_mask
        #         )[0])

        #         y = x = self.norm2(x)
        #         y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        #         y = self.dropout(self.conv2(y).transpose(-1,1))

        #         return self.norm3(x+y)



        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
        #                         d_model, n_heads, mix=mix),
        #             AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation,
        #         )
        #         for l in range(d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(d_model)
        # )

        # TODO bayesian linear layer
        # self.decoder_layer = BBBLinear(32, 16)  # Bayesian Linear Layer

        # consolidate embedding dimension
        self.fc_out1 = nn.Linear(embed_size, embed_size // 2)
        self.fc_out2 = nn.Linear(embed_size // 2, 1)

        # prediction
        self.out = nn.Linear((self.num_elements*3), output_size) # 3 for the temporal, spatial, and spatiotemporal encoder modules


    def forward(self, x, dropout):
        batch_size = len(x)

        #process/embed input for temporal module
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_temporal = self.element_embedding(x).reshape(batch_size, self.num_elements, self.embed_size)
        x_temporal = nn.functional.dropout(self.pos_embedding(positions) + x_temporal, dropout)

        #process/embed input for spatial module
        x_spatial = torch.transpose(x, 1, 2).reshape(batch_size, self.num_var, self.seq_len)
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatial = self.element_embedding(x_spatial).reshape(batch_size, self.num_elements, self.embed_size)
        x_spatial = nn.functional.dropout(self.variable_embedding(vars) + x_spatial, dropout)


        #process/embed input for spatio-temporal module
        positions = torch.arange(0, self.seq_len).expand(batch_size, self.num_var, self.seq_len).reshape(batch_size, self.num_var * self.seq_len).to(self.device)
        x_spatio_temporal = self.element_embedding(x).reshape(batch_size, self.seq_len* self.num_var, self.embed_size)
        x_spatio_temporal = nn.functional.dropout(self.pos_embedding(positions) + x_spatio_temporal, dropout)

        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)
        enc_out = torch.cat((out1, out2, out3), 1)
        # out = torch.cat((out1, out2, out3), 1)

        # TODO add decoder 
        # dec_out, a = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        out = self.fc_out1(out)
        out = nn.functional.leaky_relu(out)
        out = self.fc_out2(out)
        out = nn.functional.leaky_relu(out)
        out = torch.flatten(out, 1)
        out = self.out(out)

        return out

def train_test(embed_size, heads, num_layers, dropout, forward_expansion, lr, batch_size, data_dir, dataset):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    datasets = ['WindFarm']
    assert (datasets.__contains__(dataset)), "Invalid dataset"

    #call dataset class
    if dataset == 'WindFarm':
        train_data = WindFarm(data_dir)

    #split into train and test
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(TEST_SPLIT * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                            sampler=train_sampler)
    test_dataloader = DataLoader(train_data, batch_size=batch_size,
                                                    sampler=test_sampler)

    #define loss function, and evaluation metrics
    mape = MeanAbsolutePercentageError().to(device)
    mae = MeanAbsoluteError().to(device)
    loss_fn = torch.nn.MSELoss()

    inputs, outputs = next(iter(train_dataloader))

    model = Transformer(inputs.shape, outputs.shape[1], embed_size=embed_size, num_layers=num_layers,
                        forward_expansion=forward_expansion, heads=heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_losses = []
    val_mae = []
    val_mape = []

    for epoch in range(NUM_EPOCHS):
        if epoch >= 5:
            print('Epoch: ', epoch)
            print('Average mse: ' + str(np.average(val_losses[-5:])))
            print('Average mape: ' + str(np.average(val_mape[-5:])))
            print('Average mae: ' + str(np.average(val_mae[-5:])))

        total_loss = 0

        #train loop
        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs.to(device), dropout)
            loss = loss_fn(output, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss

        total_loss = 0
        train_acc = 0
        total_mae = 0
        total_mape = 0
        div = 1

        #test loop
        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                output = model(inputs.to(device), 0)
                loss = loss_fn(output, labels.to(device))

                total_mae = total_mae + mae(output, labels.to(device))
                total_mape = total_mape + mape(output, labels.to(device))
                total_loss = total_loss + loss
                #div is used when the number of samples in a batch is less
                #than the batch size
                div = div + (len(inputs)/batch_size)

        val_losses.append(total_loss.item()/div)
        val_mae.append(total_mae.item()/div)
        val_mape.append(total_mape.item()/div)
    
    return None

if __name__ == "__main__":
    # test Wind Farm

    #export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    # torch.mps.set_per_process_memory_fraction(0.0)
    d = 8 # 32
    h = 2 #4
    num_layers = 3
    forward_expansion = 1
    dropout = 0.1
    lr = 0.0001
    batch_size = 8 #64
    data_dir = "/Users/ahenry/Documents/toolboxes/wind-hybrid-open-controller/examples/floris_case_studies/lut"
    dataset = 'WindFarm'

    train_test(d, h, num_layers, dropout, forward_expansion, lr, batch_size, data_dir, dataset)