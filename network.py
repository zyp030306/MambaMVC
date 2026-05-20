import torch.nn as nn
import torch
from models.mamba_nets.mm_bimamba import Mamba as MMBiMamba
from models.mamba_nets.bimamba import Mamba as BiMamba
class MMMambaEncoderLayer(nn.Module):
    """
    双视图Mamba编码层：输入两个视图特征，分别编码并实现残差与归一化。
    """
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config is not None

        bidirectional = mamba_config.pop('bidirectional')

        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )

        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x1, x2, x1_inference_params=None, x2_inference_params=None):
        # x1, x2: [B, L, d_model]
        x1_out1, x2_out1 = self.mamba(x1, x2, x1_inference_params, x2_inference_params)
        x1_out = x1 + self.norm1(x1_out1)
        x2_out = x2 + self.norm2(x2_out1)
        return x1_out, x2_out

class MM_Mamba_DualView(nn.Module):
    """
    多层双视图Mamba编码器结构，适用于多视图聚类或表征学习。
    """
    def __init__(
            self,
            num_layers,
            d_model,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')

        mamba_layers = []
        for i in range(num_layers):
            mamba_layers.append(MMMambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(
            self,
            x1, x2,
            x1_inference_params=None,
            x2_inference_params=None
    ):
        """
        x1, x2: 双视图输入, shape: [B, L, d_model]
        """
        out1 = x1
        out2 = x2
        for mamba_layer in self.mamba_layers:
            out1, out2 = mamba_layer(
                out1, out2,
                x1_inference_params,
                x2_inference_params
            )
        return out1, out2

class MambaEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            d_ffn,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        # if activation == 'Swish':
        #     activation = Swish
        # elif activation == "GELU":
        #     activation = torch.nn.GELU
        # else:
        #     activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(
            self,
            x, inference_params=None
    ):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out

class Mamba_singleview(nn.Module):

    def __init__(
            self,
            num_layers,
            d_model,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')

        mamba_list = []
        # print(output_sizes)
        for i in range(num_layers):

            mamba_list.append(MambaEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)

    def forward(
            self,
            x,
            inference_params=None,
    ):
        out = x

        for mamba_layer in self.mamba_layers:
            out = mamba_layer(
                out,
                inference_params=inference_params,
            )

        return out
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class GSFL(nn.Module):
    def __init__(self,in_feature_dim,class_num):
        super(GSFL,self).__init__()
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=in_feature_dim, nhead=1,dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=1)
        self.cluster = nn.Sequential(
            nn.Linear(in_feature_dim,class_num),
            nn.Softmax(dim=1)
        )
    def forward(self,C):
        t = self.TransformerEncoder(C)
        return t

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        self.As = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.As.append(nn.Parameter(torch.Tensor(class_num, feature_dim)).to(device))  # 质心
            torch.nn.init.xavier_normal_(self.As[v].data)

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.GSFL = GSFL(feature_dim * self.view,class_num)
        self.alpha = 1.0
        self.cluster = nn.Sequential(
                    nn.Linear(feature_dim*self.view, class_num),
                    nn.Softmax(dim=1)
                )
        self.GBMamba = MM_Mamba_DualView(
            num_layers=1,
            d_model=feature_dim * self.view,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config={
                'd_state': 16,
                'expand': 2,
                'd_conv': 4,
                'bidirectional': True
            })
        self.CBMamba = Mamba_singleview(
            num_layers=1,
            d_model=feature_dim,
            d_ffn=1024,
            activation='Swish',
            dropout=0.1,
            causal=False,
            mamba_config={
                'd_state': 16,
                'expand': 2,
                'd_conv': 4,
                'bidirectional': True
            })

    def forward(self,X):
        Zs = []
        X_hat = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            Zs.append(Z)
            X_hat.append(self.decoders[v](Z))

        Z = torch.cat(Zs, dim=1)
        t = self.GSFL(Z)
        t = t.unsqueeze(1)
        tempz = Z.unsqueeze(1)
        t, _ = self.GBMamba(t, tempz)
        hs = torch.stack(Zs,dim=1)
        hs = self.CBMamba(hs)
        hs = [hs[:, v, :] for v in range(hs.shape[1])]
        t = t.squeeze(1)
        t = self.cluster(t)
        P = self.target_distribution(t)
        Qs = []
        for v in range(self.view):
            q = 1.0 / (1.0 + torch.sum(torch.pow(hs[v].unsqueeze(1) - self.As[v], 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            Qs.append(q)
        return X_hat, P, Qs, hs

    def target_distribution(self,p):
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()

