import torch

from einops import einsum, rearrange, reduce
from models.base_layers import Dense, ResidualLayer
from utils.utils import to_dense_batch
from models.na_pooling import Exchanging, Porjecting
import dgl
import pickle

class VirtualAtom(torch.nn.Module):
    def __init__(self, emb_size_atom: int, num_nodes,ratio=0.1,num_hidden=2, activation=None, name=None): # identifier in case a ScalingFactor is applied to Ewald output
        super().__init__()

        self.pre_residual = ResidualLayer(emb_size_atom, nLayers=2, activation=activation)

        # NOTE: using half size of interaction embedding size

        self.linear_heads = self.get_mlp(emb_size_atom, emb_size_atom, num_hidden, activation)

        self.number_atoms = num_nodes
        self.ratio = ratio

        self.proj_layer = Porjecting(
            emb_size_atom,
            num_heads=1,
            num_seeds=int(self.number_atoms * self.ratio),
            Conv=None,
            layer_norm=True,
        )
        self.interaction_layer = Exchanging(
            emb_size_atom,
            emb_size_atom,
            num_heads=4,
            Conv=None,
            layer_norm=True,
        )

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

   

    def forward(self, bg, h: torch.Tensor):
        """
        h: embedding
        x: pos embedding
        return h_na
        """ 
        hres = self.pre_residual(h)

        # * use hres to perform message agg and passing
        h_update = self.get_NAs_emb(bg, hres)

        # Apply update function
        for layer in self.linear_heads:
            h_update = layer(h_update)

        return h_update

    def get_NAs_emb(self, bg, x):
        batch_x, ori_mask = to_dense_batch(bg, x)
        mask = (~ori_mask).unsqueeze(1).to(dtype=x.dtype) * -1e9
        # * S for node cluster allocation matrix
        NAs_emb, S = self.proj_layer(batch_x, None, mask)
        NAs_emb = self.interaction_layer(NAs_emb, None, None)[0]
        h = reduce(
            einsum(
                rearrange(S, "(b h) c n -> h b c n", h=1),
                NAs_emb,
                "h b c n, b c d -> h b n d",
            ),
            "h b n d -> b n d",
            "mean",
        )[ori_mask]
        return h


