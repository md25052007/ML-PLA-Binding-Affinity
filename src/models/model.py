# @Time    : 2024/6/6
# @Author  : rylynn
# @Email   : 

from dgllife.model.gnn import GAT, AttentiveFPGNN
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, SAGEConv
import dgl
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from models.virtual_atom_block import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# QM dimension constants (must match qm_features.py)
QM_ATOM_FEAT_DIM = 25
QM_MOL_FEAT_DIM  = 7

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h



class DTIConvGraph3(nn.Module):
    def __init__(self, param,in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
        self.param = param
    # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim+1, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())
        self.nablock = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        # self.conv1 = SAGEConv(in_dim,in_dim,'mean')
        # self.conv2 = SAGEConv(in_dim,in_dim,'mean')


        for _ in range(self.param['cp_nalayer']):
            self.convs.append(SAGEConv(in_dim, in_dim, 'mean'))
        for _ in range(self.param['cp_nalayer']):
            self.nablock.append(VirtualAtom(emb_size_atom=in_dim, 
                                            num_hidden=self.param['cp_num_hidden'],
                                            activation=self.param['cp_activation'],
                                            num_nodes=self.param['cp_num_nodes'],
                                            ratio=self.param['cp_ratio']))        

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        # bg.ndata['h'] = atom_feats
        for i in range(self.param['cp_nalayer']):
            h = self.convs[i](bg, atom_feats)
            h_na = self.nablock[i](bg, h)
            h = h + h_na
        bg.ndata['h'] = h
        bg.edata['e'] = bond_feats

        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, param, in_dim, out_dim, dropout): 
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(param,in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            # weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum  # normal version
        # return h_g_sum, weights  # temporary version


class EdgeWeightedSumAndMax(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g  # normal version
        # return h_g, weights  # temporary version


class AttentiveGRU1(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        # update_all node level
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        # update func gru
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            # num_of_atoms * node_feats @ w1(node_geats, hidden_dim)
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)


    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        # node_feats = self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])
        # return node_feats + h_na
        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)
    # dst src . paper is src||dst
    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))
        # return self.attentive_gru(g, logits, node_feats)


class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self,param, type):
        super(ModifiedAttentiveFPGNNV2, self).__init__()
        self.param = param
        self.init_context = GetContext(self.param['node_feat_size'], self.param['edge_feat_size_3d'], self.param['graph_feat_size'], self.param['dropout'])
        self.gnn_layers = nn.ModuleList()
        self.na_layers = nn.ModuleList()
        self.sum_node_feats = 0
        self.usena = False
        self.skip_connection_factor = 1 
        for _ in range(self.param['num_layers'] - 1):
            self.gnn_layers.append(GNNLayer(self.param['graph_feat_size'],self.param['graph_feat_size'], self.param['dropout']))
        if type == 'liga':
            for _ in range(self.param['num_layers']):
                self.na_layers.append(VirtualAtom(emb_size_atom=self.param['graph_feat_size'], num_hidden=self.param['liga_num_hidden'],activation=self.param['liga_activation'],num_nodes=self.param['liga_num_nodes'],ratio=self.param['liga_ratio']))        
        else:
            for _ in range(self.param['num_layers']):
                self.na_layers.append(VirtualAtom(emb_size_atom=self.param['graph_feat_size'], num_hidden=self.param['prot_num_hidden'],activation=self.param['prot_activation'],num_nodes=self.param['prot_num_nodes'],ratio=self.param['prot_ratio']))        
    
    def forward(self, g, node_feats, edge_feats):
        if self.usena:
            h_gnn = self.init_context(g, node_feats, edge_feats)
            h_na = self.na_layers[0](g, h_gnn)

            node_feats = self.skip_connection_factor * (h_gnn + h_na)


            for i in range(self.param['num_layers'] - 1):
                H_GNN = self.gnn_layers[i](g, node_feats)
                H_NA = self.na_layers[i+1](g, H_GNN)
                node_feats = self.skip_connection_factor * (H_GNN + H_NA)
            return node_feats
        else:
            node_feats = self.init_context(g, node_feats, edge_feats)
            self.sum_node_feats = node_feats
            for gnn in self.gnn_layers:
                node_feats = gnn(g, node_feats)
                self.sum_node_feats = self.sum_node_feats + node_feats

            return self.sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self,param,type):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()
        self.gnn = ModifiedAttentiveFPGNNV2(param=param,type=type)


    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return sum_node_feats


class DTIPredictor(nn.Module):
    """
    DTA Prediction na + mape + MISATO QM features
    
    Parameters
    -----------------
    node_feat_size      : ligand graph node feat dim  (40 base + 25 QM-atom = 65)
    edge_feat_size      : graph edge feat dimension
    num_layers          : num GNN layers including GetContext
    graph_feat_size     : hidden dimension for both protein and ligand
    res_hidden_dim      : codebook protein hidden dim  [graph level]
    out_dim_g3          : prot-liga interaction graph hidden dim
    d_FC_layer          : FC layer dimension
    n_FC_layer          : number of FC layers
    dropout             : dropout rate
    n_tasks             : regression target count (1 for affinity)
    use_qm_mol_feats    : whether to fuse molecule-level QM scalars into FC
    """
    def __init__(self, param):
        super(DTIPredictor, self).__init__()
        self.param = param
        self.use_qm_mol_feats = param.get('use_qm_mol_feats', True)

        # ── Ligand node-feature projection (base=40 + QM-atom=25 → node_feat_size) ──
        ligand_raw_dim = param['node_feat_size'] + QM_ATOM_FEAT_DIM  # 65
        self.ligand_input_proj = nn.Sequential(
            nn.Linear(ligand_raw_dim, param['node_feat_size']),
            nn.LayerNorm(param['node_feat_size']),
            nn.SiLU(),
        )

        self.liga_conv = ModifiedAttentiveFPPredictorV2(param=self.param, type='liga')
        self.prot_conv = ModifiedAttentiveFPPredictorV2(param=self.param, type='prot')

        self.residue_embedding_fc = nn.Sequential(nn.Linear(self.param['prot_hidden_dim'] * 2, self.param['graph_feat_size']), 
                                                 nn.ReLU(),
                                                 nn.BatchNorm1d(self.param['graph_feat_size']))

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(self.param, self.param['graph_feat_size'], self.param['outdim_g3'], self.param['dropout'])

        # ── Molecule-level QM fusion MLP ────────────────────────────────────
        if self.use_qm_mol_feats:
            self.qm_mol_proj = nn.Sequential(
                nn.Linear(QM_MOL_FEAT_DIM, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
            )
            qm_mol_out = 32
        else:
            self.qm_mol_proj = None
            qm_mol_out = 0

        # MLP predictor: outdim_g3*2 (readout) + graph_feat_size (residue) + qm_mol_out
        fc_in_dim = self.param['outdim_g3'] * 2 + self.param['graph_feat_size'] + qm_mol_out
        self.FC = FC(fc_in_dim, self.param['d_FC_layer'], self.param['n_FC_layer'], self.param['dropout'], self.param['n_tasks'])

        # read out
        self.readout = EdgeWeightedSumAndMax(self.param['outdim_g3'])

    
    def forward(self, bg1, bg2, bg3, residue_feats):
        """
        @param bg1: ligand graph  (node feat = 65: 40 base + 25 QM-atom;
                                   ndata may also contain 'qm_mol' (N,7))
        @param bg2: pocket graph
        @param bg3: interact graph
        @param residue_feats: residue graph readout (B, prot_hidden_dim)
        """
        # ── Ligand node features (project 65 → node_feat_size) ────────────────
        atom_feats1_raw = bg1.ndata.pop('h')   # (N_lig, 65)

        # Pool molecule-level QM before we pop it (mean over atoms per graph)
        if self.use_qm_mol_feats and 'qm_mol' in bg1.ndata:
            qm_mol_node = bg1.ndata.pop('qm_mol')  # (N_lig_total, 7)
            bg1.ndata['qm_mol_tmp'] = qm_mol_node
            qm_mol_graph = dgl.mean_nodes(bg1, 'qm_mol_tmp')  # (B, 7)
            bg1.ndata.pop('qm_mol_tmp')
        else:
            if 'qm_mol' in bg1.ndata:
                bg1.ndata.pop('qm_mol')
            qm_mol_graph = None

        bond_feats1 = bg1.edata.pop('e')
        atom_feats2 = bg2.ndata.pop('h')
        bond_feats2 = bg2.edata.pop('e')

        atom_feats1 = self.ligand_input_proj(atom_feats1_raw)   # (N_lig, node_feat_size)

        # atom feats GNN
        atom_feats1 = self.liga_conv(bg1, atom_feats1, bond_feats1)
        atom_feats2 = self.prot_conv(bg2, atom_feats2, bond_feats2)
        residue_feats = self.residue_embedding_fc(residue_feats)

        bg1.ndata['h'] = atom_feats1
        bg2.ndata['h'] = atom_feats2
        bg1_ls = dgl.unbatch(bg1)
        bg2_ls = dgl.unbatch(bg2)
        bg3_ls = dgl.unbatch(bg3)
        for i in range(len(bg1_ls)):
            bg3_ls[i].ndata['h'] = torch.cat([bg1_ls[i].ndata['h'], bg2_ls[i].ndata['h']], dim=0)
        bg3 = dgl.batch(bg3_ls)
        atom_feats3 = bg3.ndata['h']
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats3, bond_feats3)

        readouts = self.readout(bg3, bond_feats3)          # (B, outdim_g3*2)
        readouts = torch.cat([readouts, residue_feats], dim=1)

        # ── Fuse molecule-level QM scalars ────────────────────────────────────
        if self.use_qm_mol_feats and qm_mol_graph is not None:
            qm_mol_emb = self.qm_mol_proj(qm_mol_graph)   # (B, 32)
            readouts = torch.cat([readouts, qm_mol_emb], dim=1)

        return self.FC(readouts)
    
 


