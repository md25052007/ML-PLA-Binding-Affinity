# @Time    : 2024/5/8
# @Author  : rylynn
# @Email   : 
# @File    : graph_constructor.py
# @desc: graph constructor 
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import numpy as np
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot, BaseBondFeaturizer
import pickle
import pandas as pd
import os
# from dgl.data.chem import BaseBondFeaturizer
from functools import partial
import warnings
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs
import multiprocessing
from itertools import repeat
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from utils.utils import *

def chirality(atom):  # the chirality手性 information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()

def graphs_from_mol(dir, key, label, graph_dic_path, rasidue_dic_path, dis_threshold=8.0, path_marker='/', add_3D=True):
    """
    :param m1: ligand molecule
    :param m2: pocket molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return:
    complex: graphs contain m1, m2 and complex
    """
    add_self_loop = False
    try:
        with open(dir, 'rb') as f:
            mol1, mol2 = pickle.load(f)
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold
        # construct graphs
        g1 = dgl.DGLGraph()  # small molecule
        g2 = dgl.DGLGraph()  # pocket

        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g1.add_nodes(num_atoms_m1)
        g2.add_nodes(num_atoms_m2)

        if add_self_loop:
            nodes1 = g1.nodes()
            g1.add_edges(nodes1, nodes1)
            nodes2 = g2.nodes()
            g2.add_edges(nodes2, nodes2)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])
        g1.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u)
            dst2.append(v)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g2.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g1.ndata['h'] = AtomFeaturizer(mol1)['h']
        g2.ndata['h'] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        g1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        g2_d = torch.tensor(dis_matrix_P[src_ls2, dst_ls2], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g1.edata['e'] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g2.edata['e'] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1.edata['e'] = torch.cat([g1.edata['e'], g1_d * 0.1], dim=-1)
        g2.edata['e'] = torch.cat([g2.edata['e'], g2_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        g1.ndata['pos'] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g2.ndata['pos'] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)

        # calculate the 3D info for g1
        src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g1.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls1 = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
        D3_info_th1 = torch.tensor(D3_info_ls1, dtype=torch.float)
        g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th1], dim=-1)

        # calculate the 3D info for g2
        src_nodes, dst_nodes = g2.find_edges(range(g2.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g2.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls2 = list(map(partial(D3_info_cal, g=g2), neighbors_ls))
        D3_info_th2 = torch.tensor(D3_info_ls2, dtype=torch.float)
        g2.edata['e'] = torch.cat([g2.edata['e'], D3_info_th2], dim=-1)
        g1.ndata.pop('pos')
        g2.ndata.pop('pos')


        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th1)) or torch.any(torch.isnan(D3_info_th2)):
            status = False
            print(f"{key} is falied")
        else:
            status = True
        # residue graph
        with open(os.path.join(rasidue_dic_path, key), 'rb') as f:
            g4 = pickle.load(f)
    except:
        g1 = None
        g2 = None
        g3 = None
        g4 = None
        status = False
    if status:
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4, 'key': key, 'label': label}, f)


def graphs_from_mol_mul(dir, key, label, graph_dic_path, rasidue_dic_path, dis_threshold=5.0, path_marker='/'):
    """
    This function is used for generating graph objects using multi-process
    :param dir: the absoute path for the complex
    :param key: the key for the complex
    :param label: the label for the complex
    :param rasidue_dic_path: saved residue graph path
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the absoute path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux
    :return:
    """
    # TODO redisue graph

    add_self_loop = False
    try:
        with open(dir, 'rb') as f:
            mol1, mol2 = pickle.load(f)
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold
        # small molecule
        # mol1 = m1
        # pocket
        # mol2 = m2

        # construct graphs1
        g = dgl.DGLGraph()
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)

        if add_self_loop:
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)
            u = bond1.GetBeginAtomIdx()
            v = bond1.GetEndAtomIdx()
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])
        dst_ls1 = np.concatenate([dst1, src1])  # 默认有向图，双向都要达到无向图。
        g.add_edges(src_ls1, dst_ls1)

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'))  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'))  # init 'h'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # if add_3D:
        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3])
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')

        # rasidue graph
        residue = os.path.join(rasidue_dic_path, key)


        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
            print(f"{key} is falied")
            print(f"{key} residue{os.path.exists(residue)} exist")
        else:
            status = True
        if not os.path.exists(residue):
            status = False
            print(key+"residue is none")
        else:
            with open(os.path.join(rasidue_dic_path, key), 'rb') as f:
                g4 = pickle.load(f)
                status = True
    except:
        g = None
        g3 = None
        g4 = None
        status = False
    if status:
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'g': g, 'g3': g3, 'g4': g4, 'key': key, 'label': label}, f)

def collate(samples):
    # return dgl.batch_hetero(samples)
    return dgl.batch(samples)

def collate_fn_v2(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    graphs1, graphs2, graphs3, graphs4, Ys, keys = map(list, zip(*data_batch))
    bg1 = dgl.batch(graphs1)
    bg2 = dgl.batch(graphs2)
    bg3 = dgl.batch(graphs3)
    bg4 = dgl.batch(graphs4)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg1, bg2, bg3, bg4, Ys, keys


def collate_fn_v2_MulPro(data_batch):
    """
    used for dataset generated from GraphDatasetV2MulPro class
    :param data_batch:
    :return:
    """
    graphs, graphs3, graphs4, Ys, keys = map(list, zip(*data_batch))
    bg = dgl.batch(graphs)
    bg3 = dgl.batch(graphs3)
    bg4 = dgl.batch(graphs4)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg, bg3, bg4, Ys, keys



class GraphsDataset(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, keys, labels, data_dirs, graph_ls_path, graph_dic_path, residue_dirs, num_process=6, dis_threshold=5.0,
                 add_3D=True, path_marker='/', ):
        """
        :param keys: the keys for the complexs, list
        :param labels: the corresponding labels for the complexs, list
        :param data_dirs: the corresponding data_dirs for the complexs, list
        :param graph_ls_path: the cache path for the total graphs objects (graphs.bin, graphs3.bin), labels, keys
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param num_process: the numer of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the atom-pair interactions
        :param add_3D: add the 3D geometric features to the edges of graphs
        :param path_marker: '\\' for windows and '/' for linux
        """
        self.origin_keys = keys
        self.origin_labels = labels
        self.origin_data_dirs = data_dirs
        #　like origin_data_dirs processed_residue graph but pickle dict。
        self.origin_residue_graph_dir = residue_dirs
        self.graph_ls_path = graph_ls_path
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.add_3D = add_3D
        self.dis_threshold = dis_threshold
        self.path_marker = path_marker
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.graph_ls_path+self.path_marker+'g1.bin'):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_path + self.path_marker + 'g1.bin', 'rb') as f:
                self.graphs1 = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'g2.bin', 'rb') as f:
                self.graphs2 = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'rb') as f:
                self.graphs3 = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'residue.bin', 'rb') as f:
                self.residue_graph = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'rb') as f:
                self.keys = pickle.load(f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'rb') as f:
                self.labels = pickle.load(f)
        else:
            graph_dic_paths = repeat(self.graph_dic_path, len(self.origin_data_dirs))
            dis_thresholds = repeat(self.dis_threshold, len(self.origin_data_dirs))
            path_markers = repeat(self.path_marker, len(self.origin_data_dirs))
            residue_graph_paths = repeat(self.origin_residue_graph_dir, len(self.origin_data_dirs))
            print('Generate complex graph...')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(graphs_from_mol,
                         zip(self.origin_data_dirs, self.origin_keys, self.origin_labels, graph_dic_paths,
                             residue_graph_paths, dis_thresholds, path_markers))
            pool.close()
            pool.join()

            # collect the generated graph for each complex
            self.graphs1 = []
            self.graphs2 = []
            self.graphs3 = []
            self.residue_graph = []
            self.labels = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.graphs1.append(graph_dic['g1'])
                    self.graphs2.append(graph_dic['g2'])
                    self.graphs3.append(graph_dic['g3'])
                    self.residue_graph.append(graph_dic['g4'])
                    self.labels.append(graph_dic['label'])
            # store to the disk
            with open(self.graph_ls_path + self.path_marker + 'g1.bin', 'wb') as f:
                pickle.dump(self.graphs1, f)
            with open(self.graph_ls_path + self.path_marker + 'g2.bin', 'wb') as f:
                pickle.dump(self.graphs2, f)
            with open(self.graph_ls_path + self.path_marker + 'g3.bin', 'wb') as f:
                pickle.dump(self.graphs3, f)
            with open(self.graph_ls_path + self.path_marker + 'residue.bin', 'wb') as f:
                pickle.dump(self.residue_graph, f)
            with open(self.graph_ls_path + self.path_marker + 'keys.bin', 'wb') as f:
                pickle.dump(self.keys, f)
            with open(self.graph_ls_path + self.path_marker + 'labels.bin', 'wb') as f:
                pickle.dump(self.labels, f)

        # delete the temporary files
        cmdline = 'rm -rf %s' % (self.graph_dic_path + self.path_marker + '*')  # graph_dic_path
        os.system(cmdline)

    def __getitem__(self, indx):
        return self.graphs1[indx], self.graphs2[indx], self.graphs3[indx], self.residue_graph[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)



def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def match_feature(x, all_for_assign):
    x_p = np.zeros((len(x), 7))

    for j in range(len(x)):
        if x[j] == 'ALA':
            x_p[j] = all_for_assign[0, :]
        elif x[j] == 'CYS':
            x_p[j] = all_for_assign[1, :]
        elif x[j] == 'ASP':
            x_p[j] = all_for_assign[2, :]
        elif x[j] == 'GLU':
            x_p[j] = all_for_assign[3, :]
        elif x[j] == 'PHE':
            x_p[j] = all_for_assign[4, :]
        elif x[j] == 'GLY':
            x_p[j] = all_for_assign[5, :]
        elif x[j] == 'HIS':
            x_p[j] = all_for_assign[6, :]
        elif x[j] == 'ILE':
            x_p[j] = all_for_assign[7, :]
        elif x[j] == 'LYS':
            x_p[j] = all_for_assign[8, :]
        elif x[j] == 'LEU':
            x_p[j] = all_for_assign[9, :]
        elif x[j] == 'MET':
            x_p[j] = all_for_assign[10, :]
        elif x[j] == 'ASN':
            x_p[j] = all_for_assign[11, :]
        elif x[j] == 'PRO':
            x_p[j] = all_for_assign[12, :]
        elif x[j] == 'GLN':
            x_p[j] = all_for_assign[13, :]
        elif x[j] == 'ARG':
            x_p[j] = all_for_assign[14, :]
        elif x[j] == 'SER':
            x_p[j] = all_for_assign[15, :]
        elif x[j] == 'THR':
            x_p[j] = all_for_assign[16, :]
        elif x[j] == 'VAL':
            x_p[j] = all_for_assign[17, :]
        elif x[j] == 'TRP':
            x_p[j] = all_for_assign[18, :]
        elif x[j] == 'TYR':
            x_p[j] = all_for_assign[19, :]

    return x_p


def read_atoms(file_name, chain="."):
    """
    return list :CA_atom_3D_ord, ajs(氨基酸标识符)
    """
    pattern = re.compile(chain)

    atoms = []
    ajs = []
    # line: ATOM      2  [CA  ]MET A   1      -7.174  35.466  26.941  1.00 40.93           C
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("ATOM"):
                # atom type
                type = line[12:16].strip()
                # atom chain
                chain = line[21:22]
                if type == "CA" and re.match(pattern, chain):
                    # 坐标
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    # 氨基酸残基标识符
                    ajs_id = line[17:20]
                    atoms.append((x, y, z))
                    ajs.append(ajs_id)
    return atoms, ajs


def compute_contacts(atoms, threshold):
    contacts = []
    """
    return contacts[(i,j) (j,i)....]
    """
    # 　间隔1个原子
    for i in range(len(atoms) - 2):
        for j in range(i + 2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i, j))
                contacts.append((j, i))
    return contacts


def knn(atoms, k=5):
    """

    :param atoms:
    :param k:
    :return: list atom id [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 0), (2, 5), (2, 6), (2, 7), (3, 6), (3, 5), (3, 1), (3, 7), (3, 0), (4, 7), (4, 6), (4, 2), (4, 8), (4, 1), (5, 8), (5, 7), (5, 3), (5, 2), (5, 9), (6, 4), (6, 8), (6, 9), (6, 3), (6, 10),
    """
    x = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            x[i, j] = dist(atoms[i], atoms[j])
    index = np.argsort(x, axis=-1)

    contacts = []
    for i in range(len(atoms)):
        num = 0
        for j in range(len(atoms)):
            if index[i, j] != i and index[i, j] != i - 1 and index[i, j] != i + 1:
                contacts.append((i, index[i, j]))
                num += 1
            if num == k:
                break

    return contacts


def pdb_to_cm(file_dir, key, graph_dic_path, threshold, all_for_assign, khop):
    atoms, x = read_atoms(file_dir)
    x_p = match_feature(x, all_for_assign)
    r_contacts = compute_contacts(atoms, threshold)  # threshold=10
    k_contacts = knn(atoms, khop)
    prot_seq = []
    # 　内循环对每一个蛋白质进行处理　构图　(0，1)( 1，0) (1，2)， (2，1)...顺序边
    for j in range(x_p.shape[0] - 1):
        prot_seq.append((j, j + 1))
        prot_seq.append((j + 1, j))
    # 直接构图，三种边。节点特征只有氨基酸理化特征。
    # prot_g = dgl.graph(prot_edge[i]).to(device)
    prot_g = dgl.heterograph({('amino_acid', 'SEQ', 'amino_acid'): prot_seq,
                              ('amino_acid', 'STR_KNN', 'amino_acid'): k_contacts,
                              ('amino_acid', 'STR_DIS', 'amino_acid'): r_contacts})
    prot_g.ndata['x'] = torch.FloatTensor(x_p)

    with open(os.path.join(graph_dic_path, key), 'wb') as f:
        pickle.dump(prot_g, f)



class ResidueDatasetDGL(object):
    """
    This class is used for generating graph objects using multi process
    """

    def __init__(self, keys, data_dirs, residue_ls_path, residue_dic_path, num_process=6, dis_threshold=10.0, khop=5):
        """
        :param keys: the keys for the protein pdb id, list
        :param data_dirs: the corresponding data_dirs for the protein pdb file
        :param graph_ls_path: the cache path for the total graphs objects (graphs.bin, graphs3.bin), labels, keys
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param num_process: the numer of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the residue-pair interactions
        :param khop: the num of k  nearilist edge
        """
        self.origin_keys = keys  # all pdb id
        self.origin_data_dirs = data_dirs  # ./data/all_protein_files
        self.residue_ls_path = residue_ls_path  # processed_data/residue_ls_path
        self.residue_dic_path = residue_dic_path  # processed_data/residue_dic_path
        self.num_process = num_process
        self.dis_threshold = dis_threshold
        self.khop = khop
        self.all_for_assign = np.loadtxt("./data/all_assign.txt")
        self._pre_process()

    def _pre_process(self):
        if not os.path.exists(self.residue_ls_path):
            os.makedirs(self.residue_ls_path)
        if not os.path.exists(self.residue_dic_path):
            os.makedirs(self.residue_dic_path)

        if os.path.exists(os.path.join(self.residue_ls_path, 'residue_g.bin')):
            print('Loading previously saved residue dgl graphs...')
            with open(os.path.join(self.residue_ls_path, 'residue_g.bin'), 'rb') as f:
                self.graphs = pickle.load(f)

        else:
            residue_dic_paths = repeat(self.residue_dic_path, len(self.origin_data_dirs))
            dis_thresholds = repeat(self.dis_threshold, len(self.origin_data_dirs))
            all_for_assigns = repeat(self.all_for_assign, len(self.origin_data_dirs))
            khops = repeat(self.khop, len(self.origin_data_dirs))
            print('Generate residue graph...')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(pdb_to_cm,
                         zip(self.origin_data_dirs, self.origin_keys, residue_dic_paths,
                             dis_thresholds, all_for_assigns, khops))
            pool.close()
            pool.join()

            # collect the generated graph for each complex
            self.graphs = []
            self.keys = os.listdir(self.residue_dic_path)
            for key in self.keys:
                with open(os.path.join(self.residue_dic_path, key), 'rb') as f:
                    graph = pickle.load(f)
                    self.graphs.append(graph)
            # store to the disk
            with open(os.path.join(self.residue_ls_path, 'residue_g.bin'), 'wb') as f:
                pickle.dump(self.graphs, f)

        # delete the temporary files
        # cmdline = 'rm -rf %s' % (self.graph_dic_path + self.path_marker + '*')  # graph_dic_path
        # os.system(cmdline)

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)


def load_pretrain_data(protein_dir, protein_id):
    # TODO arguments
    # protein_dir = "./all_protein_files"
    all_protein_id = pd.read_csv(protein_id)['id']
    all_protein_dirs = []
    for key in all_protein_id:
        all_protein_dirs.append(os.path.join(protein_dir, f'{key}_protein.pdb'))
    protein_data = ResidueDatasetDGL(keys=all_protein_id, data_dirs=all_protein_dirs,
                                     residue_ls_path='./data/binding_affinity/processed_data/residue_ls_path',
                                     residue_dic_path='./data/binding_affinity/processed_data/residue_dic_path')

    return protein_data

