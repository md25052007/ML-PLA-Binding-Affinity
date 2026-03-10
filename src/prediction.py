# @Time    : 2024/5/27
# @Author  : rylynn
# @Email   : 
# @File    : prediction.py
# @desc: use for predict test set affinity
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.graph_constructor import *
from utils.utils import *
from models.model import *
from models.codebook import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
import torch
import pandas as pd
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
from numba import njit

# Concordance index (C-index)
@njit
def CI(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair != 0:
        return summ / pair
    else:
        return 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class GraphSet(object):
    """
    load data set
    """

    def __init__(self, graph_ls_path):
        self.graph_ls_path = graph_ls_path
        self.loadgraph()

    def loadgraph(self):
        # Base path for graph files
        base_path = lambda file_name: os.path.join(self.graph_ls_path, file_name)

        # Check if the first graph file exists
        if os.path.exists(base_path('g1.bin')):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(base_path('g1.bin'), 'rb') as f:
                self.graphs1 = pickle.load(f)
            with open(base_path('g2.bin'), 'rb') as f:
                self.graphs2 = pickle.load(f)
            with open(base_path('g3.bin'), 'rb') as f:
                self.graphs3 = pickle.load(f)
            with open(base_path('residue.bin'), 'rb') as f:
                self.residue_graph = pickle.load(f)
            with open(base_path('keys.bin'), 'rb') as f:
                self.keys = pickle.load(f)
            with open(base_path('labels.bin'), 'rb') as f:
                self.labels = pickle.load(f)

    def __getitem__(self, indx):
        return self.graphs1[indx], self.graphs2[indx], self.graphs3[indx], self.residue_graph[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)

def metrics_reg(targets,predicts):
    rmse = np.sqrt(mean_squared_error(targets, predicts))
    mae = mean_absolute_error(y_true=targets,y_pred=predicts)
    r = pearsonr(targets, predicts)[0]
    r2 = r2_score(targets, predicts)

    x = [ [item] for item in predicts]
    lr = LinearRegression()
    lr.fit(X=x,y=targets)
    y_ = lr.predict(x)
    sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5
    ci = CI(targets,predicts)
    return mae,rmse,r,sd,r2,ci


def run_a_eval_epoch(model, validation_dataloader, device, vae_model):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg1, bg2, bg3, bg4, Ys, keys = batch
            bg1, bg2, bg3, bg4, Ys = bg1.to(device), bg2.to(device), bg3.to(device), bg4.to(device), Ys.to(device)
            prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer, bg4).to(device)
            outputs = model(bg1, bg2, bg3, prot_embed)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key


if __name__ == '__main__':
    saved_model = './model_save/bestmodel'
    test_dataset = GraphSet(graph_ls_path='./data/binding_affinity/test2016/graph_ls_path')
    result_dir = './result'
    check_writable(result_dir, False)

    print('the number of test data:', len(test_dataset))    

    vae_path = os.path.join(saved_model,'VAE','vae_model.ckpt')
    dta_path = os.path.join(saved_model,'DTI','dti_model.pth')
    configs = load_config(os.path.join(saved_model,'DTI','config.yaml'))

    seed = configs['seed']
    gpuid = configs['gpuid']
    batch_size = configs['batch_size']
    num_workers = configs['num_workers']

    # todo load vae model
    vae_model = CodeBook(configs).to(device)
    vae_model.load_state_dict(torch.load(vae_path))

    stat_res = []
    set_random_seed(seed)

    # model
    DTIModel = DTIPredictor(param=configs)
    print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
    DTIModel.to(device)
    DTIModel.load_state_dict(torch.load(dta_path)['model_state_dict'])


    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn_v2, drop_last=False)

    test_true, test_pred, test_keys = run_a_eval_epoch(DTIModel, test_dataloader, device,vae_model)


    
    test_true = np.concatenate([np.array(sub).flatten() for sub in test_true])
    test_pred = np.concatenate([np.array(sub).flatten() for sub in test_pred])
    test_keys = np.concatenate([np.array(sub).flatten() for sub in test_keys])


    pd_te = pd.DataFrame(
        {'key': test_keys, 'test_true': test_true, 'test_pred': test_pred})


    pd_te.to_csv('./result/test.csv', index=False)


    test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
        r2_score(test_true, test_pred), \
        mean_absolute_error(test_true, test_pred), \
        pearsonr(test_true, test_pred)

    print('***best model***')
    mae,rmse,rp,sd,r2,ci = metrics_reg(test_true, test_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Pearson correlation coefficient:", rp)
    print("Standard Deviation:", sd)
    print("R2 score:", r2)
    print("CI score:", ci)


    print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f\t test_ci:%.4f\t test_sd:%.4f" % (
        test_rmse, test_r2, test_mae, test_rp[0],ci,sd))

    stat_res.append(['test', test_rmse, test_r2, test_mae, test_rp[0],ci,sd])

    stat_res_pd = pd.DataFrame(stat_res, columns=['group', 'rmse', 'r2', 'mae', 'rp','ci','sd'])
    stat_res_pd.to_csv('./result/res.csv',index=False)
    