# @Time    : 2024/7/15
# @Author  : rylynn
# @Email   : 
# @File    : train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from dataset.graph_constructor import *
from utils.utils import *
from models.model import *
from models.codebook import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import torch
import pandas as pd
import torch.backends.cudnn
from dgl.data.utils import split_dataset
import matplotlib.pyplot as plt
# import nni

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

path_marker = '/'
limit = None
num_process = 48
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_list = []

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def pretrain_vae(vae_config, device):
    residue_set = load_pretrain_data(protein_dir='./data/all_protein_files', protein_id='./data/all_protein.csv')
    output_dir = "./model_save/{}/VAE/".format(timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.yaml"), 'a+') as tf:
        yaml.dump(vae_config, tf)
    vae_dataloader = DataLoader(residue_set, batch_size=vae_config['CB_batch_size'], shuffle=True, collate_fn=collate)
    vae_model = CodeBook(vae_config).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=float(vae_config['learning_rate']),
                                     weight_decay=float(vae_config['weight_decay']))

    for epoch in range(1, vae_config["pre_epoch"] + 1):
        for iter_num, batch_graph in enumerate(vae_dataloader):

            batch_graph = batch_graph.to(device)

            z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
            loss_vae = e_q_loss + recon_loss + mask_loss * vae_config['mask_loss']

            vae_optimizer.zero_grad()
            loss_vae.backward()
            vae_optimizer.step()

            if (epoch - 1) % vae_config['log_num'] == 0 and iter_num == 0:
                print(
                    "\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.write(
                    "Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(
                        epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.flush()

    torch.save(vae_model.state_dict(), os.path.join(output_dir, f'vae_model.ckpt'))

    del vae_model
    torch.cuda.empty_cache()

def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device, vae_model):
    loss_epoch = 0
    n = 0
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg1, bg2, bg3, bg4, Ys, keys = batch
        bg1, bg2, bg3, bg4, Ys = bg1.to(device), bg2.to(device), bg3.to(device), bg4.to(device), Ys.to(device)
        prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer, bg4).to(device)
        outputs = model(bg1, bg2, bg3, prot_embed)
        loss = loss_fn(outputs, Ys)
        loss_epoch += loss.item()

        loss.backward()
        optimizer.step()
        n += 1
    loss_list.append(loss_epoch / n)
    print('epoch:', epoch, ' loss:', loss_epoch / n)


def run_a_eval_epoch(model, validation_dataloader, device, vae_model):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            bg1, bg2, bg3, bg4, Ys, keys = batch
            bg1, bg2, bg3, bg4, Ys = bg1.to(device), bg2.to(device), bg3.to(device), bg4.to(device), Ys.to(device)
            prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer, bg4).to(device)
            outputs = model(bg1, bg2, bg3, prot_embed)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_config_path', type=str, default='./configs/config.yaml')
    argparser.add_argument('--ckpt_path', type=str, default=None)
    args = argparser.parse_args()

    configs = load_config(args.model_config_path)
    # nni get_next_parameters
    # configs.update(nni.get_next_parameters())

    seed = configs['seed']
    # load train config
    gpuid = configs['gpuid']
    lr = configs['lr']
    epochs = configs['epoches']
    batch_size = configs['batch_size']
    num_workers = configs['num_workers']
    model_save_dir = configs['save_dir']
    l2 = configs['l2']

    # early stop config
    patience = configs['patience']
    tolerance = configs['tolerance']
    repetitions = configs['repetitions']


    all_data = pd.read_csv('./data/labels_gign12904.csv')
    test_data = pd.read_csv('./data/labels_casf2016.csv')
    check_writable(model_save_dir, False)
    check_writable('./stats', False)

    # data
    train_dir = './data/binding_affinity/train/complex'
    valid_dir = './data/binding_affinity/validation/complex'
    test_dir = './data/binding_affinity/test2016/complex'

    # training data
    train_keys = os.listdir(train_dir)
    train_labels = []
    train_data_dirs = []
    valid_train_keys = []
    for key in train_keys:
        if len(all_data[all_data['id'] == key]) == 0: continue
        valid_train_keys.append(key)
        train_labels.append(
            all_data[all_data['id'] == key]['affinity'].values[0])
        train_data_dirs.append(train_dir + path_marker + key)
    train_keys = valid_train_keys

    # validtion data
    valid_keys = os.listdir(valid_dir)
    valid_labels = []
    valid_data_dirs = []
    valid_valid_keys = []
    for key in valid_keys:
        if len(all_data[all_data['id'] == key]) == 0: continue
        valid_valid_keys.append(key)
        valid_labels.append(
            all_data[all_data['id'] == key]['affinity'].values[0])
        valid_data_dirs.append(valid_dir + path_marker + key)
    valid_keys = valid_valid_keys

    # testing data
    test_keys = os.listdir(test_dir)
    test_labels = []
    test_data_dirs = []
    valid_test_keys = []
    for key in test_keys:
        if len(test_data[test_data['id'] == key]) == 0: continue
        valid_test_keys.append(key)
        test_labels.append(
            test_data[test_data['id'] == key]['affinity'].values[0])
        test_data_dirs.append(test_dir + path_marker + key)
    test_keys = valid_test_keys

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    # device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
    set_random_seed(seed)
    vae_model = CodeBook(configs).to(device)
    if args.ckpt_path is None:
        pretrain_vae(vae_config=configs, device=device)
        vae_model.load_state_dict(
            torch.load(os.path.join("./model_save/{}/VAE/".format(timestamp), f'vae_model.ckpt'), map_location=device))
    else:
        vae_model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    # generating the graph objective using multi process
    train_dataset = GraphsDataset(keys=train_keys[:limit], labels=train_labels[:limit],
                                         data_dirs=train_data_dirs[:limit],
                                         graph_ls_path='./data/binding_affinity/train/graph_ls_path',
                                         graph_dic_path='./data/binding_affinity/train/graph_dic_path',
                                         residue_dirs='./data/binding_affinity/processed_data/residue_dic_path',
                                         num_process=num_process, path_marker=path_marker)

    valid_dataset = GraphsDataset(keys=valid_keys[:limit], labels=valid_labels[:limit],
                                         data_dirs=valid_data_dirs[:limit],
                                         graph_ls_path='./data/binding_affinity/validation/graph_ls_path',
                                         graph_dic_path='./data/binding_affinity/validation/graph_dic_path',
                                         residue_dirs='./data/binding_affinity/processed_data/residue_dic_path',
                                         num_process=num_process, path_marker=path_marker)
    test_dataset = GraphsDataset(keys=test_keys[:limit], labels=test_labels[:limit],
                                        data_dirs=test_data_dirs[:limit],
                                        graph_ls_path='./data/binding_affinity/test2016/graph_ls_path',
                                        graph_dic_path='./data/binding_affinity/test2016/graph_dic_path',
                                        residue_dirs='./data/binding_affinity/processed_data/residue_dic_path',
                                        num_process=num_process, path_marker=path_marker)

    stat_res = []
 
    print('the number of train data:', len(train_dataset))
    print('the number of valid data:', len(valid_dataset))
    print('the number of test data:', len(test_dataset))
    train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                    collate_fn=collate_fn_v2, drop_last=False)
    valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                    collate_fn=collate_fn_v2, drop_last=False)

    # model
    DTIModel = DTIPredictor(param=configs)

    print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
    print(DTIModel)
    DTIModel.to(device)
    optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)
    dti_model_dir = './model_save/{}/DTI/'.format(timestamp)
    check_writable(dti_model_dir)
    stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=f"{dti_model_dir}dti_model.pth")
    loss_fn = nn.MSELoss()
    # save train config
    with open(os.path.join(dti_model_dir, "config.yaml"), 'a+') as tf:
        yaml.dump(configs, tf)


    for epoch in range(epochs):
        st = time.time()
        # train
        run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device, vae_model)

        # validation
        train_true, train_pred, _ = run_a_eval_epoch(DTIModel, train_dataloader, device, vae_model)
        valid_true, valid_pred, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device, vae_model)

        train_true_flatten = np.concatenate([np.array(sub).flatten() for sub in train_true])
        train_pred_flatten = np.concatenate([np.array(sub).flatten() for sub in train_pred])

        valid_true_flatten = np.concatenate([np.array(sub).flatten() for sub in valid_true])
        valid_pred_flatten = np.concatenate([np.array(sub).flatten() for sub in valid_pred])


        train_rmse = np.sqrt(mean_squared_error(train_true_flatten, train_pred_flatten))
        valid_rmse = np.sqrt(mean_squared_error(valid_true_flatten, valid_pred_flatten))
        # nni.report_intermediate_result(valid_rmse)
        early_stop = stopper.step(valid_rmse, DTIModel)
        end = time.time()
        if early_stop:
            break
        print(
            "epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (
                epoch, train_rmse, valid_rmse, end - st))
        
        f_log = open(file=(dti_model_dir+"log.txt"), mode="a")
        str_log = 'epoch:' + str(epoch) + ' train_rmse: ' + str(train_rmse) + ' val_rmse: ' + str(valid_rmse)+'\n'
        f_log.write(str_log)
        f_log.close()
    # nni.report_final_result(valid_rmse)

    plt.plot(loss_list)
    plt.ylabel('Loss')
    plt.xlabel("time")
    plt.savefig(dti_model_dir+'loss.png')
    # plt.show()


    # load the best model
    stopper.load_checkpoint(DTIModel)
    train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                   collate_fn=collate_fn_v2, drop_last=False)
    valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                   collate_fn=collate_fn_v2, drop_last=False)
    test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=collate_fn_v2, drop_last=False)
    train_true, train_pred, train_keys = run_a_eval_epoch(
        DTIModel, train_dataloader, device,vae_model)
    valid_true, valid_pred, valid_keys = run_a_eval_epoch(
        DTIModel, valid_dataloader, device,vae_model)
    test_true, test_pred, test_keys = run_a_eval_epoch(
        DTIModel, test_dataloader, device,vae_model)


    train_true = np.concatenate(
        [np.array(sub).flatten() for sub in train_true])
    train_pred = np.concatenate(
        [np.array(sub).flatten() for sub in train_pred])
    train_keys = np.concatenate(
        [np.array(sub).flatten() for sub in train_keys])
    
    valid_true = np.concatenate(
        [np.array(sub).flatten() for sub in valid_true])
    valid_pred = np.concatenate(
        [np.array(sub).flatten() for sub in valid_pred])
    valid_keys = np.concatenate(
        [np.array(sub).flatten() for sub in valid_keys])
    
    test_true = np.concatenate(
        [np.array(sub).flatten() for sub in test_true])
    test_pred = np.concatenate(
        [np.array(sub).flatten() for sub in test_pred])
    test_keys = np.concatenate(
        [np.array(sub).flatten() for sub in test_keys])

    pd_tr = pd.DataFrame(
        {'key': train_keys, 'train_true': train_true, 'train_pred': train_pred})
    pd_va = pd.DataFrame(
        {'key': valid_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
    pd_te = pd.DataFrame(
        {'key': test_keys, 'test_true': test_true, 'test_pred': test_pred})

    pd_tr.to_csv('./stats/{}_trin.csv'.format(timestamp), index=False)
    pd_va.to_csv('./stats/{}_val.csv'.format(timestamp), index=False)
    pd_te.to_csv('./stats/{}_test.csv'.format(timestamp), index=False)
    train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
        r2_score(train_true, train_pred), \
        mean_absolute_error(train_true, train_pred), \
        pearsonr(train_true, train_pred)
    valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(mean_squared_error(valid_true, valid_pred)), \
        r2_score(valid_true, valid_pred), \
        mean_absolute_error(valid_true, valid_pred), \
        pearsonr(valid_true, valid_pred)
    test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
        r2_score(test_true, test_pred), \
        mean_absolute_error(test_true, test_pred), \
        pearsonr(test_true, test_pred)

    print('***best model***')
    print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
        train_rmse, train_r2, train_mae, train_rp[0]))
    print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
        valid_rmse, valid_r2, valid_mae, valid_rp[0]))
    print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
        test_rmse, test_r2, test_mae, test_rp[0]))
    stat_res.append(['train', train_rmse, train_r2, train_mae, train_rp[0]])
    stat_res.append(['valid', valid_rmse, valid_r2, valid_mae, valid_rp[0]])
    stat_res.append(['test', test_rmse, test_r2, test_mae, test_rp[0]])

    stat_res_pd = pd.DataFrame(stat_res, columns=['group', 'rmse', 'r2', 'mae', 'rp'])
    stat_res_pd.to_csv('./stats/{}_metrics.csv'.format(timestamp),index=False)
   