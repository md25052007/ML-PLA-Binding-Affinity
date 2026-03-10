import os
import sys
from rdkit import Chem
import pickle
import argparse
import yaml
from multiprocessing import Pool
from itertools import repeat
import shutil
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config



def process_complex(folder_name, source_folder, target_folder):
    compound_path = os.path.join(source_folder, folder_name)

    if os.path.isdir(compound_path):
        ligand_file = os.path.join(compound_path, f"{folder_name}_ligand.sdf")
        pocket_file = os.path.join(compound_path, f"{folder_name}_pocket.pdb")

        ligand = Chem.MolFromMolFile(ligand_file)
        pocket = Chem.MolFromPDBFile(pocket_file)

        if ligand is None or pocket is None:
            print(f"Error reading files in folder {folder_name}. Skipping.")
            return

        complex_file_name = os.path.join(target_folder, folder_name)
        with open(complex_file_name, 'wb') as complex_file:
            pickle.dump([ligand, pocket], complex_file)


def generate_complexs(mol_path, complex_path, num_processes=12):
    print("Start processing complexes.")

    if not os.path.exists(complex_path):
        os.makedirs(complex_path)

    folder_names = os.listdir(mol_path)
    num_folders = len(folder_names)
    args = zip(folder_names, repeat(mol_path, num_folders),
               repeat(complex_path, num_folders))

    with Pool(num_processes) as pool:
        pool.starmap(process_complex, args)

    print("Completed processing all complexes.")


def split_set(set_path, train_path, val_path):
    # 定义源目录和目标目录
    source_dir = set_path
    # train_dir = './data/binding_affinity/test/complex'
    # val_dir = './data/binding_affinity/validation/complex'
    train_dir = train_path
    validation_dir = val_path

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if os.path.isfile(
        os.path.join(source_dir, f))]

    random.shuffle(files)
    split_index = int(0.9 * len(files))
    train_files = files[:split_index]
    validation_files = files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

    for f in validation_files:
        shutil.copy(os.path.join(source_dir, f),
                    os.path.join(validation_dir, f))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--complex_config_path',
                           type=str, default='./configs/complex.yaml')
    args = argparser.parse_args()

    complex_config = load_config(args.complex_config_path)
    # dataset path
    # train_path = complex_config['train_set_path']
    # test2016_path = complex_config['test_set2016_path']

    train_complexs = complex_config['train_complex_path']
    # test2016_complexs = complex_config['test2016_complex_path']

    # print("process train set")
    # generate_complexs(train_path, train_complexs)
    # print(20 * "=")
    # print("process test 2016 set")
    # generate_complexs(test2016_path, test2016_complexs)

    # split data set
    print("split data set")
    # print("split data set")
    train_dir = complex_config['train_dir']
    val_dir = complex_config['valid_dir']
    split_set(train_complexs, train_dir, val_dir)
    print(len(os.listdir(train_dir)))
    print(len(os.listdir(val_dir)))

