import os
import os.path as osp
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', type=str, help='path to a folder to be flattened')

args = parser.parse_args()
folders = os.listdir(args.folder)
for folder in tqdm(folders):
    tqdm.write(f'Processing {folder}')
    for subfolder in os.listdir(osp.join(args.folder, folder)):
        for image in os.listdir(osp.join(args.folder, folder, subfolder)):
            os.rename(osp.join(args.folder, folder, subfolder, image), osp.join(args.folder, image))
    os.remove(osp.join(args.folder, folder))
