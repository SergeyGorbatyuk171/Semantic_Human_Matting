import os
import os.path as osp
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('imdir', type=str, help='Directory with images')
parser.add_argument('maskdir', type=str, help='Directory with masks')


def filter(src, dst, dst_ext='png'):
    dst_files = set(os.listdir(dst))
    for f in os.listdir(src):
        fname = f.rsplit('.', maxsplit=1)[0]
        if (f'{fname}.{dst_ext[:3]}' not in dst_files) and (f'{fname}.{dst_ext[3:]}' not in dst_files):
            os.remove(osp.join(src, f))


args = parser.parse_args()
filter(args.imdir, args.maskdir, dst_ext='pngPNG')
filter(args.maskdir, args.imdir, dst_ext='jpgJPG')
