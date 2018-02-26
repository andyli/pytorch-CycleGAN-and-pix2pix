import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageDraw
import PIL
from pdb import set_trace as st
import random, re

def get_image_paths(source_images_dir):
    paths = []
    for f in os.listdir(source_images_dir):
        fpath = os.path.join(source_images_dir, f)
        if not os.path.isdir(fpath):
            continue
        for img in os.listdir(fpath):
            name, ext = os.path.splitext(img)
            if ext != ".png":
                continue
            imgf = os.path.join(fpath, img)
            paths.append(imgf)
    return paths

class TCLFacesDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.paths = get_image_paths(self.root)

        self.paths = sorted(self.paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        index = index % len(self.paths)
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return {'A': img, 'B': img,
                'A_paths': path, 'B_paths': path}

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'TCLFacesDataset'
