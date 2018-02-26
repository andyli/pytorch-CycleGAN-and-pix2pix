import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html, util

source_images_dir = "/media/andy/Data/Downloads/第一批处理后的840"
#source_images_dir = "/media/andy/Data/Downloads/第二批处理后的160"
dest_images_dir = os.path.join(os.path.dirname(source_images_dir), os.path.basename(source_images_dir) + "_genglasses")

if not os.path.exists(dest_images_dir):
    os.makedirs(dest_images_dir)

class TCLFacesVisualizer(Visualizer):
    def save_images(self, webpage, visuals, image_path):
        short_path = os.path.relpath(image_path[0], source_images_dir)

        webpage.add_header(short_path)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            if label in ['real_B', 'fake_A']:
                path = os.path.join(label, short_path)
                save_path = os.path.join(dest_images_dir, webpage.get_image_dir(), path)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                util.save_image(image_numpy, save_path)

                ims.append(path)
                txts.append(label)
                links.append(path)
        webpage.add_images(ims, txts, links, width=self.win_size)



opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = source_images_dir


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = TCLFacesVisualizer(opt)
# create website
web_dir = dest_images_dir
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= len(dataset):
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... {} {}'.format(i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
