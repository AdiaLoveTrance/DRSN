from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='./outputs/camelyon17AtB/config.yaml',
    help="net configuration")
parser.add_argument(
    '--input',
    type=str,
    default='./datasets/CAMELYON17/testB/',
    help="input images path")
parser.add_argument(
    '--sc_name',
    type=str,
    default='./datasets/CAMELYON17/testB_sc.pt',
    help="average style code name"
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default='./outputs/camelyon17AtB/checkpoints/gen_00000001.pt',
    help="checkpoint of autoencoders")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
style_dim = config['gen']['style_dim']
trainer = MUNIT_Trainer(config)

state_dict = torch.load(opts.checkpoint)
# trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
style_encode = trainer.gen_b.enc_style

new_size = config['new_size']
imgs_num = len(os.listdir(opts.input))
# Dataset loader
batch_size = 16
test_loader = get_data_loader_folder(
    opts.input,
    batch_size,
    False,
    new_size,
    config['crop_image_height'],
    config['crop_image_width'],
    config['num_workers'],
    True,
    return_path=True)

style_code = Variable(torch.zeros((1, style_dim, 1, 1)).cuda())
print(style_code.size())
with torch.no_grad():
    for i, (test_images, test_images_path) in enumerate(test_loader):
        with torch.no_grad():
            test_images = test_images.cuda()
            style = style_encode(test_images)
            style = torch.sum(style, dim=0, keepdim=True)
            print(style.shape)
            style_code += (style / imgs_num)

print(style_code)
torch.save(style_code, opts.sc_name)