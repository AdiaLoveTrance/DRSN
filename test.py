from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer
import argparse
import torchvision.utils as vutils
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='./outputs/camelyon17AtB/config.yaml',
    help="net configuration")
parser.add_argument(
    '--input',
    type=str,
    default='./datasets/CAMELYON17/testA/',
    help="input images path")
parser.add_argument(
    '--output-folder',
    type=str,
    default='./results/camelyon17AtB/',
    help="output image path")
parser.add_argument(
    '--checkpoint',
    type=str,
    default='./outputs/camelyon17AtB/checkpoints/gen_00000008.pt',
    help="checkpoint of autoencoders")
parser.add_argument(
    '--style',
    type=str,
    default='../datasets/CAMELYON17/trainB/patient_043_node_0_196.PNG',
    help="style image path")
parser.add_argument(
    '--sc_name',
    default='./datasets/CAMELYON17/testB_sc.pt',
    help="average style code file"
)
parser.add_argument(
    '--use_avgsc',
    action='store_true',
    default=False
)
parser.add_argument('--seed', type=int, default=10, help="random seed")

opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

# Setup model and data loader
style_dim = config['gen']['style_dim']
trainer = MUNIT_Trainer(config)



state_dict = torch.load(opts.checkpoint)
trainer.gen_b.load_state_dict(state_dict['b'])


trainer.cuda()
trainer.eval()

# decode function
content_encode = trainer.gen_b.enc_content
style_encode = trainer.gen_b.enc_style
decode = trainer.gen_b.decode

new_size = config['new_size']

# Dataset loader
batch_size = 4
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

if not opts.use_avgsc:
    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor(
        ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        style_image = transform(Image.open(opts.style).convert(
                    'RGB')).unsqueeze(0).cuda() if opts.style != '' else None
        style = style_encode(style_image)
else:
    style = torch.load(opts.sc_name)

for i, (test_images, test_images_path) in enumerate(test_loader):
    with torch.no_grad():
        test_images = test_images.cuda()
        # Start testing
        content = content_encode(test_images)

        s = torch.stack(batch_size * [style])
        outputs = decode(content, s)
        outputs = (outputs + 1) / 2.
        for i in range(outputs.size()[0]):
            test_image_name = os.path.splitext(
                os.path.basename(test_images_path[i]))[0]
            print(test_image_name)
            path = os.path.join(
                opts.output_folder,
                '{}.png'.format(test_image_name))
            vutils.save_image(
                outputs[i].data, path, padding=0, normalize=True)

torch.cuda.empty_cache()
