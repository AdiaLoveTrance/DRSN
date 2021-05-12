from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images
import argparse
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
from tensorboardX import SummaryWriter
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='configs/camelyon17AtB.yaml',
    help='Path to the config file.')
parser.add_argument(
    '--output_path',
    type=str,
    default='.',
    help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True
# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
trainer = MUNIT_Trainer(config)
print(trainer.gen_b)
print(trainer.dis_b)

trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(
    config)
train_display_images_a = torch.stack(
    [train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack(
    [train_loader_b.dataset[i] for i in range(1)]).cuda()
test_display_images_a = torch.stack(
    [test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack(
    [test_loader_b.dataset[i] for i in range(1)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = SummaryWriter(
    os.path.join(opts.output_path, "logs"))
output_directory = opts.output_path
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# copy config file to output folder
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

# Start training
iterations = trainer.resume(checkpoint_directory,
                            hyperparameters=config) if opts.resume else 0
epoch = iterations if iterations!=0 else 0

while True:
    print("------Start epoch ", epoch, "------")
    for it, (images_a, images_b) in enumerate(
            zip(train_loader_a, train_loader_b)):
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, config)
        torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        trainer.update_learning_rate()
        iterations += 1

    with torch.no_grad():
        test_image_outputs = trainer.sample(
            test_display_images_a, test_display_images_b)
        train_image_outputs = trainer.sample(
            train_display_images_a, train_display_images_b)
    write_2images(
        test_image_outputs,
        display_size,
        image_directory,
        'test_%08d' %
        epoch)
    write_2images(
        train_image_outputs,
        display_size,
        image_directory,
        'train_%08d' %
        epoch)

    epoch += 1
    # Save network weights
    trainer.save(checkpoint_directory, epoch)

    if epoch >= config['epoch']:
        sys.exit('Finish training')

train_writer.close()
torch.cuda.empty_cache()
