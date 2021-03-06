from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import torch.nn.init as init


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(
            # os.path.join(
            #     conf['data_root'],
            #     'trainA'),
            conf['trainA'],
            batch_size,
            True,
            new_size_a,
            height,
            width,
            num_workers,
            True)
        test_loader_a = get_data_loader_folder(
            # os.path.join(
            #     conf['data_root'],
            #     'testA'),
            conf['testA'],
            batch_size,
            False,
            new_size_a,
            new_size_a,
            new_size_a,
            num_workers,
            True)
        train_loader_b = get_data_loader_folder(
            # os.path.join(
            #     conf['data_root'],
            #     'trainB'),
            conf['trainB'],
            batch_size,
            True,
            new_size_b,
            height,
            width,
            num_workers,
            True)
        test_loader_b = get_data_loader_folder(
            # os.path.join(
            #     conf['data_root'],
            #     'testB'),
            conf['testB'],
            batch_size,
            False,
            new_size_b,
            new_size_b,
            new_size_b,
            num_workers,
            True)
    else:
        train_loader_a = get_data_loader_list(
            conf['data_folder_train_a'],
            conf['data_list_train_a'],
            batch_size,
            True,
            new_size_a,
            height,
            width,
            num_workers,
            True)
        test_loader_a = get_data_loader_list(
            conf['data_folder_test_a'],
            conf['data_list_test_a'],
            batch_size,
            False,
            new_size_a,
            new_size_a,
            new_size_a,
            num_workers,
            True)
        train_loader_b = get_data_loader_list(
            conf['data_folder_train_b'],
            conf['data_list_train_b'],
            batch_size,
            True,
            new_size_b,
            height,
            width,
            num_workers,
            True)
        test_loader_b = get_data_loader_list(
            conf['data_folder_test_b'],
            conf['data_list_test_b'],
            batch_size,
            False,
            new_size_b,
            new_size_b,
            new_size_b,
            num_workers,
            True)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                         height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [
        transforms.RandomCrop(
            (height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(
        new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + \
        transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers)
    return loader


def get_data_loader_folder(
        input_folder,
        batch_size,
        train,
        new_size=None,
        height=256,
        width=256,
        num_workers=4,
        crop=True,
        return_path=False):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [
        transforms.RandomCrop(
            (height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(
        new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + \
        transform_list if train else transform_list
    # ??????ColorJitter??????????????????????????????
    # transform_list = [transforms.ColorJitter(hue=0.2)] + transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(
        input_folder,
        transform=transform,
        return_paths=return_path)
    print("load ", len(dataset), " images")
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def __write_images(image_outputs, display_image_num, file_name):
    # expand gray-scale images to 3 channels
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num]
                              for images in image_outputs], 0)
    image_grid = vutils.make_grid(
        image_tensor.data,
        nrow=display_image_num,
        padding=0,
        normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    # n = len(image_outputs)
    __write_images(image_outputs, display_image_num, '%s/gen_a2b_%s.jpg' %
                   (image_directory, postfix))
    # __write_images(image_outputs[n //
    #                              2:n], display_image_num, '%s/gen_b2a_%s.jpg' %
    #                (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [
        attr for attr in dir(trainer) if not callable(getattr(trainer, attr))
                                         and not attr.startswith("__")
                                         and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    loss_gen_dir = {}
    loss_dis_dir = {}
    for m in members:
        if 'gen' in m:
            loss_gen_dir[m] = getattr(trainer, m)
        elif 'dis' in m:
            loss_dis_dir[m] = getattr(trainer, m)
    train_writer.add_scalars('Generator', loss_gen_dir, iterations + 1)
    train_writer.add_scalars('Discriminator', loss_dis_dir, iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(
            dirname,
            f) for f in os.listdir(dirname) if os.path.isfile(
            os.path.join(
                dirname,
                f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparameters['step_size'],
            gamma=hyperparameters['gamma'],
            last_epoch=iterations)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented',
            hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

