from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, vgg_preprocess, get_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import os
import pytorch_ssim


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_b = AdaINGen(
            hyperparameters['input_dim_b'],
            hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_b = MsImageDis(
            hyperparameters['input_dim_b'],
            hyperparameters['new_size'],
            hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        self.reg_param = hyperparameters['reg_param']
        self.beta_step = hyperparameters['beta_step']
        self.target_kl = hyperparameters['target_kl']
        self.gan_type = hyperparameters['gan_type']

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_b.parameters())
        gen_params = list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=lr, betas=(
            beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=lr, betas=(
            beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.gen_b.apply(weights_init(hyperparameters['init']))
        self.dis_b.apply(weights_init('gaussian'))

        # SSIM Loss
        self.ssim_loss = pytorch_ssim.SSIM()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_l1(self, input, target, mask):
        return torch.sum(torch.abs(input - target)) / torch.sum(mask)

    def forward(self, x_a, x_b):
        self.eval()
        s_b = self.gen_b.enc_style(x_b)
        c_a = self.gen_b.enc_content(x_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab

    def gen_update(self, x_a, x_b, hyperparameters):
        toogle_grad(self.dis_b, False)
        toogle_grad(self.gen_b, True)
        self.dis_b.train()
        self.gen_b.train()
        self.gen_opt.zero_grad()
        s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
        # encode
        c_a = self.gen_b.enc_content(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        x_ab.requires_grad_()
        # reconstruction loss
        self.loss_gen_recon_x_ab_ssim = - self.ssim_loss.forward(x_a, x_ab)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        # GAN loss
        _, _, d_fake = self.dis_b(x_ab)
        # d_fake = d_fake['out']
        self.loss_gen_adv_b = self.compute_loss(d_fake, 1)
        # total loss
        self.loss_gen_total = self.loss_gen_adv_b + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
            hyperparameters['recon_x_ab'] * self.loss_gen_recon_x_ab_ssim
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) -
             self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_ab = []
        s_b = self.gen_b.enc_style(x_b)
        for i in range(x_a.size(0)):
            c_a = self.gen_b.enc_content(x_a[i].unsqueeze(0))
            x_ab.append(self.gen_b.decode(c_a, s_b))
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_ab

    def dis_update(self, x_a, x_b, hyperparameters):
        toogle_grad(self.gen_b, False)
        toogle_grad(self.dis_b, True)
        self.gen_b.train()
        self.dis_b.train()
        self.dis_opt.zero_grad()

        # On real data
        x_b.requires_grad_()
        d_real_dict = self.dis_b(x_b)
        d_real = d_real_dict[2]
        dloss_real = self.compute_loss(d_real, 1)
        reg = 0.
        # Both grad penal and vgan!
        dloss_real.backward(retain_graph=True)
        # hard coded 10 weight for grad penal.
        reg += 10. * compute_grad2(d_real, x_b).mean()
        mu = d_real_dict[0]
        logstd = d_real_dict[1]
        kl_real = kl_loss(mu, logstd).mean()

        # On fake data
        with torch.no_grad():
            s_b = torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda()
            c_a = self.gen_b.enc_content(x_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        x_ab.requires_grad_()
        d_fake_dict = self.dis_b(x_ab)
        d_fake = d_fake_dict[2]
        dloss_fake = self.compute_loss(d_fake, 0)
        dloss_fake.backward(retain_graph=True)
        mu_fake = d_fake_dict[0]
        logstd_fake = d_fake_dict[1]
        kl_fake = kl_loss(mu_fake, logstd_fake).mean()
        avg_kl = 0.5 * (kl_real + kl_fake)
        reg += self.reg_param * avg_kl
        reg.backward()

        self.update_beta(avg_kl)
        self.dis_opt.step()

        self.loss_dis_total = (dloss_real + dloss_fake)
        return self.loss_dis_total.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def update_beta(self, avg_kl):
        with torch.no_grad():
            new_beta = self.reg_param - self.beta_step * \
                (self.target_kl - avg_kl)  # self.target_kl is constrain I_c,
            new_beta = max(new_beta, 0)
            # print('setting beta from %.2f to %.2f' % (self.reg_param, new_beta))
            self.reg_param = new_beta

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(
            self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(
            self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % iterations)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % iterations)
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)


# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def kl_loss(mu, logstd):
    # mu and logstd are b x k x d x d
    # make them into b*d*d x k

    dim = mu.shape[1]
    mu = mu.permute(0, 2, 3, 1).contiguous()
    logstd = logstd.permute(0, 2, 3, 1).contiguous()
    mu = mu.view(-1, dim)
    logstd = logstd.view(-1, dim)

    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std**2 + mu**2), dim=-1) - (0.5 * dim)

    return kl


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

