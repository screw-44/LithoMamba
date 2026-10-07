

import torch
import torch.nn as nn

from model.mamba.networks.vision_mamba import MambaUnet

from model.network_module import NLayerDiscriminator, GANLoss, VanillaDiscriminator

class MambaGAN(nn.Module):
    def __init__(self, opt):
        super(MambaGAN, self).__init__()
        self.opt = opt


        self.net_g = MambaUnet(opt.load_size, 1).to(opt.device)
        if opt.is_train and not opt.continue_train and opt.pretrained_path:
            self.net_g.load_from(opt.pretrained_path)

        if opt.is_train:
            # self.net_d = MambaD(opt, input_size=opt.load_size, in_chans=2).to(opt.device) # This is for MambaDiscriminator
            self.net_d = NLayerDiscriminator(input_nc=2).to(opt.device) # This is for region_loss/normal etc.
            # self.net_d = VanillaDiscriminator(opt.load_size).to(opt.device)             # valila discriminator for baseline compare
            # 判别器先用之前？ 对，然后抄 一下Pix2PixHD_newloss实现一下新的 网络结构（未来考虑下把discriminator也换成Vmamba）
            self.criterionGAN = GANLoss(opt, use_lsgan=True).to(opt.device) # use_lsgan设置为False，使用BCELoss，MSELoss生成太平滑了
            self.criterionL1 = nn.L1Loss().to(opt.device)

            self.CriterionCos = nn.CosineSimilarity().to(opt.device)
            self.CriterionKL = nn.KLDivLoss().to(opt.device)

            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def discriminate(self, layout, sem):
        # note: for discriminator, sem must be sem.detach()
        return self.net_d(torch.cat((layout, sem), 1))

    def forward(self, layout):
        return self.net_g(layout)

    def backward_g(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem)
        loss_g_fakeIsTrue = self.criterionGAN(pred_fake, True)

        loss_g_l1 = self.criterionL1(fake_sem, sem)
        return loss_g_fakeIsTrue + loss_g_l1 * self.opt.lambda_l1

    def backward_d(self, layout, fake_sem, sem):
        pred_fake = self.discriminate(layout, fake_sem.detach())
        loss_d_fakeIsFake = self.criterionGAN(pred_fake, False)

        pred_true = self.discriminate(layout, sem)
        loss_d_trueIsTrue = self.criterionGAN(pred_true, True)

        return (loss_d_fakeIsFake + loss_d_trueIsTrue) * 0.1

