"""Train the released Mamba generator and local convolutional discriminator."""

from pathlib import Path
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from data.data_loader import AlignedDatasetLoader
from options.train_options import TrainOptions
from util.util import load_network, save_network


def main():
    opt = TrainOptions().parse()
    if opt.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('LithoMamba requires an NVIDIA CUDA GPU and mamba-ssm selective-scan kernels.')
    torch.cuda.set_device(opt.gpu_ids)
    if opt.epochs < 1 or opt.display_freq < 1:
        raise ValueError('--epochs and --display_freq must be positive.')
    if opt.lambda_l1 < 0 or opt.lambda_edge < 0:
        raise ValueError('Loss weights must be non-negative.')
    # Parse --help before importing optional CUDA extensions.
    from model.mamba.mamba_gan import MambaGAN
    data_loader = AlignedDatasetLoader(opt)
    dataset = data_loader.load_data()
    print('# training dataset size: {}'.format(len(data_loader)))
    model = MambaGAN(opt).to(opt.device)
    model.train()
    experiment_path = Path(opt.checkpoints_dir) / opt.name
    iter_path = experiment_path / 'iter.txt'
    start_epoch = 1
    if opt.continue_train:
        # This is a model-weight warm start; optimizer/scaler state is not stored.
        load_network(model.net_g, str(experiment_path), 'G', opt.which_epoch)
        load_network(model.net_d, str(experiment_path), 'D', opt.which_epoch)
        if opt.which_epoch == 'latest':
            if not iter_path.is_file():
                raise FileNotFoundError('Latest warm start requires {}'.format(iter_path))
            start_epoch = int(np.loadtxt(iter_path, delimiter=',', dtype=int)[0])
        else:
            start_epoch = int(opt.which_epoch) + 1
    scaler = torch.amp.GradScaler('cuda', enabled=opt.fp16)
    for epoch in range(start_epoch, opt.epochs + 1):
        for data in tqdm(dataset, desc='Epoch {}'.format(epoch)):
            layout, sem = data['layout'].to(opt.device), data['sem'].to(opt.device)
            for parameter in model.net_d.parameters():
                parameter.requires_grad = True
            model.optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.fp16):
                fake_sem = model(layout)
                d_loss = model.backward_d(layout, fake_sem, sem)
            scaler.scale(d_loss).backward()
            scaler.step(model.optimizer_d)
            for parameter in model.net_d.parameters():
                parameter.requires_grad = False
            model.optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=opt.fp16):
                g_loss = model.backward_g(layout, fake_sem, sem)
                if opt.lambda_edge:
                    edge_fake = torch.where(fake_sem >= opt.edge_threshold, fake_sem, 0)
                    edge_real = torch.where(sem >= opt.edge_threshold, sem, 0)
                    g_loss = g_loss + opt.lambda_edge * model.criterionL1(edge_fake, edge_real)
            scaler.scale(g_loss).backward()
            scaler.step(model.optimizer_g)
            scaler.update()
        print('Epoch: {} has g_loss: {:.6f}, d_loss: {:.6f}'.format(
            epoch, g_loss.item(), d_loss.item()))
        if epoch % opt.display_freq == 0 or epoch == opt.epochs:
            for label, network in [('G', model.net_g), ('D', model.net_d)]:
                save_network(network, str(experiment_path), label, epoch)
                save_network(network, str(experiment_path), label, 'latest')
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
            previews = torch.cat((layout, fake_sem.detach(), sem), dim=0)
            grid = torchvision.utils.make_grid((previews + 1) / 2, nrow=layout.shape[0])
            torchvision.utils.save_image(grid, experiment_path / ('epoch_{}.png'.format(epoch)))


if __name__ == '__main__':
    main()
