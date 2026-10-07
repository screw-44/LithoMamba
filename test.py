"""Generate SEM predictions from a trained LithoMamba generator checkpoint."""

from pathlib import Path
import torch
import torchvision
from tqdm import tqdm
from data.data_loader import AlignedDatasetLoader
from options.test_options import TestOptions
from util.util import load_network


def main():
    opt = TestOptions().parse(save=False)
    if opt.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('LithoMamba requires an NVIDIA CUDA GPU and mamba-ssm selective-scan kernels.')
    torch.cuda.set_device(opt.gpu_ids)
    from model.mamba.mamba_gan import MambaGAN
    dataset = AlignedDatasetLoader(opt).load_data()
    model = MambaGAN(opt).to(opt.device)
    load_network(model.net_g, str(Path(opt.checkpoints_dir) / opt.name), 'G', opt.which_epoch)
    model.eval()
    image_id = 0
    with torch.inference_mode():
        for data in tqdm(dataset):
            if image_id >= opt.num_test:
                break
            predictions = model(data['layout'].to(opt.device)).cpu()
            for prediction, source in zip(predictions, data['path']):
                if image_id >= opt.num_test:
                    break
                relative = Path(source).relative_to(opt.layout_image_dir).with_suffix('.png')
                destination = Path(opt.results_dir) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image((prediction + 1) / 2, destination)
                image_id += 1


if __name__ == '__main__':
    main()
