"""Load grayscale layout/SEM pairs with shared spatial augmentation."""

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision import transforms
from torchvision.transforms import functional as TF


def make_dataset(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError('Image directory does not exist: {}'.format(root))
    return sorted(path for path in root.rglob('*')
                  if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS)


def index_images(directory):
    root = Path(directory)
    indexed = {}
    for path in make_dataset(directory):
        key = path.relative_to(root).with_suffix('').as_posix()
        if key in indexed:
            raise ValueError('Duplicate image pairing key: {}'.format(key))
        indexed[key] = path
    return indexed


class AlignedDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        layouts = index_images(opt.layout_image_dir)
        sems = index_images(opt.sem_image_dir)
        if not layouts:
            raise ValueError('No layout/SEM image pairs found.')
        if layouts.keys() != sems.keys():
            missing_sem = sorted(layouts.keys() - sems.keys())[:5]
            missing_layout = sorted(sems.keys() - layouts.keys())[:5]
            raise ValueError('Image pairs must match relative filenames without extensions. '
                             'Missing SEM: {}; missing layout: {}'.format(missing_sem, missing_layout))
        if opt.load_size < 1 or opt.crop_size < 1:
            raise ValueError('Image sizes must be positive.')
        if 'crop' in opt.resize_or_crop and opt.crop_size > opt.load_size:
            raise ValueError('--crop_size must not exceed --load_size.')
        if not 0 <= opt.flip_ratio <= 1:
            raise ValueError('--flip_ratio must be between 0 and 1.')
        self.pairs = [(layouts[key], sems[key]) for key in sorted(layouts)]

    def __getitem__(self, index):
        layout_path, sem_path = self.pairs[index]
        with Image.open(layout_path) as image:
            layout = image.convert('L')
        with Image.open(sem_path) as image:
            sem = image.convert('L')
        if layout.size != sem.size:
            raise ValueError('Paired image dimensions differ: {} and {}'.format(layout_path, sem_path))
        # Retain the released square resizing convention, then augment both images together.
        size = [self.opt.load_size, self.opt.load_size]
        layout = TF.resize(layout, size, transforms.InterpolationMode.BILINEAR)
        sem = TF.resize(sem, size, transforms.InterpolationMode.BILINEAR)
        if 'crop' in self.opt.resize_or_crop:
            if self.opt.is_train:
                parameters = transforms.RandomCrop.get_params(layout, (self.opt.crop_size,) * 2)
                layout, sem = TF.crop(layout, *parameters), TF.crop(sem, *parameters)
            else:
                layout = TF.center_crop(layout, self.opt.crop_size)
                sem = TF.center_crop(sem, self.opt.crop_size)
        if self.opt.is_train:
            if torch.rand(()).item() < self.opt.flip_ratio:
                layout, sem = TF.hflip(layout), TF.hflip(sem)
            if torch.rand(()).item() < self.opt.flip_ratio:
                layout, sem = TF.vflip(layout), TF.vflip(sem)
        layout = TF.normalize(TF.to_tensor(layout), [0.5], [0.5])
        sem = TF.normalize(TF.to_tensor(sem), [0.5], [0.5])
        return {'layout': layout, 'sem': sem, 'path': str(layout_path)}

    def __len__(self):
        return len(self.pairs)

    @property
    def name(self):
        return 'AlignedDataset'
