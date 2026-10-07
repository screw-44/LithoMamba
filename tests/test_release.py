from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from data.dataset import AlignedDataset
from data.data_loader import AlignedDatasetLoader
from model.network_module import GANLoss
from util.util import load_network, save_network


def dataset_options(tmp_path, **overrides):
    layout = tmp_path / 'layout'
    sem = tmp_path / 'sem'
    layout.mkdir(exist_ok=True)
    sem.mkdir(exist_ok=True)
    values = dict(layout_image_dir=str(layout), sem_image_dir=str(sem),
                  data_root=str(tmp_path), batch_size=8, load_size=32,
                  crop_size=24, resize_or_crop='resize_and_crop',
                  is_train=True, flip_ratio=0.5)
    values.update(overrides)
    return SimpleNamespace(**values)


def write_pair(opt, name='sample'):
    pixels = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    Image.fromarray(pixels).save(Path(opt.layout_image_dir) / (name + '.png'))
    Image.fromarray(pixels).save(Path(opt.sem_image_dir) / (name + '.bmp'))


def test_adversarial_loss_uses_every_sample():
    prediction = torch.tensor([0.0, 0.5], requires_grad=True).view(2, 1, 1, 1)
    prediction.retain_grad()
    loss = GANLoss(SimpleNamespace(device='cpu'))(prediction, True)
    assert loss.item() == pytest.approx(0.625)
    loss.backward()
    assert torch.all(prediction.grad != 0)


def test_adversarial_loss_supports_multiscale_outputs():
    criterion = GANLoss(SimpleNamespace(device='cpu'))
    loss = criterion([[torch.zeros(2, 1, 2, 2)], [torch.ones(2, 1, 1, 1)]], True)
    assert loss.item() == pytest.approx(1.0)


def test_small_dataset_keeps_all_samples(tmp_path):
    opt = dataset_options(tmp_path)
    write_pair(opt)
    dataset = AlignedDataset(opt)
    assert len(dataset) == 1


def test_loader_respects_dataset_limit_and_keeps_final_batch(tmp_path):
    opt = dataset_options(tmp_path, batch_size=2, max_dataset_size=3,
                          shuffle=False, num_workers=0)
    for index in range(5):
        write_pair(opt, 'sample_{}'.format(index))
    loader = AlignedDatasetLoader(opt)
    batches = list(loader.load_data())
    assert len(loader) == 3
    assert [len(batch['path']) for batch in batches] == [2, 1]


def test_inference_preprocessing_is_deterministic(tmp_path):
    opt = dataset_options(tmp_path, is_train=False)
    write_pair(opt)
    dataset = AlignedDataset(opt)
    torch.manual_seed(1)
    first = dataset[0]
    torch.manual_seed(123)
    second = dataset[0]
    assert torch.equal(first['layout'], second['layout'])
    assert torch.equal(first['layout'], first['sem'])


def test_transforms_keep_pairs_aligned_and_do_not_reseed_rng(tmp_path):
    opt = dataset_options(tmp_path)
    write_pair(opt)
    dataset = AlignedDataset(opt)
    torch.manual_seed(42)
    first = dataset[0]
    after_first = torch.get_rng_state().clone()
    torch.manual_seed(42)
    second = dataset[0]
    assert torch.equal(first['layout'], first['sem'])
    assert torch.equal(first['layout'], second['layout'])
    assert torch.equal(after_first, torch.get_rng_state())


def test_unpaired_filenames_raise_actionable_error(tmp_path):
    opt = dataset_options(tmp_path)
    write_pair(opt)
    (Path(opt.sem_image_dir) / 'sample.bmp').rename(Path(opt.sem_image_dir) / 'other.bmp')
    with pytest.raises(ValueError, match='pair|match'):
        AlignedDataset(opt)


def test_checkpoint_roundtrip_preserves_cpu_device(tmp_path):
    network = torch.nn.Linear(3, 2)
    expected = {key: value.clone() for key, value in network.state_dict().items()}
    save_network(network, str(tmp_path), 'G', 'latest')
    assert next(network.parameters()).device.type == 'cpu'
    restored = torch.nn.Linear(3, 2)
    load_network(restored, str(tmp_path), 'G', 'latest')
    assert all(torch.equal(restored.state_dict()[key], value) for key, value in expected.items())


@pytest.mark.parametrize('script', ['train_mamba.py', 'test.py'])
def test_cli_help_needs_no_dataset_or_cuda_extensions(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, script, '--help'], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--pretrained_path' in result.stdout
