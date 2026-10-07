from torch.utils.data import DataLoader, Subset
from data.dataset import AlignedDataset


class AlignedDatasetLoader:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = AlignedDataset(opt)
        print("dataset {} was created".format(self.dataset.name))
        if self.opt.max_dataset_size < len(self.dataset):
            self.dataset = Subset(self.dataset, range(max(0, int(self.opt.max_dataset_size))))
        if len(self.dataset) == 0:
            raise ValueError('No image pairs selected; check --max_dataset_size.')
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=self.opt.num_workers,
        )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def name():
        return "AlignedDatasetLoader"

    def load_data(self):
        return self.dataloader
