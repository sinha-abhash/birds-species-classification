from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from birds_species_classification import config


class DataReader:
    def __init__(self, train_path: str, validation_path: str, test_path: str) -> None:
        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path

    def get_data_loader(self, transforms=None) -> Tuple[DataLoader, DataLoader]:
        train_dataset = ImageFolder(root=self.train_path, transform=transforms)
        validation_dataset = ImageFolder(
            root=self.validation_path, transform=transforms
        )
        test_dataset = ImageFolder(root=self.test_path, transform=transforms)

        train_dataloader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
        )

        return train_dataloader, validation_dataloader, test_dataloader
