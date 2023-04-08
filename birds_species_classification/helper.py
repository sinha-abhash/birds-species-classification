import torch
import torch.nn as nn
from torchvision import transforms

from matplotlib import pyplot as plt

from birds_species_classification import config


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((config.INPUT_IMG_SIZE, config.INPUT_IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0) -> None:
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss):
        if (val_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def calculate_test_accuracy(model: nn.modules.Module, test_data) -> float:
    model.eval()
    correct_preds = 0
    for img, label in test_data:
        img = img.to(config.DEVICE)
        label = label.to(config.DEVICE)
        output = model(img)
        _, pred = torch.max(output, 1)
        correct_preds += torch.sum(pred == label)
    return correct_preds.double() / len(test_data.dataset)


def save_plots(train_history: list, val_history: list, save_path: str, file_name: str):
    assert len(train_history) == len(val_history)
    train_history = torch.tensor(train_history, device="cpu")
    val_history = torch.tensor(val_history, device="cpu")
    x_value = list(range(len(train_history)))
    plt.plot(x_value, train_history, label="train_acc")
    plt.plot(x_value, val_history, label="val_acc")
    plt.xlabel("#Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{save_path}/{file_name}.png")
