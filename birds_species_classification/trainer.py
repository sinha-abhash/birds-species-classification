from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import logging

from birds_species_classification import config
from birds_species_classification.helper import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: str,
        device: torch.device,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.model = model.to(device=device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.set_optimizer(optimizer_name=optimizer)
        self.device = device
        self.early_stopping = EarlyStopping(tolerance=3, min_delta=0.10)

    def set_optimizer(self, optimizer_name: str):
        params_to_update = []
        for _, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        if optimizer_name == "sgd":
            return SGD(params_to_update, lr=config.LEARNING_RATE, momentum=0.8)
        elif optimizer_name == "adam":
            return Adam(params_to_update, lr=config.LEARNING_RATE)

    def train(
        self, num_epochs: int, train_dataset: DataLoader, val_dataset: DataLoader
    ):
        val_acc_history, train_acc_history = [], []
        for epoch in range(num_epochs):
            start_time = timer()
            train_loss, train_acc = self.train_epoch(dataloader=train_dataset)
            end_time = timer()

            val_loss, val_acc = self.evaluate(dataloader=val_dataset)
            self.early_stopping(train_loss, val_loss)

            self.logger.info(
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, "
                f"Val loss: {val_loss:.3f}, Val Accuracy: {val_acc:.3f} "
                f"Epoch time={(end_time - start_time):.3f}"
            )
            val_acc_history.append(val_acc)
            train_acc_history.append(train_acc)

            if self.early_stopping.early_stop is True:
                self.logger.info(f"Stopping early: {val_loss}, {train_loss}, {epoch}")
                break
        return train_acc_history, val_acc_history

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                _, preds = torch.max(outputs, 1)
                self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # _, y_targets = torch.max(labels, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc
