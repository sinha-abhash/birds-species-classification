#!/usr/bin/env python

from argparse import ArgumentParser
import logging

import pandas as pd
import torch

from birds_species_classification.reader import DataReader
from birds_species_classification.model import ImageClassifier
from birds_species_classification.trainer import Trainer
from birds_species_classification import config
from birds_species_classification.helper import (
    get_transforms,
    calculate_test_accuracy,
    save_plots,
)

logging.basicConfig(level=logging.INFO)


def get_num_classes(birds_csv_path: str):
    birds_df = pd.read_csv(birds_csv_path)
    return birds_df["labels"].unique().shape[0]


def run(args):
    logger = logging.getLogger(__name__)
    logger.info("Load train and validation data!")
    dr = DataReader(
        train_path=args.train, validation_path=args.valid, test_path=args.test
    )
    train_dataloader, validation_dataloader, test_dataloader = dr.get_data_loader(
        transforms=get_transforms()
    )

    logger.info("Create Model")
    img_classifier = ImageClassifier(num_classes=get_num_classes(args.csv_file))
    model = img_classifier.get_model(model_name=args.model_name)
    trainer = Trainer(model=model, optimizer=args.optimizer, device=config.DEVICE)
    train_acc_history, val_acc_history = trainer.train(
        num_epochs=args.epochs,
        train_dataset=train_dataloader,
        val_dataset=validation_dataloader,
    )

    test_accuracy = calculate_test_accuracy(model, test_dataloader)
    logger.info(
        f"Test Accuracy of {args.model_name} after {args.epochs} is {test_accuracy:.3f}"
    )

    file_name = f"{args.optimizer}_{args.epochs}_{args.model_name}"
    torch.save(
        model.state_dict(),
        f"{args.model_path}/{file_name}.ph",
    )
    save_plots(
        train_history=train_acc_history,
        val_history=val_acc_history,
        save_path=args.plot_path,
        file_name=file_name,
    )
    logger.info("Completed")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--train", "-t", type=str, required=True, help="Path to Train folder"
    )
    parser.add_argument(
        "--valid", "-v", type=str, required=True, help="Path to Validatin folder"
    )
    parser.add_argument(
        "--test", "-te", type=str, required=True, help="Path to Test folder"
    )
    parser.add_argument(
        "--csv_file", "-c", type=str, required=True, help="Path to birds.csv file"
    )

    parser.add_argument(
        "--model_name",
        "-n",
        type=str,
        default="resnet18",
        help="Name of the model to fine tune",
    )
    parser.add_argument(
        "--model_path", "-m", type=str, required=True, help="Path to save trained model"
    )
    parser.add_argument(
        "--plot_path", "-p", type=str, required=True, help="Path to save history plots"
    )

    parser.add_argument(
        "--optimizer", "-o", type=str, default="sgd", help="Name of optimizer to use"
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=5,
        help="Number of epochs to train the model",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
