from torchvision import models
import torch.nn as nn


class ImageClassifier:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def set_parameter_requires_grad(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def get_model(self, model_name: str):
        model = None
        if model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.set_parameter_requires_grad(model)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.set_parameter_requires_grad(model)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif model_name == "efficientnet":
            model = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.DEFAULT
            )
            model.classifier[1] = nn.Linear(1280, self.num_classes)
        return model
