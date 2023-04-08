import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
