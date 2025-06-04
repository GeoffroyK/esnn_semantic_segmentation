import torch
import torch.nn as nn
import argparse
from dataset.dsec_ess_dataset import ESS_DSEC_Dataset
from model.unet_model import SpikingUNet
from spikingjelly.activation_based import neuron, surrogate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the DSEC dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    return parser.parse_args()

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):

    for epoch in range(epochs):
        model.train() # Training pass through the training data
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        model.eval() # Validation pass through the validation data
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Validation Loss after epoch {epoch+1}: {val_loss:.4f}')

def main():
    args = parse_args()
    # Initialize dataset
    train_dataset = ESS_DSEC_Dataset(data_dir='/media/geoffroy/T7/Thèse/Datasets/DSEC/train_semantic_segmentation/train')
    val_dataset = ESS_DSEC_Dataset(data_dir='/media/geoffroy/T7/Thèse/Datasets/DSEC/train_semantic_segmentation/test')
        
    print(f"Number of samples in dataset: {len(train_dataset.labels)}")

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(batch_size=32, shuffle=False, num_workers=2)

    # Initialize model
    model = SpikingUNet(n_channels=2, n_classes=11)  # Example model initialization
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, val_loader, optimizer, criterion, args.epochs)

if __name__ == "__main__":
    #main()
    model = SpikingUNet(n_channels=2, n_classes=11)  # Example model initialization
    x = torch.randn(1, 2, 256, 256)  # Example input tensor
    y = model(x)  # Forward pass
    print(y.shape)  # Output shapes
    print(y)  # Output tensor values
    gt = torch.randint(0, 11, (1, 256, 256))  # Example ground truth tensor
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y, gt)  # Compute loss
    print(f'Loss: {loss.item()}')  # Print loss value


