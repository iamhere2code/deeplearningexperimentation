import torch
import torchvision
import torchvision.transforms as transforms

def main():
    print("Initializing data transformations...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print("Setting batch size...")
    batch_size = 32

    print("Loading training dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # CIFAR is database to be used, data, specifies whether or not its a training set
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=True, num_workers=4) # Shuffles elements before each batch, how many workers to work on unpacking info
