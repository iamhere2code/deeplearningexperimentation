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

     print("Loading test dataset...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, # FALSE means this is a test dataset
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    print("Defining class labels...")
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # Classes to be labled

    import matplotlib.pyplot as plt
    import numpy as np

    print("Defining helper function to display images...")
    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5 # normalizing the pixel range 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # Rearranging the axis of the original np image
        plt.show() # Showing the lot
