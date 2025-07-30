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

    print("Fetching some random training images...")
    # get some random training images
    dataiter = iter(trainloader) # Iterating through datasets and picking one batch (of photos and labels) at a time 
    images, labels = next(dataiter) # Labeling what's inside the data batch

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    print("Defining the CNN model...")
    # Defining the CNN
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module): # Base class for all nueral networks
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5) # Applies a 2D convolution over an input signal composed of several input planes.
            # IMPORTANT VOCAB
            # 2D Convolution: Puts a small matrix over a 2D image that transform inpiut data by highlighting or extracting certain features
            self.pool = nn.MaxPool2d(2, 2) # Shrinks the image by keeping only the most important/notable pixel out of each group of 2x2 pixels
            self.conv2 = nn.Conv2d(6, 16, 5) # 6 layer image, has 5x5 patches, and 16 different input channels to analyze the image in different ways
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # Flattens image into one long list of numbers to be analyzed by 120 neutrons. 400 images (16 5x5 batches)
            self.fc2 = nn.Linear(120, 84) # Take 120 values from last layer and pass it into next layer with 84 more neruons
            self.fc3 = nn.Linear(84, 10) # Take 84 values from last layer and pass it into next layer with 10 more neruons

        def forward(self, x):
            # x is the image
            x = self.pool(F.relu(self.conv1(x))) 
            # conv1 = first convolutional layer that looks at each 5x5 part of the image to find simple things like edges and colors. Will output 6 different B&W images
            # relu = Rectified Linear Unit --> helps computer to ignore useless things like negative space
            # pool = pooling layer, looks at 2x2 pixel groups and only keeps the pixel with the highest number
            x = self.pool(F.relu(self.conv2(x))) # Same thing as above line, but with a different convolutional layer
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            # fc1 = flatten list of numbers
            #r relu activation = ignores weak signals
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    print("CNN model defined.")

    print("Setting up loss function and optimizer...")
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    # Cross Entropy: measures how far off neural networks are from the real answer. The function outputs a list of scores (logits) per class about model's confidence.
    # Then, softmax function is applied to convert to probability. If computer is confidently wrong, it will be penalized higher.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # SGD = stochastic gradient descent. net.parameters tells you what you can change in the model (essentially all trainable pieces of the network)
    # lr = learning rate, how big each step is of the gradient descent (here, we are taking little steps to update weights)
    # momemntum: adding a bit of speed to the optimazatin process

    print("Starting training process...")
    for epoch in range(2):  # loop over the dataset multiple times(2 epochs)
        print(f"Epoch {epoch + 1} in progress...")
        running_loss = 0.0 # How much total loss overtime running the model
        for i, data in enumerate(trainloader, 0): # Loop and counter at the same time. i is index of batch, data is actual batch
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data # unpacking data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0

    print("Finished Training")