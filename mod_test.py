import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import mod_net
from torch.backends import cudnn


# Set up the test transformation
def test_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


# Define the test function
def test(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Top 1 accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

            # Top 5 accuracy
            _, top5 = outputs.topk(5, 1, True, True)
            correct_top5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_top1 / total))
    print('Top-5 Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_top5 / total))


import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True, help='Model path')
    args = parser.parse_args()

    # Load the trained model
    model_path = args.m
    # Make sure to load the correct model based on your training script
    model = mod_net.Model(num_classes=100)  # or vanilla_net.Model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Download and load the CIFAR100 test dataset
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform())
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Run the test
    test(model, test_loader, device)
