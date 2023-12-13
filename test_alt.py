import argparse
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import vanilla_net
import mod_net


# Function to compute top-k accuracy
def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# Testing loop
def test(model, test_loader, device):
    top1_accuracy = 0.0
    top5_accuracy = 0.0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
            total += labels.size(0)

    top1_accuracy = top1_accuracy / total
    top5_accuracy = top5_accuracy / total
    return top1_accuracy.item(), top5_accuracy.item()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, required=True, help='Decoder path')
    parser.add_argument('-m', type=str, required=True, help='[V/M]')  # V for vanilla, M for modified
    parser.add_argument('-cuda', type=str, default='N', help='[Y/N]')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda == 'Y' else 'cpu')

    # Define the test dataset and loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR100 mean and std
    ])

    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False,
                             num_workers=2)  # Ensure batch size matches training

    # Loading the model depending on the argument chosen
    args.m = args.m.lower()
    model = None
    if args.m == 'v':
        model = vanilla_net.Model().to(device)
        state_dict = torch.load(args.s)
        # Filter out unnecessary keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:  # Load modified model
        model = mod_net.Model().to(device)
        state_dict = torch.load(args.s)
        # Filter out unnecessary keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)

    if model is not None:
        model.eval()  # Set the model to evaluation mode
        top1, top5 = test(model, test_loader, device)
        print(f'{args.m.upper()} Model - Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%')
    else:
        print("Model loading failed.")
