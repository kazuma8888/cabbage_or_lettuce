import os
import torch
import glob
import cv2
import pickle
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from collections import OrderedDict

from transformer import ComposeTransform, SimpleTransform
from dataset import TestDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(img = None):
    # Transform 組み立て
    transform = ComposeTransform([
        SimpleTransform(debug=False),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset組み立て
    img = cv2.imread('IMG_0047.JPG', 1)
    test_set = TestDataset(img, transform=transform)

    # Dataloader組み立て
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, 
                                            shuffle=False, num_workers=4)

    # Model組み立て
    with open('resnet.pkl', 'rb') as f:
        net = pickle.load(f)
    #net = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False, force_reload=True).to(device)
    #net.load_state_dict(torch.load('model.pth'), strict=False)

    # MainFunction実行
    y_preds = pred_net(net, test_loader, device=device)
    y_preds = y_preds.to('cpu').detach().numpy().copy()
    return y_preds


def pred_net(net, test_loader, device= 'cpu'):
    y_preds = []
    net = net.to(device)
    for i, x in enumerate(test_loader):
        x = x.to(device)
        h = net(x)
        _, y_pred = h.max(1)
        y_preds.append(y_pred)
    return torch.cat(y_preds,dim=0)

if __name__ == '__main__':
    main()