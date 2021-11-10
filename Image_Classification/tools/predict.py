# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from ..models import alexnet


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    img_path = "../data/tulip.jpg"
    assert os.path.exists(img_path), f"file {img_path} dose not exist."
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)

    # [C, H, W] -> [1, C, H, W]
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file {json_path} does not exist."
    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    model = alexnet(num_classes=5)
    model_weight_path = "./weights/alexnet.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}  prob: {:.3f}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())

    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()