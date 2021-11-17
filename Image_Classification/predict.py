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
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.base_model import BaseModel


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    img_path = args.img_path
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

    model = BaseModel(name=args.model_name, num_classes=args.num_classes).to(device)

    model.load_state_dict(torch.load(args.model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "real: {}   predict: {}   prob: {:.3f}".format(args.real_label, class_indict[str(predict_cla)],
                                                               predict[predict_cla].numpy())
    plt.title(print_res)
    plt.xticks([])
    plt.yticks([])
    print(print_res)
    plt.savefig('./data/predict.jpg', bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./data/tulip.jpg')
    parser.add_argument('--real_label', type=str, default='tulip')
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--model_weight_path', type=str, default='./weights/resnet34.pth')

    args = parser.parse_args()
    main(args)
