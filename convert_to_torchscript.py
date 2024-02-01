#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from argparse import Namespace
import warnings
import yaml
import torch
import cv2
from rich import print
from imutils import paths
from rich.progress import track
from sklearn.metrics import accuracy_score

from lprnet import LPRNet, numpy2tensor, decode

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    with open('config/idn_config.yaml') as f:
        args = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    load_model_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprnet = LPRNet(args).to(device).eval()
    lprnet.load_state_dict(torch.load(args.pretrained)['state_dict'])
    script = lprnet.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, "lprNet_v2.0_GPU.pt")