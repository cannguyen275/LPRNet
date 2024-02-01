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
    device = torch.device('cpu')
    lprnet = LPRNet(args).to(device).eval()
    lprnet.load_state_dict(torch.load(args.pretrained)['state_dict'])
    output_onnx = "test.onnx"
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(1, 3, 50, 100)
    torch_out = torch.onnx._export(lprnet,
                                   inputs,
                                   output_onnx,
                                   verbose=True,
                                   input_names=input_names,
                                   output_names=output_names,
                                   # example_outputs=True,  # to show sample output dimension
                                   keep_initializers_as_inputs=True,  # to avoid error _Map_base::at
                                   # opset_version=7, # need to change to 11, to deal with tensorflow fix_size input
                                #    dynamic_axes={
                                #        "input0": [2, 3],
                                #        "output0": [1, 2]
                                #    }
                                   )