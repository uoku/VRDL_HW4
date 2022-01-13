import os
import torch
from option import args
import model
import utility
import imageio
from data import common
import cv2
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda')
    checkpoint = utility.checkpoint(args)
    _model = model.Model(args)
    images = os.listdir("../testimg")

    for image in images:
        print(image)
        image_path = '../testimg/' + image
        lr = imageio.imread(image_path)
        lr = common.set_channel(lr)[0]
        lr = common.np2Tensor(lr, rgb_range=args.rgb_range)[0]
        lr = lr[np.newaxis, ...]
        lr = lr.to(device)

        _model.to(device)
        with torch.no_grad():
            sr = _model(lr, 3)
            sr = utility.quantize(sr, args.rgb_range)

            normalized = sr[0].mul(255 / args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
            imageio.imwrite(
                "./result/"+image[:-4]+"_pred"+image[-4:], tensor_cpu.numpy())
