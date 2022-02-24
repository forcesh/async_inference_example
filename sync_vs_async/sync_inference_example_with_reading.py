import time

import cv2
import numpy as np
import torch
import torchvision
from settings import *


def inference(model, img, i):
    print(i, 'start')
    batch = cv2.imread('/home/forcesh/Projects/ore-recognizer/data/img.jpg')
    batch = cv2.resize(batch, (w, h))
    batch = np.asarray([batch for __ in range(b_size)])
    batch = np.transpose(batch, (0, 3, 1, 2))
    batch = torch.Tensor(batch).to('cuda')
    with torch.no_grad():
        result = model(batch)
    print(i, 'end')
    return result.cpu()


def get_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    model = torchvision.models.resnet18(pretrained=True).cuda()
    img = torch.ones(b_size, 3, h, w, device='cuda')

    results = []

    with torch.no_grad():
        for __ in range(10):
            model(img)

    start = get_sync_time()

    for i in range(300):
        prel_results = inference(model, img, i)
        results.extend(prel_results)

    end = get_sync_time()
    print(end - start)


if __name__ == '__main__':
    main()
