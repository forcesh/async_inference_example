import asyncio
import concurrent.futures as concurrent_futures
import time

import cv2
import numpy as np
import torch
import torchvision
from settings import *


async def inference(model, img, loop, executor, i):
    print(i, 'start')
    # if data preprocessing is too easy then async inference works slower
    # (for instance, batch = (b_size, 3, h, w))
    batch = np.random.rand(1920, 1200, 3)
    batch = cv2.resize(batch, (w, h))
    batch = np.asarray([batch for __ in range(b_size)])
    batch = np.transpose(batch, (0, 3, 1, 2))
    batch = torch.Tensor(batch).to('cuda')
    with torch.no_grad():
        result = await loop.run_in_executor(executor, model.forward, batch)
    print(i, 'end')
    return result.cpu()


def get_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


async def main():
    model = torchvision.models.resnet18(pretrained=True).cuda()
    img = torch.ones(b_size, 3, h, w, device='cuda')

    executor = concurrent_futures.ThreadPoolExecutor(max_workers=workers)
    loop = asyncio.get_event_loop()
    tasks, results = [], []

    with torch.no_grad():
        for __ in range(10):
            model(img)

    start = get_sync_time()

    for i in range(300):
        tasks.append(
            asyncio.create_task(inference(model, img, loop, executor, i)))

        if len(tasks) == workers:
            prel_results = await asyncio.gather(*tasks)
            results.extend(prel_results)
            del tasks[:]

    if len(tasks) > 0:
        prel_results = await asyncio.gather(*tasks)
        results.extend(prel_results)
        del tasks[:]

    end = get_sync_time()
    print(end - start)


if __name__ == '__main__':
    asyncio.run(main())
