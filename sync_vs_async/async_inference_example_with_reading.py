import asyncio
import concurrent.futures as concurrent_futures
import time

import cv2
import numpy as np
import torch
import torchvision
from settings import *


async def inference(model, loop, executor, i):
    print(i, 'start')
    batch = cv2.imread('/home/forcesh/Projects/ore-recognizer/data/img.jpg')
    batch = cv2.resize(batch, (w, h))
    batch = np.asarray([batch for __ in range(b_size)])
    batch = np.transpose(batch, (0, 3, 1, 2))
    batch = torch.Tensor(batch).to('cuda')
    with torch.no_grad():
        """Arrange for func to be called in the specified executor.

        The executor argument should be an concurrent.futures.Executor instance
        """
        result = await loop.run_in_executor(executor, model.forward, batch)

    print(i, 'end')
    return result.cpu()


def get_sync_time() -> float:
    if torch.cuda.is_available():
        """Waits for all kernels in all streams on a CUDA device to
        complete."""
        torch.cuda.synchronize()
    '''
    time.time() should NOT be used for comparing relative times. It’s not reliable because it’s adjustable.
    time.process_time(), time.perf_counter() aren't adjustable

    time.perf_counter()
    Return the value (in fractional seconds) of a performance counter,
    i.e. a clock with the highest available resolution to measure a short duration.
    It does include time elapsed during sleep and is system-wide.
    The reference point of the returned value is undefined,
    so that only the difference between the results of two calls is valid.

    time.process_time()
    Return the value (in fractional seconds) of the sum of the system and user CPU time of the current process.
    It does not include time elapsed during sleep. It is process-wide by definition.
    The reference point of the returned value is undefined,
    so that only the difference between the results of two calls is valid.
    '''
    return time.perf_counter()


async def main():
    model = torchvision.models.resnet18(pretrained=True).cuda()
    img = torch.ones(b_size, 3, h, w, device='cuda')
    '''
    The event loop is the core of every asyncio application.
    Event loops run asynchronous tasks and callbacks, perform network IO operations, and run subprocesses.
    '''
    loop = asyncio.get_event_loop()
    '''
      concurrent.futures.Executor instance;
      we can also use ProcessPoolExecutor
      '''
    executor = concurrent_futures.ThreadPoolExecutor(max_workers=workers)

    tasks, results = [], []

    with torch.no_grad():
        for __ in range(10):
            model(img)

    start = get_sync_time()

    for i in range(300):
        """coroutine to task."""
        tasks.append(asyncio.create_task(inference(model, loop, executor, i)))

        if len(tasks) == workers:
            """Run awaitable objects in the aws sequence concurrently.

            If any awaitable in aws is a coroutine, it is automatically
            scheduled as a Task. If all awaitables are completed successfully,
            the result is an aggregate list of returned values. The order of
            result values corresponds to the order of awaitables in aws.
            """
            prel_results = await asyncio.gather(*tasks)
            '''
            There are three main types of awaitable objects: coroutines(async def function), Tasks, and Futures.

            Tasks are used to schedule coroutines concurrently.
            When a coroutine is wrapped into a Task with functions like asyncio.create_task() the coroutine is automatically scheduled to run soon:

            A Future is a special low-level awaitable object that represents an eventual result of an asynchronous operation.
            When a Future object is awaited it means that the coroutine will wait until the Future is resolved in some other place.
            Future objects in asyncio are needed to allow callback-based code to be used with async/await.

            Normally there is no need to create Future objects at the application level code.
            '''
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
