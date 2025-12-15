# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

from typing import Callable, List, Optional

import torch


# from https://github.com/openai/triton/blob/95d9b7f4ae21710dc899e1de6a579b2136ea4f3d/python/triton/testing.py#L19
def do_bench_cudagraph(
    fn: Callable, rep: int = 20, grad_to_none: Optional[List[torch.Tensor]] = None
) -> float:
    """
    Benchmark the runtime of the provided function.
    Args:
        fn: Function to benchmark
        rep: Repetition time (in ms)
        grad_to_none: Reset the gradient of the provided tensor to None
    Returns:
        Benchmarked runtime in ms
    """
    if torch.cuda.current_stream() == torch.cuda.default_stream():
        raise RuntimeError(
            "Cannot capture graph in default stream. "
            "Please use side stream in benchmark code."
        )
    # warmup
    fn()
    # step 1 - we estimate the amount of time the kernel call takes
    # NOTE: this estimate isn't super accurate because the GPU isn't warmed up at this point
    #       but it is probably good enough
    if grad_to_none is not None:
        for x in grad_to_none:
            x.detach_()
            x.requires_grad_(True)
            x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event)
    n_repeat = max(1, int(rep / estimate_ms))
    # step 2 - construct a cuda graph with `n_repeat` unrolled function calls to minimize
    # host overhead
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
    torch.cuda.synchronize()
    # measure time and return
    ret = []
    n_retries = 10
    for _ in range(n_retries):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        g.replay()
        end_event.record()
        torch.cuda.synchronize()
        ret += [start_event.elapsed_time(end_event) / n_repeat]
    return torch.mean(torch.tensor(ret)).item()
