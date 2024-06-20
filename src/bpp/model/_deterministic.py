import torch

from torch import Tensor
from typing import Any, Sequence, Generator
from contextlib import contextmanager, ContextDecorator


def _get_devices(elements: Sequence[Any]) -> list[int]:
    """ """

    devices = []

    if torch.cuda._initialized:
        for element in elements:
            if isinstance(element, Tensor) and element.is_cuda:
                device = element.get_device()
                if device not in devices:
                    devices.append(device)

    return devices


def _get_states(devices: Sequence[int]) -> list[Tensor]:
    """ """

    states = [torch.get_rng_state()]

    if torch.cuda._initialized:
        for device in devices:
            with torch.cuda.device(device):
                states.append(torch.cuda.get_rng_state())

    return states


def _set_states(
    devices: Sequence[int],
    states: Sequence[Tensor],
) -> None:
    """ """

    torch.set_rng_state(states[0])

    if devices and states[1:]:
        for device, gpu_state in zip(devices, states[1:]):
            with torch.cuda.device(device):
                torch.cuda.set_rng_state(gpu_state)


@contextmanager
def _dummy_context() -> Generator[None, None, None]:
    """ """

    yield


def deterministic_context(*elements: Any, enabled: bool = True) -> ContextDecorator:
    """ """

    if not enabled:
        return _dummy_context

    devices = _get_devices(elements)
    states = _get_states(devices)

    @contextmanager
    def context() -> Generator[None, None, None]:
        with torch.random.fork_rng(devices=devices, enabled=True):
            _set_states(devices, states)
            yield

    return context


def reproducible_context(
    *elements: Any,
    preserve_rng_state: bool = True,
) -> ContextDecorator:
    """ """

    cpu_autocast = torch.is_autocast_cpu_enabled()
    gpu_autocast = torch.is_autocast_enabled()
    cpu_dtype = torch.get_autocast_cpu_dtype()
    gpu_dtype = torch.get_autocast_gpu_dtype()

    deterministic = deterministic_context(*elements, enabled=preserve_rng_state)

    @contextmanager
    def context() -> Generator[None, None, None]:
        with torch.autocast("cpu", enabled=cpu_autocast, dtype=cpu_dtype):
            with torch.autocast("cuda", enabled=gpu_autocast, dtype=gpu_dtype):
                with deterministic():
                    yield

    return context
