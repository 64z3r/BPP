"""
torch_geometric.nn.models.rev_gnn
"""

from typing import Any, Sequence, Callable
from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Function

from ._tensor import tensors_of, free, allocate, detached, unsqueeze_grads
from ._arguments import pack_args, split_args, restore_args
from ._deterministic import reproducible_context


class ReversibleFunction(Function):
    """ """

    @staticmethod
    def forward(
        ctx: Any,
        fn: Module,
        fn_reverse: Module,
        free_inputs: bool,
        free_outputs: bool,
        preserve_rng_state: bool,
        num_inputs: int,
        *args: Any,
    ) -> tuple[Any, ...]:
        """ """

        inputs = args[:num_inputs]
        parameters = args[num_inputs:]

        ctx.fn = fn
        ctx.fn_reverse = fn_reverse
        ctx.free_inputs = free_inputs
        ctx.free_outputs = free_outputs

        # Create context for reproducible computations. Needed in backwards
        #   pass due to autocast and dropout.
        ctx.reproducible_context = reproducible_context(
            *inputs,
            *parameters,
            preserve_rng_state=preserve_rng_state,
        )

        with torch.no_grad():
            outputs = ctx.fn(*detached(inputs))
            assert isinstance(outputs, Sequence)

        outputs = tuple(detached(outputs))
        extra_args = inputs[len(outputs) :]

        if ctx.free_inputs:
            for element in tensors_of(inputs[: len(outputs)], without=outputs):
                free(element)
            inputs = inputs[: len(outputs)]
        else:
            inputs = []

        # Prepare values for backwards pass.
        packed_args, slices = pack_args(extra_args, inputs, outputs, parameters)
        tensors, non_tensors = split_args(packed_args)
        ctx.save_for_backward(*tensors)
        ctx.non_tensors = non_tensors
        ctx.extra_args = slices[0]
        ctx.inputs = slices[1]
        ctx.outputs = slices[2]
        ctx.parameters = slices[3]

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """ """

        # Restore values from forwards pass.
        args = tuple(restore_args(ctx.saved_tensors, ctx.non_tensors))
        extra_args = args[ctx.extra_args]
        fwd_inputs = args[ctx.inputs]
        fwd_outputs = args[ctx.outputs]
        parameters = args[ctx.parameters]

        # Compute inputs from outputs (in reverse).
        with torch.no_grad():
            with ctx.reproducible_context():
                inputs = ctx.fn_reverse(*fwd_outputs, *extra_args)
                assert isinstance(inputs, Sequence)

            if ctx.free_inputs:
                for dst, src in zip(fwd_inputs, inputs):
                    allocate(dst)
                    dst.data.set_(src.data.to(dst.dtype))

            if ctx.free_outputs:
                for element in tensors_of(fwd_outputs, without=inputs):
                    free(element)

        # Construct graph for gradient computation; i.e., recompute outputs and
        #   record graph.
        with torch.set_grad_enabled(True):
            inputs = tuple(detached((*inputs, *extra_args)))

            for element, requires_grad in zip(inputs, ctx.needs_input_grad[6:]):
                if isinstance(element, Tensor):
                    element.requires_grad = requires_grad

            with ctx.reproducible_context():
                outputs = ctx.fn(*inputs)
                assert isinstance(outputs, Sequence)

        # Prepare inputs and output and compute gradients.
        outputs = tuple(tensors_of(outputs, with_grads_only=True))
        inputs = tuple(tensors_of(inputs, with_grads_only=True))

        gradients = tuple(
            unsqueeze_grads(
                torch.autograd.grad(
                    outputs=outputs,
                    inputs=(*inputs, *parameters),
                    grad_outputs=grad_outputs,
                ),
                ctx.needs_input_grad,
            )
        )

        # import warnings

        # for i, grad in enumerate(gradients):

        #     if grad is None:
        #         print(f"({i:3d}) gradient is none")
        #         continue
        #     with warnings.catch_warnings(action="ignore"):
        #         print(
        #             f"({i:3d}) "
        #             f"mean={grad.mean().item(): 13.6e} "
        #             f"std={grad.std().item(): 13.6e} "
        #             f"min={grad.min().item(): 13.6e} "
        #             f"max={grad.max().item(): 13.6e}"
        #         )

        return gradients


class ReversibleModule(Module):
    """ """

    def __init__(
        self,
        disable: bool = False,
        free_inputs: bool = False,
        free_outputs: bool = True,
        preserve_rng_state: bool = True,
        forward_has_only_one_output: bool = True,
    ) -> None:
        """ """

        super().__init__()
        self.disable = disable
        self.free_inputs = free_inputs
        self.free_outputs = free_outputs
        self.preserve_rng_state = preserve_rng_state
        self.forward_has_only_one_output = forward_has_only_one_output

    def forward(self, *args: Any) -> Any:
        """ """

        return self._fn_apply(args, self._forward, self._reverse)

    def reverse(self, *args: Any) -> Any:
        """ """

        return self._fn_apply(args, self._reverse, self._forward)

    def _forward(self, *args: Any) -> tuple[Any, ...]:
        """
        **Note**: must return a tuple, even in case of a single value.
        """

        raise NotImplementedError()

    def _reverse(self, *args: Any) -> tuple[Any, ...]:
        """
        **Note**: must return a tuple, even in case of a single value.
        """

        raise NotImplementedError()

    def _fn_apply(
        self,
        inputs: Sequence[Any],
        fn: Callable[..., Any],
        fn_reverse: Callable[..., Any],
    ) -> Any:
        """ """

        if not self.disable:
            outputs = ReversibleFunction.apply(
                fn,
                fn_reverse,
                self.free_inputs,
                self.free_outputs,
                self.preserve_rng_state,
                len(inputs),
                *inputs,
                *tuple(tensors_of(self.parameters(), with_grads_only=True)),
            )
        else:
            outputs = fn(*inputs)

        if self.forward_has_only_one_output:
            if len(outputs) > 1:
                warn(
                    f"'forward_has_only_one_output' is set but forward has "
                    f"returned {len(outputs)} outputs"
                )
            return outputs[0]
        else:
            return outputs
