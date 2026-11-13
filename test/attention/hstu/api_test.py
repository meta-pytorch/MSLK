# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HSTU attention API tests: op registration, fake impls, validation,
torch.compile, autograd wiring and routing.

Meta-tensor tests run on any machine. The autograd wiring tests need a GPU;
they replace MSLK's backend with stubs that write known constants, so they
check which gradient reaches which input, not the attention math.

``force_backend`` is process-wide and not thread-safe: do not run these tests
in parallel threads of one process.
"""

import contextlib
import importlib.resources
import inspect
import unittest
from typing import Any, Dict, Iterator, Optional
from unittest import mock

import mslk.attention.hstu as hstu
import torch
from mslk.aiter.routing import force_backend, MSLK
from mslk.attention.dispatch import hstu_cross, hstu_grouped, hstu_ragged

META = torch.device("meta")
BF16 = torch.bfloat16

T, H, D = 64, 2, 32
B = 4
TQ = 24
G, Z = 2, 3


def _t(shape: Any, device: torch.device, dtype: torch.dtype = BF16) -> torch.Tensor:
    if device.type == "meta":
        return torch.empty(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def _offsets(n: int, total: int, device: torch.device) -> torch.Tensor:
    # Evenly split ``total`` tokens into ``n`` sequences.
    return torch.linspace(0, total, n + 1, device="cpu").to(torch.int32).to(device)


def _ragged_args(device: torch.device = META, shared_kv: bool = True) -> Dict:
    q, k = _t((T, H, D), device), _t((T, H, D), device)
    return dict(
        N=T // B,
        alpha=1.0,
        q=q,
        k=k,
        v=k if shared_kv else _t((T, H, D), device),
        seq_offsets=_offsets(B, T, device),
        invalid_attn_mask_type="lower_triangular",
        num_targets=None,
        attn_scale=None,
        max_attn_len=4,
        contextual_seq_len=0,
        full_attn_size=0,
        num_softmax_heads=H,
        shared_kv=shared_kv,
    )


def _ragged_bwd_args(fwd: Dict, **overrides: Any) -> Dict:
    shared = fwd["shared_kv"]
    args = dict(
        dout=torch.empty_like(fwd["v"]),
        q=fwd["q"],
        k=fwd["k"],
        v=fwd["v"],
        dq=torch.empty_like(fwd["q"]),
        dk=torch.empty_like(fwd["k"]),
        dv=None if shared else torch.empty_like(fwd["v"]),
        seq_offsets=fwd["seq_offsets"],
        num_targets=None,
        attn_scale=None,
        N=fwd["N"],
        alpha=fwd["alpha"],
        max_attn_len=fwd["max_attn_len"],
        invalid_attn_mask_type=fwd["invalid_attn_mask_type"],
        contextual_seq_len=0,
        sort_by_length_indices=None,
        full_attn_size=0,
        num_softmax_heads=fwd["num_softmax_heads"],
        M=None,
        Out=None,
        shared_kv=shared,
    )
    args.update(overrides)
    return args


def _cross_args(
    device: torch.device = META, shared_kv: bool = True, v_none: bool = False
) -> Dict:
    k = _t((T, H, D), device)
    if shared_kv:
        v = None if v_none else k
    else:
        v = _t((T, H, D), device)
    return dict(
        max_seq_len=T // B,
        alpha=1.0,
        q=_t((TQ, H, D), device),
        k=k,
        v=v,
        seq_offsets=_offsets(B, T, device),
        seq_offsets_q=_offsets(B, TQ, device),
        max_q_len=TQ // B,
        attn_scale=torch.ones(1, device=device, dtype=torch.float32),
        G=1,
        shared_kv=shared_kv,
        num_softmax_heads=H,
        enable_tma=True,
        truncate_method="none",
        num_targets=None,
        causal=False,
    )


def _cross_bwd_args(fwd: Dict) -> Dict:
    return dict(
        dout=torch.empty_like(fwd["q"]),
        q=fwd["q"],
        k=fwd["k"],
        v=fwd["v"],
        seq_offsets=fwd["seq_offsets"],
        attn_scale=fwd["attn_scale"],
        max_seq_len=fwd["max_seq_len"],
        alpha=fwd["alpha"],
        max_q_len=fwd["max_q_len"],
        seq_offsets_q=fwd["seq_offsets_q"],
        num_targets=None,
        causal=False,
        shared_kv=fwd["shared_kv"],
        G=1,
        num_softmax_heads=fwd["num_softmax_heads"],
        M=None,
        Delta=None,
        stride_mm=0,
        truncate_method="none",
        enable_tma=True,
    )


def _grouped_args(device: torch.device = META, shared_kv: bool = True) -> Dict:
    k = _t((T, H, D), device)
    per_group = torch.full((G,), 4, device=device, dtype=torch.int32)
    return dict(
        Z=Z,
        q=_t((T, H, D), device),
        k=k,
        v=k if shared_kv else _t((T, H, D), device),
        seq_offsets=_offsets(G * Z, T, device),
        group_offsets=torch.tensor([0, Z, 2 * Z], dtype=torch.int32).to(device),
        num_targets=None,
        max_seq_len_tensor=per_group,
        max_attn_len_tensor=per_group,
        full_attn_size_tensor=torch.zeros_like(per_group),
        contextual_seq_len_tensor=torch.zeros_like(per_group),
        alpha=1.0,
        invalid_attn_mask_type="lower_triangular",
        attn_scale_tensor=None,
    )


def _public_grouped(args: Dict) -> Dict:
    """Op argument names -> the public grouped functions' (reference) names."""
    names = {"q": "group_q", "k": "group_k", "v": "group_v"}
    names["num_targets"] = "group_num_targets"
    return {names.get(n, n): t for n, t in args.items()}


def _grouped_bwd_args(fwd: Dict, **overrides: Any) -> Dict:
    shared = fwd["v"] is fwd["k"]
    args = dict(
        dout=torch.empty_like(fwd["v"]),
        Z=fwd["Z"],
        q=fwd["q"],
        k=fwd["k"],
        v=fwd["v"],
        dq=torch.empty_like(fwd["q"]),
        dk=torch.empty_like(fwd["k"]),
        dv=None if shared else torch.empty_like(fwd["v"]),
        **{
            n: fwd[n]
            for n in (
                "seq_offsets",
                "group_offsets",
                "num_targets",
                "max_seq_len_tensor",
                "max_attn_len_tensor",
                "full_attn_size_tensor",
                "contextual_seq_len_tensor",
                "alpha",
                "invalid_attn_mask_type",
                "attn_scale_tensor",
            )
        },
        Out=None,
        shared_kv=shared,
    )
    args.update(overrides)
    return args


class SignatureTest(unittest.TestCase):
    """Public functions keep the reference signatures (names, order, defaults)."""

    REFERENCE = {
        "ragged_hstu_attention_fwd": (
            "N, alpha, q, k, v, seq_offsets, invalid_attn_mask_type, num_targets, "
            "attn_scale, max_attn_len, contextual_seq_len, full_attn_size, "
            "num_softmax_heads=0, shared_kv=False"
        ),
        "ragged_hstu_attention_bwd": (
            "dout, q, k, v, dq, dk, dv, seq_offsets, num_targets, attn_scale, N, "
            "alpha, max_attn_len, invalid_attn_mask_type, contextual_seq_len, "
            "sort_by_length_indices, full_attn_size, num_softmax_heads=0, M=None, "
            "Out=None, shared_kv=False"
        ),
        "cross_hstu_attention_fwd": (
            "max_seq_len, alpha, q, k, v, seq_offsets, seq_offsets_q, max_q_len, "
            "attn_scale, G, shared_kv=False, num_softmax_heads=0, enable_tma=True, "
            "truncate_method='none', num_targets=None, causal=False"
        ),
        "cross_hstu_attention_bwd": (
            "dout, q, k, v, seq_offsets, attn_scale, max_seq_len, alpha, max_q_len, "
            "seq_offsets_q, num_targets, causal, shared_kv, G, num_softmax_heads=0, "
            "M=None, Delta=None, stride_mm=0, truncate_method='none', "
            "enable_tma=True"
        ),
        "grouped_ragged_hstu_attention_fwd": (
            "Z, group_q, group_k, group_v, seq_offsets, group_offsets, "
            "group_num_targets, max_seq_len_tensor, max_attn_len_tensor, "
            "full_attn_size_tensor, contextual_seq_len_tensor, alpha, "
            "invalid_attn_mask_type, attn_scale_tensor=None"
        ),
    }

    def test_reference_signatures(self) -> None:
        for name, expected in self.REFERENCE.items():
            with self.subTest(name=name):
                params = inspect.signature(getattr(hstu, name)).parameters.values()
                actual = ", ".join(
                    p.name
                    if p.default is inspect.Parameter.empty
                    else f"{p.name}={p.default!r}"
                    for p in params
                )
                self.assertEqual(actual, expected)


class FakeImplTest(unittest.TestCase):
    """Fake impls give the documented output shapes on meta tensors."""

    def test_ragged(self) -> None:
        args = _ragged_args()
        out, M = torch.ops.mslk.hstu_ragged_fwd(**args)
        self.assertEqual(out.shape, args["v"].shape)
        self.assertEqual(out.dtype, BF16)
        self.assertEqual(M.shape, (T, H))
        self.assertEqual(M.dtype, torch.float32)
        torch.ops.mslk.hstu_ragged_bwd(**_ragged_bwd_args(args))

    def test_cross(self) -> None:
        for shared_kv, n_grads in ((True, 2), (False, 3)):
            args = _cross_args(shared_kv=shared_kv)
            out, M = torch.ops.mslk.hstu_cross_fwd(**args)
            self.assertEqual(out.shape, (TQ, H, D))
            self.assertEqual(M.shape, (TQ, H))
            grads = torch.ops.mslk.hstu_cross_bwd(**_cross_bwd_args(args))
            self.assertEqual(len(grads), n_grads)
            self.assertEqual(grads[0].shape, args["q"].shape)
            self.assertEqual(grads[1].shape, args["k"].shape)

    def test_cross_value_head_dim(self) -> None:
        args = _cross_args(shared_kv=False)
        args["v"] = _t((T, H, 2 * D), META)
        out, _ = torch.ops.mslk.hstu_cross_fwd(**args)
        self.assertEqual(out.shape, (TQ, H, 2 * D))

    def test_grouped(self) -> None:
        args = _grouped_args()
        out = torch.ops.mslk.hstu_grouped_fwd(**args)
        self.assertEqual(out.shape, args["v"].shape)
        torch.ops.mslk.hstu_grouped_bwd(**_grouped_bwd_args(args))


class ValidationTest(unittest.TestCase):
    """Bad arguments are rejected by the ops (checked here on meta tensors)."""

    def _raises(self, op: Any, args: Dict, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            op(**args)

    def test_ragged_fwd(self) -> None:
        op = torch.ops.mslk.hstu_ragged_fwd
        cases = [
            ({"q": _t((T, H, D), META, torch.float64)}, "bfloat16, float16 or float32"),
            (
                {"q": _t((T, D, H), META).transpose(1, 2)},
                "contiguous in the last dimension",
            ),
            ({"v": _t((T, H, D), META)}, "same tensor object as k"),
            ({"num_softmax_heads": 1}, "num_softmax_heads"),
            ({"invalid_attn_mask_type": "causal"}, "invalid_attn_mask_type"),
            (
                {"num_targets": torch.empty(B + 1, device=META, dtype=torch.int32)},
                "num_targets must have length",
            ),
            (
                {"seq_offsets": torch.empty(B + 1, device=META, dtype=torch.float32)},
                "seq_offsets must be int32 or int64",
            ),
            ({"max_attn_len": -1}, "max_attn_len must be >= 0"),
            ({"k": _t((T, H, 2 * D), META)}, "heads and head dim of q"),
        ]
        for override, pattern in cases:
            with self.subTest(pattern=pattern):
                args = _ragged_args()
                args.update(override)
                if "k" in override:
                    args["v"] = args["k"]
                self._raises(op, args, pattern)

    def test_ragged_bwd_shared_kv_contract(self) -> None:
        op = torch.ops.mslk.hstu_ragged_bwd
        shared = _ragged_args(shared_kv=True)
        self._raises(
            op,
            _ragged_bwd_args(shared, dv=torch.empty_like(shared["k"])),
            "dv must be None",
        )
        separate = _ragged_args(shared_kv=False)
        self._raises(op, _ragged_bwd_args(separate, dv=None), "requires a dv")
        bwd = _ragged_bwd_args(separate)
        bwd["dv"] = bwd["dk"]
        self._raises(op, bwd, "distinct")
        self._raises(
            op,
            _ragged_bwd_args(separate, dout=_t((T, H, 2 * D), META)),
            "dout must have shape",
        )

    def test_cross(self) -> None:
        op = torch.ops.mslk.hstu_cross_fwd
        args = _cross_args(shared_kv=False)
        args["v"] = None
        self._raises(op, args, "requires v")
        args = _cross_args(shared_kv=True)
        args["v"] = _t((T, H, D), META)
        self._raises(op, args, "None or the same tensor as k")
        args = _cross_args()
        args["seq_offsets_q"] = torch.empty(B + 2, device=META, dtype=torch.int32)
        self._raises(op, args, "seq_offsets_q must have length")
        args = _cross_args()
        args["G"] = 0
        self._raises(op, args, "G must be >= 1")

    def test_grouped(self) -> None:
        op = torch.ops.mslk.hstu_grouped_fwd
        args = _grouped_args()
        args["seq_offsets"] = _offsets(G * Z + 1, T, META)
        self._raises(op, args, "seq_offsets must have length")
        args = _grouped_args()
        args["max_attn_len_tensor"] = torch.zeros(G + 1, device=META, dtype=torch.int32)
        self._raises(op, args, "max_attn_len_tensor must have length")
        args = _grouped_args()
        args["attn_scale_tensor"] = torch.ones(G + 1, device=META)
        self._raises(op, args, "one entry per group")
        args = _grouped_args()
        args["Z"] = 0
        self._raises(op, args, "Z must be >= 1")

    def test_grouped_bwd_shared_kv_contract(self) -> None:
        op = torch.ops.mslk.hstu_grouped_bwd
        shared = _grouped_args(shared_kv=True)
        self._raises(
            op,
            _grouped_bwd_args(shared, dv=torch.empty_like(shared["k"])),
            "dv must be None",
        )
        separate = _grouped_args(shared_kv=False)
        self._raises(
            op, _grouped_bwd_args(separate, shared_kv=True), "same tensor object"
        )


class CompileTest(unittest.TestCase):
    """torch.compile traces every op through its fake impl."""

    @staticmethod
    def _compile(fn: Any) -> Any:
        return torch.compile(fn, backend="eager", fullgraph=True)

    def test_forward_ops(self) -> None:
        for op, args in (
            (torch.ops.mslk.hstu_ragged_fwd, _ragged_args()),
            (torch.ops.mslk.hstu_cross_fwd, _cross_args()),
            (torch.ops.mslk.hstu_grouped_fwd, _grouped_args()),
        ):
            with self.subTest(op=str(op)):
                torch._dynamo.reset()
                self._compile(lambda a: op(**a))(args)

    def test_backward_ops(self) -> None:
        for op, args in (
            (torch.ops.mslk.hstu_ragged_bwd, _ragged_bwd_args(_ragged_args())),
            (torch.ops.mslk.hstu_cross_bwd, _cross_bwd_args(_cross_args())),
            (
                torch.ops.mslk.hstu_grouped_bwd,
                _grouped_bwd_args(_grouped_args(shared_kv=False)),
            ),
        ):
            with self.subTest(op=str(op)):
                torch._dynamo.reset()
                self._compile(lambda a: op(**a))(args)

    def test_autograd_through_aot(self) -> None:
        # Traces forward and backward, including the dv mutation and the
        # shared-K/V alias across saved tensors.
        for shared_kv in (True, False):
            with self.subTest(shared_kv=shared_kv):
                torch._dynamo.reset()
                args = _ragged_args(shared_kv=shared_kv)
                for name in ("q", "k", "v"):
                    args[name].requires_grad_(True)

                @torch.compile(backend="aot_eager", fullgraph=True)
                def step(a: Dict) -> torch.Tensor:
                    out, _ = hstu.ragged_hstu_attention(**a)
                    return out.float().sum()

                step(args).backward()
                self.assertEqual(args["q"].grad.shape, args["q"].shape)
                self.assertEqual(args["k"].grad.shape, args["k"].shape)


class AutogradMetaTest(unittest.TestCase):
    """Which inputs receive gradients, on meta tensors."""

    def test_grouped_separate_v_gets_gradient(self) -> None:
        args = _grouped_args(shared_kv=False)
        for name in ("q", "k", "v"):
            args[name].requires_grad_(True)
        hstu.grouped_ragged_hstu_attention(
            **_public_grouped(args)
        ).float().sum().backward()
        self.assertIsNotNone(args["v"].grad)

    def test_cross_shared_v_none(self) -> None:
        args = _cross_args(shared_kv=True, v_none=True)
        args["q"].requires_grad_(True)
        args["k"].requires_grad_(True)
        out, _ = hstu.cross_hstu_attention(**args)
        out.float().sum().backward()
        self.assertIsNotNone(args["k"].grad)


# ── Autograd wiring on a GPU, with stub backends ────────────────────────
#
# Stub backward backends write constants: dq=1; with separate V dk=2, dv=3;
# with shared K/V dk=5 (dK+dV). A doubly counted shared gradient shows as 10.


def _ragged_fwd_stub(q: torch.Tensor, *, v: torch.Tensor, **_: Any) -> Any:
    return torch.zeros_like(v), q.new_zeros(q.shape[:2], dtype=torch.float32)


def _ragged_bwd_stub(
    q: torch.Tensor,
    *,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: Optional[torch.Tensor],
    **_: Any,
) -> None:
    dq.fill_(1)
    if dv is None:
        dk.fill_(5)
    else:
        dk.fill_(2)
        dv.fill_(3)


def _grouped_fwd_stub(q: torch.Tensor, *, v: torch.Tensor, **_: Any) -> Any:
    return torch.zeros_like(v)


def _cross_fwd_stub(
    q: torch.Tensor, *, k: torch.Tensor, v: Optional[torch.Tensor], **_: Any
) -> Any:
    value = k if v is None else v
    out = q.new_zeros((q.shape[0], q.shape[1], value.shape[2]))
    return out, q.new_zeros(q.shape[:2], dtype=torch.float32)


def _cross_bwd_stub(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    shared_kv: bool,
    **_: Any,
) -> Any:
    dq = torch.ones_like(q)
    if shared_kv:
        return [dq, torch.full_like(k, 5)]
    assert v is not None
    return [dq, torch.full_like(k, 2), torch.full_like(v, 3)]


@contextlib.contextmanager
def _stubbed(module: Any, fwd: Any, bwd: Any) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(module.FWD_ROUTING.backends, {MSLK: fwd}))
        stack.enter_context(mock.patch.dict(module.BWD_ROUTING.backends, {MSLK: bwd}))
        stack.enter_context(force_backend(module.OP, MSLK))
        yield


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class AutogradWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")

    def _leaf(self, args: Dict, names: Any) -> None:
        for name in names:
            if args[name] is not None:
                args[name].requires_grad_(True)

    def _assert_all(self, t: Optional[torch.Tensor], value: float) -> None:
        self.assertIsNotNone(t)
        torch.testing.assert_close(t.float(), torch.full_like(t.float(), value))

    def test_ragged(self) -> None:
        with _stubbed(hstu_ragged, _ragged_fwd_stub, _ragged_bwd_stub):
            args = _ragged_args(self.device, shared_kv=True)
            self._leaf(args, ("q", "k"))
            out, _ = hstu.ragged_hstu_attention(**args)
            out.sum().backward()  # an expanded, non-contiguous gradient
            self._assert_all(args["q"].grad, 1)
            self._assert_all(args["k"].grad, 5)

            args = _ragged_args(self.device, shared_kv=False)
            self._leaf(args, ("q", "k", "v"))
            out, _ = hstu.ragged_hstu_attention(**args)
            out.sum().backward()
            self._assert_all(args["k"].grad, 2)
            self._assert_all(args["v"].grad, 3)

    def test_cross(self) -> None:
        with _stubbed(hstu_cross, _cross_fwd_stub, _cross_bwd_stub):
            for v_none in (False, True):
                with self.subTest(shared_kv=True, v_none=v_none):
                    args = _cross_args(self.device, shared_kv=True, v_none=v_none)
                    self._leaf(args, ("q", "k"))
                    out, _ = hstu.cross_hstu_attention(**args)
                    out.sum().backward()
                    self._assert_all(args["q"].grad, 1)
                    self._assert_all(args["k"].grad, 5)

            args = _cross_args(self.device, shared_kv=False)
            self._leaf(args, ("q", "k", "v"))
            out, _ = hstu.cross_hstu_attention(**args)
            out.sum().backward()
            self._assert_all(args["k"].grad, 2)
            self._assert_all(args["v"].grad, 3)

    def test_cross_bwd_aliases_dv_when_shared(self) -> None:
        with _stubbed(hstu_cross, _cross_fwd_stub, _cross_bwd_stub):
            args = _cross_bwd_args(_cross_args(self.device, shared_kv=True))
            dq, dk, dv = hstu.cross_hstu_attention_bwd(**args)
            self.assertIs(dv, dk)

    def test_grouped(self) -> None:
        with _stubbed(hstu_grouped, _grouped_fwd_stub, _ragged_bwd_stub):
            args = _grouped_args(self.device, shared_kv=True)
            self._leaf(args, ("q", "k"))
            hstu.grouped_ragged_hstu_attention(**_public_grouped(args)).sum().backward()
            self._assert_all(args["k"].grad, 5)

            args = _grouped_args(self.device, shared_kv=False)
            self._leaf(args, ("q", "k", "v"))
            hstu.grouped_ragged_hstu_attention(**_public_grouped(args)).sum().backward()
            self._assert_all(args["q"].grad, 1)
            self._assert_all(args["k"].grad, 2)
            self._assert_all(args["v"].grad, 3)


class RoutingTest(unittest.TestCase):
    """Routing modules: q-first calling convention, shared fwd/bwd keys, and
    table headers that match the declared columns."""

    CASES = (
        (hstu_ragged, _ragged_args, lambda a: _ragged_bwd_args(a)),
        (hstu_cross, _cross_args, lambda a: _cross_bwd_args(a)),
        (hstu_grouped, _grouped_args, lambda a: _grouped_bwd_args(a)),
    )

    def test_force_backend(self) -> None:
        for module, make_fwd, make_bwd in self.CASES:
            with self.subTest(op=module.OP), force_backend(module.OP, MSLK):
                fwd = make_fwd()
                bwd = make_bwd(fwd)
                q = fwd.pop("q")
                self.assertEqual(module.FWD_ROUTING.backend_for(q, **fwd), MSLK)
                self.assertEqual(
                    module.BWD_ROUTING.backend_for(bwd.pop("q"), **bwd), MSLK
                )

    def test_forward_and_backward_share_the_key(self) -> None:
        for module, make_fwd, make_bwd in self.CASES:
            with self.subTest(op=module.OP):
                fwd = make_fwd()
                bwd = make_bwd(fwd)
                fwd_key = module.key(fwd.pop("q"), **fwd)
                bwd_key = module.key(bwd.pop("q"), **bwd)
                self.assertEqual(fwd_key, bwd_key)

    def test_table_columns(self) -> None:
        for module, _, _ in self.CASES:
            with self.subTest(op=module.OP):
                table = module.FWD_ROUTING.table
                self.assertEqual(table.key_columns, module.KEY_COLUMNS)
                self.assertEqual(table.bucket_column, module.BUCKET_COLUMN)
                text = (
                    importlib.resources.files("mslk.attention.dispatch")
                    .joinpath(f"tables/{module.OP}.csv")
                    .read_text()
                )
                header = text.splitlines()[0].split(",")
                self.assertEqual(
                    header,
                    [
                        "gfx",
                        "cu_num",
                        *module.KEY_COLUMNS,
                        module.BUCKET_COLUMN,
                        "backend",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
