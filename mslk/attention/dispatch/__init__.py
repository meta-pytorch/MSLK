# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves each HSTU attention op that has more than one, per call.

One module per op: it lists the op's backends, decides which of them can take
a call's arguments, and picks among those from the op's tuning table in
``tables/`` (see :mod:`mslk.aiter.routing`). The op's registered implementation
calls the module; the backends themselves live with their kind.
Importing these modules imports no backend.

Conventions shared by the HSTU ops:

- Calling convention. A routing is called as ``ROUTING(q, **arguments)``:
  ``q`` first, because :class:`~mslk.aiter.routing.OpRouting` reads the device
  from the first argument (the ops' schemas start with integers), and every
  other argument by its schema name. Backends, ``key`` and ``accepts``
  predicates take the same form.
- Forward and backward route together. Each op has a forward and a backward
  routing with the same op name (so :func:`~mslk.aiter.routing.force_backend`
  and ``MSLK_DISABLE_AITER_OPS`` cover both), the same table and the same
  ``key`` function. A backend's ``accepts`` predicate must be shared by both
  directions too. The backward may then rely on the forward's saved outputs
  (``M``) coming from the same backend, as long as the routing state
  (``force_backend``, environment switches) does not change between them.
- Saved forward outputs. ``M`` has the shape and dtype the fake
  implementation declares, ``[T, H]`` float32, for every backend.
- Shapes a table does not cover go to MSLK's backend, the
  :class:`~mslk.aiter.routing.OpRouting` fallback.
- Architecture. Backends decline architectures they do not support through
  their ``accepts`` predicate or by raising, never by computing a wrong result.
"""
