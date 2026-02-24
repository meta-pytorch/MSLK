# CuteDSL TMA (Tensor Memory Accelerator) Guide

## What is TMA?

TMA is a hardware unit (Hopper+) that asynchronously copies tiles between global memory (GMEM)
and shared memory (SMEM). Advantages over manual copies:
- **Single-threaded issuance**: one thread issues the copy; no per-thread address math
- **Automatic out-of-bounds predication**: remainder tiles handled by hardware
- **Asynchronous execution**: runs in the async proxy, enabling warp specialization

TMA operates via a **descriptor** (CUtensorMap) created on the host that encodes the GMEM tensor
shape/strides and the SMEM tile layout. The kernel receives this descriptor and issues
`cp.async.bulk.tensor` PTX instructions referencing it.

**Stride requirement**: non-contiguous GMEM dimensions must have strides that are multiples
of 16 bytes (e.g., for float32 row-major (M,N), N must be divisible by 4).

---

## Two-Step Process

### Step 1: Host — Create TMA Atom + TMA Tensor

```python
# For MMA-aware loads (knows how MMA tiles A/B):
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    op,                    # CopyBulkTensorTileG2SOp or G2SMulticastOp
    gmem_tensor,           # The full GMEM tensor (with shape + strides)
    smem_layout_one_stage, # SMEM layout for ONE stage (no stage dim)
    mma_tiler_mnk,         # (TILE_M, TILE_N, TILE_K) — CTA tile shape
    tiled_mma,             # The TiledMma that will consume the data
    cluster_shape_vmnk,    # For multicast: cluster shape (optional)
)
# make_tiled_tma_atom_A projects onto M,K dims; _B projects onto N,K dims
# A multicasts across N-mode; B multicasts across M-mode
```

Returns two objects:
1. **CopyAtom** (`tma_atom`): wraps the PTX copy instruction + TMA descriptor
2. **TMA Tensor** (`tma_tensor`): a coordinate tensor with "basis stride elements"
   (ArithTuple values instead of data pointers). Used for partitioning.

For non-MMA operations (epilogue stores, standalone copies):
```python
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(),  # or G2SOp, or ReduceAdd
    gmem_tensor,
    smem_layout,
    cta_tiler,          # Just a tile shape, not MMA-aware
)
```

### Step 2: Kernel — Partition, Issue Copy, Synchronize

```python
# 1. Partition SMEM and GMEM tensors for TMA coordinates
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a,
    cta_coord,                         # CTA position in cluster (0 if no cluster)
    cta_layout,                        # Layout of CTAs in cluster (make_layout(1) if trivial)
    cute.group_modes(sA, 0, 3),        # SMEM tensor (group all non-stage modes)
    cute.group_modes(tCgA, 0, 3),      # GMEM tensor (MMA-partitioned coordinates)
)
# tAsA shape: ((atom_vals, rest_vals), NUM_STAGES)  — SMEM side
# tAgA shape: ((atom_vals, rest_vals), NUM_K_TILES) — GMEM side (ArithTuple coords)

# 2. Issue copy (single thread only!)
cute.copy(
    tma_atom_a,
    tAgA[(None, k_tile_idx)],      # Source: GMEM coordinates for this K-tile
    tAsA[(None, stage_idx)],       # Dest: SMEM buffer at this pipeline stage
    tma_bar_ptr=barrier_ptr,       # mbarrier for completion tracking
    mcast_mask=mask,               # uint16 multicast mask (optional)
)
```

---

## ArithTuple Coordinates (The Key Concept)

TMA does NOT work with raw GMEM pointers. Instead, it uses **ArithTuple** coordinate tensors.

When you call `tma_load.get_tma_tensor(shape(gmem))` (C++) or the CuteDSL equivalent
`make_tiled_tma_atom_*`, you get a tensor whose "elements" are coordinate tuples like
`(row, col)` rather than data values. The TMA descriptor maps these coordinates to
actual GMEM addresses internally.

```
# A 16x16 ArithTuple tensor for CTA at block (0, 7):
ArithTuple(0, 112) o (_16, _16):(_1@1, _1@0)
  (0,112)  (1,112)  ...  (15,112)
  (0,113)  (1,113)  ...  (15,113)
  ...
  (0,127)  (1,127)  ...  (15,127)
```

These coordinate tensors support the same CuTe algebra (local_tile, partition, slice)
as regular tensors. The difference: out-of-bounds coordinates are silently predicated
by the TMA hardware — no manual bounds checking needed.

In CuteDSL, this is handled inside `tma_partition()`: it takes MMA-partitioned GMEM
tensors (which are already coordinate tensors from `thr_mma.partition_A(gA)`) and
produces correctly indexed source/destination tensors.

---

## Synchronization

### TMA Load: mbarrier (async memory barrier)

TMA load writes to SMEM asynchronously. You must wait for completion via an mbarrier.

```
Timeline:  [thread 0 issues TMA] ----async copy----> [SMEM written]
           [all threads]         wait_barrier ------> [can read SMEM]
```

In CuteDSL, this is abstracted by **pipelines**:

```python
# Create pipeline (host-side setup, runs in kernel)
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=4,                # Double/quad buffering
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=num_tma_copy_bytes, # Expected bytes per TMA operation
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
).make_participants()

# Producer (load warp, usually warp 0):
handle = ab_producer.acquire_and_advance()  # Get empty slot + barrier
cute.copy(tma_atom, src, dst, tma_bar_ptr=handle.barrier)

# Consumer (MMA warp):
handle = ab_consumer.wait_and_advance()     # Wait for data ready
# ... use data in sA[..., handle.index] ...
handle.release()                            # Signal slot is free
```

The pipeline manages:
- mbarrier initialization, arrive counts, transaction bytes
- Phase bit flipping across stages
- Producer/consumer synchronization

### TMA Store: fence (proxy fence)

TMA store reads from SMEM. You must ensure SMEM writes are visible BEFORE issuing TMA store.

```
Timeline:  [all threads write SMEM] -> fence -> [thread 0 issues TMA store]
```

```python
# All threads write SMEM data
__syncthreads()
tma_store_fence()          # fence.proxy.async.shared::cta

# Single thread issues TMA store
if threadIdx.x == 0:
    cute.copy(tma_store, smem_src, gmem_dst_coords)
    tma_store_arrive()     # Commit the store group
    tma_store_wait(0)      # Wait for all stores to complete
```

**Summary table:**

| Operation | Direction    | Sync method  | When to sync        |
|-----------|-------------|-------------|---------------------|
| TMA Load  | GMEM→SMEM  | mbarrier    | After the copy      |
| TMA Store | SMEM→GMEM  | proxy fence | Before the copy     |

---

## SMEM Layouts and Swizzle

TMA requires the SMEM layout to match what the MMA unit expects. CuteDSL provides helpers:

```python
# Creates a ComposedLayout = (outer_shape, swizzle_pattern)
a_smem_layout = sm100_utils.make_smem_layout_a(
    tiled_mma, mma_tiler_mnk, a_dtype, num_stages
)
# Shape: (MMA_MODES, TILE_M, TILE_K, NUM_STAGES) with bank-conflict-avoiding swizzle

# For TMA atom creation, remove the stage dimension:
a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])

# In kernel, allocate with swizzle:
sA = smem.allocate_tensor(
    element_type=io_dtype,
    layout=a_smem_layout.outer,     # Logical shape
    byte_alignment=128,
    swizzle=a_smem_layout.inner,    # Physical swizzle pattern
)
```

The swizzle is chosen automatically based on dtype and major mode to avoid SMEM bank conflicts
when the MMA unit reads the data.

---

## Partitioning Flow (GEMM example end-to-end)

```python
# === HOST ===
# 1. Create GMEM tensors from torch
a_tensor = from_dlpack(a, assumed_align=32).mark_layout_dynamic(leading_dim=1)

# 2. Create TMA atoms
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    a_tensor, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma,
)

# 3. Pass to kernel: tma_atom, tma_tensor (= coordinate tensor), smem_layout

# === KERNEL ===
# 4. Tile GMEM by CTA coordinate → get this CTA's tile (still coordinates)
gA = cute.local_tile(mA_mkl, mma_tiler_mnk, (bidx, bidy, None), proj=(1, None, 1))
# gA shape: (TILE_M, TILE_K, NUM_K_TILES)

# 5. Partition for MMA thread mapping → ArithTuple coordinates
tCgA = thr_mma.partition_A(gA)
# tCgA shape: (MMA, MMA_M, MMA_K, NUM_K_TILES)

# 6. Partition for TMA → separate SMEM and GMEM index tensors
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a, 0, cute.make_layout(1),
    cute.group_modes(sA, 0, 3),     # Group (MMA, MMA_M, MMA_K) → single mode
    cute.group_modes(tCgA, 0, 3),   # Group (MMA, MMA_M, MMA_K) → single mode
)
# tAsA: ((vals), NUM_STAGES)  — indexes into SMEM stages
# tAgA: ((vals), NUM_K_TILES) — indexes into GMEM K-tiles

# 7. Issue TMA copy: pick K-tile from GMEM, stage from SMEM
cute.copy(tma_atom_a, tAgA[(None, k_idx)], tAsA[(None, stage_idx)],
          tma_bar_ptr=barrier_ptr)
```

---

## TMA Store Reduce

Change one line to get atomic reduce instead of plain store:

```python
# Plain store:
tma_store = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), ...)
# Reduce-add (for split-K GEMM accumulation):
tma_reduce = cpasync.make_tiled_tma_atom(
    cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD), ...
)
```

Supported: ADD, MIN, MAX, AND, OR, XOR, INC, DEC. Maps to `cp.reduce.async.bulk.tensor`.

---

## TMA Load Multicast

When multiple CTAs in a cluster need the same GMEM tile (e.g., in GEMM, all CTAs sharing
the same M-row need the same B tile), multicast loads the data once and delivers to all.

```python
# Host: use Multicast op + cluster shape
op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
tma_atom_b, _ = cute.nvgpu.make_tiled_tma_atom_B(
    op, b_tensor, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma,
    cluster_shape_vmnk,  # e.g., (1, 2, 1, 1) — 2 CTAs along M
)

# Kernel: compute multicast mask and pass to copy
mcast_mask = cpasync.create_tma_multicast_mask(
    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1  # M-mode for B
)
cute.copy(tma_atom_b, src, dst, tma_bar_ptr=bar, mcast_mask=mcast_mask)
```

Each CTA in the cluster uses its `block_rank_in_cluster` as the TMA partition coordinate
so the data lands at the correct offset in each CTA's SMEM.

---

## Dynamic TMA Descriptor Updates

For variable-length batching (different shapes per batch), update the descriptor at runtime:

```python
# Copy descriptor to SMEM once at init
cpasync.copy_tensormap(tma_atom, smem_desc_ptr)

# Per-batch: update shape/stride/base_ptr in the SMEM copy
cpasync.update_tma_descriptor(tma_atom, new_gmem_tensor, smem_desc_ptr)
cpasync.fence_tma_desc_release()          # Release fence
cpasync.fence_tma_desc_acquire(smem_desc_ptr)  # Acquire fence
# Now safe to use updated descriptor for TMA copies
```

---

## TMA Operation Types Summary

| CuteDSL Op Class                         | PTX Instruction              | Direction   |
|------------------------------------------|------------------------------|-------------|
| `CopyBulkTensorTileG2SOp`               | `cp.async.bulk.tensor`       | GMEM→SMEM  |
| `CopyBulkTensorTileG2SMulticastOp`      | `cp.async.bulk.tensor` +mcast| GMEM→SMEM  |
| `CopyBulkTensorTileS2GOp`               | `cp.async.bulk.tensor`       | SMEM→GMEM  |
| `CopyReduceBulkTensorTileS2GOp(ADD)`    | `cp.reduce.async.bulk.tensor`| SMEM→GMEM  |

---

## Key Code References

All CuteDSL library paths below use the shorthand `<cutlass>/` for the third-party package root:
`fbsource/third-party/pypi/nvidia-cutlass-dsl/<VERSION>/_whl_common_files/nvidia_cutlass_dsl/python_packages/cutlass/`

The version changes over time — check `fbsource/third-party/pypi/nvidia-cutlass-dsl/VERSION.bzl` for the current default.

Notebooks and examples are NOT in the third-party package. They live in `fbcode/peis/cutlass/examples/python/CuTeDSL/`.

### CuteDSL Library (third-party)

| What                           | Path from `<cutlass>/`                                   |
|--------------------------------|----------------------------------------------------------|
| `make_tiled_tma_atom` (generic)| `cute/nvgpu/cpasync/helpers.py:43`                       |
| `tma_partition()`              | `cute/nvgpu/cpasync/helpers.py:177`                      |
| `create_tma_multicast_mask()`  | `cute/nvgpu/cpasync/helpers.py:204`                      |
| TMA descriptor update/fence    | `cute/nvgpu/cpasync/helpers.py:264-360`                  |
| `make_tiled_tma_atom_A/B`      | `cute/nvgpu/helpers.py:36,148`                           |
| TMA copy ops (G2S/S2G/Reduce)  | `cute/nvgpu/cpasync/copy.py`                             |
| SMEM layout helpers            | `utils/blackwell_helpers.py`                             |
| Pipeline classes               | `pipeline/`                                              |

### Examples (fbcode/peis/cutlass — internal, not in third-party package)

| What                           | Path                                                                           |
|--------------------------------|--------------------------------------------------------------------------------|
| Simplest GEMM with TMA         | `peis/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py` |
| Dense GEMM example             | `peis/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py`                |
| Flash MHA example              | `peis/cutlass/examples/python/CuTeDSL/blackwell/fmha.py`                      |

### Production Kernels (fbcode)

| What                           | Path                                                     |
|--------------------------------|----------------------------------------------------------|
| `tma_get_copy_fn` (high-level) | `ads_mkl/ops/cute_dsl/fa4/src/copy_utils.py:315`        |
| Flash Attention bwd (TMA)      | `ads_mkl/ops/cute_dsl/fa4/src/flash_bwd_sm100.py`       |
| GEMM SM100 (TMA)               | `ads_mkl/ops/cute_dsl/quack/gemm_sm100.py`              |
