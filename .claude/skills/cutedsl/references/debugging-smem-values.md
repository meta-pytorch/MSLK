# Debugging SMEM Values in CuTe DSL

## Quick Pattern: Print SMEM as Hex

```python
# Recast to Uint32, create flat view, print with offset
flat = cute.make_tensor(
    cute.recast_ptr(sSMEM.iterator, dtype=cutlass.Uint32),
    cute.make_layout(total_bytes // 4),
)
offset = stage_index * (stage_stride_bytes // 4)
cute.printf("[tag=%d] %08x %08x\n", stage_index, flat[offset], flat[offset+1])
```

## Calculating Stage Stride

From layout `shape:stride`, the last stride value is the stage stride in bytes:
```
((((32,4),1),(32,1)),1,4,4):((((16,4),0),(0,0)),0,1,512)
                                                    ^^^
                                        stage stride = 512 bytes = 128 uint32s
```

## Common Issue: Data Zeros but Should Be Non-Zero

**Symptom:** SFV zeros, V data correct → Issue is SFV TMA, not sync

**Likely cause:** Wrong `local_tile` coordinate order
```python
# WRONG - iterates on wrong dimension
gSFV = cute.local_tile(mSFV, tiler, (None, 0))

# CORRECT
gSFV = cute.local_tile(mSFV, tiler, (0, None))
```

## Don'ts

- Don't print gmem from TMA descriptors (`tSFVgSFV.iterator` is a tuple, not pointer)
- Don't guess stage strides - derive from layout or known sizes

## Key Lesson

When porting code (e.g., GDPA → FA4), **diff the critical paths first**. A simple coordinate swap `(None, 0)` vs `(0, None)` caused hours of debugging.
