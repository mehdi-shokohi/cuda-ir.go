package cuda

import (
	"sync/atomic"
	"unsafe"
)

// Integer atomics: use Go's sync/atomic directly — llgo lowers
// atomic.AddInt32/CompareAndSwapInt32/LoadInt32/StoreInt32/... on
// device pointers to native atomicrmw/cmpxchg, i.e. PTX atom.* instructions:
//
//	atomic.AddInt32(counter.Ptr(0), 1)
//
// Float atomics have no Go equivalent, so they are provided here.

//go:linkname atomicAddF32 cudair.atomic.fadd.float
func atomicAddF32(p *float32, v float32) float32

//go:linkname atomicAddF64 cudair.atomic.fadd.double
func atomicAddF64(p *float64, v float64) float64

//go:linkname atomicIncU32 llvm.nvvm.atomic.load.inc.32.p0
func atomicIncU32(p *uint32, v uint32) uint32

//go:linkname atomicDecU32 llvm.nvvm.atomic.load.dec.32.p0
func atomicDecU32(p *uint32, v uint32) uint32

// AtomicAddFloat32 is atomicAdd(float*, float); returns the old value.
func AtomicAddFloat32(p *float32, v float32) float32 { return atomicAddF32(p, v) }

// AtomicAddFloat64 is atomicAdd(double*, double); returns the old value.
func AtomicAddFloat64(p *float64, v float64) float64 { return atomicAddF64(p, v) }

// AtomicInc is atomicInc: old >= limit ? 0 : old+1; returns old.
func AtomicInc(p *uint32, limit uint32) uint32 { return atomicIncU32(p, limit) }

// AtomicDec is atomicDec: (old == 0 || old > limit) ? limit : old-1; returns old.
func AtomicDec(p *uint32, limit uint32) uint32 { return atomicDecU32(p, limit) }

// ---- atomicMin / atomicMax (atomicrmw min|max|umin|umax; IR helpers)

//go:linkname atomicMinI32 cudair.atomic.min.i32
func atomicMinI32(p *int32, v int32) int32

//go:linkname atomicMaxI32 cudair.atomic.max.i32
func atomicMaxI32(p *int32, v int32) int32

//go:linkname atomicMinU32 cudair.atomic.umin.i32
func atomicMinU32(p *uint32, v uint32) uint32

//go:linkname atomicMaxU32 cudair.atomic.umax.i32
func atomicMaxU32(p *uint32, v uint32) uint32

//go:linkname atomicMinI64 cudair.atomic.min.i64
func atomicMinI64(p *int64, v int64) int64

//go:linkname atomicMaxI64 cudair.atomic.max.i64
func atomicMaxI64(p *int64, v int64) int64

//go:linkname atomicMinU64 cudair.atomic.umin.i64
func atomicMinU64(p *uint64, v uint64) uint64

//go:linkname atomicMaxU64 cudair.atomic.umax.i64
func atomicMaxU64(p *uint64, v uint64) uint64

// AtomicMinInt32 is atomicMin(int*); returns the old value. The other
// Min/Max functions are the unsigned and 64-bit variants.
func AtomicMinInt32(p *int32, v int32) int32     { return atomicMinI32(p, v) }
func AtomicMaxInt32(p *int32, v int32) int32     { return atomicMaxI32(p, v) }
func AtomicMinUint32(p *uint32, v uint32) uint32 { return atomicMinU32(p, v) }
func AtomicMaxUint32(p *uint32, v uint32) uint32 { return atomicMaxU32(p, v) }
func AtomicMinInt64(p *int64, v int64) int64     { return atomicMinI64(p, v) }
func AtomicMaxInt64(p *int64, v int64) int64     { return atomicMaxI64(p, v) }
func AtomicMinUint64(p *uint64, v uint64) uint64 { return atomicMinU64(p, v) }
func AtomicMaxUint64(p *uint64, v uint64) uint64 { return atomicMaxU64(p, v) }

// ---- float exchange / CAS: the integer atomics on the bit pattern

// AtomicSwapFloat32 is atomicExch(float*); returns the old value.
func AtomicSwapFloat32(p *float32, v float32) float32 {
	return Float32FromBits(atomic.SwapUint32((*uint32)(unsafe.Pointer(p)), Float32Bits(v)))
}

// AtomicSwapFloat64 is atomicExch(double*).
func AtomicSwapFloat64(p *float64, v float64) float64 {
	return Float64FromBits(atomic.SwapUint64((*uint64)(unsafe.Pointer(p)), Float64Bits(v)))
}

// AtomicCASFloat32 is atomicCAS(float*, old, new) as a Go compare-and-swap
// (true if it stored). Compares bit patterns, like CUDA.
func AtomicCASFloat32(p *float32, old, new float32) bool {
	return atomic.CompareAndSwapUint32((*uint32)(unsafe.Pointer(p)), Float32Bits(old), Float32Bits(new))
}

// AtomicCASFloat64 is atomicCAS(double*, old, new).
func AtomicCASFloat64(p *float64, old, new float64) bool {
	return atomic.CompareAndSwapUint64((*uint64)(unsafe.Pointer(p)), Float64Bits(old), Float64Bits(new))
}

// ---- scoped atomics and acquire/release (cuda::atomic_ref memory orders)

//go:linkname atomicAddI32Block cudair.atomic.add.i32.block
func atomicAddI32Block(p *int32, v int32) int32

//go:linkname atomicAddI32System cudair.atomic.add.i32.system
func atomicAddI32System(p *int32, v int32) int32

//go:linkname atomicAddI64Block cudair.atomic.add.i64.block
func atomicAddI64Block(p *int64, v int64) int64

//go:linkname atomicAddI64System cudair.atomic.add.i64.system
func atomicAddI64System(p *int64, v int64) int64

// AtomicAddInt32Block is atomicAdd_block: only atomic w.r.t. threads of the
// same block (cheaper). sync/atomic.AddInt32 is the device-scope atomicAdd.
func AtomicAddInt32Block(p *int32, v int32) int32 { return atomicAddI32Block(p, v) }

// AtomicAddInt32System is atomicAdd_system: atomic w.r.t. the host and
// other devices too (for unified / pinned memory).
func AtomicAddInt32System(p *int32, v int32) int32 { return atomicAddI32System(p, v) }
func AtomicAddInt64Block(p *int64, v int64) int64  { return atomicAddI64Block(p, v) }
func AtomicAddInt64System(p *int64, v int64) int64 { return atomicAddI64System(p, v) }

//go:linkname loadAcquireI32 cudair.load.acquire.i32
func loadAcquireI32(p *int32) int32

//go:linkname loadAcquireI64 cudair.load.acquire.i64
func loadAcquireI64(p *int64) int64

//go:linkname storeReleaseI32 cudair.store.release.i32
func storeReleaseI32(p *int32, v int32)

//go:linkname storeReleaseI64 cudair.store.release.i64
func storeReleaseI64(p *int64, v int64)

// LoadAcquireInt32 is ld.acquire.gpu: an atomic load that orders later
// memory accesses after it (cuda::atomic_ref<int>::load(memory_order_acquire)).
func LoadAcquireInt32(p *int32) int32 { return loadAcquireI32(p) }
func LoadAcquireInt64(p *int64) int64 { return loadAcquireI64(p) }

// StoreReleaseInt32 is st.release.gpu: an atomic store that orders earlier
// memory accesses before it.
func StoreReleaseInt32(p *int32, v int32) { storeReleaseI32(p, v) }
func StoreReleaseInt64(p *int64, v int64) { storeReleaseI64(p, v) }
