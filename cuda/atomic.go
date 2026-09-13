package cuda

import _ "unsafe" // for go:linkname

// Integer atomics: use Go's sync/atomic directly — llgo lowers
// atomic.AddInt32/CompareAndSwapInt32/LoadInt32/StoreInt32/... on
// device pointers to native atomicrmw/cmpxchg, i.e. PTX atom.* instructions:
//
//	atomic.AddInt32(counter.Ptr(0), 1)
//
// Float atomics have no Go equivalent, so they are provided here.

//go:linkname atomicAddF32 llvm.nvvm.atomic.load.add.f32.p0
func atomicAddF32(p *float32, v float32) float32

//go:linkname atomicAddF64 llvm.nvvm.atomic.load.add.f64.p0
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
