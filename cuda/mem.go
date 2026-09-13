package cuda

import "unsafe"

// ---- dynamic shared memory (extern __shared__)

// DynShared declares dynamic shared memory: the size comes from the launch
// (gocudrv: LaunchConfig.SharedMemBytes) instead of the type. Use it as a
// package-level variable; every DynShared variable of a package aliases
// the same block, as in CUDA.
//
//	var buf cuda.DynShared[float32]
//
//	func K(...) {
//		b := buf.Buf()          // Buf[float32] over the block's memory
//		b.Set(cuda.ThreadIdxX(), ...)
type DynShared[T any] struct{ v [1]T } // the compiler replaces it with an extern [] array

// Ptr returns the base of the block's dynamic shared memory.
func (s *DynShared[T]) Ptr() *T { return &s.v[0] }

// Buf returns an unchecked view of the dynamic shared memory.
func (s *DynShared[T]) Buf() Buf[T] { return Buf[T]{s.Ptr()} }

// ---- constant memory (__constant__)

// Constant declares __constant__ memory: read-only in kernels, written by
// the host through the module (gocudrv: mod.Global("Name") + WriteGlobal).
// Use it as an exported package-level variable so the host can find it by
// name; a composite-literal initialiser becomes the initial contents.
//
//	var Coef = cuda.Constant[[4]float32]{V: [4]float32{1, 2, 3, 4}}
//
//	func K(...) { c := Coef.Get(); y := c[0]*x + c[1] ... }
type Constant[T any] struct{ V T }

// Get returns a pointer to the constant data (reads become ld.const).
func (c *Constant[T]) Get() *T { return &c.V }

// ---- read-only / volatile loads (__ldg, volatile)

//go:linkname ldgF32 cudair.ldg.float
func ldgF32(p *float32) float32

//go:linkname ldgF64 cudair.ldg.double
func ldgF64(p *float64) float64

//go:linkname ldgI32 cudair.ldg.i32
func ldgI32(p *int32) int32

//go:linkname ldgI64 cudair.ldg.i64
func ldgI64(p *int64) int64

// LdgF32 is __ldg(const float*): a load through the read-only data cache
// (ld.global.nc). The memory must not be written during the kernel.
func LdgF32(p *float32) float32 { return ldgF32(p) }
func LdgF64(p *float64) float64 { return ldgF64(p) }
func LdgI32(p *int32) int32     { return ldgI32(p) }
func LdgI64(p *int64) int64     { return ldgI64(p) }

//go:linkname volatileLoadI32 cudair.load.volatile.i32
func volatileLoadI32(p *int32) int32

//go:linkname volatileLoadI64 cudair.load.volatile.i64
func volatileLoadI64(p *int64) int64

//go:linkname volatileLoadF32 cudair.load.volatile.float
func volatileLoadF32(p *float32) float32

//go:linkname volatileLoadF64 cudair.load.volatile.double
func volatileLoadF64(p *float64) float64

//go:linkname volatileStoreI32 cudair.store.volatile.i32
func volatileStoreI32(p *int32, v int32)

//go:linkname volatileStoreI64 cudair.store.volatile.i64
func volatileStoreI64(p *int64, v int64)

//go:linkname volatileStoreF32 cudair.store.volatile.float
func volatileStoreF32(p *float32, v float32)

//go:linkname volatileStoreF64 cudair.store.volatile.double
func volatileStoreF64(p *float64, v float64)

// VolatileLoadInt32 & co are `volatile` accesses: never cached in a
// register, always re-read from / written to memory (ld.volatile /
// st.volatile), for flags polled across threads.
func VolatileLoadInt32(p *int32) int32       { return volatileLoadI32(p) }
func VolatileLoadInt64(p *int64) int64       { return volatileLoadI64(p) }
func VolatileLoadFloat32(p *float32) float32 { return volatileLoadF32(p) }
func VolatileLoadFloat64(p *float64) float64 { return volatileLoadF64(p) }
func VolatileStoreInt32(p *int32, v int32)   { volatileStoreI32(p, v) }
func VolatileStoreInt64(p *int64, v int64)   { volatileStoreI64(p, v) }
func VolatileStoreFloat32(p *float32, v float32) {
	volatileStoreF32(p, v)
}
func VolatileStoreFloat64(p *float64, v float64) {
	volatileStoreF64(p, v)
}

// ---- 128-bit vector accesses (float4 / int4)

//go:linkname ldV4F32 cudair.ld.v4.float
func ldV4F32(dst *[4]float32, src *float32)

//go:linkname stV4F32 cudair.st.v4.float
func stV4F32(dst *float32, src *[4]float32)

//go:linkname ldV4I32 cudair.ld.v4.i32
func ldV4I32(dst *[4]int32, src *int32)

//go:linkname stV4I32 cudair.st.v4.i32
func stV4I32(dst *int32, src *[4]int32)

// LoadFloat4 reads four consecutive floats with one 128-bit access
// (ld.global.v4.f32, like reading a float4). p must be 16-byte aligned.
func LoadFloat4(p *float32) [4]float32 {
	var v [4]float32
	ldV4F32(&v, p)
	return v
}

// StoreFloat4 writes four consecutive floats with one 128-bit access.
func StoreFloat4(p *float32, v [4]float32) { stV4F32(p, &v) }

// LoadInt4 / StoreInt4 are the int4 variants.
func LoadInt4(p *int32) [4]int32 {
	var v [4]int32
	ldV4I32(&v, p)
	return v
}

func StoreInt4(p *int32, v [4]int32) { stV4I32(p, &v) }

// ---- asynchronous global -> shared copies (cp.async, sm_80+)

//go:linkname cpAsyncCA4 cudair.cp.async.ca.4
func cpAsyncCA4(dst, src unsafe.Pointer)

//go:linkname cpAsyncCA8 cudair.cp.async.ca.8
func cpAsyncCA8(dst, src unsafe.Pointer)

//go:linkname cpAsyncCA16 cudair.cp.async.ca.16
func cpAsyncCA16(dst, src unsafe.Pointer)

//go:linkname cpAsyncCG16 cudair.cp.async.cg.16
func cpAsyncCG16(dst, src unsafe.Pointer)

// CpAsync4/8/16 are __pipeline_memcpy_async / cp.async.ca: copy 4, 8 or 16
// bytes from global (src) to shared memory (dst) without going through
// registers. Group the copies with CpAsyncCommit and wait with
// CpAsyncWaitGroup / CpAsyncWaitAll before reading dst.
func CpAsync4(dst, src unsafe.Pointer)  { cpAsyncCA4(dst, src) }
func CpAsync8(dst, src unsafe.Pointer)  { cpAsyncCA8(dst, src) }
func CpAsync16(dst, src unsafe.Pointer) { cpAsyncCA16(dst, src) }

// CpAsync16CG is the L2-only (cache-global) 16-byte variant.
func CpAsync16CG(dst, src unsafe.Pointer) { cpAsyncCG16(dst, src) }

// CpAsyncCommit is cp.async.commit_group.
//
//go:linkname CpAsyncCommit llvm.nvvm.cp.async.commit.group
func CpAsyncCommit()

// CpAsyncWaitAll is cp.async.wait_all.
//
//go:linkname CpAsyncWaitAll llvm.nvvm.cp.async.wait.all
func CpAsyncWaitAll()

//go:linkname cpAsyncWaitGroup llvm.nvvm.cp.async.wait.group
func cpAsyncWaitGroup(n int32)

// CpAsyncWait1 / CpAsyncWait2 are cp.async.wait_group 1 / 2: wait until at
// most one / two of the committed groups are still pending (the PTX
// instruction takes an immediate, hence one function per count).
func CpAsyncWait1() { cpAsyncWaitGroup(1) }
func CpAsyncWait2() { cpAsyncWaitGroup(2) }
