// Package memory exercises the v0.3 features: dynamic shared memory,
// __constant__ and __device__ globals reachable from the host, launch
// bounds, device printf, panic/copy/slices, __ldg / volatile / float4,
// half precision, cp.async and a grid-wide barrier.
package memory

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

const blockSize = 256

// ---- extern __shared__

var dyn cuda.DynShared[float32]

// DynReverse reverses each block's elements through dynamic shared memory;
// the host launches with SharedMemBytes = blockDim.x * 4.
func DynReverse(in, out cuda.Buf[float32], n int32) {
	tid := cuda.ThreadIdxX()
	i := cuda.GlobalIdX()
	b := dyn.Buf()
	if i < n {
		b.Set(tid, in.At(i))
	}
	cuda.SyncThreads()
	j := cuda.BlockDimX() - 1 - tid
	if i < n {
		out.Set(i, b.At(j))
	}
}

// ---- __constant__ and __device__ globals

// Coef is __constant__ memory with an initialiser; the host may overwrite
// it with mod.Global("Coef") + WriteGlobal.
var Coef = cuda.Constant[[4]float32]{V: [4]float32{1, 2, 3, 4}}

// Scale is a __device__ global the host writes before the launch.
var Scale float32 = 1

// Counter is a __device__ global the host reads after the launch.
var Counter int32

// Poly computes out = Scale * (c0 + c1 x + c2 x^2 + c3 x^3) and counts
// elements in Counter.
func Poly(x, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	c := Coef.Get()
	v := x.At(i)
	out.Set(i, Scale*(c[0]+v*(c[1]+v*(c[2]+v*c[3]))))
	cuda.AtomicAddInt32Block(&Counter, 0) // no-op, keeps the symbol exercised
	if cuda.ThreadIdxX() == 0 {
		cuda.AtomicMaxInt32(&Counter, n)
	}
}

// ---- __launch_bounds__ / __maxnreg__

// Bounded is __launch_bounds__(128, 2) __maxnreg__(32): the test checks the
// PTX for .maxntid / .minnctapersm / .maxnreg.
//
//cuda:launch_bounds 128 2
//cuda:maxnreg 32
func Bounded(out cuda.Buf[int32]) { out.Set(cuda.GlobalIdX(), cuda.BlockDimX()) }

// ---- printf

// Print prints from thread 0 of each block.
func Print(x cuda.Buf[float32]) {
	if cuda.ThreadIdxX() == 0 {
		b := cuda.BlockIdxX()
		cuda.Printf("block %d: x=%.2f n=%lld hex=%x\n",
			cuda.Args().Int(b).Float(x.At(b)).Int64(int64(b)*1000000000000).Uint(0xbeef))
	}
}

// ---- Go semantics: panic, copy, slices, local arrays that escape

// PanicCopy copies a window of `in` into a local array with copy(), sums it
// with range, and panics (trap -> launch error) when bad != 0.
func PanicCopy(in, out cuda.Buf[float32], n, bad int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	if bad != 0 && i == 0 {
		panic("bad input")
	}
	var win [4]float32
	copy(win[:], in.View(i&^3, 4)) // the aligned window of 4 around i
	var s float32
	for _, v := range win {
		s += v
	}
	out.Set(i, s)
}

// Slices takes Go slices (host: host.ArgSlice); indexing is bounds-checked.
func Slices(out []float32, in []float32) {
	i := int(cuda.GlobalIdX())
	if i < len(out) {
		out[i] = in[i] * 2
	}
}

// SliceOOB indexes past the end: Go bounds check -> trap -> launch error.
func SliceOOB(out []float32) {
	i := int(cuda.GlobalIdX())
	out[i+len(out)] = 1
}

// ---- __ldg, volatile, float4

// Vec4 scales 4 floats per thread with one 128-bit load and store
// (ld.global.v4.f32 in the PTX). n is the number of float4s.
func Vec4(in, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := cuda.LoadFloat4(in.Ptr(i * 4))
	for k := range v {
		v[k] *= 2
	}
	cuda.StoreFloat4(out.Ptr(i*4), v)
}

// Ldg sums neighbours through the read-only cache (ld.global.nc).
func Ldg(in cuda.Buf[float32], idx cuda.Buf[int32], out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	j := idx.Ldg(i)
	out.Set(i, in.Ldg(i)+cuda.LdgF32(in.Ptr(j)))
}

// Volatile: thread 0 of the block publishes a value with a volatile store
// that the other threads poll with volatile loads (ld.volatile / st.volatile).
func Volatile(flag cuda.Buf[int32], out cuda.Buf[int32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	b := cuda.BlockIdxX()
	if cuda.ThreadIdxX() == 0 {
		flag.SetVolatile(b, b+1)
	}
	for flag.Volatile(b) == 0 {
	}
	out.Set(i, cuda.VolatileLoadInt32(flag.Ptr(b)))
}

// ---- half precision

// HalfOps: out[i*2] = x*x+1 in binary16, out[i*2+1] = x*2+x in bfloat16.
func HalfOps(x, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	h := cuda.FloatToHalf(x.At(i))
	one := cuda.FloatToHalf(1)
	out.Set(i*2, h.FMA(h, one).Float32())
	b := cuda.FloatToBFloat16(x.At(i))
	two := cuda.FloatToBFloat16(2)
	out.Set(i*2+1, b.Mul(two).Add(b).Float32())
}

// ---- cp.async

var tile cuda.Shared[[blockSize]float32]

// CpAsyncReverse fills the block's tile with cp.async and writes it back
// reversed. n must be a multiple of blockSize.
func CpAsyncReverse(in, out cuda.Buf[float32], n int32) {
	t := tile.Get()
	tid := cuda.ThreadIdxX()
	i := cuda.GlobalIdX()
	if i < n {
		cuda.CpAsync(&t[tid], in.Ptr(i))
	}
	cuda.CpAsyncCommit()
	cuda.CpAsyncWaitAll()
	cuda.SyncThreads()
	if i < n {
		out.Set(i, t[blockSize-1-tid])
	}
}

// ---- grid-wide barrier (cooperative launch)

// GridSum: phase 1 every block writes data[block]; after the grid barrier
// every block sums all of data into out[block]. Launch cooperatively.
func GridSum(bar *cuda.GridBarrier, data, out cuda.Buf[int32]) {
	b := cuda.BlockIdxX()
	if cuda.ThreadIdxX() == 0 {
		data.Set(b, b+1)
	}
	bar.Sync()
	if cuda.ThreadIdxX() == 0 {
		var s int32
		for k := int32(0); k < cuda.GridDimX(); k++ {
			s += data.At(k)
		}
		out.Set(b, s)
	}
}
