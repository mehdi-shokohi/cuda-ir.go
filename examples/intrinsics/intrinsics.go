// Package intrinsics exercises the v0.2 device API: special registers,
// integer intrinsics (and math/bits), warp match/reduce/64-bit shuffles,
// named barriers, atomicMin/Max & friends, the extended libdevice math.
package intrinsics

import (
	"math"
	"math/bits"
	"sync/atomic"
	"unsafe"

	"github.com/mehdi-shokohi/cuda-ir.go/cuda"
)

// Sregs writes a bit set per thread: each bit is one special-register
// check that passed. out[n..2n) receives the SM id for inspection.
func Sregs(out cuda.Buf[int32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	lane := uint32(cuda.LaneID())
	var ok int32
	if cuda.SMID() < cuda.NumSMs() {
		ok |= 1
	}
	if cuda.LaneMaskEq() == 1<<lane {
		ok |= 2
	}
	if cuda.LaneMaskLt() == (1<<lane)-1 {
		ok |= 4
	}
	if cuda.LaneMaskLe() == (1<<lane)|((1<<lane)-1) {
		ok |= 8
	}
	if cuda.LaneMaskGt() == ^((1 << lane) | ((1 << lane) - 1)) {
		ok |= 16
	}
	if cuda.LaneMaskGe() == ^((1 << lane) - 1) {
		ok |= 32
	}
	if cuda.WarpID() < cuda.NumWarps() {
		ok |= 64
	}
	if cuda.GridID() >= 0 {
		ok |= 128
	}
	out.Set(i, ok)
	out.Set(n+i, cuda.SMID())
}

// IntMath: out[i*8+k] = the k-th integer intrinsic of in[i].
func IntMath(in cuda.Buf[uint32], out cuda.Buf[uint32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	x := in.At(i)
	o := i * 8
	out.Set(o+0, uint32(cuda.Popc(x)))
	out.Set(o+1, uint32(cuda.Clz(x)))
	out.Set(o+2, uint32(cuda.Ffs(int32(x))))
	out.Set(o+3, cuda.Brev(x))
	out.Set(o+4, cuda.MulHiU(x, 0x9E3779B9))
	out.Set(o+5, cuda.FunnelShiftL(x, ^x, 7))
	out.Set(o+6, cuda.BytePerm(x, ^x, 0x7654))
	out.Set(o+7, cuda.SadU(x, 12345, 7)+cuda.HAddU(x, 0xffffffff)+uint32(cuda.Popc64(uint64(x)<<32|uint64(x))))
}

// Bits uses Go's math/bits in a kernel (the compiler maps it to the same
// intrinsics): out[i*4+k].
func Bits(in cuda.Buf[uint32], out cuda.Buf[uint32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	x := in.At(i)
	o := i * 4
	out.Set(o+0, uint32(bits.OnesCount32(x)+bits.LeadingZeros32(x)*64+bits.TrailingZeros32(x)*4096))
	out.Set(o+1, bits.Reverse32(x))
	out.Set(o+2, bits.RotateLeft32(x, -5))
	out.Set(o+3, uint32(bits.Len64(uint64(x)<<3)+bits.OnesCount8(uint8(x))*128)+uint32(bits.ReverseBytes16(uint16(x)))<<16)
}

// WarpOps: warp reductions and matches; n must be a multiple of 32.
// out[i*8+k]: 0 ReduceAdd, 1 popc(MatchAny(x&3)), 2 MatchAll pred,
// 3 ShflDownWidth(x, 1, 16), 4 ReduceMax, 5 ReduceXor, 6 ShflWidth(lane 3 of 8)
// out64[i] = ShflXor64(x*1000003, 1)
func WarpOps(in cuda.Buf[int32], out cuda.Buf[int32], out64 cuda.Buf[int64], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	x := in.At(i)
	o := i * 8
	out.Set(o+0, cuda.ReduceAdd(cuda.FullMask, x))
	out.Set(o+1, cuda.Popc(cuda.MatchAny(cuda.FullMask, x&3)))
	_, pred := cuda.MatchAll(cuda.FullMask, 7)
	m2, pred2 := cuda.MatchAll(cuda.FullMask, x&1) // differs across lanes
	var p int32
	if pred {
		p = 1
	}
	if pred2 || m2 != 0 {
		p |= 2
	}
	out.Set(o+2, p)
	out.Set(o+3, cuda.ShflDownWidth(cuda.FullMask, x, 1, 16))
	out.Set(o+4, cuda.ReduceMax(cuda.FullMask, x))
	out.Set(o+5, int32(cuda.ReduceXor(cuda.FullMask, uint32(x))))
	out.Set(o+6, cuda.ShflWidth(cuda.FullMask, x, 3, 8))
	out.Set(o+7, int32(cuda.MatchAny64(cuda.FullMask, int64(x)<<40)))
	out64.Set(i, cuda.ShflXor64(cuda.FullMask, int64(x)*1000003, 1))
}

const blockSize = 128

var tile cuda.Shared[[64]int32]

// Barriers: named barrier producer/consumer between warps 0-1 and 2-3 of a
// 128-thread block, plus __syncthreads_count/and/or and __isShared/__isGlobal.
// out[block*8]: consumer sum, count, and, or, isShared(tile), isGlobal(out),
// isShared(out), NanoSleep marker.
func Barriers(out cuda.Buf[int32]) {
	t := tile.Get()
	tid := cuda.ThreadIdxX()
	b := cuda.BlockIdxX()
	if tid < 64 {
		t[tid] = tid + 1 // producers
		cuda.ArriveBarrier(1, 128)
	} else {
		cuda.SyncBarrier(1, 128) // consumers wait for the producers
		if tid == 64 {
			var s int32
			for k := int32(0); k < 64; k++ {
				s += t[k]
			}
			out.Set(b*8+0, s)
		}
	}
	cnt := cuda.SyncThreadsCount(tid%2 == 0)
	and := cuda.SyncThreadsAnd(tid < 200)
	or := cuda.SyncThreadsOr(tid == 77)
	cuda.NanoSleep(100)
	if tid == 0 {
		out.Set(b*8+1, cnt)
		out.Set(b*8+2, b2i(and))
		out.Set(b*8+3, b2i(or))
		out.Set(b*8+4, b2i(cuda.IsShared(unsafe.Pointer(&t[0]))))
		out.Set(b*8+5, b2i(cuda.IsGlobal(unsafe.Pointer(out.Ptr(0)))))
		out.Set(b*8+6, b2i(cuda.IsShared(unsafe.Pointer(out.Ptr(0)))))
		out.Set(b*8+7, 1)
	}
}

func b2i(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// Atomics: min/max/scoped/float exchange+CAS/acquire-release.
// out32: 0 min, 1 max, 2 umin, 3 umax, 4 system add count, 5 release flag,
// 8+block: block-scope counts. out64: 0 min, 1 max, 2 umax. outf: 0 last
// exchanged value, 1 max via CAS, 2 fadd sum.
func Atomics(vals cuda.Buf[int32], out32 cuda.Buf[int32], out64 cuda.Buf[int64], outf cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := vals.At(i)
	cuda.AtomicMinInt32(out32.Ptr(0), v)
	cuda.AtomicMaxInt32(out32.Ptr(1), v)
	cuda.AtomicMinUint32((*uint32)(unsafe.Pointer(out32.Ptr(2))), uint32(v))
	cuda.AtomicMaxUint32((*uint32)(unsafe.Pointer(out32.Ptr(3))), uint32(v))
	cuda.AtomicAddInt32System(out32.Ptr(4), 1)
	cuda.AtomicAddInt32Block(out32.Ptr(8+cuda.BlockIdxX()), 1)
	cuda.AtomicMinInt64(out64.Ptr(0), int64(v)*1000000)
	cuda.AtomicMaxInt64(out64.Ptr(1), int64(v)*1000000)
	cuda.AtomicMaxUint64((*uint64)(unsafe.Pointer(out64.Ptr(2))), uint64(int64(v)))
	f := float32(v) * 0.5
	cuda.AtomicSwapFloat32(outf.Ptr(0), f)
	for { // atomicMax on floats via CAS, the classic pattern
		old := cuda.VolatileLoadFloat32(outf.Ptr(1))
		if f <= old || cuda.AtomicCASFloat32(outf.Ptr(1), old, f) {
			break
		}
	}
	cuda.AtomicAddFloat32(outf.Ptr(2), 1)
	if i == 0 {
		cuda.StoreReleaseInt32(out32.Ptr(5), 1)
	}
	_ = cuda.LoadAcquireInt32(out32.Ptr(5))
	atomic.AddInt32(out32.Ptr(6), 1)
}

// MathExt: out[i*16+k] = the k-th function of x (x in [0,1)), checked
// against Go's math on the host.
func MathExt(x cuda.Buf[float32], out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := x.At(i)
	o := i * 16
	out.Set(o+0, cuda.Asin(v))
	out.Set(o+1, cuda.Atan2(v, 1))
	out.Set(o+2, cuda.Cbrt(v))
	out.Set(o+3, cuda.Hypot(v, 1))
	out.Set(o+4, cuda.Log1p(v))
	out.Set(o+5, cuda.Expm1(v))
	out.Set(o+6, cuda.Fmod(v*10, 3))
	out.Set(o+7, cuda.Sinpi(v))
	fr, e := cuda.Frexp(v*100 + 1)
	out.Set(o+8, cuda.Ldexp(fr, e))
	ip, f := cuda.Modf(v*100 + 1)
	out.Set(o+9, ip+f)
	s, c := cuda.Sincos(v * 7)
	out.Set(o+10, s*s+c*c)
	out.Set(o+11, cuda.Erf(v)+cuda.Erfc(v))
	out.Set(o+12, float32(cuda.Float32ToInt32RN(v*10)))
	out.Set(o+13, cuda.Rint(v*10)+cuda.Saturate(v*3-1))
	out.Set(o+14, float32(math.Sqrt(float64(v)))+math.Float32frombits(math.Float32bits(v)))
	out.Set(o+15, cuda.Tgamma(v+2)/(v+1)+cuda.Lgamma(v+1)) // Gamma(x+2)/(x+1) = Gamma(x+1)
}

// Math64: the float64 twins, out[i*8+k].
func Math64(x cuda.Buf[float64], out cuda.Buf[float64], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := x.At(i)
	o := i * 8
	out.Set(o+0, cuda.Tan64(v))
	out.Set(o+1, cuda.Floor64(v*10)+cuda.Ceil64(v*10)+cuda.Trunc64(-v*10)+cuda.Round64(v*10))
	out.Set(o+2, cuda.Min64(v, 0.5)+cuda.Max64(v, 0.5))
	out.Set(o+3, cuda.Rsqrt64(v+1))
	out.Set(o+4, cuda.Exp2_64(v)+cuda.Log2_64(v+1))
	out.Set(o+5, cuda.Tanh64(v)+cuda.Erf64(v))
	out.Set(o+6, cuda.Atan2_64(v, 1)+cuda.Cbrt64(v)+cuda.Hypot64(v, 1))
	out.Set(o+7, cuda.Log10_64(v+1)+cuda.Log1p64(v)+cuda.Expm1_64(v))
}
