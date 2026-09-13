package cuda

import _ "unsafe" // for go:linkname

// Integer intrinsics (__popc, __clz, __brev, __mulhi, ...). math/bits works
// in kernels too: `gocuda build` maps its functions to the same intrinsics.

//go:linkname ctpop32 llvm.ctpop.i32
func ctpop32(x uint32) uint32

//go:linkname ctpop64 llvm.ctpop.i64
func ctpop64(x uint64) uint64

//go:linkname ctlz32 llvm.ctlz.i32
func ctlz32(x uint32, zeroPoison bool) uint32

//go:linkname ctlz64 llvm.ctlz.i64
func ctlz64(x uint64, zeroPoison bool) uint64

//go:linkname bitrev32 llvm.bitreverse.i32
func bitrev32(x uint32) uint32

//go:linkname bitrev64 llvm.bitreverse.i64
func bitrev64(x uint64) uint64

//go:linkname fshl32 llvm.fshl.i32
func fshl32(hi, lo, shift uint32) uint32

//go:linkname fshr32 llvm.fshr.i32
func fshr32(hi, lo, shift uint32) uint32

// Popc is __popc: number of set bits.
func Popc(x uint32) int32 { return int32(ctpop32(x)) }

// Popc64 is __popcll.
func Popc64(x uint64) int32 { return int32(ctpop64(x)) }

// Clz is __clz: leading zero bits (32 for x == 0).
func Clz(x uint32) int32 { return int32(ctlz32(x, false)) }

// Clz64 is __clzll.
func Clz64(x uint64) int32 { return int32(ctlz64(x, false)) }

// Ffs is __ffs: 1-based position of the lowest set bit, 0 for x == 0.
//
//go:linkname Ffs __nv_ffs
func Ffs(x int32) int32

// Ffs64 is __ffsll.
//
//go:linkname Ffs64 __nv_ffsll
func Ffs64(x int64) int32

// Brev is __brev: bit reversal.
func Brev(x uint32) uint32 { return bitrev32(x) }

// Brev64 is __brevll.
func Brev64(x uint64) uint64 { return bitrev64(x) }

// BytePerm is __byte_perm(x, y, s): byte selection from the 8 bytes of x:y.
//
//go:linkname BytePerm __nv_byte_perm
func BytePerm(x, y, s uint32) uint32

// FunnelShiftL is __funnelshift_l(lo, hi, shift): the high 32 bits of
// (hi:lo) << (shift & 31).
func FunnelShiftL(lo, hi, shift uint32) uint32 { return fshl32(hi, lo, shift) }

// FunnelShiftR is __funnelshift_r(lo, hi, shift): the low 32 bits of
// (hi:lo) >> (shift & 31).
func FunnelShiftR(lo, hi, shift uint32) uint32 { return fshr32(hi, lo, shift) }

// MulHi is __mulhi: the high 32 bits of the 64-bit product.
//
//go:linkname MulHi __nv_mulhi
func MulHi(a, b int32) int32

// MulHiU is __umulhi.
//
//go:linkname MulHiU __nv_umulhi
func MulHiU(a, b uint32) uint32

// MulHi64 is __mul64hi: the high 64 bits of the 128-bit product.
//
//go:linkname MulHi64 __nv_mul64hi
func MulHi64(a, b int64) int64

// MulHiU64 is __umul64hi.
//
//go:linkname MulHiU64 __nv_umul64hi
func MulHiU64(a, b uint64) uint64

// Mul24 is __mul24: product of the low 24 bits (fast on old GPUs).
//
//go:linkname Mul24 __nv_mul24
func Mul24(a, b int32) int32

// Mul24U is __umul24.
//
//go:linkname Mul24U __nv_umul24
func Mul24U(a, b uint32) uint32

// Sad is __sad(a, b, c): |a-b| + c.
//
//go:linkname Sad __nv_sad
func Sad(a, b int32, c uint32) uint32

// SadU is __usad.
//
//go:linkname SadU __nv_usad
func SadU(a, b, c uint32) uint32

// HAdd is __hadd: (a+b)>>1 without overflow.
//
//go:linkname HAdd __nv_hadd
func HAdd(a, b int32) int32

// RHAdd is __rhadd: (a+b+1)>>1 without overflow.
//
//go:linkname RHAdd __nv_rhadd
func RHAdd(a, b int32) int32

// HAddU is __uhadd.
//
//go:linkname HAddU __nv_uhadd
func HAddU(a, b uint32) uint32
