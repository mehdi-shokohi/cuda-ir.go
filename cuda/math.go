package cuda

import "unsafe"

// Math for kernels. The Go "math" package is not available on the GPU;
// these map to LLVM intrinsics (native PTX instructions) or to NVIDIA's
// libdevice (__nv_*), which `gocuda build` links in automatically.

//go:linkname Sqrt llvm.sqrt.f32
func Sqrt(x float32) float32

//go:linkname Sqrt64 llvm.sqrt.f64
func Sqrt64(x float64) float64

//go:linkname Abs llvm.fabs.f32
func Abs(x float32) float32

//go:linkname Abs64 llvm.fabs.f64
func Abs64(x float64) float64

//go:linkname Floor llvm.floor.f32
func Floor(x float32) float32

//go:linkname Ceil llvm.ceil.f32
func Ceil(x float32) float32

//go:linkname Trunc llvm.trunc.f32
func Trunc(x float32) float32

//go:linkname Round llvm.round.f32
func Round(x float32) float32

// FMA is a*b+c with a single rounding.
//
//go:linkname FMA llvm.fma.f32
func FMA(a, b, c float32) float32

//go:linkname FMA64 llvm.fma.f64
func FMA64(a, b, c float64) float64

//go:linkname Min llvm.minnum.f32
func Min(a, b float32) float32

//go:linkname Max llvm.maxnum.f32
func Max(a, b float32) float32

//go:linkname Rsqrt __nv_rsqrtf
func Rsqrt(x float32) float32

//go:linkname Sin __nv_sinf
func Sin(x float32) float32

//go:linkname Cos __nv_cosf
func Cos(x float32) float32

//go:linkname Tan __nv_tanf
func Tan(x float32) float32

//go:linkname Exp __nv_expf
func Exp(x float32) float32

//go:linkname Exp2 __nv_exp2f
func Exp2(x float32) float32

//go:linkname Log __nv_logf
func Log(x float32) float32

//go:linkname Log2 __nv_log2f
func Log2(x float32) float32

//go:linkname Pow __nv_powf
func Pow(x, y float32) float32

//go:linkname Tanh __nv_tanhf
func Tanh(x float32) float32

//go:linkname Erf __nv_erff
func Erf(x float32) float32

//go:linkname Sin64 __nv_sin
func Sin64(x float64) float64

//go:linkname Cos64 __nv_cos
func Cos64(x float64) float64

//go:linkname Exp64 __nv_exp
func Exp64(x float64) float64

//go:linkname Log64 __nv_log
func Log64(x float64) float64

//go:linkname Pow64 __nv_pow
func Pow64(x, y float64) float64

// Fast, lower-precision variants (__sinf, __expf, ...; CUDA's --use_fast_math).

//go:linkname FastSin __nv_fast_sinf
func FastSin(x float32) float32

//go:linkname FastCos __nv_fast_cosf
func FastCos(x float32) float32

//go:linkname FastExp __nv_fast_expf
func FastExp(x float32) float32

//go:linkname FastLog __nv_fast_logf
func FastLog(x float32) float32

//go:linkname FastPow __nv_fast_powf
func FastPow(x, y float32) float32

// ---- bit casts (__float_as_int & co; math.Float32bits also works in kernels)

// Float32Bits is __float_as_uint.
func Float32Bits(x float32) uint32 { return *(*uint32)(unsafe.Pointer(&x)) }

// Float32FromBits is __uint_as_float.
func Float32FromBits(b uint32) float32 { return *(*float32)(unsafe.Pointer(&b)) }

// Float64Bits is __double_as_longlong.
func Float64Bits(x float64) uint64 { return *(*uint64)(unsafe.Pointer(&x)) }

// Float64FromBits is __longlong_as_double.
func Float64FromBits(b uint64) float64 { return *(*float64)(unsafe.Pointer(&b)) }

// ---- remaining float32 libdevice functions

//go:linkname Asin __nv_asinf
func Asin(x float32) float32

//go:linkname Acos __nv_acosf
func Acos(x float32) float32

//go:linkname Atan __nv_atanf
func Atan(x float32) float32

//go:linkname Atan2 __nv_atan2f
func Atan2(y, x float32) float32

//go:linkname Sinh __nv_sinhf
func Sinh(x float32) float32

//go:linkname Cosh __nv_coshf
func Cosh(x float32) float32

//go:linkname Asinh __nv_asinhf
func Asinh(x float32) float32

//go:linkname Acosh __nv_acoshf
func Acosh(x float32) float32

//go:linkname Atanh __nv_atanhf
func Atanh(x float32) float32

//go:linkname Cbrt __nv_cbrtf
func Cbrt(x float32) float32

//go:linkname Hypot __nv_hypotf
func Hypot(x, y float32) float32

//go:linkname Log10 __nv_log10f
func Log10(x float32) float32

//go:linkname Log1p __nv_log1pf
func Log1p(x float32) float32

//go:linkname Expm1 __nv_expm1f
func Expm1(x float32) float32

//go:linkname Exp10 __nv_exp10f
func Exp10(x float32) float32

//go:linkname Erfc __nv_erfcf
func Erfc(x float32) float32

//go:linkname Erfinv __nv_erfinvf
func Erfinv(x float32) float32

//go:linkname Lgamma __nv_lgammaf
func Lgamma(x float32) float32

//go:linkname Tgamma __nv_tgammaf
func Tgamma(x float32) float32

//go:linkname Fmod __nv_fmodf
func Fmod(x, y float32) float32

//go:linkname Remainder __nv_remainderf
func Remainder(x, y float32) float32

//go:linkname Copysign __nv_copysignf
func Copysign(x, y float32) float32

// Rint rounds to nearest even (rintf).
//
//go:linkname Rint __nv_rintf
func Rint(x float32) float32

//go:linkname Nearbyint __nv_nearbyintf
func Nearbyint(x float32) float32

// Ldexp is ldexpf: x * 2^e.
//
//go:linkname Ldexp __nv_ldexpf
func Ldexp(x float32, e int32) float32

//go:linkname frexpf __nv_frexpf
func frexpf(x float32, e *int32) float32

//go:linkname modff __nv_modff
func modff(x float32, ip *float32) float32

//go:linkname sincosf __nv_sincosf
func sincosf(x float32, s, c *float32)

//go:linkname isnanf __nv_isnanf
func isnanf(x float32) int32

//go:linkname isinff __nv_isinff
func isinff(x float32) int32

//go:linkname finitef __nv_finitef
func finitef(x float32) int32

// Frexp is frexpf: x = frac * 2^exp with frac in [0.5, 1).
func Frexp(x float32) (frac float32, exp int32) {
	var e int32
	f := frexpf(x, &e)
	return f, e
}

// Modf is modff: integer and fractional parts.
func Modf(x float32) (integer, frac float32) {
	var ip float32
	f := modff(x, &ip)
	return ip, f
}

// Sincos is sincosf.
func Sincos(x float32) (sin, cos float32) {
	var s, c float32
	sincosf(x, &s, &c)
	return s, c
}

// IsNaN / IsInf / IsFinite are isnan / isinf / isfinite.
func IsNaN(x float32) bool    { return isnanf(x) != 0 }
func IsInf(x float32) bool    { return isinff(x) != 0 }
func IsFinite(x float32) bool { return finitef(x) != 0 }

//go:linkname Fdim __nv_fdimf
func Fdim(x, y float32) float32

// Sinpi / Cospi are sin(pi*x) / cos(pi*x).
//
//go:linkname Sinpi __nv_sinpif
func Sinpi(x float32) float32

//go:linkname Cospi __nv_cospif
func Cospi(x float32) float32

// NormCdf is normcdff, the standard normal CDF.
//
//go:linkname NormCdf __nv_normcdff
func NormCdf(x float32) float32

//go:linkname J0 __nv_j0f
func J0(x float32) float32

//go:linkname J1 __nv_j1f
func J1(x float32) float32

// ---- rounding-mode intrinsics and fast paths

// FastDiv is __fdividef(x, y).
//
//go:linkname FastDiv __nv_fast_fdividef
func FastDiv(x, y float32) float32

//go:linkname FastTan __nv_fast_tanf
func FastTan(x float32) float32

//go:linkname FastExp10 __nv_fast_exp10f
func FastExp10(x float32) float32

//go:linkname FastLog2 __nv_fast_log2f
func FastLog2(x float32) float32

//go:linkname FastLog10 __nv_fast_log10f
func FastLog10(x float32) float32

//go:linkname fastSincosf __nv_fast_sincosf
func fastSincosf(x float32, s, c *float32)

// FastSincos is __sincosf.
func FastSincos(x float32) (sin, cos float32) {
	var s, c float32
	fastSincosf(x, &s, &c)
	return s, c
}

// RcpRN is __frcp_rn: 1/x rounded to nearest.
//
//go:linkname RcpRN __nv_frcp_rn
func RcpRN(x float32) float32

// SqrtRN is __fsqrt_rn (IEEE sqrt; Sqrt may use the approximate PTX sqrt).
//
//go:linkname SqrtRN __nv_fsqrt_rn
func SqrtRN(x float32) float32

//go:linkname RsqrtRN __nv_frsqrt_rn
func RsqrtRN(x float32) float32

// AddRN / AddRZ / MulRN / MulRZ are __fadd_rn & co: the operation with a
// fixed rounding mode and no contraction into an FMA.
//
//go:linkname AddRN __nv_fadd_rn
func AddRN(x, y float32) float32

//go:linkname AddRZ __nv_fadd_rz
func AddRZ(x, y float32) float32

//go:linkname MulRN __nv_fmul_rn
func MulRN(x, y float32) float32

// Saturate is __saturatef: clamp to [0, 1].
//
//go:linkname Saturate __nv_saturatef
func Saturate(x float32) float32

// Float32ToInt32RN is __float2int_rn (round to nearest even); a Go
// conversion int32(x) truncates like __float2int_rz.
//
//go:linkname Float32ToInt32RN __nv_float2int_rn
func Float32ToInt32RN(x float32) int32

//go:linkname Float32ToUint32RN __nv_float2uint_rn
func Float32ToUint32RN(x float32) uint32

// ---- float64 twins

//go:linkname Tan64 __nv_tan
func Tan64(x float64) float64

//go:linkname Floor64 llvm.floor.f64
func Floor64(x float64) float64

//go:linkname Ceil64 llvm.ceil.f64
func Ceil64(x float64) float64

//go:linkname Trunc64 llvm.trunc.f64
func Trunc64(x float64) float64

//go:linkname Round64 llvm.round.f64
func Round64(x float64) float64

//go:linkname Min64 llvm.minnum.f64
func Min64(a, b float64) float64

//go:linkname Max64 llvm.maxnum.f64
func Max64(a, b float64) float64

//go:linkname Rsqrt64 __nv_rsqrt
func Rsqrt64(x float64) float64

//go:linkname Exp2_64 __nv_exp2
func Exp2_64(x float64) float64

//go:linkname Log2_64 __nv_log2
func Log2_64(x float64) float64

//go:linkname Tanh64 __nv_tanh
func Tanh64(x float64) float64

//go:linkname Erf64 __nv_erf
func Erf64(x float64) float64

//go:linkname Asin64 __nv_asin
func Asin64(x float64) float64

//go:linkname Acos64 __nv_acos
func Acos64(x float64) float64

//go:linkname Atan64 __nv_atan
func Atan64(x float64) float64

//go:linkname Atan2_64 __nv_atan2
func Atan2_64(y, x float64) float64

//go:linkname Sinh64 __nv_sinh
func Sinh64(x float64) float64

//go:linkname Cosh64 __nv_cosh
func Cosh64(x float64) float64

//go:linkname Cbrt64 __nv_cbrt
func Cbrt64(x float64) float64

//go:linkname Hypot64 __nv_hypot
func Hypot64(x, y float64) float64

//go:linkname Log10_64 __nv_log10
func Log10_64(x float64) float64

//go:linkname Log1p64 __nv_log1p
func Log1p64(x float64) float64

//go:linkname Expm1_64 __nv_expm1
func Expm1_64(x float64) float64

//go:linkname Erfc64 __nv_erfc
func Erfc64(x float64) float64

//go:linkname Lgamma64 __nv_lgamma
func Lgamma64(x float64) float64

//go:linkname Tgamma64 __nv_tgamma
func Tgamma64(x float64) float64

//go:linkname Fmod64 __nv_fmod
func Fmod64(x, y float64) float64

//go:linkname Copysign64 __nv_copysign
func Copysign64(x, y float64) float64

//go:linkname Rint64 __nv_rint
func Rint64(x float64) float64

//go:linkname Ldexp64 __nv_ldexp
func Ldexp64(x float64, e int32) float64

//go:linkname isnand __nv_isnand
func isnand(x float64) int32

//go:linkname isinfd __nv_isinfd
func isinfd(x float64) int32

func IsNaN64(x float64) bool { return isnand(x) != 0 }
func IsInf64(x float64) bool { return isinfd(x) != 0 }
