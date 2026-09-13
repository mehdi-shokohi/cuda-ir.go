package cuda

import _ "unsafe" // for go:linkname

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
