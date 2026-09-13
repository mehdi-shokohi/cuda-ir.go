package cuda

import _ "unsafe" // for go:linkname

// Half is an IEEE binary16 (__half) and BFloat16 a bfloat16 (__nv_bfloat16);
// both are stored as their 16-bit pattern (Go has no such types). Arithmetic
// is native on sm_53+ (half) / sm_80+ (bfloat16).
type Half uint16
type BFloat16 uint16

//go:linkname halfToF32 cudair.half.to.f32
func halfToF32(h Half) float32

//go:linkname f32ToHalf cudair.f32.to.half
func f32ToHalf(x float32) Half

//go:linkname halfAdd cudair.half.add
func halfAdd(a, b Half) Half

//go:linkname halfSub cudair.half.sub
func halfSub(a, b Half) Half

//go:linkname halfMul cudair.half.mul
func halfMul(a, b Half) Half

//go:linkname halfDiv cudair.half.div
func halfDiv(a, b Half) Half

//go:linkname halfFMA cudair.half.fma
func halfFMA(a, b, c Half) Half

//go:linkname bf16ToF32 cudair.bfloat.to.f32
func bf16ToF32(h BFloat16) float32

//go:linkname f32ToBF16 cudair.f32.to.bfloat
func f32ToBF16(x float32) BFloat16

//go:linkname bf16Add cudair.bfloat.add
func bf16Add(a, b BFloat16) BFloat16

//go:linkname bf16Mul cudair.bfloat.mul
func bf16Mul(a, b BFloat16) BFloat16

//go:linkname bf16FMA cudair.bfloat.fma
func bf16FMA(a, b, c BFloat16) BFloat16

// FloatToHalf is __float2half (round to nearest even).
func FloatToHalf(x float32) Half { return f32ToHalf(x) }

// Float32 is __half2float.
func (h Half) Float32() float32 { return halfToF32(h) }

// Add/Sub/Mul/Div/FMA are __hadd/__hsub/__hmul/__hdiv/__hfma.
func (h Half) Add(o Half) Half    { return halfAdd(h, o) }
func (h Half) Sub(o Half) Half    { return halfSub(h, o) }
func (h Half) Mul(o Half) Half    { return halfMul(h, o) }
func (h Half) Div(o Half) Half    { return halfDiv(h, o) }
func (h Half) FMA(b, c Half) Half { return halfFMA(h, b, c) }

// FloatToBFloat16 is __float2bfloat16.
func FloatToBFloat16(x float32) BFloat16 { return f32ToBF16(x) }

// Float32 is __bfloat162float.
func (h BFloat16) Float32() float32 { return bf16ToF32(h) }

func (h BFloat16) Add(o BFloat16) BFloat16    { return bf16Add(h, o) }
func (h BFloat16) Mul(o BFloat16) BFloat16    { return bf16Mul(h, o) }
func (h BFloat16) FMA(b, c BFloat16) BFloat16 { return bf16FMA(h, b, c) }
