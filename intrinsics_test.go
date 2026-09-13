package cudair_test

import (
	"fmt"
	"math"
	"math/bits"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
)

// TestIntrinsics runs examples/intrinsics: one subtest per roadmap group,
// each result checked against a CPU model.
func TestIntrinsics(t *testing.T) {
	ctx, mod, _ := loadKernels(t, "github.com/mehdi-shokohi/cuda-ir.go/examples/intrinsics")
	const n = 4096
	cfg := cuda.LaunchConfig1D(n, 256)

	t.Run("Sregs", func(t *testing.T) { // %smid %nsmid %warpid %nwarpid %gridid %lanemask_*
		dout := upload(t, ctx, make([]int32, 2*n))
		launch(t, ctx, mod, "Sregs", cfg, cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		sms := map[int32]bool{}
		for i := 0; i < n; i++ {
			if out[i] != 255 {
				t.Fatalf("thread %d: checks = %08b, want 11111111", i, out[i])
			}
			sms[out[n+i]] = true
		}
		fmt.Printf("Sregs: all 8 checks pass on %d threads; %d distinct SMs used\n", n, len(sms))
	})

	t.Run("IntMath", func(t *testing.T) { // __popc __clz __ffs __brev __umulhi __funnelshift_l __byte_perm __usad __uhadd __popcll
		in := make([]uint32, n)
		for i := range in {
			in[i] = uint32(i*2654435761) ^ uint32(i>>3)
		}
		in[0], in[1] = 0, 0xffffffff
		din := upload(t, ctx, in)
		dout := upload(t, ctx, make([]uint32, 8*n))
		launch(t, ctx, mod, "IntMath", cfg, cuda.Arg(din), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		names := []string{"Popc", "Clz", "Ffs", "Brev", "MulHiU", "FunnelShiftL", "BytePerm", "SadU+HAddU+Popc64"}
		for i, x := range in {
			ffs := uint32(0)
			if x != 0 {
				ffs = uint32(bits.TrailingZeros32(x) + 1)
			}
			sad := x - 12345
			if x < 12345 {
				sad = 12345 - x
			}
			want := []uint32{
				uint32(bits.OnesCount32(x)), uint32(bits.LeadingZeros32(x)), ffs, bits.Reverse32(x),
				uint32((uint64(x) * 0x9E3779B9) >> 32), (^x)<<7 | x>>25, ^x,
				sad + 7 + uint32((uint64(x)+0xffffffff)>>1) + uint32(2*bits.OnesCount32(x)),
			}
			for k := range want {
				if out[i*8+k] != want[k] {
					t.Fatalf("%s(%#x) = %#x, want %#x", names[k], x, out[i*8+k], want[k])
				}
			}
		}
		fmt.Printf("IntMath: %d values x %d intrinsics match\n", n, len(names))
	})

	t.Run("MathBits", func(t *testing.T) { // Go's math/bits inside a kernel
		in := make([]uint32, n)
		for i := range in {
			in[i] = uint32(i*40503) ^ uint32(i<<7)
		}
		din := upload(t, ctx, in)
		dout := upload(t, ctx, make([]uint32, 4*n))
		launch(t, ctx, mod, "Bits", cfg, cuda.Arg(din), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i, x := range in {
			want := []uint32{
				uint32(bits.OnesCount32(x) + bits.LeadingZeros32(x)*64 + bits.TrailingZeros32(x)*4096),
				bits.Reverse32(x), bits.RotateLeft32(x, -5),
				uint32(bits.Len64(uint64(x)<<3)+bits.OnesCount8(uint8(x))*128) + uint32(bits.ReverseBytes16(uint16(x)))<<16,
			}
			for k := range want {
				if out[i*4+k] != want[k] {
					t.Fatalf("bits[%d](%#x) = %#x, want %#x", k, x, out[i*4+k], want[k])
				}
			}
		}
		fmt.Printf("MathBits: OnesCount/LeadingZeros/TrailingZeros/Reverse/RotateLeft/Len64/ReverseBytes16 match on %d values\n", n)
	})

	t.Run("WarpOps", func(t *testing.T) { // __reduce_*_sync __match_any/all_sync 64-bit and width shuffles
		in := make([]int32, n)
		for i := range in {
			in[i] = int32((i*7)%13) - 3
		}
		din := upload(t, ctx, in)
		dout := upload(t, ctx, make([]int32, 8*n))
		dout64 := upload(t, ctx, make([]int64, n))
		launch(t, ctx, mod, "WarpOps", cfg, cuda.Arg(din), cuda.Arg(dout), cuda.Arg(dout64), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		out64 := download(t, dout64)
		for w := 0; w < n/32; w++ {
			warp := in[w*32 : w*32+32]
			var sum, max int32 = 0, math.MinInt32
			var xor uint32
			for _, v := range warp {
				sum += v
				xor ^= uint32(v)
				if v > max {
					max = v
				}
			}
			for l := 0; l < 32; l++ {
				i := w*32 + l
				x := warp[l]
				same, same64 := 0, uint32(0)
				for m, v := range warp {
					if v&3 == x&3 {
						same++
					}
					if v == x {
						same64 |= 1 << m
					}
				}
				down := x
				if l%16 != 15 {
					down = warp[l+1]
				}
				want := []int32{sum, int32(same), 1, down, max, int32(xor), warp[(l/8)*8+3], int32(same64)}
				for k := range want {
					if out[i*8+k] != want[k] {
						t.Fatalf("warp %d lane %d op %d = %d, want %d", w, l, k, out[i*8+k], want[k])
					}
				}
				if want := int64(warp[l^1]) * 1000003; out64[i] != want {
					t.Fatalf("ShflXor64 lane %d = %d, want %d", l, out64[i], want)
				}
			}
		}
		fmt.Printf("WarpOps: ReduceAdd/Max/Xor, MatchAny/MatchAll(+64), ShflDownWidth/ShflWidth, ShflXor64 match on %d warps\n", n/32)
	})

	t.Run("Barriers", func(t *testing.T) { // bar.sync/bar.arrive named barriers, __syncthreads_count/and/or, __isShared/__isGlobal, __nanosleep
		const blocks = 4
		dout := upload(t, ctx, make([]int32, 8*blocks))
		launch(t, ctx, mod, "Barriers", cuda.LaunchConfig1D(blocks*128, 128), cuda.Arg(dout))
		out := download(t, dout)
		want := []int32{2080, 64, 1, 1, 1, 1, 0, 1}
		for b := 0; b < blocks; b++ {
			for k := range want {
				if out[b*8+k] != want[k] {
					t.Fatalf("block %d check %d = %d, want %d (%v)", b, k, out[b*8+k], want[k], out[b*8:b*8+8])
				}
			}
		}
		fmt.Printf("Barriers: producer/consumer sum=%d count=%d and=%d or=%d isShared(tile)=%d isGlobal(out)=%d isShared(out)=%d\n",
			out[0], out[1], out[2], out[3], out[4], out[5], out[6])
	})

	t.Run("Atomics", func(t *testing.T) { // atomicMin/Max (s32 u32 s64 u64), _block/_system scopes, float exch/CAS, ld.acquire/st.release
		vals := make([]int32, n)
		for i := range vals {
			vals[i] = int32((i*7919)%2001) - 1000
		}
		dv := upload(t, ctx, vals)
		o32 := make([]int32, 8+n/256)
		o32[0], o32[1], o32[2], o32[3] = math.MaxInt32, math.MinInt32, -1, 0
		d32 := upload(t, ctx, o32)
		d64 := upload(t, ctx, []int64{math.MaxInt64, math.MinInt64, 0})
		df := upload(t, ctx, []float32{0, -1e30, 0})
		launch(t, ctx, mod, "Atomics", cfg, cuda.Arg(dv), cuda.Arg(d32), cuda.Arg(d64), cuda.Arg(df), cuda.ArgValue(int32(n)))
		out32, out64, outf := download(t, d32), download(t, d64), download(t, df)
		var min, max int32 = math.MaxInt32, math.MinInt32
		var umin, umax uint32 = math.MaxUint32, 0
		var umax64 uint64
		seen := map[float32]bool{}
		for _, v := range vals {
			min, max = int32(math.Min(float64(min), float64(v))), int32(math.Max(float64(max), float64(v)))
			if uint32(v) < umin {
				umin = uint32(v)
			}
			if uint32(v) > umax {
				umax = uint32(v)
			}
			if uint64(int64(v)) > umax64 {
				umax64 = uint64(int64(v))
			}
			seen[float32(v)*0.5] = true
		}
		check := func(name string, got, want any) {
			if fmt.Sprint(got) != fmt.Sprint(want) {
				t.Errorf("%s = %v, want %v", name, got, want)
			}
		}
		check("AtomicMinInt32", out32[0], min)
		check("AtomicMaxInt32", out32[1], max)
		check("AtomicMinUint32", uint32(out32[2]), umin)
		check("AtomicMaxUint32", uint32(out32[3]), umax)
		check("AtomicAddInt32System count", out32[4], int32(n))
		check("StoreRelease flag", out32[5], int32(1))
		check("sync/atomic count", out32[6], int32(n))
		for b := 0; b < n/256; b++ {
			check(fmt.Sprintf("AtomicAddInt32Block[%d]", b), out32[8+b], int32(256))
		}
		check("AtomicMinInt64", out64[0], int64(min)*1000000)
		check("AtomicMaxInt64", out64[1], int64(max)*1000000)
		check("AtomicMaxUint64", uint64(out64[2]), umax64)
		if !seen[outf[0]] {
			t.Errorf("AtomicSwapFloat32 left %v, not one of the inputs", outf[0])
		}
		check("float max via AtomicCASFloat32", outf[1], float32(max)*0.5)
		check("AtomicAddFloat32 (atomicrmw fadd)", outf[2], float32(n))
		fmt.Printf("Atomics: min=%d max=%d umin=%d umax=%d system=%d block[0]=%d min64=%d umax64=%d fmax=%v\n",
			out32[0], out32[1], uint32(out32[2]), uint32(out32[3]), out32[4], out32[8], out64[0], uint64(out64[2]), outf[1])
	})

	t.Run("MathExt", func(t *testing.T) { // asinf atan2f cbrtf hypotf log1pf expm1f fmodf sinpif frexpf/ldexpf modff sincosf erfcf __float2int_rn rintf __saturatef tgammaf lgammaf + Go math.Sqrt/Float32bits
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(i) / n
		}
		dx := upload(t, ctx, x)
		dout := upload(t, ctx, make([]float32, 16*n))
		launch(t, ctx, mod, "MathExt", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		names := []string{"Asin", "Atan2", "Cbrt", "Hypot", "Log1p", "Expm1", "Fmod", "Sinpi", "Ldexp(Frexp)", "Modf", "Sincos", "Erf+Erfc", "Float32ToInt32RN", "Rint+Saturate", "math.Sqrt+Float32bits", "Tgamma/Lgamma"}
		for i, v := range x {
			f := float64(v)
			v10, v100 := float64(v*10), float64(v*100+1) // float32 products, as the GPU computes them
			want := []float64{
				math.Asin(f), math.Atan2(f, 1), math.Cbrt(f), math.Hypot(f, 1), math.Log1p(f), math.Expm1(f),
				math.Mod(v10, 3), math.Sin(math.Pi * f), v100, v100, 1, 1, math.RoundToEven(v10),
				math.RoundToEven(v10) + math.Min(math.Max(float64(v*3-1), 0), 1),
				math.Sqrt(f) + f, math.Gamma(f+1) + lgamma(f+1),
			}
			for k := range want {
				if !near(float64(out[i*16+k]), want[k], 2e-4) {
					t.Fatalf("%s(%v) = %v, want %v", names[k], v, out[i*16+k], want[k])
				}
			}
		}
		fmt.Printf("MathExt: %d functions match Go's math on %d inputs\n", len(names), n)
	})

	t.Run("Math64", func(t *testing.T) { // float64 twins
		x := make([]float64, n)
		for i := range x {
			x[i] = float64(i) / n
		}
		dx := upload(t, ctx, x)
		dout := upload(t, ctx, make([]float64, 8*n))
		launch(t, ctx, mod, "Math64", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i, v := range x {
			want := []float64{
				math.Tan(v), math.Floor(v*10) + math.Ceil(v*10) + math.Trunc(-v*10) + math.Round(v*10),
				v + 0.5, 1 / math.Sqrt(v+1), math.Exp2(v) + math.Log2(v+1), math.Tanh(v) + math.Erf(v),
				math.Atan2(v, 1) + math.Cbrt(v) + math.Hypot(v, 1), math.Log10(v+1) + math.Log1p(v) + math.Expm1(v),
			}
			for k := range want {
				if !near(out[i*8+k], want[k], 1e-9) {
					t.Fatalf("Math64[%d](%v) = %v, want %v", k, v, out[i*8+k], want[k])
				}
			}
		}
		fmt.Printf("Math64: Tan64 Floor64 Ceil64 Trunc64 Round64 Min64 Max64 Rsqrt64 Exp2_64 Log2_64 Tanh64 Erf64 Atan2_64 Cbrt64 Hypot64 Log10_64 Log1p64 Expm1_64 match on %d inputs\n", n)
	})
}

func lgamma(x float64) float64 { l, _ := math.Lgamma(x); return l }
