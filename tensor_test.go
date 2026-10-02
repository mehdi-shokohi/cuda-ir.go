package cudair_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
)

const tensorPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/tensor"

// matSize is the matrix order every matmul subtest uses: a multiple of 32, so
// the k loop of the m16n8k32 integer shape closes, and of 16 so the m16n8k16
// and m16n16k16 tiles do.
const matSize = 64

// TestTensor runs examples/tensor: the raw mma.sync shapes, ldmatrix, and the
// wmma element types beyond half. Everything here is sm_80, so it executes on
// the test GPU; each subtest checks the PTX for the instruction it is about
// and the result against a CPU model.
func TestTensor(t *testing.T) {
	ctx, mod, res := loadKernels(t, tensorPkg)
	ptx := string(res.PTX)

	// wantPTX fails unless every instruction is in the module.
	wantPTX := func(t *testing.T, ins ...string) {
		t.Helper()
		for _, s := range ins {
			if !strings.Contains(ptx, s) {
				t.Fatalf("PTX lacks %q", s)
			}
		}
	}

	// a, b and bias are the operands every matmul subtest uses, as float32.
	// The values are small integers, so half, bfloat16 and tf32 all hold them
	// exactly and every result below is an exact integer: the comparisons can
	// be ==, with no tolerance to hide a layout mistake.
	//
	// They must also be asymmetric. An operand built from i+j or i*j makes
	// A[i+8][j] == A[i][j+8], and the mma A fragment holds exactly those two
	// elements in different registers — a matmul over such data still comes
	// out right when two registers of the fragment are swapped. elem below
	// mixes i, j and i*j with coprime weights so no index permutation of a
	// fragment survives.
	const n = matSize
	af, bf, bias := make([]float32, n*n), make([]float32, n*n), make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			af[i*n+j] = float32(elem(i, j, 7, 13, 3, 5) - 2) // -2..2
			bf[i*n+j] = float32(elem(i, j, 11, 5, 7, 5) - 2) // -2..2
			bias[i*n+j] = float32(elem(i, j, 3, 7, 5, 11) - 5)
		}
	}
	// want = A*B, wantBias = A*B + bias; |want| <= 64*4 = 256.
	want, wantBias := make([]float32, n*n), make([]float32, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float32
			for k := 0; k < n; k++ {
				s += af[i*n+k] * bf[k*n+j]
			}
			want[i*n+j], wantBias[i*n+j] = s, s+bias[i*n+j]
		}
	}
	transpose := func(m []float32) []float32 {
		out := make([]float32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				out[j*n+i] = m[i*n+j]
			}
		}
		return out
	}
	// one warp per 16x8 output tile for the mma shapes
	mmaCfg := cuda.LaunchConfig1D((n/16)*(n/8)*32, 128)
	// one warp per 16x16 output tile for the wmma shapes
	wmmaCfg := cuda.LaunchConfig1D((n/16)*(n/16)*32, 128)

	// checkF32 compares a device float32 result against w element by element.
	checkF32 := func(t *testing.T, name string, got, w []float32) {
		t.Helper()
		for i := range w {
			if got[i] != w[i] {
				t.Fatalf("%s: C[%d][%d] = %v, want %v", name, i/n, i%n, got[i], w[i])
			}
		}
	}
	checkI32 := func(t *testing.T, name string, got, w []int32) {
		t.Helper()
		for i := range w {
			if got[i] != w[i] {
				t.Fatalf("%s: C[%d][%d] = %d, want %d", name, i/n, i%n, got[i], w[i])
			}
		}
	}

	t.Run("MmaF16", func(t *testing.T) { // mma.sync m16n8k16, half in, float accumulate
		wantPTX(t, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
		ah, bh := toHalves(af), toHalves(bf)
		bth := toHalves(transpose(bf))
		da, db, dbt := upload(t, ctx, ah), upload(t, ctx, bh), upload(t, ctx, bth)
		dbias := upload(t, ctx, bias)

		dc := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaF16", mmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaF16", download(t, dc), want)

		// the accumulator read from memory (MmaLoadAcc) and written back
		dc2 := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaF16Bias", mmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dbias), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaF16Bias", download(t, dc2), wantBias)

		// B read column-major from its transpose (MmaLoadB16Col)
		dc3 := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaF16Trans", mmaCfg, cuda.Arg(da), cuda.Arg(dbt), cuda.Arg(dc3), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaF16Trans", download(t, dc3), want)
		fmt.Printf("MmaF16: %dx%d half matmul on mma.m16n8k16 matches the CPU (row-major B, B^T, loaded accumulator)\n", n, n)
	})

	t.Run("MmaF16Half", func(t *testing.T) { // the same shape accumulating in half
		wantPTX(t, "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16")
		da, db := upload(t, ctx, toHalves(af)), upload(t, ctx, toHalves(bf))
		dc := upload(t, ctx, make([]uint16, n*n))
		launch(t, ctx, mod, "MmaF16Half", mmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		out := download(t, dc)
		for i := range want {
			// |want| <= 256, so the f16 accumulator is exact too
			if got := halfToFloat32(out[i]); got != want[i] {
				t.Fatalf("MmaF16Half: C[%d][%d] = %v, want %v", i/n, i%n, got, want[i])
			}
		}
		fmt.Printf("MmaF16Half: the half accumulator (f16.f16.f16.f16) matches the CPU exactly\n")
	})

	t.Run("MmaBF16", func(t *testing.T) { // mma.sync m16n8k16, bfloat16 in
		wantPTX(t, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
		da, db := upload(t, ctx, toBF16s(af)), upload(t, ctx, toBF16s(bf))
		dc := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaBF16", mmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaBF16", download(t, dc), want)
		fmt.Printf("MmaBF16: bfloat16 shares the half fragment layout and matches the CPU\n")
	})

	t.Run("MmaTF32", func(t *testing.T) { // mma.sync m16n8k8, tf32 in
		wantPTX(t, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", "cvt.rna.tf32.f32")
		da, db, dbt := upload(t, ctx, af), upload(t, ctx, bf), upload(t, ctx, transpose(bf))
		dc := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaTF32", mmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaTF32", download(t, dc), want)
		dc2 := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "MmaTF32Trans", mmaCfg, cuda.Arg(da), cuda.Arg(dbt), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		checkF32(t, "MmaTF32Trans", download(t, dc2), want)
		fmt.Printf("MmaTF32: float32 rounded to tf32 on mma.m16n8k8 matches the CPU (B row-major and B^T)\n")
	})

	t.Run("MmaInt8", func(t *testing.T) { // mma.sync m16n8k32, s8 / u8 in, s32 out
		wantPTX(t, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
			"mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32")
		as, bs := make([]int8, n*n), make([]int8, n*n)
		au, bu, but := make([]uint8, n*n), make([]uint8, n*n), make([]uint8, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				as[i*n+j], bs[i*n+j] = int8(af[i*n+j]), int8(bf[i*n+j])
				au[i*n+j], bu[i*n+j] = uint8(elem(i, j, 7, 13, 3, 5)), uint8(elem(i, j, 11, 5, 7, 3))
				but[j*n+i] = bu[i*n+j]
			}
		}
		wantS, wantU := make([]int32, n*n), make([]int32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				var ss, su int32
				for k := 0; k < n; k++ {
					ss += int32(as[i*n+k]) * int32(bs[k*n+j])
					su += int32(au[i*n+k]) * int32(bu[k*n+j])
				}
				wantS[i*n+j], wantU[i*n+j] = ss, su
			}
		}
		dc := upload(t, ctx, make([]int32, n*n))
		launch(t, ctx, mod, "MmaS8", mmaCfg, cuda.Arg(upload(t, ctx, as)), cuda.Arg(upload(t, ctx, bs)), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkI32(t, "MmaS8", download(t, dc), wantS)

		dc2 := upload(t, ctx, make([]int32, n*n))
		launch(t, ctx, mod, "MmaU8", mmaCfg, cuda.Arg(upload(t, ctx, au)), cuda.Arg(upload(t, ctx, but)), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		checkI32(t, "MmaU8", download(t, dc2), wantU)
		fmt.Printf("MmaInt8: s8 and u8 (.satfinite) matmuls on mma.m16n8k32 match the CPU exactly\n")
	})

	t.Run("LdMatrix", func(t *testing.T) { // ldmatrix.m8n8.x1/x2/x4 (+.trans)
		wantPTX(t, "ldmatrix.sync.aligned.m8n8.x4.shared.b16", "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
			"ldmatrix.sync.aligned.m8n8.x2.shared.b16", "ldmatrix.sync.aligned.m8n8.x1.shared.b16")
		// four 8x8 matrices of distinct halfs, laid out back to back
		in := make([]uint16, 4*64)
		for i := range in {
			in[i] = float32ToHalf(float32(i + 1))
		}
		dout := upload(t, ctx, make([]uint32, 4*32*4))
		launch(t, ctx, mod, "LdMatrixRegs", cuda.LaunchConfig1D(32, 32), cuda.Arg(upload(t, ctx, in)), cuda.Arg(dout))
		out := download(t, dout)
		// element (r, c) of matrix m
		at := func(m, r, c int) uint32 { return uint32(in[m*64+r*8+c]) }
		pack := func(lo, hi uint32) uint32 { return lo | hi<<16 }
		for lane := 0; lane < 32; lane++ {
			g, s := lane/4, lane%4 // group = row, lane in group = column pair
			for i := 0; i < 4; i++ {
				// x4: register i holds matrix i at (g, 2s) and (g, 2s+1)
				if got, w := out[lane*4+i], pack(at(i, g, 2*s), at(i, g, 2*s+1)); got != w {
					t.Fatalf("ldmatrix.x4: lane %d reg %d = %#08x, want %#08x", lane, i, got, w)
				}
				// x4.trans: the same with the matrix transposed
				if got, w := out[32*4+lane*4+i], pack(at(i, 2*s, g), at(i, 2*s+1, g)); got != w {
					t.Fatalf("ldmatrix.x4.trans: lane %d reg %d = %#08x, want %#08x", lane, i, got, w)
				}
			}
			for i := 0; i < 2; i++ { // x2: matrices 0 and 1
				if got, w := out[2*32*4+lane*2+i], pack(at(i, g, 2*s), at(i, g, 2*s+1)); got != w {
					t.Fatalf("ldmatrix.x2: lane %d reg %d = %#08x, want %#08x", lane, i, got, w)
				}
			}
			if got, w := out[3*32*4+lane], pack(at(0, g, 2*s), at(0, g, 2*s+1)); got != w { // x1
				t.Fatalf("ldmatrix.x1: lane %d = %#08x, want %#08x", lane, got, w)
			}
		}
		fmt.Printf("LdMatrix: x1/x2/x4 and x4.trans put every element in the register the ISA says, for all 32 lanes\n")
	})

	t.Run("MmaLdMatrix", func(t *testing.T) { // ldmatrix feeding mma.sync
		// one 16x16 by 16x8 product with both operands staged through shared
		// memory: the fragments ldmatrix builds must equal MmaLoadA16's and
		// MmaLoadB16's, or the product is wrong.
		a16, b16 := make([]float32, 16*16), make([]float32, 16*8)
		for i := 0; i < 16; i++ {
			for j := 0; j < 16; j++ {
				a16[i*16+j] = float32(elem(i, j, 5, 11, 3, 7) - 3)
			}
			for j := 0; j < 8; j++ {
				b16[i*8+j] = float32(elem(i, j, 13, 7, 5, 5) - 2)
			}
		}
		w := make([]float32, 16*8)
		for i := 0; i < 16; i++ {
			for j := 0; j < 8; j++ {
				for k := 0; k < 16; k++ {
					w[i*8+j] += a16[i*16+k] * b16[k*8+j]
				}
			}
		}
		dc := upload(t, ctx, make([]float32, 16*8))
		launch(t, ctx, mod, "MmaLdMatrix", cuda.LaunchConfig1D(32, 32),
			cuda.Arg(upload(t, ctx, toHalves(a16))), cuda.Arg(upload(t, ctx, toHalves(b16))), cuda.Arg(dc))
		out := download(t, dc)
		for i := range w {
			if out[i] != w[i] {
				t.Fatalf("MmaLdMatrix: C[%d][%d] = %v, want %v", i/8, i%8, out[i], w[i])
			}
		}
		fmt.Printf("MmaLdMatrix: a 16x16 * 16x8 product fed by ldmatrix.x4 / x2.trans matches the CPU\n")
	})

	t.Run("WmmaBF16", func(t *testing.T) { // wmma m16n16k16 with bfloat16 operands
		wantPTX(t, "wmma.load.a.sync.aligned.row.m16n16k16.bf16", "wmma.load.b.sync.aligned.col.m16n16k16.bf16",
			"wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32",
			"wmma.mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32")
		da, db := upload(t, ctx, toBF16s(af)), upload(t, ctx, toBF16s(bf))
		dbt := upload(t, ctx, toBF16s(transpose(bf)))
		dbias := upload(t, ctx, bias)
		dc := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "WmmaBF16", wmmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dbias), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkF32(t, "WmmaBF16", download(t, dc), wantBias)
		dc2 := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "WmmaBF16Trans", wmmaCfg, cuda.Arg(da), cuda.Arg(dbt), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		checkF32(t, "WmmaBF16Trans", download(t, dc2), want)
		fmt.Printf("WmmaBF16: %dx%d bfloat16 matmul + bias on wmma matches the CPU (row.row and row.col)\n", n, n)
	})

	t.Run("WmmaTF32", func(t *testing.T) { // wmma m16n16k8 with tf32 operands
		wantPTX(t, "wmma.load.a.sync.aligned.row.m16n16k8.tf32", "wmma.load.b.sync.aligned.row.m16n16k8.tf32",
			"wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32", "wmma.store.d.sync.aligned.row.m16n16k8.f32")
		da, db := upload(t, ctx, af), upload(t, ctx, bf)
		dc := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "WmmaTF32", wmmaCfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkF32(t, "WmmaTF32", download(t, dc), want)
		fmt.Printf("WmmaTF32: the m16n16k8 tf32 geometry matches the CPU\n")
	})

	t.Run("WmmaInt8", func(t *testing.T) { // wmma m16n16k16 with s8 / u8 operands
		wantPTX(t, "wmma.load.a.sync.aligned.row.m16n16k16.s8", "wmma.load.a.sync.aligned.row.m16n16k16.u8",
			"wmma.load.b.sync.aligned.col.m16n16k16.u8", "wmma.load.c.sync.aligned.row.m16n16k16.s32",
			"wmma.mma.sync.aligned.row.row.m16n16k16.s32.s8.s8.s32",
			"wmma.mma.sync.aligned.row.col.m16n16k16.s32.u8.u8.s32",
			"wmma.store.d.sync.aligned.row.m16n16k16.s32")
		as, bs := make([]int8, n*n), make([]int8, n*n)
		au, but := make([]uint8, n*n), make([]uint8, n*n)
		biasI := make([]int32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				as[i*n+j], bs[i*n+j] = int8(af[i*n+j]), int8(bf[i*n+j])
				au[i*n+j] = uint8(elem(i, j, 7, 13, 3, 5))
				but[j*n+i] = uint8(elem(i, j, 11, 5, 7, 3))
				biasI[i*n+j] = int32(elem(i, j, 3, 7, 5, 11) - 5)
			}
		}
		wantS, wantU := make([]int32, n*n), make([]int32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				ss, su := int32(0), biasI[i*n+j]
				for k := 0; k < n; k++ {
					ss += int32(as[i*n+k]) * int32(bs[k*n+j])
					su += int32(au[i*n+k]) * int32(but[j*n+k])
				}
				wantS[i*n+j], wantU[i*n+j] = ss, su
			}
		}
		dc := upload(t, ctx, make([]int32, n*n))
		launch(t, ctx, mod, "WmmaS8", wmmaCfg, cuda.Arg(upload(t, ctx, as)), cuda.Arg(upload(t, ctx, bs)), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		checkI32(t, "WmmaS8", download(t, dc), wantS)
		dc2 := upload(t, ctx, make([]int32, n*n))
		launch(t, ctx, mod, "WmmaU8", wmmaCfg, cuda.Arg(upload(t, ctx, au)), cuda.Arg(upload(t, ctx, but)),
			cuda.Arg(upload(t, ctx, biasI)), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		checkI32(t, "WmmaU8", download(t, dc2), wantU)
		fmt.Printf("WmmaInt8: s8 and u8 matmuls with the s32 accumulator match the CPU exactly\n")
	})
}

// elem is the asymmetric small-integer generator the operands are built
// from: (wi*i + wj*j + wij*i*j) mod m. Mixing i, j and i*j with coprime
// weights means no row/column offset of the index relates two elements, so a
// permuted fragment cannot pass as the right one (see the note in TestTensor).
func elem(i, j, wi, wj, wij, m int) int { return (wi*i + wj*j + wij*i*j) % m }

// toHalves converts a float32 matrix to IEEE binary16 patterns.
func toHalves(x []float32) []uint16 {
	out := make([]uint16, len(x))
	for i, v := range x {
		out[i] = float32ToHalf(v)
	}
	return out
}

// toBF16s converts a float32 matrix to bfloat16 patterns.
func toBF16s(x []float32) []uint16 {
	out := make([]uint16, len(x))
	for i, v := range x {
		out[i] = float32ToBF16(v)
	}
	return out
}

// float32ToBF16 truncates to bfloat16 with round to nearest even.
func float32ToBF16(f float32) uint16 {
	b := math.Float32bits(f)
	h := uint16(b >> 16)
	if rem := b & 0xffff; rem > 0x8000 || rem == 0x8000 && h&1 == 1 {
		h++
	}
	return h
}

// halfToFloat32 is the inverse of float32ToHalf (no NaN in the tests).
func halfToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp, mant := uint32(h>>10&0x1f), uint32(h&0x3ff)
	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// subnormal: normalise
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
