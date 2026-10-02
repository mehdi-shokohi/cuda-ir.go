// Package tensor exercises the raw tensor-core instructions: mma.sync in
// every shape LLVM exposes for sm_80 (m16n8k16 f16/bf16, m16n8k8 tf32,
// m16n8k32 s8/u8), ldmatrix, and the wmma element types beyond half
// (bf16, tf32, s8/u8). stmatrix needs sm_90 and lives in examples/hopper.
package tensor

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// ---- mma.sync m16n8k16: half and bfloat16 inputs, float accumulate
//
// One warp per 16x8 output tile: launch (n/16)*(n/8) warps. a and b are
// row-major n x n, c is row-major float n x n.

// MmaF16 is C = A*B with half inputs.
func MmaF16(a, b cuda.Buf[cuda.Half], c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF32
	for k := int32(0); k < n; k += 16 {
		fa := cuda.MmaLoadA16(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB16(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaF16(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaF16Bias is C = A*B + Bias, loading the accumulator from memory.
func MmaF16Bias(a, b cuda.Buf[cuda.Half], bias, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	acc := cuda.MmaLoadAcc(bias.Ptr(row*16*n+col*8), n)
	for k := int32(0); k < n; k += 16 {
		fa := cuda.MmaLoadA16(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB16(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaF16(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaF16Trans is C = A*Bt^T with bt row-major n x n, so B is read
// column-major (the contiguous MmaLoadB16Col path).
func MmaF16Trans(a, bt cuda.Buf[cuda.Half], c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF32
	for k := int32(0); k < n; k += 16 {
		fa := cuda.MmaLoadA16(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB16Col(bt.Ptr(col*8*n+k), n)
		acc = cuda.MmaF16(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaF16Half is C = A*B accumulating in half instead of float.
func MmaF16Half(a, b, c cuda.Buf[cuda.Half], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF16
	for k := int32(0); k < n; k += 16 {
		fa := cuda.MmaLoadA16(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB16(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaF16Half(fa, fb, acc)
	}
	cuda.MmaStoreAccF16(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaBF16 is C = A*B with bfloat16 inputs (same fragment layout as half).
func MmaBF16(a, b cuda.Buf[cuda.BFloat16], c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF32
	for k := int32(0); k < n; k += 16 {
		fa := cuda.MmaLoadA16(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB16(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaBF16(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// ---- mma.sync m16n8k8: tf32 inputs, float accumulate (k steps of 8)

// MmaTF32 is C = A*B with float32 inputs rounded to tf32.
func MmaTF32(a, b, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF32
	for k := int32(0); k < n; k += 8 {
		fa := cuda.MmaLoadATF32(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadBTF32(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaTF32(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaTF32Trans reads B column-major from a row-major bt.
func MmaTF32Trans(a, bt, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccF32
	for k := int32(0); k < n; k += 8 {
		fa := cuda.MmaLoadATF32(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadBTF32Col(bt.Ptr(col*8*n+k), n)
		acc = cuda.MmaTF32(fa, fb, acc)
	}
	cuda.MmaStoreAcc(c.Ptr(row*16*n+col*8), n, acc)
}

// ---- mma.sync m16n8k32: int8 inputs, int32 accumulate (k steps of 32)

// MmaS8 is C = A*B with signed byte inputs.
func MmaS8(a, b cuda.Buf[int8], c cuda.Buf[int32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccI32
	for k := int32(0); k < n; k += 32 {
		fa := cuda.MmaLoadA8(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB8(b.Ptr(k*n+col*8), n)
		acc = cuda.MmaS8(fa, fb, acc)
	}
	cuda.MmaStoreAccI32(c.Ptr(row*16*n+col*8), n, acc)
}

// MmaU8 is C = A*B with unsigned byte inputs, B read column-major from a
// row-major bt, and .satfinite accumulation.
func MmaU8(a, bt cuda.Buf[uint8], c cuda.Buf[int32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 8
	row, col := warp/tiles, warp%tiles
	if row >= n/16 {
		return
	}
	var acc cuda.MmaAccI32
	for k := int32(0); k < n; k += 32 {
		fa := cuda.MmaLoadA8(a.Ptr(row*16*n+k), n)
		fb := cuda.MmaLoadB8Col(bt.Ptr(col*8*n+k), n)
		acc = cuda.MmaU8Sat(fa, fb, acc)
	}
	cuda.MmaStoreAccI32(c.Ptr(row*16*n+col*8), n, acc)
}

// ---- ldmatrix

var tile cuda.Shared[[4 * 64]cuda.Half] // four 8x8 matrices, row-major

// LdMatrixRegs copies 4*64 halfs into shared memory, reads them back with
// ldmatrix.x4 (and .x4.trans, .x2, .x1) and writes each lane's registers to
// out, so the test can check the register layout element by element.
// out holds 4 (variants) * 32 (lanes) * 4 (registers) uint32s.
func LdMatrixRegs(in cuda.Buf[cuda.Half], out cuda.Buf[uint32]) {
	t := tile.Get()
	for i := cuda.ThreadIdxX(); i < 4*64; i += cuda.BlockDimX() {
		t[i] = in.At(i)
	}
	cuda.SyncThreads()
	lane := cuda.LaneID()
	// x4: lanes 0-7 give the rows of matrix 0, 8-15 of matrix 1, ...
	row := &t[lane*8]
	v4 := cuda.LdMatrix4(row)
	v4t := cuda.LdMatrix4Trans(row)
	// x2 / x1 only consume the first 16 / 8 lanes' addresses
	v2 := cuda.LdMatrix2(row)
	v1 := cuda.LdMatrix1(row)
	for i := int32(0); i < 4; i++ {
		out.Set(lane*4+i, v4[i])
		out.Set(32*4+lane*4+i, v4t[i])
	}
	out.Set(2*32*4+lane*2, v2[0])
	out.Set(2*32*4+lane*2+1, v2[1])
	out.Set(3*32*4+lane, v1)
}

var atile cuda.Shared[[16 * 16]cuda.Half] // one 16x16 A tile, row-major
var btile cuda.Shared[[16 * 8]cuda.Half]  // one 16x8 B tile, row-major

// MmaLdMatrix is C = A*B for a single 16x16 by 16x8 product, with both
// operands staged through shared memory and loaded by ldmatrix instead of
// MmaLoadA16 / MmaLoadB16 — the layout the two produce must agree.
func MmaLdMatrix(a, b cuda.Buf[cuda.Half], c cuda.Buf[float32]) {
	at, bt := atile.Get(), btile.Get()
	for i := cuda.ThreadIdxX(); i < 16*16; i += cuda.BlockDimX() {
		at[i] = a.At(i)
	}
	for i := cuda.ThreadIdxX(); i < 16*8; i += cuda.BlockDimX() {
		bt[i] = b.At(i)
	}
	cuda.SyncThreads()
	lane := cuda.LaneID()
	// A: four 8x8 matrices at (0,0), (8,0), (0,8), (8,8) of the 16x16 tile,
	// which is exactly ldmatrix.x4 over rows lane%16 and columns 8*(lane/16).
	fa := cuda.MmaAFromRegs(cuda.LdMatrix4(&at[(lane%16)*16+8*(lane/16)]))
	// B: 16x8 read column-major, i.e. two 8x8 matrices transposed.
	fb := cuda.MmaBFromRegs(cuda.LdMatrix2Trans(&bt[(lane%16)*8]))
	acc := cuda.MmaF16(fa, fb, cuda.MmaAccF32{})
	cuda.MmaStoreAcc(c.Ptr(0), 8, acc)
}

// ---- wmma with the other element types (m16n16k16 bf16 / s8 / u8,
// m16n16k8 tf32): one warp per 16x16 output tile, as in examples/advanced.

// WmmaBF16 is C = A*B + Bias with bfloat16 inputs and float accumulate.
func WmmaBF16(a, b cuda.Buf[cuda.BFloat16], bias, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	acc := cuda.WmmaLoadC(bias.Ptr(row*16*n+col*16), n)
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadABF16(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBBF16(b.Ptr(k*n+col*16), n)
		acc = cuda.WmmaMmaBF16(fa, fb, acc)
	}
	cuda.WmmaStore(c.Ptr(row*16*n+col*16), n, acc)
}

// WmmaBF16Trans reads B column-major from a row-major bt.
func WmmaBF16Trans(a, bt cuda.Buf[cuda.BFloat16], c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	var acc cuda.FragC
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadABF16(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBBF16Col(bt.Ptr(col*16*n+k), n)
		acc = cuda.WmmaMmaBF16RowCol(fa, fb, acc)
	}
	cuda.WmmaStore(c.Ptr(row*16*n+col*16), n, acc)
}

// WmmaTF32 is C = A*B with float32 inputs rounded to tf32; the geometry is
// m16n16k8, so k advances by 8.
func WmmaTF32(a, b, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	var acc cuda.FragC
	for k := int32(0); k < n; k += 8 {
		fa := cuda.WmmaLoadATF32(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBTF32(b.Ptr(k*n+col*16), n)
		acc = cuda.WmmaMmaTF32(fa, fb, acc)
	}
	cuda.WmmaStoreTF32(c.Ptr(row*16*n+col*16), n, acc)
}

// WmmaS8 is C = A*B with signed byte inputs and int32 accumulate.
func WmmaS8(a, b cuda.Buf[int8], c cuda.Buf[int32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	var acc cuda.FragCI32
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadAS8(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBS8(b.Ptr(k*n+col*16), n)
		acc = cuda.WmmaMmaS8(fa, fb, acc)
	}
	cuda.WmmaStoreI32(c.Ptr(row*16*n+col*16), n, acc)
}

// WmmaU8 is the unsigned twin, loading the accumulator from memory and
// reading B column-major from a row-major bt.
func WmmaU8(a, bt cuda.Buf[uint8], bias, c cuda.Buf[int32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	acc := cuda.WmmaLoadCI32(bias.Ptr(row*16*n+col*16), n)
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadAU8(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBU8Col(bt.Ptr(col*16*n+k), n)
		acc = cuda.WmmaMmaU8RowCol(fa, fb, acc)
	}
	cuda.WmmaStoreI32(c.Ptr(row*16*n+col*16), n, acc)
}
