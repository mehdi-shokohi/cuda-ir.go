package cuda

import _ "unsafe" // for go:linkname

// ---- tensor cores: warp matrix multiply-accumulate (wmma, sm_70+)
//
// D = A(16x16, half) * B(16x16, half) + C(16x16, float): the whole warp
// takes part in every call, as with nvcuda::wmma. A fragment is an opaque
// per-thread slice of the tile; ldm is the leading dimension of the matrix
// in memory, in elements (a multiple of 8 for half, 4 for float).
//
//	a := cuda.WmmaLoadA(A.Ptr(row*16*k), k)      // row-major A[m][k]
//	b := cuda.WmmaLoadB(B.Ptr(col*16), n)        // row-major B[k][n]
//	var c cuda.FragC                             // zero accumulator
//	c = cuda.WmmaMma(a, b, c)
//	cuda.WmmaStore(D.Ptr(row*16*n+col*16), n, c) // row-major D[m][n]

// FragA is a wmma::fragment<matrix_a, 16, 16, 16, half>: 8 half2 registers.
type FragA struct{ v [8]uint32 }

// FragB is a wmma::fragment<matrix_b, 16, 16, 16, half>.
type FragB struct{ v [8]uint32 }

// FragC is a wmma::fragment<accumulator, 16, 16, 16, float>: 8 floats per
// thread. The zero value is a zero matrix (wmma::fill_fragment(c, 0)).
type FragC struct{ v [8]float32 }

// Fill sets every element to x (wmma::fill_fragment).
func (c *FragC) Fill(x float32) {
	for i := range c.v {
		c.v[i] = x
	}
}

// Elems is this thread's slice of the accumulator (fragment.x[]); which
// matrix elements they are is unspecified, as in CUDA.
func (c *FragC) Elems() *[8]float32 { return &c.v }

//go:linkname wmmaLoadARow cudair.wmma.load.a.row
func wmmaLoadARow(f *FragA, p *Half, ldm int32)

//go:linkname wmmaLoadACol cudair.wmma.load.a.col
func wmmaLoadACol(f *FragA, p *Half, ldm int32)

//go:linkname wmmaLoadBRow cudair.wmma.load.b.row
func wmmaLoadBRow(f *FragB, p *Half, ldm int32)

//go:linkname wmmaLoadBCol cudair.wmma.load.b.col
func wmmaLoadBCol(f *FragB, p *Half, ldm int32)

//go:linkname wmmaLoadCRow cudair.wmma.load.c.row
func wmmaLoadCRow(f *FragC, p *float32, ldm int32)

//go:linkname wmmaLoadCCol cudair.wmma.load.c.col
func wmmaLoadCCol(f *FragC, p *float32, ldm int32)

//go:linkname wmmaStoreRow cudair.wmma.store.d.row
func wmmaStoreRow(p *float32, f *FragC, ldm int32)

//go:linkname wmmaStoreCol cudair.wmma.store.d.col
func wmmaStoreCol(p *float32, f *FragC, ldm int32)

//go:linkname wmmaMmaRowRow cudair.wmma.mma.row.row
func wmmaMmaRowRow(d *FragC, a *FragA, b *FragB, c *FragC)

//go:linkname wmmaMmaRowCol cudair.wmma.mma.row.col
func wmmaMmaRowCol(d *FragC, a *FragA, b *FragB, c *FragC)

//go:linkname wmmaMmaColRow cudair.wmma.mma.col.row
func wmmaMmaColRow(d *FragC, a *FragA, b *FragB, c *FragC)

//go:linkname wmmaMmaColCol cudair.wmma.mma.col.col
func wmmaMmaColCol(d *FragC, a *FragA, b *FragB, c *FragC)

// WmmaLoadA is wmma::load_matrix_sync for a row-major A tile at p (16x16
// halfs, rows ldm elements apart). WmmaLoadACol reads a column-major tile.
func WmmaLoadA(p *Half, ldm int32) FragA {
	var f FragA
	wmmaLoadARow(&f, p, ldm)
	return f
}

func WmmaLoadACol(p *Half, ldm int32) FragA {
	var f FragA
	wmmaLoadACol(&f, p, ldm)
	return f
}

// WmmaLoadB / WmmaLoadBCol load the B tile (row-major / column-major).
func WmmaLoadB(p *Half, ldm int32) FragB {
	var f FragB
	wmmaLoadBRow(&f, p, ldm)
	return f
}

func WmmaLoadBCol(p *Half, ldm int32) FragB {
	var f FragB
	wmmaLoadBCol(&f, p, ldm)
	return f
}

// WmmaLoadC / WmmaLoadCCol load a float accumulator tile (mem_row_major /
// mem_col_major).
func WmmaLoadC(p *float32, ldm int32) FragC {
	var f FragC
	wmmaLoadCRow(&f, p, ldm)
	return f
}

func WmmaLoadCCol(p *float32, ldm int32) FragC {
	var f FragC
	wmmaLoadCCol(&f, p, ldm)
	return f
}

// WmmaStore / WmmaStoreCol are wmma::store_matrix_sync (mem_row_major /
// mem_col_major) of an accumulator to p.
func WmmaStore(p *float32, ldm int32, c FragC)    { wmmaStoreRow(p, &c, ldm) }
func WmmaStoreCol(p *float32, ldm int32, c FragC) { wmmaStoreCol(p, &c, ldm) }

// WmmaMma is wmma::mma_sync: a*b + c with a loaded by WmmaLoadA (row) and b
// by WmmaLoadB (row). The layout in the name must match how the fragments
// were loaded: WmmaMmaRowCol for WmmaLoadA + WmmaLoadBCol, and so on.
func WmmaMma(a FragA, b FragB, c FragC) FragC {
	var d FragC
	wmmaMmaRowRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaRowCol(a FragA, b FragB, c FragC) FragC {
	var d FragC
	wmmaMmaRowCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaColRow(a FragA, b FragB, c FragC) FragC {
	var d FragC
	wmmaMmaColRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaColCol(a FragA, b FragB, c FragC) FragC {
	var d FragC
	wmmaMmaColCol(&d, &a, &b, &c)
	return d
}

// ---- wmma with the other element types: bf16 and tf32 (sm_80+) and the
// 8-bit integer shapes (sm_72+).
//
// The accumulator is 8 registers in every geometry, so bf16 and tf32 reuse
// FragC (float) and only the integer shapes need FragCI32. A/B fragments are
// 4 registers for bf16 and tf32 (m16n16k16 / m16n16k8) and 2 for s8/u8.

// FragABF16 / FragBBF16 are wmma::fragment<matrix_a|b, 16, 16, 16,
// nv_bfloat16>: 4 registers, 8 bfloat16s per lane.
type FragABF16 struct{ v [4]uint32 }
type FragBBF16 struct{ v [4]uint32 }

// FragATF32 / FragBTF32 are the m16n16k8 tf32 operands: 4 registers.
type FragATF32 struct{ v [4]uint32 }
type FragBTF32 struct{ v [4]uint32 }

// FragAI8 / FragBI8 are the m16n16k16 u8/s8 operands: 2 registers, 8 bytes
// per lane.
type FragAI8 struct{ v [2]uint32 }
type FragBI8 struct{ v [2]uint32 }

// FragCI32 is wmma::fragment<accumulator, 16, 16, 16, int>: 8 int32s per
// lane. The zero value is a zero matrix.
type FragCI32 struct{ v [8]int32 }

// Fill sets every element to x.
func (c *FragCI32) Fill(x int32) {
	for i := range c.v {
		c.v[i] = x
	}
}

// Elems is this thread's slice of the accumulator.
func (c *FragCI32) Elems() *[8]int32 { return &c.v }

//go:linkname wmmaLoadABF16Row cudair.wmma.load.bf16.a.row
func wmmaLoadABF16Row(f *FragABF16, p *BFloat16, ldm int32)

//go:linkname wmmaLoadABF16Col cudair.wmma.load.bf16.a.col
func wmmaLoadABF16Col(f *FragABF16, p *BFloat16, ldm int32)

//go:linkname wmmaLoadBBF16Row cudair.wmma.load.bf16.b.row
func wmmaLoadBBF16Row(f *FragBBF16, p *BFloat16, ldm int32)

//go:linkname wmmaLoadBBF16Col cudair.wmma.load.bf16.b.col
func wmmaLoadBBF16Col(f *FragBBF16, p *BFloat16, ldm int32)

//go:linkname wmmaMmaBF16RowRow cudair.wmma.mma.bf16.row.row
func wmmaMmaBF16RowRow(d *FragC, a *FragABF16, b *FragBBF16, c *FragC)

//go:linkname wmmaMmaBF16RowCol cudair.wmma.mma.bf16.row.col
func wmmaMmaBF16RowCol(d *FragC, a *FragABF16, b *FragBBF16, c *FragC)

//go:linkname wmmaMmaBF16ColRow cudair.wmma.mma.bf16.col.row
func wmmaMmaBF16ColRow(d *FragC, a *FragABF16, b *FragBBF16, c *FragC)

//go:linkname wmmaMmaBF16ColCol cudair.wmma.mma.bf16.col.col
func wmmaMmaBF16ColCol(d *FragC, a *FragABF16, b *FragBBF16, c *FragC)

// WmmaLoadABF16 / WmmaLoadBBF16 (and the Col forms) are
// wmma::load_matrix_sync for a 16x16 bfloat16 tile at p.
func WmmaLoadABF16(p *BFloat16, ldm int32) FragABF16 {
	var f FragABF16
	wmmaLoadABF16Row(&f, p, ldm)
	return f
}

func WmmaLoadABF16Col(p *BFloat16, ldm int32) FragABF16 {
	var f FragABF16
	wmmaLoadABF16Col(&f, p, ldm)
	return f
}

func WmmaLoadBBF16(p *BFloat16, ldm int32) FragBBF16 {
	var f FragBBF16
	wmmaLoadBBF16Row(&f, p, ldm)
	return f
}

func WmmaLoadBBF16Col(p *BFloat16, ldm int32) FragBBF16 {
	var f FragBBF16
	wmmaLoadBBF16Col(&f, p, ldm)
	return f
}

// WmmaMmaBF16 is wmma::mma_sync for bfloat16 operands accumulating in float;
// the layout in the name must match how the fragments were loaded.
func WmmaMmaBF16(a FragABF16, b FragBBF16, c FragC) FragC {
	var d FragC
	wmmaMmaBF16RowRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaBF16RowCol(a FragABF16, b FragBBF16, c FragC) FragC {
	var d FragC
	wmmaMmaBF16RowCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaBF16ColRow(a FragABF16, b FragBBF16, c FragC) FragC {
	var d FragC
	wmmaMmaBF16ColRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaBF16ColCol(a FragABF16, b FragBBF16, c FragC) FragC {
	var d FragC
	wmmaMmaBF16ColCol(&d, &a, &b, &c)
	return d
}

//go:linkname wmmaLoadATF32Row cudair.wmma.load.tf32.a.row
func wmmaLoadATF32Row(f *FragATF32, p *float32, ldm int32)

//go:linkname wmmaLoadATF32Col cudair.wmma.load.tf32.a.col
func wmmaLoadATF32Col(f *FragATF32, p *float32, ldm int32)

//go:linkname wmmaLoadBTF32Row cudair.wmma.load.tf32.b.row
func wmmaLoadBTF32Row(f *FragBTF32, p *float32, ldm int32)

//go:linkname wmmaLoadBTF32Col cudair.wmma.load.tf32.b.col
func wmmaLoadBTF32Col(f *FragBTF32, p *float32, ldm int32)

//go:linkname wmmaLoadCTF32Row cudair.wmma.load.tf32.c.row
func wmmaLoadCTF32Row(f *FragC, p *float32, ldm int32)

//go:linkname wmmaLoadCTF32Col cudair.wmma.load.tf32.c.col
func wmmaLoadCTF32Col(f *FragC, p *float32, ldm int32)

//go:linkname wmmaStoreTF32Row cudair.wmma.store.tf32.d.row
func wmmaStoreTF32Row(p *float32, f *FragC, ldm int32)

//go:linkname wmmaStoreTF32Col cudair.wmma.store.tf32.d.col
func wmmaStoreTF32Col(p *float32, f *FragC, ldm int32)

//go:linkname wmmaMmaTF32RowRow cudair.wmma.mma.tf32.row.row
func wmmaMmaTF32RowRow(d *FragC, a *FragATF32, b *FragBTF32, c *FragC)

//go:linkname wmmaMmaTF32RowCol cudair.wmma.mma.tf32.row.col
func wmmaMmaTF32RowCol(d *FragC, a *FragATF32, b *FragBTF32, c *FragC)

//go:linkname wmmaMmaTF32ColRow cudair.wmma.mma.tf32.col.row
func wmmaMmaTF32ColRow(d *FragC, a *FragATF32, b *FragBTF32, c *FragC)

//go:linkname wmmaMmaTF32ColCol cudair.wmma.mma.tf32.col.col
func wmmaMmaTF32ColCol(d *FragC, a *FragATF32, b *FragBTF32, c *FragC)

// WmmaLoadATF32 / WmmaLoadBTF32 (and the Col forms) load the m16n16k8 tf32
// operands: A is 16x8 and B 8x16 float32, rounded to tf32 by the load.
func WmmaLoadATF32(p *float32, ldm int32) FragATF32 {
	var f FragATF32
	wmmaLoadATF32Row(&f, p, ldm)
	return f
}

func WmmaLoadATF32Col(p *float32, ldm int32) FragATF32 {
	var f FragATF32
	wmmaLoadATF32Col(&f, p, ldm)
	return f
}

func WmmaLoadBTF32(p *float32, ldm int32) FragBTF32 {
	var f FragBTF32
	wmmaLoadBTF32Row(&f, p, ldm)
	return f
}

func WmmaLoadBTF32Col(p *float32, ldm int32) FragBTF32 {
	var f FragBTF32
	wmmaLoadBTF32Col(&f, p, ldm)
	return f
}

// WmmaLoadCTF32 / WmmaStoreTF32 move the m16n16k8 float accumulator; the
// fragment is the same FragC as the m16n16k16 shapes, but the instruction
// carries its own geometry.
func WmmaLoadCTF32(p *float32, ldm int32) FragC {
	var f FragC
	wmmaLoadCTF32Row(&f, p, ldm)
	return f
}

func WmmaLoadCTF32Col(p *float32, ldm int32) FragC {
	var f FragC
	wmmaLoadCTF32Col(&f, p, ldm)
	return f
}

func WmmaStoreTF32(p *float32, ldm int32, c FragC)    { wmmaStoreTF32Row(p, &c, ldm) }
func WmmaStoreTF32Col(p *float32, ldm int32, c FragC) { wmmaStoreTF32Col(p, &c, ldm) }

// WmmaMmaTF32 is wmma::mma_sync for the m16n16k8 tf32 shape.
func WmmaMmaTF32(a FragATF32, b FragBTF32, c FragC) FragC {
	var d FragC
	wmmaMmaTF32RowRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaTF32RowCol(a FragATF32, b FragBTF32, c FragC) FragC {
	var d FragC
	wmmaMmaTF32RowCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaTF32ColRow(a FragATF32, b FragBTF32, c FragC) FragC {
	var d FragC
	wmmaMmaTF32ColRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaTF32ColCol(a FragATF32, b FragBTF32, c FragC) FragC {
	var d FragC
	wmmaMmaTF32ColCol(&d, &a, &b, &c)
	return d
}

//go:linkname wmmaLoadAS8Row cudair.wmma.load.s8.a.row
func wmmaLoadAS8Row(f *FragAI8, p *int8, ldm int32)

//go:linkname wmmaLoadAS8Col cudair.wmma.load.s8.a.col
func wmmaLoadAS8Col(f *FragAI8, p *int8, ldm int32)

//go:linkname wmmaLoadBS8Row cudair.wmma.load.s8.b.row
func wmmaLoadBS8Row(f *FragBI8, p *int8, ldm int32)

//go:linkname wmmaLoadBS8Col cudair.wmma.load.s8.b.col
func wmmaLoadBS8Col(f *FragBI8, p *int8, ldm int32)

//go:linkname wmmaLoadAU8Row cudair.wmma.load.u8.a.row
func wmmaLoadAU8Row(f *FragAI8, p *uint8, ldm int32)

//go:linkname wmmaLoadAU8Col cudair.wmma.load.u8.a.col
func wmmaLoadAU8Col(f *FragAI8, p *uint8, ldm int32)

//go:linkname wmmaLoadBU8Row cudair.wmma.load.u8.b.row
func wmmaLoadBU8Row(f *FragBI8, p *uint8, ldm int32)

//go:linkname wmmaLoadBU8Col cudair.wmma.load.u8.b.col
func wmmaLoadBU8Col(f *FragBI8, p *uint8, ldm int32)

//go:linkname wmmaLoadCI32Row cudair.wmma.load.s32.c.row
func wmmaLoadCI32Row(f *FragCI32, p *int32, ldm int32)

//go:linkname wmmaLoadCI32Col cudair.wmma.load.s32.c.col
func wmmaLoadCI32Col(f *FragCI32, p *int32, ldm int32)

//go:linkname wmmaStoreI32Row cudair.wmma.store.s32.d.row
func wmmaStoreI32Row(p *int32, f *FragCI32, ldm int32)

//go:linkname wmmaStoreI32Col cudair.wmma.store.s32.d.col
func wmmaStoreI32Col(p *int32, f *FragCI32, ldm int32)

//go:linkname wmmaMmaS8RowRow cudair.wmma.mma.s8.row.row
func wmmaMmaS8RowRow(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaS8RowCol cudair.wmma.mma.s8.row.col
func wmmaMmaS8RowCol(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaS8ColRow cudair.wmma.mma.s8.col.row
func wmmaMmaS8ColRow(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaS8ColCol cudair.wmma.mma.s8.col.col
func wmmaMmaS8ColCol(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaU8RowRow cudair.wmma.mma.u8.row.row
func wmmaMmaU8RowRow(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaU8RowCol cudair.wmma.mma.u8.row.col
func wmmaMmaU8RowCol(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaU8ColRow cudair.wmma.mma.u8.col.row
func wmmaMmaU8ColRow(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

//go:linkname wmmaMmaU8ColCol cudair.wmma.mma.u8.col.col
func wmmaMmaU8ColCol(d *FragCI32, a *FragAI8, b *FragBI8, c *FragCI32)

// WmmaLoadAS8 / WmmaLoadBS8 / WmmaLoadAU8 / WmmaLoadBU8 (and the Col forms)
// load a 16x16 tile of signed or unsigned bytes.
func WmmaLoadAS8(p *int8, ldm int32) FragAI8 {
	var f FragAI8
	wmmaLoadAS8Row(&f, p, ldm)
	return f
}

func WmmaLoadAS8Col(p *int8, ldm int32) FragAI8 {
	var f FragAI8
	wmmaLoadAS8Col(&f, p, ldm)
	return f
}

func WmmaLoadBS8(p *int8, ldm int32) FragBI8 {
	var f FragBI8
	wmmaLoadBS8Row(&f, p, ldm)
	return f
}

func WmmaLoadBS8Col(p *int8, ldm int32) FragBI8 {
	var f FragBI8
	wmmaLoadBS8Col(&f, p, ldm)
	return f
}

func WmmaLoadAU8(p *uint8, ldm int32) FragAI8 {
	var f FragAI8
	wmmaLoadAU8Row(&f, p, ldm)
	return f
}

func WmmaLoadAU8Col(p *uint8, ldm int32) FragAI8 {
	var f FragAI8
	wmmaLoadAU8Col(&f, p, ldm)
	return f
}

func WmmaLoadBU8(p *uint8, ldm int32) FragBI8 {
	var f FragBI8
	wmmaLoadBU8Row(&f, p, ldm)
	return f
}

func WmmaLoadBU8Col(p *uint8, ldm int32) FragBI8 {
	var f FragBI8
	wmmaLoadBU8Col(&f, p, ldm)
	return f
}

// WmmaLoadCI32 / WmmaStoreI32 move the int32 accumulator (mem_row_major /
// mem_col_major).
func WmmaLoadCI32(p *int32, ldm int32) FragCI32 {
	var f FragCI32
	wmmaLoadCI32Row(&f, p, ldm)
	return f
}

func WmmaLoadCI32Col(p *int32, ldm int32) FragCI32 {
	var f FragCI32
	wmmaLoadCI32Col(&f, p, ldm)
	return f
}

func WmmaStoreI32(p *int32, ldm int32, c FragCI32)    { wmmaStoreI32Row(p, &c, ldm) }
func WmmaStoreI32Col(p *int32, ldm int32, c FragCI32) { wmmaStoreI32Col(p, &c, ldm) }

// WmmaMmaS8 / WmmaMmaU8 are wmma::mma_sync for the 8-bit integer shapes,
// accumulating in int32.
func WmmaMmaS8(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaS8RowRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaS8RowCol(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaS8RowCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaS8ColRow(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaS8ColRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaS8ColCol(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaS8ColCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaU8(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaU8RowRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaU8RowCol(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaU8RowCol(&d, &a, &b, &c)
	return d
}

func WmmaMmaU8ColRow(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaU8ColRow(&d, &a, &b, &c)
	return d
}

func WmmaMmaU8ColCol(a FragAI8, b FragBI8, c FragCI32) FragCI32 {
	var d FragCI32
	wmmaMmaU8ColCol(&d, &a, &b, &c)
	return d
}
