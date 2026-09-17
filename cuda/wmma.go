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
