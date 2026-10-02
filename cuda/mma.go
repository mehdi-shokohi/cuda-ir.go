package cuda

import _ "unsafe" // for go:linkname

// ---- tensor cores: raw mma.sync shapes (sm_80+)
//
// wmma (see wmma.go) keeps the fragment layout opaque and only works on
// 16x16x16 tiles. mma.sync is the instruction the libraries use: the warp
// computes D[16][8] = A[16][K] * B[K][8] + C[16][8] and the per-lane
// fragment layout is fixed by the ISA, so a kernel can build fragments any
// way it likes — element by element, with a vector load, or with ldmatrix.
//
// Only the row:col operand layout exists in hardware (A row-major,
// B column-major), which is what MmaLoadA* / MmaLoadB* produce.
//
//	var acc cuda.MmaAccF32                       // 16x8 f32 accumulator
//	for k := int32(0); k < n; k += 16 {
//		a := cuda.MmaLoadA16(A.Ptr(row*16*n+k), n)  // 16x16 half tile
//		b := cuda.MmaLoadB16(B.Ptr(k*n+col*8), n)   // 16x8 half tile
//		acc = cuda.MmaF16(a, b, acc)
//	}
//	cuda.MmaStoreAcc(C.Ptr(row*16*n+col*8), n, acc)

// MmaA is the A operand of an mma.sync m16n8* instruction: four 32-bit
// registers, holding 8 halfs / 8 bfloat16s / 4 tf32s / 16 int8s per lane
// depending on the shape.
type MmaA struct{ v [4]uint32 }

// MmaB is the B operand: two registers.
type MmaB struct{ v [2]uint32 }

// MmaAccF32 is a 16x8 f32 accumulator (C and D): four floats per lane.
// The zero value is a zero matrix.
type MmaAccF32 struct{ v [4]float32 }

// MmaAccI32 is a 16x8 s32 accumulator, for the integer shapes.
type MmaAccI32 struct{ v [4]int32 }

// MmaAccF16 is a 16x8 f16 accumulator: two registers (4 halfs) per lane.
type MmaAccF16 struct{ v [2]uint32 }

// Regs is this lane's registers; which matrix elements they hold is fixed by
// the ISA and documented with the MmaLoad* helpers that fill them.
func (a *MmaA) Regs() *[4]uint32      { return &a.v }
func (b *MmaB) Regs() *[2]uint32      { return &b.v }
func (c *MmaAccF16) Regs() *[2]uint32 { return &c.v }

// Elems is this lane's four accumulator elements.
func (c *MmaAccF32) Elems() *[4]float32 { return &c.v }
func (c *MmaAccI32) Elems() *[4]int32   { return &c.v }

// Fill sets every accumulator element to x.
func (c *MmaAccF32) Fill(x float32) {
	for i := range c.v {
		c.v[i] = x
	}
}

func (c *MmaAccI32) Fill(x int32) {
	for i := range c.v {
		c.v[i] = x
	}
}

//go:linkname mmaF16 cudair.mma.m16n8k16.f16.f32
func mmaF16(d *MmaAccF32, a *MmaA, b *MmaB, c *MmaAccF32)

//go:linkname mmaF16Half cudair.mma.m16n8k16.f16.f16
func mmaF16Half(d *MmaAccF16, a *MmaA, b *MmaB, c *MmaAccF16)

//go:linkname mmaBF16 cudair.mma.m16n8k16.bf16
func mmaBF16(d *MmaAccF32, a *MmaA, b *MmaB, c *MmaAccF32)

//go:linkname mmaTF32 cudair.mma.m16n8k8.tf32
func mmaTF32(d *MmaAccF32, a *MmaA, b *MmaB, c *MmaAccF32)

//go:linkname mmaS8 cudair.mma.m16n8k32.s8
func mmaS8(d *MmaAccI32, a *MmaA, b *MmaB, c *MmaAccI32)

//go:linkname mmaU8 cudair.mma.m16n8k32.u8
func mmaU8(d *MmaAccI32, a *MmaA, b *MmaB, c *MmaAccI32)

//go:linkname mmaS8Sat cudair.mma.m16n8k32.s8.sat
func mmaS8Sat(d *MmaAccI32, a *MmaA, b *MmaB, c *MmaAccI32)

//go:linkname mmaU8Sat cudair.mma.m16n8k32.u8.sat
func mmaU8Sat(d *MmaAccI32, a *MmaA, b *MmaB, c *MmaAccI32)

// MmaF16 is mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32: half inputs,
// float accumulate. a and b must come from MmaLoadA16 / MmaLoadB16 (or hold
// the same layout).
func MmaF16(a MmaA, b MmaB, c MmaAccF32) MmaAccF32 {
	var d MmaAccF32
	mmaF16(&d, &a, &b, &c)
	return d
}

// MmaF16Half is the same shape accumulating in half
// (m16n8k16.row.col.f16.f16.f16.f16).
func MmaF16Half(a MmaA, b MmaB, c MmaAccF16) MmaAccF16 {
	var d MmaAccF16
	mmaF16Half(&d, &a, &b, &c)
	return d
}

// MmaBF16 is m16n8k16.row.col.f32.bf16.bf16.f32: bfloat16 inputs (loaded by
// MmaLoadA16 / MmaLoadB16, whose layout is the same as for half), float
// accumulate.
func MmaBF16(a MmaA, b MmaB, c MmaAccF32) MmaAccF32 {
	var d MmaAccF32
	mmaBF16(&d, &a, &b, &c)
	return d
}

// MmaTF32 is m16n8k8.row.col.f32.tf32.tf32.f32: A is 16x8 and B 8x8 (see
// MmaLoadATF32 / MmaLoadBTF32).
func MmaTF32(a MmaA, b MmaB, c MmaAccF32) MmaAccF32 {
	var d MmaAccF32
	mmaTF32(&d, &a, &b, &c)
	return d
}

// MmaS8 / MmaU8 are m16n8k32.row.col.s32.{s8,u8}.{s8,u8}.s32: A is 16x32 and
// B 32x8 bytes (MmaLoadA8 / MmaLoadB8), accumulating in int32. The Sat forms
// add .satfinite, clamping instead of wrapping.
func MmaS8(a MmaA, b MmaB, c MmaAccI32) MmaAccI32 {
	var d MmaAccI32
	mmaS8(&d, &a, &b, &c)
	return d
}

func MmaU8(a MmaA, b MmaB, c MmaAccI32) MmaAccI32 {
	var d MmaAccI32
	mmaU8(&d, &a, &b, &c)
	return d
}

func MmaS8Sat(a MmaA, b MmaB, c MmaAccI32) MmaAccI32 {
	var d MmaAccI32
	mmaS8Sat(&d, &a, &b, &c)
	return d
}

func MmaU8Sat(a MmaA, b MmaB, c MmaAccI32) MmaAccI32 {
	var d MmaAccI32
	mmaU8Sat(&d, &a, &b, &c)
	return d
}

// ---- building the fragments
//
// Every m16n8* shape splits the warp into eight groups of four lanes:
// group = laneID/4 picks the row (and row+8) of A and C, lane%4 picks the
// column within the group's slice. The helpers below do that arithmetic;
// the element order inside a register is low half first.

// F16 is a 16-bit tensor-core element type: Half or BFloat16. Both have the
// same fragment layout, so one set of loads serves MmaF16 and MmaBF16.
type F16 interface{ Half | BFloat16 }

// Int8 is an 8-bit tensor-core element type.
type Int8 interface{ int8 | uint8 }

// MmaLoadA16 is the A operand of m16n8k16 read from a row-major 16x16 tile
// of 16-bit elements at p, rows ldm elements apart.
func MmaLoadA16[T F16](p *T, ldm int32) MmaA {
	g, t := LaneID()>>2, LaneID()&3
	var a MmaA
	a.v[0] = pack16(p, g*ldm+t*2)
	a.v[1] = pack16(p, (g+8)*ldm+t*2)
	a.v[2] = pack16(p, g*ldm+t*2+8)
	a.v[3] = pack16(p, (g+8)*ldm+t*2+8)
	return a
}

// MmaLoadB16 is the B operand of m16n8k16 read from a row-major 16x8 tile
// (K x N) at p: the natural layout when B is stored row-major, as in
// C[m][n] = sum_k A[m][k]*B[k][n].
func MmaLoadB16[T F16](p *T, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = uint32(load16(p, t*2*ldm+g)) | uint32(load16(p, (t*2+1)*ldm+g))<<16
	b.v[1] = uint32(load16(p, (t*2+8)*ldm+g)) | uint32(load16(p, (t*2+9)*ldm+g))<<16
	return b
}

// MmaLoadB16Col is the B operand read from a column-major 16x8 tile, i.e. a
// row-major 8x16 tile of B transposed at p (rows ldm elements apart). This
// is the contiguous case and what a transposed weight matrix gives.
func MmaLoadB16Col[T F16](p *T, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = pack16(p, g*ldm+t*2)
	b.v[1] = pack16(p, g*ldm+t*2+8)
	return b
}

// MmaLoadATF32 / MmaLoadBTF32 are the m16n8k8 operands: a row-major 16x8 A
// tile and a row-major 8x8 B tile of float32, rounded to tf32 (cvt.rna).
func MmaLoadATF32(p *float32, ldm int32) MmaA {
	g, t := LaneID()>>2, LaneID()&3
	var a MmaA
	a.v[0] = FloatToTF32(Load(p, g*ldm+t))
	a.v[1] = FloatToTF32(Load(p, (g+8)*ldm+t))
	a.v[2] = FloatToTF32(Load(p, g*ldm+t+4))
	a.v[3] = FloatToTF32(Load(p, (g+8)*ldm+t+4))
	return a
}

func MmaLoadBTF32(p *float32, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = FloatToTF32(Load(p, t*ldm+g))
	b.v[1] = FloatToTF32(Load(p, (t+4)*ldm+g))
	return b
}

// MmaLoadBTF32Col reads B from a column-major 8x8 tile (B transposed,
// row-major, at p).
func MmaLoadBTF32Col(p *float32, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = FloatToTF32(Load(p, g*ldm+t))
	b.v[1] = FloatToTF32(Load(p, g*ldm+t+4))
	return b
}

// MmaLoadA8 is the A operand of m16n8k32 read from a row-major 16x32 tile of
// bytes at p.
func MmaLoadA8[T Int8](p *T, ldm int32) MmaA {
	g, t := LaneID()>>2, LaneID()&3
	var a MmaA
	a.v[0] = pack8(p, g*ldm+t*4)
	a.v[1] = pack8(p, (g+8)*ldm+t*4)
	a.v[2] = pack8(p, g*ldm+t*4+16)
	a.v[3] = pack8(p, (g+8)*ldm+t*4+16)
	return a
}

// MmaLoadB8 is the B operand of m16n8k32 read from a row-major 32x8 tile
// (K x N) of bytes at p.
func MmaLoadB8[T Int8](p *T, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = strided8(p, t*4*ldm+g, ldm)
	b.v[1] = strided8(p, (t*4+16)*ldm+g, ldm)
	return b
}

// MmaLoadB8Col is the B operand read from a column-major 32x8 tile, i.e. a
// row-major 8x32 tile of B transposed at p.
func MmaLoadB8Col[T Int8](p *T, ldm int32) MmaB {
	g, t := LaneID()>>2, LaneID()&3
	var b MmaB
	b.v[0] = pack8(p, g*ldm+t*4)
	b.v[1] = pack8(p, g*ldm+t*4+16)
	return b
}

// MmaLoadAcc / MmaStoreAcc move a 16x8 f32 accumulator to and from a
// row-major tile at p, rows ldm elements apart.
func MmaLoadAcc(p *float32, ldm int32) MmaAccF32 {
	g, t := LaneID()>>2, LaneID()&3
	var c MmaAccF32
	c.v[0] = Load(p, g*ldm+t*2)
	c.v[1] = Load(p, g*ldm+t*2+1)
	c.v[2] = Load(p, (g+8)*ldm+t*2)
	c.v[3] = Load(p, (g+8)*ldm+t*2+1)
	return c
}

func MmaStoreAcc(p *float32, ldm int32, c MmaAccF32) {
	g, t := LaneID()>>2, LaneID()&3
	Store(p, g*ldm+t*2, c.v[0])
	Store(p, g*ldm+t*2+1, c.v[1])
	Store(p, (g+8)*ldm+t*2, c.v[2])
	Store(p, (g+8)*ldm+t*2+1, c.v[3])
}

// MmaLoadAccI32 / MmaStoreAccI32 are the same for an s32 accumulator.
func MmaLoadAccI32(p *int32, ldm int32) MmaAccI32 {
	g, t := LaneID()>>2, LaneID()&3
	var c MmaAccI32
	c.v[0] = Load(p, g*ldm+t*2)
	c.v[1] = Load(p, g*ldm+t*2+1)
	c.v[2] = Load(p, (g+8)*ldm+t*2)
	c.v[3] = Load(p, (g+8)*ldm+t*2+1)
	return c
}

func MmaStoreAccI32(p *int32, ldm int32, c MmaAccI32) {
	g, t := LaneID()>>2, LaneID()&3
	Store(p, g*ldm+t*2, c.v[0])
	Store(p, g*ldm+t*2+1, c.v[1])
	Store(p, (g+8)*ldm+t*2, c.v[2])
	Store(p, (g+8)*ldm+t*2+1, c.v[3])
}

// MmaLoadAccF16 / MmaStoreAccF16 are the same for an f16 accumulator.
func MmaLoadAccF16[T F16](p *T, ldm int32) MmaAccF16 {
	g, t := LaneID()>>2, LaneID()&3
	var c MmaAccF16
	c.v[0] = pack16(p, g*ldm+t*2)
	c.v[1] = pack16(p, (g+8)*ldm+t*2)
	return c
}

func MmaStoreAccF16[T F16](p *T, ldm int32, c MmaAccF16) {
	g, t := LaneID()>>2, LaneID()&3
	Store(p, g*ldm+t*2, T(uint16(c.v[0])))
	Store(p, g*ldm+t*2+1, T(uint16(c.v[0]>>16)))
	Store(p, (g+8)*ldm+t*2, T(uint16(c.v[1])))
	Store(p, (g+8)*ldm+t*2+1, T(uint16(c.v[1]>>16)))
}

// load16 reads element i of p as its 16-bit pattern.
func load16[T F16](p *T, i int32) uint16 { return uint16(Load(p, i)) }

// pack16 reads elements i and i+1 into one register (low half first).
func pack16[T F16](p *T, i int32) uint32 {
	return uint32(load16(p, i)) | uint32(load16(p, i+1))<<16
}

// pack8 reads four consecutive bytes from i into one register.
func pack8[T Int8](p *T, i int32) uint32 {
	return uint32(uint8(Load(p, i))) | uint32(uint8(Load(p, i+1)))<<8 |
		uint32(uint8(Load(p, i+2)))<<16 | uint32(uint8(Load(p, i+3)))<<24
}

// strided8 reads four bytes ldm apart starting at i into one register.
func strided8[T Int8](p *T, i, ldm int32) uint32 {
	return uint32(uint8(Load(p, i))) | uint32(uint8(Load(p, i+ldm)))<<8 |
		uint32(uint8(Load(p, i+2*ldm)))<<16 | uint32(uint8(Load(p, i+3*ldm)))<<24
}

//go:linkname f2tf32 llvm.nvvm.f2tf32.rna
func f2tf32(x float32) uint32

// FloatToTF32 rounds a float32 to tf32 (cvt.rna.tf32.f32), the register form
// the m16n8k8 and wmma m16n16k8 tf32 shapes take.
func FloatToTF32(x float32) uint32 { return f2tf32(x) }

// ---- ldmatrix / stmatrix (sm_75+ / sm_90+)
//
// One instruction for the whole warp moves one, two or four 8x8 matrices of
// 16-bit elements between shared memory and registers, in exactly the layout
// mma.sync wants. Each lane passes the address of one row: lanes 0-7 give the
// rows of the first matrix, 8-15 the second, and so on, and a lane whose
// matrix is not being loaded may pass any address in the tile.
//
//	rows := tile.Get()
//	a := cuda.LdMatrix4(&rows[cuda.LaneID()%8*ldm], ...)  // -> 4 registers

//go:linkname ldmatrix1 cudair.ldmatrix.x1
func ldmatrix1(f *uint32, p *Half)

//go:linkname ldmatrix2 cudair.ldmatrix.x2
func ldmatrix2(f *[2]uint32, p *Half)

//go:linkname ldmatrix4 cudair.ldmatrix.x4
func ldmatrix4(f *[4]uint32, p *Half)

//go:linkname ldmatrix1T cudair.ldmatrix.x1trans
func ldmatrix1T(f *uint32, p *Half)

//go:linkname ldmatrix2T cudair.ldmatrix.x2trans
func ldmatrix2T(f *[2]uint32, p *Half)

//go:linkname ldmatrix4T cudair.ldmatrix.x4trans
func ldmatrix4T(f *[4]uint32, p *Half)

//go:linkname stmatrix1 cudair.stmatrix.x1
func stmatrix1(p *Half, f *uint32)

//go:linkname stmatrix2 cudair.stmatrix.x2
func stmatrix2(p *Half, f *[2]uint32)

//go:linkname stmatrix4 cudair.stmatrix.x4
func stmatrix4(p *Half, f *[4]uint32)

//go:linkname stmatrix1T cudair.stmatrix.x1trans
func stmatrix1T(p *Half, f *uint32)

//go:linkname stmatrix2T cudair.stmatrix.x2trans
func stmatrix2T(p *Half, f *[2]uint32)

//go:linkname stmatrix4T cudair.stmatrix.x4trans
func stmatrix4T(p *Half, f *[4]uint32)

// LdMatrix1 is ldmatrix.sync.aligned.m8n8.x1.shared.b16: one 8x8 matrix of
// 16-bit elements, p being this lane's row. The result is the lane's single
// register, in mma.sync A/B order.
func LdMatrix1[T F16](p *T) uint32 {
	var f uint32
	ldmatrix1(&f, As[Half](p))
	return f
}

// LdMatrix2 loads two 8x8 matrices (lanes 0-7 and 8-15 give the rows).
func LdMatrix2[T F16](p *T) [2]uint32 {
	var f [2]uint32
	ldmatrix2(&f, As[Half](p))
	return f
}

// LdMatrix4 loads four 8x8 matrices, one per group of eight lanes: the whole
// A operand of an mma.m16n8k16, or two B operands.
func LdMatrix4[T F16](p *T) [4]uint32 {
	var f [4]uint32
	ldmatrix4(&f, As[Half](p))
	return f
}

// LdMatrix1Trans / LdMatrix2Trans / LdMatrix4Trans add .trans: each 8x8
// matrix is transposed on the way to the registers.
func LdMatrix1Trans[T F16](p *T) uint32 {
	var f uint32
	ldmatrix1T(&f, As[Half](p))
	return f
}

func LdMatrix2Trans[T F16](p *T) [2]uint32 {
	var f [2]uint32
	ldmatrix2T(&f, As[Half](p))
	return f
}

func LdMatrix4Trans[T F16](p *T) [4]uint32 {
	var f [4]uint32
	ldmatrix4T(&f, As[Half](p))
	return f
}

// StMatrix1 / StMatrix2 / StMatrix4 are stmatrix.sync.aligned.m8n8.xN
// (sm_90+): the inverse of LdMatrixN, writing the registers back to shared
// memory with p this lane's row.
func StMatrix1[T F16](p *T, v uint32)    { stmatrix1(As[Half](p), &v) }
func StMatrix2[T F16](p *T, v [2]uint32) { stmatrix2(As[Half](p), &v) }
func StMatrix4[T F16](p *T, v [4]uint32) { stmatrix4(As[Half](p), &v) }

// StMatrix1Trans / StMatrix2Trans / StMatrix4Trans add .trans.
func StMatrix1Trans[T F16](p *T, v uint32)    { stmatrix1T(As[Half](p), &v) }
func StMatrix2Trans[T F16](p *T, v [2]uint32) { stmatrix2T(As[Half](p), &v) }
func StMatrix4Trans[T F16](p *T, v [4]uint32) { stmatrix4T(As[Half](p), &v) }

// MmaAFromRegs / MmaBFromRegs wrap ldmatrix results as mma operands.
func MmaAFromRegs(v [4]uint32) MmaA { return MmaA{v} }
func MmaBFromRegs(v [2]uint32) MmaB { return MmaB{v} }
