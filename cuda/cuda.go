// Package cuda is the device-side API for Go kernels compiled with
// `gocuda build`. Everything here maps to NVPTX intrinsics or to plain
// pointer arithmetic, so a kernel needs no llgo runtime on the GPU.
//
// Rules for kernel functions:
//   - exported, returns nothing
//   - parameters: Buf[T], *T, int32/uint32/int64/uint64, float32/float64
//   - no make/append/maps/strings/interfaces/goroutines/defer
//   - a Go panic (nil deref, index out of range) becomes a PTX trap
package cuda

import "unsafe"

// ---- thread / block indices (CUDA threadIdx, blockIdx, blockDim, gridDim)

//go:linkname ThreadIdxX llvm.nvvm.read.ptx.sreg.tid.x
func ThreadIdxX() int32

//go:linkname ThreadIdxY llvm.nvvm.read.ptx.sreg.tid.y
func ThreadIdxY() int32

//go:linkname ThreadIdxZ llvm.nvvm.read.ptx.sreg.tid.z
func ThreadIdxZ() int32

//go:linkname BlockIdxX llvm.nvvm.read.ptx.sreg.ctaid.x
func BlockIdxX() int32

//go:linkname BlockIdxY llvm.nvvm.read.ptx.sreg.ctaid.y
func BlockIdxY() int32

//go:linkname BlockIdxZ llvm.nvvm.read.ptx.sreg.ctaid.z
func BlockIdxZ() int32

//go:linkname BlockDimX llvm.nvvm.read.ptx.sreg.ntid.x
func BlockDimX() int32

//go:linkname BlockDimY llvm.nvvm.read.ptx.sreg.ntid.y
func BlockDimY() int32

//go:linkname BlockDimZ llvm.nvvm.read.ptx.sreg.ntid.z
func BlockDimZ() int32

//go:linkname GridDimX llvm.nvvm.read.ptx.sreg.nctaid.x
func GridDimX() int32

//go:linkname GridDimY llvm.nvvm.read.ptx.sreg.nctaid.y
func GridDimY() int32

//go:linkname GridDimZ llvm.nvvm.read.ptx.sreg.nctaid.z
func GridDimZ() int32

//go:linkname WarpSize llvm.nvvm.read.ptx.sreg.warpsize
func WarpSize() int32

//go:linkname LaneID llvm.nvvm.read.ptx.sreg.laneid
func LaneID() int32

// GlobalIdX is blockIdx.x*blockDim.x + threadIdx.x.
func GlobalIdX() int32 { return BlockIdxX()*BlockDimX() + ThreadIdxX() }

// GlobalIdY is blockIdx.y*blockDim.y + threadIdx.y.
func GlobalIdY() int32 { return BlockIdxY()*BlockDimY() + ThreadIdxY() }

// GlobalIdZ is blockIdx.z*blockDim.z + threadIdx.z.
func GlobalIdZ() int32 { return BlockIdxZ()*BlockDimZ() + ThreadIdxZ() }

// GridStrideX is the total number of threads in x (for grid-stride loops).
func GridStrideX() int32 { return GridDimX() * BlockDimX() }

// ---- device memory

// Buf is a typed view of device memory. As a kernel parameter it is
// exactly one 64-bit device pointer, so the host passes a CUdeviceptr.
// Indexing is unchecked, like C.
type Buf[T any] struct{ ptr *T }

// BufOf wraps a raw pointer.
func BufOf[T any](p *T) Buf[T] { return Buf[T]{p} }

// Ptr returns the raw pointer of the element i.
func (b Buf[T]) Ptr(i int32) *T {
	return (*T)(unsafe.Add(unsafe.Pointer(b.ptr), uintptr(i)*unsafe.Sizeof(*b.ptr)))
}

// At reads element i.
func (b Buf[T]) At(i int32) T { return *b.Ptr(i) }

// Set writes element i.
func (b Buf[T]) Set(i int32, v T) { *b.Ptr(i) = v }

// Slice returns a view starting at element off.
func (b Buf[T]) Slice(off int32) Buf[T] { return Buf[T]{b.Ptr(off)} }

// Load / Store are the same for a raw *T parameter.
func Load[T any](p *T, i int32) T {
	return *(*T)(unsafe.Add(unsafe.Pointer(p), uintptr(i)*unsafe.Sizeof(*p)))
}

func Store[T any](p *T, i int32, v T) {
	*(*T)(unsafe.Add(unsafe.Pointer(p), uintptr(i)*unsafe.Sizeof(*p))) = v
}
