package host

import (
	"fmt"
	"unsafe"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudasys"
)

// ---- TMA tensor maps (cuTensorMapEncodeTiled)
//
// A tensor map is the 128-byte descriptor cp.async.bulk.tensor copies
// through: it carries the tensor's base address, its shape in global memory
// and the shape of the tile ("box") each copy moves. The host encodes one per
// tensor before launching; the kernel takes it as a grid-constant parameter
// and names it in the copy.
//
//	tm, err := host.EncodeTiled(ctx, dsrc, host.TiledTensor{
//		DataType:      host.TensorMapFloat32,
//		GlobalDim:     []uint64{w, h},     // innermost first, in elements
//		GlobalStrides: []uint64{w * 4},    // bytes, dims 1..rank-1
//		BoxDim:        []uint32{32, 32},   // the shared-memory tile
//	})
//	launch(..., tm.Arg(), ...)
//
// The kernel side is cuda.TensorMap behind //cuda:grid_constant.

// tensorMapBytes is sizeof(CUtensorMap). The descriptor is opaque; only its
// size and 64-byte alignment are part of the ABI.
const tensorMapBytes = 128

// TensorMapDataType is CUtensorMapDataType: the element type of the tensor
// the map describes.
type TensorMapDataType uint32

const (
	TensorMapUint8      TensorMapDataType = 0
	TensorMapUint16     TensorMapDataType = 1
	TensorMapUint32     TensorMapDataType = 2
	TensorMapInt32      TensorMapDataType = 3
	TensorMapUint64     TensorMapDataType = 4
	TensorMapInt64      TensorMapDataType = 5
	TensorMapFloat16    TensorMapDataType = 6
	TensorMapFloat32    TensorMapDataType = 7
	TensorMapFloat64    TensorMapDataType = 8
	TensorMapBFloat16   TensorMapDataType = 9
	TensorMapFloat32FTZ TensorMapDataType = 10
	TensorMapTFloat32   TensorMapDataType = 11
	TensorMapTF32FTZ    TensorMapDataType = 12
)

// size is the element size in bytes, and 0 for an unknown type.
func (d TensorMapDataType) size() int {
	switch d {
	case TensorMapUint8:
		return 1
	case TensorMapUint16, TensorMapFloat16, TensorMapBFloat16:
		return 2
	case TensorMapUint32, TensorMapInt32, TensorMapFloat32, TensorMapFloat32FTZ,
		TensorMapTFloat32, TensorMapTF32FTZ:
		return 4
	case TensorMapUint64, TensorMapInt64, TensorMapFloat64:
		return 8
	}
	return 0
}

// TensorMapInterleave is CUtensorMapInterleave.
type TensorMapInterleave uint32

const (
	InterleaveNone TensorMapInterleave = 0
	Interleave16B  TensorMapInterleave = 1
	Interleave32B  TensorMapInterleave = 2
)

// TensorMapSwizzle is CUtensorMapSwizzle: the shared-memory swizzling pattern
// the copy applies, which must match what the consuming mma expects.
type TensorMapSwizzle uint32

const (
	SwizzleNone TensorMapSwizzle = 0
	Swizzle32B  TensorMapSwizzle = 1
	Swizzle64B  TensorMapSwizzle = 2
	Swizzle128B TensorMapSwizzle = 3
)

// TensorMapL2Promotion is CUtensorMapL2promotion.
type TensorMapL2Promotion uint32

const (
	L2PromotionNone TensorMapL2Promotion = 0
	L2Promotion64B  TensorMapL2Promotion = 1
	L2Promotion128B TensorMapL2Promotion = 2
	L2Promotion256B TensorMapL2Promotion = 3
)

// TensorMapOOBFill is CUtensorMapFloatOOBfill: what a copy reads where the
// tile hangs off the edge of the tensor.
type TensorMapOOBFill uint32

const (
	OOBFillNone TensorMapOOBFill = 0
	// OOBFillNaN fills out-of-bounds elements with a NaN that an mma treats
	// as a zero (CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA).
	OOBFillNaN TensorMapOOBFill = 1
)

// TiledTensor describes a tiled tensor map. Rank is len(GlobalDim), 1 to 5.
// Dimensions are innermost first, matching the driver and the PTX coordinate
// order (and so reversed from the way a Go [h][w]T index reads).
type TiledTensor struct {
	DataType TensorMapDataType

	// GlobalDim is the tensor's extent in elements, Rank entries.
	GlobalDim []uint64
	// GlobalStrides is the stride in bytes of dimensions 1..Rank-1 (so
	// Rank-1 entries; dimension 0 is contiguous). Each must be a multiple
	// of 16.
	GlobalStrides []uint64
	// BoxDim is the tile a single copy moves, in elements, Rank entries,
	// each 1..256.
	BoxDim []uint32
	// ElementStrides sub-samples each dimension. nil means all 1.
	ElementStrides []uint32

	Interleave  TensorMapInterleave
	Swizzle     TensorMapSwizzle
	L2Promotion TensorMapL2Promotion
	OOBFill     TensorMapOOBFill
}

// TensorMap is an encoded CUtensorMap, kept 64-byte aligned as the driver
// requires. Pass it to a kernel with Arg.
type TensorMap struct {
	backing []byte      // heap-allocated and oversized; keeps d alive
	d       *[16]uint64 // the 64-byte-aligned descriptor inside backing
}

// newTensorMap returns a zeroed, 64-byte-aligned descriptor. Go guarantees
// no alignment above 8 for a []byte, so the slice is oversized and the
// descriptor placed at the first aligned offset inside it. The allocation is
// on the heap, which does not move, so the interior pointer stays valid.
func newTensorMap() *TensorMap {
	b := make([]byte, tensorMapBytes+63)
	off := int((64 - uintptr(unsafe.Pointer(&b[0]))%64) % 64)
	return &TensorMap{backing: b, d: (*[16]uint64)(unsafe.Pointer(&b[off]))}
}

// Arg passes the descriptor to a kernel parameter declared as a
// grid-constant cuda.TensorMap. gocudrv copies the bytes into its argument
// pack, so the TensorMap need not outlive the launch call.
func (m *TensorMap) Arg() cuda.KernelArg {
	return cuda.ArgRaw(unsafe.Pointer(m.d), tensorMapBytes)
}

// Bytes is the encoded descriptor, for copying into a device buffer or a
// __constant__ global instead of passing it as a parameter. The slice aliases
// the TensorMap; it is not a copy.
func (m *TensorMap) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(m.d)), tensorMapBytes)
}

// EncodeTiled encodes a tensor map over b, checking that the element size of
// T matches d.DataType.
func EncodeTiled[T cuda.Supported](ctx *cuda.Context, b *cuda.Buffer[T], d TiledTensor) (*TensorMap, error) {
	if b == nil {
		return nil, fmt.Errorf("host: EncodeTiled: nil buffer")
	}
	var zero T
	if got, want := int(unsafe.Sizeof(zero)), d.DataType.size(); want != 0 && got != want {
		return nil, fmt.Errorf("host: EncodeTiled: buffer element is %d bytes, DataType is %d", got, want)
	}
	return EncodeTiledPtr(ctx, b.DevicePtr(), d)
}

// EncodeTiledPtr encodes a tensor map over a raw device pointer, which must be
// 16-byte aligned.
func EncodeTiledPtr(ctx *cuda.Context, ptr cudasys.CUdeviceptr, d TiledTensor) (*TensorMap, error) {
	if err := loadDriver(); err != nil {
		return nil, err
	}
	if err := d.validate(ptr); err != nil {
		return nil, err
	}

	rank := len(d.GlobalDim)
	elemStrides := d.ElementStrides
	if elemStrides == nil {
		elemStrides = make([]uint32, rank)
		for i := range elemStrides {
			elemStrides[i] = 1
		}
	}
	// A rank-1 tensor has no strides, so there is no element to point at;
	// the driver ignores the argument in that case.
	strides := d.GlobalStrides
	if len(strides) == 0 {
		strides = []uint64{0}
	}

	m := newTensorMap()
	err := OnContext(ctx, func() error {
		return check("cuTensorMapEncodeTiled", cuTensorMapEncodeTiled(
			m.d,
			uint32(d.DataType),
			uint32(rank),
			ptr,
			&d.GlobalDim[0],
			&strides[0],
			&d.BoxDim[0],
			&elemStrides[0],
			uint32(d.Interleave),
			uint32(d.Swizzle),
			uint32(d.L2Promotion),
			uint32(d.OOBFill),
		))
	})
	if err != nil {
		return nil, err
	}
	return m, nil
}

// validate checks the constraints the driver documents but reports only as a
// bare CUDA_ERROR_INVALID_VALUE.
func (d TiledTensor) validate(ptr cudasys.CUdeviceptr) error {
	rank := len(d.GlobalDim)
	switch {
	case rank < 1 || rank > 5:
		return fmt.Errorf("host: EncodeTiled: rank %d, want 1..5", rank)
	case len(d.BoxDim) != rank:
		return fmt.Errorf("host: EncodeTiled: %d BoxDim for rank %d", len(d.BoxDim), rank)
	case len(d.GlobalStrides) != rank-1:
		return fmt.Errorf("host: EncodeTiled: %d GlobalStrides for rank %d, want %d", len(d.GlobalStrides), rank, rank-1)
	case d.ElementStrides != nil && len(d.ElementStrides) != rank:
		return fmt.Errorf("host: EncodeTiled: %d ElementStrides for rank %d", len(d.ElementStrides), rank)
	case uint64(ptr)%16 != 0:
		return fmt.Errorf("host: EncodeTiled: device pointer %#x is not 16-byte aligned", uint64(ptr))
	}
	for i, n := range d.BoxDim {
		if n < 1 || n > 256 {
			return fmt.Errorf("host: EncodeTiled: BoxDim[%d] = %d, want 1..256", i, n)
		}
	}
	for i, s := range d.GlobalStrides {
		if s%16 != 0 {
			return fmt.Errorf("host: EncodeTiled: GlobalStrides[%d] = %d, want a multiple of 16", i, s)
		}
	}
	return nil
}
