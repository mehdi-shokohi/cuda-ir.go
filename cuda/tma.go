package cuda

import "unsafe"

// ---- TMA tensor copies (cp.async.bulk.tensor, sm_90+)
//
// A plain CpAsyncBulkG2S copies a flat run of bytes. A tensor copy instead
// moves a rectangular tile out of a multi-dimensional tensor, with the shape,
// the strides and the swizzling all held in a descriptor the host encodes
// once (host.EncodeTiled) — so the kernel names a tile by its coordinates and
// the hardware walks the tensor, clamping at the edges.
//
//	var tile cuda.Shared[[32 * 32]float32]
//	var bar cuda.Shared[cuda.MBarrier]
//
//	func K(m cuda.Buf[cuda.TensorMap], out cuda.Buf[float32]) {
//		t, b := tile.Get(), bar.Get()
//		if cuda.ThreadIdxX() == 0 {
//			b.Init(1)
//			cuda.FenceMBarrierInit()
//			b.ArriveExpectTx(cuda.Bytes[float32](32 * 32))
//			cuda.CpAsyncBulkTensor2D(&t[0], m.Ptr(0), x0, y0, b)
//		}
//		cuda.SyncThreads()
//		b.Wait(...)
//
// The descriptor must live in constant, grid-constant parameter or global
// memory. Go cannot give a struct the 64-byte alignment the hardware wants
// from a parameter, so keep it in global memory: a Buf[TensorMap] the host
// fills with host.TensorMap.Bytes(), whose cuMemAlloc'd address is aligned
// well past 64.

// TensorMap is the 128-byte TMA descriptor (CUtensorMap) that
// host.EncodeTiled produces. A kernel only ever takes its address, never
// reads it directly.
type TensorMap struct {
	_ [16]uint64
}

//go:linkname cpAsyncBulkTensorG2S1D cudair.cp.async.bulk.tensor.g2s.1d
func cpAsyncBulkTensorG2S1D(dst, m unsafe.Pointer, c0 int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkTensorG2S2D cudair.cp.async.bulk.tensor.g2s.2d
func cpAsyncBulkTensorG2S2D(dst, m unsafe.Pointer, c0, c1 int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkTensorG2S3D cudair.cp.async.bulk.tensor.g2s.3d
func cpAsyncBulkTensorG2S3D(dst, m unsafe.Pointer, c0, c1, c2 int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkTensorG2S4D cudair.cp.async.bulk.tensor.g2s.4d
func cpAsyncBulkTensorG2S4D(dst, m unsafe.Pointer, c0, c1, c2, c3 int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkTensorG2S5D cudair.cp.async.bulk.tensor.g2s.5d
func cpAsyncBulkTensorG2S5D(dst, m unsafe.Pointer, c0, c1, c2, c3, c4 int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkTensorS2G1D cudair.cp.async.bulk.tensor.s2g.1d
func cpAsyncBulkTensorS2G1D(src, m unsafe.Pointer, c0 int32)

//go:linkname cpAsyncBulkTensorS2G2D cudair.cp.async.bulk.tensor.s2g.2d
func cpAsyncBulkTensorS2G2D(src, m unsafe.Pointer, c0, c1 int32)

//go:linkname cpAsyncBulkTensorS2G3D cudair.cp.async.bulk.tensor.s2g.3d
func cpAsyncBulkTensorS2G3D(src, m unsafe.Pointer, c0, c1, c2 int32)

//go:linkname cpAsyncBulkTensorS2G4D cudair.cp.async.bulk.tensor.s2g.4d
func cpAsyncBulkTensorS2G4D(src, m unsafe.Pointer, c0, c1, c2, c3 int32)

//go:linkname cpAsyncBulkTensorS2G5D cudair.cp.async.bulk.tensor.s2g.5d
func cpAsyncBulkTensorS2G5D(src, m unsafe.Pointer, c0, c1, c2, c3, c4 int32)

// CpAsyncBulkTensor1D..5D are
// cp.async.bulk.tensor.Nd.shared::cta.global.tile: copy the tile of the
// tensor m describes whose origin is at the given element coordinates into
// shared dst, completing on bar's transaction count. Coordinates are in
// elements and innermost-dimension first, the same order as
// host.TiledTensor's GlobalDim. Issued by one thread, after
// bar.ArriveExpectTx for the tile's byte count.
//
// A tile that hangs off the edge of the tensor is filled per the map's
// OOBFill rather than faulting.
func CpAsyncBulkTensor1D[T any](dst *T, m *TensorMap, c0 int32, bar *MBarrier) {
	cpAsyncBulkTensorG2S1D(unsafe.Pointer(dst), unsafe.Pointer(m), c0, unsafe.Pointer(bar))
}

func CpAsyncBulkTensor2D[T any](dst *T, m *TensorMap, c0, c1 int32, bar *MBarrier) {
	cpAsyncBulkTensorG2S2D(unsafe.Pointer(dst), unsafe.Pointer(m), c0, c1, unsafe.Pointer(bar))
}

func CpAsyncBulkTensor3D[T any](dst *T, m *TensorMap, c0, c1, c2 int32, bar *MBarrier) {
	cpAsyncBulkTensorG2S3D(unsafe.Pointer(dst), unsafe.Pointer(m), c0, c1, c2, unsafe.Pointer(bar))
}

func CpAsyncBulkTensor4D[T any](dst *T, m *TensorMap, c0, c1, c2, c3 int32, bar *MBarrier) {
	cpAsyncBulkTensorG2S4D(unsafe.Pointer(dst), unsafe.Pointer(m), c0, c1, c2, c3, unsafe.Pointer(bar))
}

func CpAsyncBulkTensor5D[T any](dst *T, m *TensorMap, c0, c1, c2, c3, c4 int32, bar *MBarrier) {
	cpAsyncBulkTensorG2S5D(unsafe.Pointer(dst), unsafe.Pointer(m), c0, c1, c2, c3, c4, unsafe.Pointer(bar))
}

// CpAsyncBulkTensorS2G1D..5D are
// cp.async.bulk.tensor.Nd.global.shared::cta.tile: write shared src back as
// the tile of m at the given coordinates. Tracked by bulk groups, so follow
// with CpAsyncBulkCommit and CpAsyncBulkWait0 (and precede with
// FenceProxyAsyncShared, since the tile was written by ordinary stores).
func CpAsyncBulkTensorS2G1D[T any](src *T, m *TensorMap, c0 int32) {
	cpAsyncBulkTensorS2G1D(unsafe.Pointer(src), unsafe.Pointer(m), c0)
}

func CpAsyncBulkTensorS2G2D[T any](src *T, m *TensorMap, c0, c1 int32) {
	cpAsyncBulkTensorS2G2D(unsafe.Pointer(src), unsafe.Pointer(m), c0, c1)
}

func CpAsyncBulkTensorS2G3D[T any](src *T, m *TensorMap, c0, c1, c2 int32) {
	cpAsyncBulkTensorS2G3D(unsafe.Pointer(src), unsafe.Pointer(m), c0, c1, c2)
}

func CpAsyncBulkTensorS2G4D[T any](src *T, m *TensorMap, c0, c1, c2, c3 int32) {
	cpAsyncBulkTensorS2G4D(unsafe.Pointer(src), unsafe.Pointer(m), c0, c1, c2, c3)
}

func CpAsyncBulkTensorS2G5D[T any](src *T, m *TensorMap, c0, c1, c2, c3, c4 int32) {
	cpAsyncBulkTensorS2G5D(unsafe.Pointer(src), unsafe.Pointer(m), c0, c1, c2, c3, c4)
}
