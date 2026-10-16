// Package hopper exercises the sm_90 features: thread block clusters with
// distributed shared memory, mbarrier, bulk asynchronous copies (TMA),
// elect.sync, programmatic dependent launch and the README's TMA-fed
// tensor-core matmul. Build with -sm sm_90 -ptx 80
// (cudair.Options{SM: "sm_90", PTX: "80"}).
package hopper

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

const blockSize = 256

// ---- clusters and distributed shared memory

var tile cuda.Shared[[blockSize]int32]

// ClusterExchange runs in clusters of 2 blocks: each block fills its shared
// tile with block*1000+tid, then reads its partner's tile through
// distributed shared memory: out[i] = partner's value. info[block] =
// rank + 10*size + 100*clusterID + 1000*numClusters.
//
//cuda:cluster_dims 2
func ClusterExchange(out, info cuda.Buf[int32]) {
	t := tile.Get()
	tid := cuda.ThreadIdxX()
	b := cuda.BlockIdxX()
	t[tid] = b*1000 + tid
	cuda.ClusterSync() // publishes the tile cluster-wide
	rank := cuda.ClusterCtaRank()
	peer := tile.InCluster(rank ^ 1)
	out.Set(cuda.GlobalIdX(), peer[tid])
	if tid == 0 {
		info.Set(b, rank+10*cuda.ClusterSize()+100*cuda.ClusterIDX()+1000*cuda.NumClustersX())
	}
	cuda.ClusterSync() // no block leaves while a peer may still read its tile
}

// ---- mbarrier as a block barrier (sm_80+)

var bar cuda.Shared[cuda.MBarrier]
var vals cuda.Shared[[blockSize]int32]

// MBarrierSum: every thread deposits in[i] in shared memory, arrives on the
// mbarrier and waits for the phase; thread 0 then sums the block.
func MBarrierSum(in, out cuda.Buf[int32]) {
	b := bar.Get()
	tid := cuda.ThreadIdxX()
	if tid == 0 {
		b.Init(cuda.BlockDimX())
		cuda.FenceMBarrierInit()
	}
	cuda.SyncThreads()
	v := vals.Get()
	v[tid] = in.At(cuda.GlobalIdX())
	tok := b.Arrive()
	b.Wait(tok)
	if tid == 0 {
		var s int32
		for k := range v {
			s += v[k]
		}
		out.Set(cuda.BlockIdxX(), s)
		b.Inval()
	}
}

// ---- TMA: bulk global -> shared -> global copies

var tmaTile cuda.Shared[[blockSize]int32]
var tmaBar cuda.Shared[cuda.MBarrier]

// TmaDouble copies each block's 1 KiB slice of `in` to shared memory with
// cp.async.bulk (completion on an mbarrier), doubles it, and bulk-copies
// it back to out. n is a multiple of blockSize.
func TmaDouble(in, out cuda.Buf[int32], n int32) {
	b := tmaBar.Get()
	t := tmaTile.Get()
	tid := cuda.ThreadIdxX()
	base := cuda.BlockIdxX() * blockSize
	if base >= n {
		return
	}
	if tid == 0 {
		b.Init(1)
		cuda.FenceMBarrierInit()
	}
	cuda.SyncThreads()
	if tid == 0 {
		b.ArriveExpectTx(cuda.Bytes[int32](blockSize))
		cuda.CpAsyncBulkG2S(&t[0], in.Ptr(base), blockSize, b)
	}
	b.WaitParity(0) // the tile has landed
	t[tid] *= 2
	cuda.FenceProxyAsyncShared() // make the generic-proxy writes visible to TMA
	cuda.SyncThreads()
	if tid == 0 {
		cuda.CpAsyncBulkS2G(out.Ptr(base), &t[0], blockSize)
		cuda.CpAsyncBulkCommit()
		cuda.CpAsyncBulkWait0()
	}
}

// ---- elect.sync

// Elect: leader[i] = the elected lane of the warp, elected[i] = 1 in that
// lane only.
func Elect(leader, elected cuda.Buf[int32]) {
	i := cuda.GlobalIdX()
	l, e := cuda.ElectSync(cuda.FullMask)
	leader.Set(i, l)
	if e {
		elected.Set(i, 1)
	} else {
		elected.Set(i, 0)
	}
}

// ---- programmatic dependent launch

// GridDep waits for prerequisite grids, does its work and signals its
// dependents (no-ops for an ordinary launch).
func GridDep(out cuda.Buf[int32]) {
	cuda.GridDepWait()
	i := cuda.GlobalIdX()
	out.Set(i, i)
	cuda.GridDepLaunchDependents()
	cuda.ThreadFenceCluster()
}

// ---- TMA tensor copies (cp.async.bulk.tensor)

const tensorTile = 32

var tensorTileBuf cuda.Shared[[tensorTile * tensorTile]float32]
var tensorTileBar cuda.Shared[cuda.MBarrier]

// TensorTileDouble moves one 32x32 tile out of a 2D tensor with a TMA tensor
// copy, doubles it in shared memory and writes it back as the tile at the
// same coordinates of a second tensor. The shape, strides and edge clamping
// all come from the two descriptors the host encoded, so the kernel only
// names the tile's origin: block (bx, by) takes the tile at
// (bx*32, by*32), and a tile that hangs off the edge is filled per the
// map's OOBFill.
//
// src and dst are Buf[cuda.TensorMap] rather than parameters because the
// descriptor needs 64-byte alignment, which Go cannot ask of a struct
// parameter but cuMemAlloc gives for free.
func TensorTileDouble(src, dst cuda.Buf[cuda.TensorMap]) {
	t, b := tensorTileBuf.Get(), tensorTileBar.Get()
	tid := cuda.ThreadIdxX()
	x0 := cuda.BlockIdxX() * tensorTile
	y0 := cuda.BlockIdxY() * tensorTile

	if tid == 0 {
		b.Init(1)
		cuda.FenceMBarrierInit()
	}
	cuda.SyncThreads()
	if tid == 0 {
		token := b.ArriveExpectTx(cuda.Bytes[float32](tensorTile * tensorTile))
		cuda.CpAsyncBulkTensor2D(&t[0], src.Ptr(0), x0, y0, b)
		b.Wait(token)
	}
	cuda.SyncThreads()

	for i := tid; i < tensorTile*tensorTile; i += cuda.BlockDimX() {
		t[i] *= 2
	}
	cuda.SyncThreads()

	if tid == 0 {
		cuda.FenceProxyAsyncShared() // the stores above, seen by the async proxy
		cuda.CpAsyncBulkTensorS2G2D(&t[0], dst.Ptr(0), x0, y0)
		cuda.CpAsyncBulkCommit()
		cuda.CpAsyncBulkWait0()
	}
}

// ClusterRankRuntime carries no //cuda:cluster_dims, so its PTX has no
// .reqnctapercluster and the cluster shape can only come from the launch
// attribute (host.LaunchEx with LaunchAttrs.ClusterDim). Block b writes
// rank + 100*size, so both the shape and the ranks within it are visible.
func ClusterRankRuntime(out cuda.Buf[int32]) {
	if cuda.ThreadIdxX() != 0 {
		return
	}
	out.Set(cuda.BlockIdxX(), cuda.ClusterCtaRank()+100*cuda.ClusterSize())
}

// ---- the README sample: TMA-fed tensor-core matmul

var mmBar cuda.Shared[cuda.MBarrier]       // __shared__ cuda::barrier
var mmTile cuda.Shared[[16 * 16]cuda.Half] // __shared__ __half tile[256]

// MatMul16: C = A * B + C for n x n row-major matrices, one 32-thread block
// (a warp) per 16x16 output tile, grid (n/16, n/16). Each A tile is streamed
// into shared memory with TMA (cp.async.bulk) completing on an mbarrier,
// then multiplied on the tensor cores with wmma. Build with -sm sm_90.
//
//cuda:restrict
//cuda:launch_bounds 32
func MatMul16(a, b cuda.Buf[cuda.Half], c cuda.Buf[float32], n int32) {
	row, col := cuda.BlockIdxY(), cuda.BlockIdxX()
	mb, t := mmBar.Get(), mmTile.Get()
	if cuda.ThreadIdxX() == 0 {
		mb.Init(1)
		cuda.FenceMBarrierInit()
	}
	cuda.SyncThreads()

	acc := cuda.WmmaLoadC(c.Ptr(row*16*n+col*16), n) // wmma::load_matrix_sync(c_frag, ...)
	for k, phase := int32(0), int32(0); k < n; k, phase = k+16, phase^1 {
		if cuda.ThreadIdxX() == 0 { // one thread issues the tile's bulk copy, row by row
			mb.ArriveExpectTx(cuda.Bytes[cuda.Half](16 * 16))
			for r := int32(0); r < 16; r++ {
				cuda.CpAsyncBulkG2S(&t[r*16], a.Ptr((row*16+r)*n+k), 16, mb)
			}
		}
		mb.WaitParity(phase) // the tile has landed
		fa := cuda.WmmaLoadA(&t[0], 16)
		fb := cuda.WmmaLoadB(b.Ptr(k*n+col*16), n)
		acc = cuda.WmmaMma(fa, fb, acc) // wmma::mma_sync
		cuda.FenceProxyAsyncShared()    // our reads of the tile are ordered before the next TMA write
		cuda.SyncThreads()
	}
	cuda.WmmaStore(c.Ptr(row*16*n+col*16), n, acc) // wmma::store_matrix_sync
}

// ---- stmatrix (sm_90+)

var stTile cuda.Shared[[4 * 64]cuda.Half] // four 8x8 matrices, row-major

// StMatrixRoundTrip reads four 8x8 half matrices out of in with ldmatrix and
// writes them back to shared memory with stmatrix, which must be its exact
// inverse: out[0:256] is the tile after a plain x4 round trip (equal to in)
// and out[256:512] after x4.trans (each 8x8 matrix transposed). x2.trans and
// x1.trans do the same for the first two / the first matrix, into out[512:]
// and out[640:]. One warp.
func StMatrixRoundTrip(in, out cuda.Buf[cuda.Half]) {
	t := stTile.Get()
	lane := cuda.LaneID()
	for i := lane; i < 4*64; i += 32 {
		t[i] = in.At(i)
	}
	cuda.SyncWarp(cuda.FullMask)

	row := &t[lane*8]
	v4 := cuda.LdMatrix4(row)
	v2 := cuda.LdMatrix2(row)
	v1 := cuda.LdMatrix1(row)
	cuda.SyncWarp(cuda.FullMask)

	cuda.StMatrix4(row, v4) // the same layout back in: the tile is unchanged
	cuda.SyncWarp(cuda.FullMask)
	for i := lane; i < 4*64; i += 32 {
		out.Set(i, t[i])
	}

	cuda.StMatrix4Trans(row, v4) // each 8x8 matrix transposed in place
	cuda.SyncWarp(cuda.FullMask)
	for i := lane; i < 4*64; i += 32 {
		out.Set(4*64+i, t[i])
	}

	cuda.StMatrix2Trans(row, v2) // only the first two matrices are written
	cuda.SyncWarp(cuda.FullMask)
	for i := lane; i < 2*64; i += 32 {
		out.Set(8*64+i, t[i])
	}

	cuda.StMatrix1Trans(row, v1) // only the first
	cuda.SyncWarp(cuda.FullMask)
	for i := lane; i < 64; i += 32 {
		out.Set(10*64+i, t[i])
	}
}
