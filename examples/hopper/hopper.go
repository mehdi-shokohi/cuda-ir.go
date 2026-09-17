// Package hopper exercises the sm_90 features: thread block clusters with
// distributed shared memory, mbarrier, bulk asynchronous copies (TMA),
// elect.sync and programmatic dependent launch. Build with -sm sm_90
// -ptx 80 (cudair.Options{SM: "sm_90", PTX: "80"}).
package hopper

import (
	"unsafe"

	"github.com/mehdi-shokohi/cuda-ir.go/cuda"
)

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
	const bytes = blockSize * 4
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
		b.ArriveExpectTx(bytes)
		cuda.CpAsyncBulkG2S(unsafe.Pointer(t), unsafe.Pointer(in.Ptr(base)), bytes, b)
	}
	b.WaitParity(0) // the tile has landed
	t[tid] *= 2
	cuda.FenceProxyAsyncShared() // make the generic-proxy writes visible to TMA
	cuda.SyncThreads()
	if tid == 0 {
		cuda.CpAsyncBulkS2G(unsafe.Pointer(out.Ptr(base)), unsafe.Pointer(t), bytes)
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
