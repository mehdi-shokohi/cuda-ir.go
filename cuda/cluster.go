package cuda

import "unsafe"

// ---- thread block clusters (sm_90+)
//
// A kernel gets a compile-time cluster shape with a doc-comment directive
// (__cluster_dims__):
//
//	//cuda:cluster_dims 2
//	func K(...) { ... }
//
// and is launched with the usual grid (a multiple of the cluster shape).
// The blocks of a cluster are co-scheduled and can read each other's
// shared memory (distributed shared memory) through Shared[T].InCluster.

//go:linkname clusterArrive llvm.nvvm.barrier.cluster.arrive
func clusterArrive()

//go:linkname clusterArriveRelaxed llvm.nvvm.barrier.cluster.arrive.relaxed
func clusterArriveRelaxed()

//go:linkname clusterWait llvm.nvvm.barrier.cluster.wait
func clusterWait()

// ClusterSync is cluster.sync(): every thread of every block in the
// cluster arrives, then waits; shared-memory writes made before it are
// visible cluster-wide after it.
func ClusterSync() {
	clusterArrive()
	clusterWait()
}

// ClusterArrive / ClusterWait are the split barrier (barrier.cluster.arrive
// / barrier.cluster.wait); ClusterArriveRelaxed arrives without the release
// fence.
func ClusterArrive()        { clusterArrive() }
func ClusterArriveRelaxed() { clusterArriveRelaxed() }
func ClusterWait()          { clusterWait() }

// ClusterIDX/Y/Z are %clusterid: the cluster's index in the grid.
//
//go:linkname ClusterIDX llvm.nvvm.read.ptx.sreg.clusterid.x
func ClusterIDX() int32

//go:linkname ClusterIDY llvm.nvvm.read.ptx.sreg.clusterid.y
func ClusterIDY() int32

//go:linkname ClusterIDZ llvm.nvvm.read.ptx.sreg.clusterid.z
func ClusterIDZ() int32

// NumClustersX/Y/Z are %nclusterid: the number of clusters in the grid.
//
//go:linkname NumClustersX llvm.nvvm.read.ptx.sreg.nclusterid.x
func NumClustersX() int32

//go:linkname NumClustersY llvm.nvvm.read.ptx.sreg.nclusterid.y
func NumClustersY() int32

//go:linkname NumClustersZ llvm.nvvm.read.ptx.sreg.nclusterid.z
func NumClustersZ() int32

// ClusterCtaRank is %cluster_ctarank (cluster.block_rank()): this block's
// linear rank within its cluster.
//
//go:linkname ClusterCtaRank llvm.nvvm.read.ptx.sreg.cluster.ctarank
func ClusterCtaRank() int32

// ClusterSize is %cluster_nctarank (cluster.num_blocks()).
//
//go:linkname ClusterSize llvm.nvvm.read.ptx.sreg.cluster.nctarank
func ClusterSize() int32

// ClusterBlockIdxX/Y/Z are %cluster_ctaid: the block's index within the
// cluster; ClusterDimX/Y/Z are %cluster_nctaid, the cluster shape.
//
//go:linkname ClusterBlockIdxX llvm.nvvm.read.ptx.sreg.cluster.ctaid.x
func ClusterBlockIdxX() int32

//go:linkname ClusterBlockIdxY llvm.nvvm.read.ptx.sreg.cluster.ctaid.y
func ClusterBlockIdxY() int32

//go:linkname ClusterBlockIdxZ llvm.nvvm.read.ptx.sreg.cluster.ctaid.z
func ClusterBlockIdxZ() int32

//go:linkname ClusterDimX llvm.nvvm.read.ptx.sreg.cluster.nctaid.x
func ClusterDimX() int32

//go:linkname ClusterDimY llvm.nvvm.read.ptx.sreg.cluster.nctaid.y
func ClusterDimY() int32

//go:linkname ClusterDimZ llvm.nvvm.read.ptx.sreg.cluster.nctaid.z
func ClusterDimZ() int32

//go:linkname mapa llvm.nvvm.mapa
func mapa(p unsafe.Pointer, rank int32) unsafe.Pointer

// MapShared is cluster.map_shared_rank(p, rank): the address of the same
// shared-memory variable in the cluster block with the given rank.
func MapShared[T any](p *T, rank int32) *T { return (*T)(mapa(unsafe.Pointer(p), rank)) }

// InCluster returns the block of rank `rank`'s instance of this shared
// variable (distributed shared memory). Synchronise with ClusterSync
// before reading another block's writes and before exiting, so the memory
// is not released while a peer still accesses it.
func (s *Shared[T]) InCluster(rank int32) *T { return (*T)(mapa(unsafe.Pointer(&s.v), rank)) }

//go:linkname isSharedCluster llvm.nvvm.isspacep.shared.cluster
func isSharedCluster(p unsafe.Pointer) bool

// IsSharedCluster is __isClusterShared(p).
func IsSharedCluster[T any](p *T) bool { return isSharedCluster(unsafe.Pointer(p)) }

//go:linkname fenceAcqRelCluster cudair.fence.acq_rel.cluster
func fenceAcqRelCluster()

// ThreadFenceCluster is a cluster-scope memory fence
// (fence.acq_rel.cluster; __threadfence() covers the whole device).
func ThreadFenceCluster() { fenceAcqRelCluster() }

// FenceSCCluster is fence.sc.cluster.
//
//go:linkname FenceSCCluster llvm.nvvm.fence.sc.cluster
func FenceSCCluster()

// ---- programmatic dependent launch (sm_90+)

// GridDepLaunchDependents is cudaTriggerProgrammaticLaunchCompletion()
// (griddepcontrol.launch_dependents): lets a dependent grid launched with
// the programmatic-stream-serialization attribute start early. A no-op
// without that attribute.
//
//go:linkname GridDepLaunchDependents llvm.nvvm.griddepcontrol.launch.dependents
func GridDepLaunchDependents()

// GridDepWait is cudaGridDependencySynchronize() (griddepcontrol.wait):
// waits until every prerequisite grid has completed and its memory is
// visible.
//
//go:linkname GridDepWait llvm.nvvm.griddepcontrol.wait
func GridDepWait()
