package cuda

import _ "unsafe" // for go:linkname

// ---- block barriers

// SyncThreads is __syncthreads().
//
//go:linkname SyncThreads llvm.nvvm.barrier0
func SyncThreads()

//go:linkname barrierAnd llvm.nvvm.barrier0.and
func barrierAnd(p int32) int32

//go:linkname barrierOr llvm.nvvm.barrier0.or
func barrierOr(p int32) int32

//go:linkname barrierPopc llvm.nvvm.barrier0.popc
func barrierPopc(p int32) int32

func b2i(p bool) int32 {
	if p {
		return 1
	}
	return 0
}

// SyncThreadsAnd is __syncthreads_and(pred).
func SyncThreadsAnd(pred bool) bool { return barrierAnd(b2i(pred)) != 0 }

// SyncThreadsOr is __syncthreads_or(pred).
func SyncThreadsOr(pred bool) bool { return barrierOr(b2i(pred)) != 0 }

// SyncThreadsCount is __syncthreads_count(pred).
func SyncThreadsCount(pred bool) int32 { return barrierPopc(b2i(pred)) }

// SyncWarp is __syncwarp(mask).
//
//go:linkname SyncWarp llvm.nvvm.bar.warp.sync
func SyncWarp(mask uint32)

// ---- memory fences

// ThreadFenceBlock is __threadfence_block().
//
//go:linkname ThreadFenceBlock llvm.nvvm.membar.cta
func ThreadFenceBlock()

// ThreadFence is __threadfence() (device scope).
//
//go:linkname ThreadFence llvm.nvvm.membar.gl
func ThreadFence()

// ThreadFenceSystem is __threadfence_system().
//
//go:linkname ThreadFenceSystem llvm.nvvm.membar.sys
func ThreadFenceSystem()

// ---- timers

// Clock is the per-SM cycle counter (clock()).
//
//go:linkname Clock llvm.nvvm.read.ptx.sreg.clock
func Clock() int32

// Clock64 is clock64().
//
//go:linkname Clock64 llvm.nvvm.read.ptx.sreg.clock64
func Clock64() int64

// GlobalTimer is the nanosecond global timer (%globaltimer).
//
//go:linkname GlobalTimer llvm.nvvm.read.ptx.sreg.globaltimer
func GlobalTimer() int64

// ---- warp-level primitives (mask selects participating lanes; FullMask = all)

const FullMask uint32 = 0xffffffff

//go:linkname shflIdxI32 llvm.nvvm.shfl.sync.idx.i32
func shflIdxI32(mask uint32, v int32, lane int32, c int32) int32

//go:linkname shflUpI32 llvm.nvvm.shfl.sync.up.i32
func shflUpI32(mask uint32, v int32, delta int32, c int32) int32

//go:linkname shflDownI32 llvm.nvvm.shfl.sync.down.i32
func shflDownI32(mask uint32, v int32, delta int32, c int32) int32

//go:linkname shflBflyI32 llvm.nvvm.shfl.sync.bfly.i32
func shflBflyI32(mask uint32, v int32, lanemask int32, c int32) int32

//go:linkname shflIdxF32 llvm.nvvm.shfl.sync.idx.f32
func shflIdxF32(mask uint32, v float32, lane int32, c int32) float32

//go:linkname shflUpF32 llvm.nvvm.shfl.sync.up.f32
func shflUpF32(mask uint32, v float32, delta int32, c int32) float32

//go:linkname shflDownF32 llvm.nvvm.shfl.sync.down.f32
func shflDownF32(mask uint32, v float32, delta int32, c int32) float32

//go:linkname shflBflyF32 llvm.nvvm.shfl.sync.bfly.f32
func shflBflyF32(mask uint32, v float32, lanemask int32, c int32) float32

// The "c" operand packs clamp (low 5 bits) and segment mask, as CUDA's
// __shfl_* helpers do: up uses 0, the others 0x1f (warp width 32).
const shflUpC, shflC = 0, 0x1f

// Shfl is __shfl_sync: value of v from lane srcLane.
func Shfl(mask uint32, v int32, srcLane int32) int32 { return shflIdxI32(mask, v, srcLane, shflC) }

// ShflUp is __shfl_up_sync.
func ShflUp(mask uint32, v int32, delta int32) int32 { return shflUpI32(mask, v, delta, shflUpC) }

// ShflDown is __shfl_down_sync.
func ShflDown(mask uint32, v int32, delta int32) int32 { return shflDownI32(mask, v, delta, shflC) }

// ShflXor is __shfl_xor_sync.
func ShflXor(mask uint32, v int32, laneMask int32) int32 {
	return shflBflyI32(mask, v, laneMask, shflC)
}

func ShflF32(mask uint32, v float32, srcLane int32) float32 {
	return shflIdxF32(mask, v, srcLane, shflC)
}
func ShflUpF32(mask uint32, v float32, delta int32) float32 {
	return shflUpF32(mask, v, delta, shflUpC)
}
func ShflDownF32(mask uint32, v float32, delta int32) float32 {
	return shflDownF32(mask, v, delta, shflC)
}
func ShflXorF32(mask uint32, v float32, laneMask int32) float32 {
	return shflBflyF32(mask, v, laneMask, shflC)
}

//go:linkname voteAll llvm.nvvm.vote.all.sync
func voteAll(mask uint32, p bool) bool

//go:linkname voteAny llvm.nvvm.vote.any.sync
func voteAny(mask uint32, p bool) bool

//go:linkname voteBallot llvm.nvvm.vote.ballot.sync
func voteBallot(mask uint32, p bool) uint32

//go:linkname activeMask llvm.nvvm.activemask
func activeMask() uint32

// All is __all_sync.
func All(mask uint32, pred bool) bool { return voteAll(mask, pred) }

// Any is __any_sync.
func Any(mask uint32, pred bool) bool { return voteAny(mask, pred) }

// Ballot is __ballot_sync.
func Ballot(mask uint32, pred bool) uint32 { return voteBallot(mask, pred) }

// ActiveMask is __activemask().
func ActiveMask() uint32 { return activeMask() }
