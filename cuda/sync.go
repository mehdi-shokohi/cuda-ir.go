package cuda

import "unsafe"

// ---- block barriers

// barrier{.cta}.sync.aligned id[, count] / .arrive / .red.{and,or,popc}
// (LLVM 22 names; the old llvm.nvvm.barrier0 family is auto-upgraded to these)

//go:linkname barrierAll llvm.nvvm.barrier.cta.sync.aligned.all
func barrierAll(id int32)

//go:linkname barrierCount llvm.nvvm.barrier.cta.sync.aligned.count
func barrierCount(id, count int32)

//go:linkname barrierArrive llvm.nvvm.barrier.cta.arrive.aligned.count
func barrierArrive(id, count int32)

//go:linkname barrierAnd llvm.nvvm.barrier.cta.red.and.aligned.all
func barrierAnd(id int32, p bool) bool

//go:linkname barrierOr llvm.nvvm.barrier.cta.red.or.aligned.all
func barrierOr(id int32, p bool) bool

//go:linkname barrierPopc llvm.nvvm.barrier.cta.red.popc.aligned.all
func barrierPopc(id int32, p bool) int32

// SyncThreads is __syncthreads().
func SyncThreads() { barrierAll(0) }

// SyncThreadsAnd is __syncthreads_and(pred).
func SyncThreadsAnd(pred bool) bool { return barrierAnd(0, pred) }

// SyncThreadsOr is __syncthreads_or(pred).
func SyncThreadsOr(pred bool) bool { return barrierOr(0, pred) }

// SyncThreadsCount is __syncthreads_count(pred).
func SyncThreadsCount(pred bool) int32 { return barrierPopc(0, pred) }

// SyncBarrier is `bar.sync id, count`: a named barrier (id 1..15; 0 is
// __syncthreads) that only `count` threads (a multiple of 32) take part in,
// for producer/consumer patterns within a block.
func SyncBarrier(id, count int32) { barrierCount(id, count) }

// ArriveBarrier is `bar.arrive id, count`: signal the named barrier without
// waiting on it (the producer side; consumers use SyncBarrier).
func ArriveBarrier(id, count int32) { barrierArrive(id, count) }

// NanoSleep is __nanosleep(ns) (sm_70+): suspend the thread for about ns.
//
//go:linkname NanoSleep llvm.nvvm.nanosleep
func NanoSleep(ns uint32)

// Trap is __trap(): abort the kernel; the launch fails with an error.
//
//go:linkname Trap llvm.trap
func Trap()

// Breakpoint is __brkpt().
//
//go:linkname Breakpoint llvm.debugtrap
func Breakpoint()

// Assume is __builtin_assume(cond): an optimisation hint; undefined
// behaviour if cond is false.
//
//go:linkname Assume llvm.assume
func Assume(cond bool)

//go:linkname isGlobal llvm.nvvm.isspacep.global
func isGlobal(p unsafe.Pointer) bool

//go:linkname isShared llvm.nvvm.isspacep.shared
func isShared(p unsafe.Pointer) bool

//go:linkname isConstant llvm.nvvm.isspacep.const
func isConstant(p unsafe.Pointer) bool

//go:linkname isLocal llvm.nvvm.isspacep.local
func isLocal(p unsafe.Pointer) bool

// IsGlobal/IsShared/IsConstant/IsLocal are __isGlobal & co: which memory
// space a pointer points into.
func IsGlobal[T any](p *T) bool   { return isGlobal(unsafe.Pointer(p)) }
func IsShared[T any](p *T) bool   { return isShared(unsafe.Pointer(p)) }
func IsConstant[T any](p *T) bool { return isConstant(unsafe.Pointer(p)) }
func IsLocal[T any](p *T) bool    { return isLocal(unsafe.Pointer(p)) }

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

// shflCWidth is the c operand for a sub-warp segment of `width` lanes
// (CUDA's `width` argument, a power of two <= 32).
func shflCWidth(width int32, c int32) int32 { return ((32 - width) << 8) | c }

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

// ---- shuffles with a segment width (CUDA's `width` argument)

// ShflWidth is __shfl_sync(mask, v, srcLane, width).
func ShflWidth(mask uint32, v int32, srcLane, width int32) int32 {
	return shflIdxI32(mask, v, srcLane, shflCWidth(width, shflC))
}

// ShflUpWidth is __shfl_up_sync(mask, v, delta, width).
func ShflUpWidth(mask uint32, v int32, delta, width int32) int32 {
	return shflUpI32(mask, v, delta, shflCWidth(width, shflUpC))
}

// ShflDownWidth is __shfl_down_sync(mask, v, delta, width).
func ShflDownWidth(mask uint32, v int32, delta, width int32) int32 {
	return shflDownI32(mask, v, delta, shflCWidth(width, shflC))
}

// ShflXorWidth is __shfl_xor_sync(mask, v, laneMask, width).
func ShflXorWidth(mask uint32, v int32, laneMask, width int32) int32 {
	return shflBflyI32(mask, v, laneMask, shflCWidth(width, shflC))
}

// ---- 64-bit shuffles: two 32-bit shuffles, as CUDA's headers do

func split64(v int64) (lo, hi int32) { return int32(uint32(v)), int32(uint32(uint64(v) >> 32)) }
func join64(lo, hi int32) int64      { return int64(uint64(uint32(hi))<<32 | uint64(uint32(lo))) }

// Shfl64 is __shfl_sync for 64-bit values.
func Shfl64(mask uint32, v int64, srcLane int32) int64 {
	lo, hi := split64(v)
	return join64(Shfl(mask, lo, srcLane), Shfl(mask, hi, srcLane))
}

// ShflUp64 is __shfl_up_sync for 64-bit values.
func ShflUp64(mask uint32, v int64, delta int32) int64 {
	lo, hi := split64(v)
	return join64(ShflUp(mask, lo, delta), ShflUp(mask, hi, delta))
}

// ShflDown64 is __shfl_down_sync for 64-bit values.
func ShflDown64(mask uint32, v int64, delta int32) int64 {
	lo, hi := split64(v)
	return join64(ShflDown(mask, lo, delta), ShflDown(mask, hi, delta))
}

// ShflXor64 is __shfl_xor_sync for 64-bit values.
func ShflXor64(mask uint32, v int64, laneMask int32) int64 {
	lo, hi := split64(v)
	return join64(ShflXor(mask, lo, laneMask), ShflXor(mask, hi, laneMask))
}

// ShflF64 & co are the float64 variants.
func ShflF64(mask uint32, v float64, srcLane int32) float64 {
	return Float64FromBits(uint64(Shfl64(mask, int64(Float64Bits(v)), srcLane)))
}
func ShflUpF64(mask uint32, v float64, delta int32) float64 {
	return Float64FromBits(uint64(ShflUp64(mask, int64(Float64Bits(v)), delta)))
}
func ShflDownF64(mask uint32, v float64, delta int32) float64 {
	return Float64FromBits(uint64(ShflDown64(mask, int64(Float64Bits(v)), delta)))
}
func ShflXorF64(mask uint32, v float64, laneMask int32) float64 {
	return Float64FromBits(uint64(ShflXor64(mask, int64(Float64Bits(v)), laneMask)))
}

// ---- match / reduce (sm_70+ / sm_80+)

//go:linkname matchAnyI32 llvm.nvvm.match.any.sync.i32
func matchAnyI32(mask uint32, v int32) uint32

//go:linkname matchAnyI64 llvm.nvvm.match.any.sync.i64
func matchAnyI64(mask uint32, v int64) uint32

//go:linkname matchAllI32 llvm.nvvm.match.all.sync.i32p
func matchAllI32(mask uint32, v int32) (uint32, bool)

//go:linkname matchAllI64 llvm.nvvm.match.all.sync.i64p
func matchAllI64(mask uint32, v int64) (uint32, bool)

// MatchAny is __match_any_sync (sm_70+): the mask of lanes in `mask` that
// hold the same value of v as this lane.
func MatchAny(mask uint32, v int32) uint32 { return matchAnyI32(mask, v) }

// MatchAny64 is __match_any_sync for 64-bit values.
func MatchAny64(mask uint32, v int64) uint32 { return matchAnyI64(mask, v) }

// MatchAll is __match_all_sync (sm_70+): mask if all lanes in `mask` hold
// the same v (and pred = true), else 0 and pred = false.
func MatchAll(mask uint32, v int32) (m uint32, pred bool) { return matchAllI32(mask, v) }

// MatchAll64 is __match_all_sync for 64-bit values.
func MatchAll64(mask uint32, v int64) (m uint32, pred bool) { return matchAllI64(mask, v) }

//go:linkname reduxAdd llvm.nvvm.redux.sync.add
func reduxAdd(v int32, mask uint32) int32

//go:linkname reduxMin llvm.nvvm.redux.sync.min
func reduxMin(v int32, mask uint32) int32

//go:linkname reduxMax llvm.nvvm.redux.sync.max
func reduxMax(v int32, mask uint32) int32

//go:linkname reduxUMin llvm.nvvm.redux.sync.umin
func reduxUMin(v uint32, mask uint32) uint32

//go:linkname reduxUMax llvm.nvvm.redux.sync.umax
func reduxUMax(v uint32, mask uint32) uint32

//go:linkname reduxAnd llvm.nvvm.redux.sync.and
func reduxAnd(v uint32, mask uint32) uint32

//go:linkname reduxOr llvm.nvvm.redux.sync.or
func reduxOr(v uint32, mask uint32) uint32

//go:linkname reduxXor llvm.nvvm.redux.sync.xor
func reduxXor(v uint32, mask uint32) uint32

// ReduceAdd is __reduce_add_sync (sm_80+): the sum of v over the lanes in
// mask, returned to every one of them. Min/Max/MinU/MaxU/And/Or/Xor likewise.
func ReduceAdd(mask uint32, v int32) int32    { return reduxAdd(v, mask) }
func ReduceMin(mask uint32, v int32) int32    { return reduxMin(v, mask) }
func ReduceMax(mask uint32, v int32) int32    { return reduxMax(v, mask) }
func ReduceMinU(mask uint32, v uint32) uint32 { return reduxUMin(v, mask) }
func ReduceMaxU(mask uint32, v uint32) uint32 { return reduxUMax(v, mask) }
func ReduceAnd(mask uint32, v uint32) uint32  { return reduxAnd(v, mask) }
func ReduceOr(mask uint32, v uint32) uint32   { return reduxOr(v, mask) }
func ReduceXor(mask uint32, v uint32) uint32  { return reduxXor(v, mask) }

// ---- elect.sync (sm_90+), float reductions (sm_100a+)

//go:linkname electSync llvm.nvvm.elect.sync
func electSync(mask uint32) (uint32, bool)

// ElectSync is elect.sync (sm_90+): picks one leader among the lanes in
// mask; `elected` is true in that lane only, `leader` is its lane id.
func ElectSync(mask uint32) (leader int32, elected bool) {
	l, e := electSync(mask)
	return int32(l), e
}

//go:linkname reduxFMin llvm.nvvm.redux.sync.fmin
func reduxFMin(v float32, mask uint32) float32

//go:linkname reduxFMax llvm.nvvm.redux.sync.fmax
func reduxFMax(v float32, mask uint32) float32

// ReduceMinF32 / ReduceMaxF32 are __reduce_min_sync / __reduce_max_sync on
// floats (redux.sync.min.f32, sm_100a+; build with -sm sm_100a).
func ReduceMinF32(mask uint32, v float32) float32 { return reduxFMin(v, mask) }
func ReduceMaxF32(mask uint32, v float32) float32 { return reduxFMax(v, mask) }
