package cuda

import "unsafe"

// ---- mbarrier (sm_80+) and bulk asynchronous copies / TMA (sm_90+)
//
// An MBarrier lives in shared memory:
//
//	var bar cuda.Shared[cuda.MBarrier]
//
//	b := bar.Get()
//	if tid == 0 { b.Init(1); cuda.FenceMBarrierInit() }
//	cuda.SyncThreads()
//	if tid == 0 {
//		b.ArriveExpectTx(tileBytes)
//		cuda.CpAsyncBulkG2S(unsafe.Pointer(t), unsafe.Pointer(src), tileBytes, b)
//	}
//	b.WaitParity(0)          // the copy has landed
//	...
//	cuda.FenceProxyAsyncShared()
//	cuda.SyncThreads()
//	if tid == 0 { cuda.CpAsyncBulkS2G(...); cuda.CpAsyncBulkCommit(); cuda.CpAsyncBulkWait0() }

// MBarrier is a 64-bit mbarrier object (cuda::barrier<thread_scope_block>).
type MBarrier uint64

// MBarrierToken is the arrival state returned by Arrive, for TestWait.
type MBarrierToken uint64

//go:linkname mbarInit cudair.mbarrier.init
func mbarInit(p unsafe.Pointer, count int32)

//go:linkname mbarInval cudair.mbarrier.inval
func mbarInval(p unsafe.Pointer)

//go:linkname mbarArrive cudair.mbarrier.arrive
func mbarArrive(p unsafe.Pointer) MBarrierToken

//go:linkname mbarArriveDrop cudair.mbarrier.arrive.drop
func mbarArriveDrop(p unsafe.Pointer) MBarrierToken

//go:linkname mbarArriveExpectTx cudair.mbarrier.arrive.expect_tx
func mbarArriveExpectTx(p unsafe.Pointer, bytes int32) MBarrierToken

//go:linkname mbarExpectTx cudair.mbarrier.expect_tx
func mbarExpectTx(p unsafe.Pointer, bytes int32)

//go:linkname mbarTestWait cudair.mbarrier.test.wait
func mbarTestWait(p unsafe.Pointer, tok MBarrierToken) bool

//go:linkname mbarTryWaitParity cudair.mbarrier.try_wait.parity
func mbarTryWaitParity(p unsafe.Pointer, parity int32) bool

//go:linkname mbarPendingCount llvm.nvvm.mbarrier.pending.count
func mbarPendingCount(tok MBarrierToken) int32

// Init is mbarrier.init: expect `count` arrivals per phase. One thread
// initialises; publish with FenceMBarrierInit + SyncThreads before use.
func (b *MBarrier) Init(count int32) { mbarInit(unsafe.Pointer(b), count) }

// Inval is mbarrier.inval, before the memory is reused for something else.
func (b *MBarrier) Inval() { mbarInval(unsafe.Pointer(b)) }

// Arrive is mbarrier.arrive: count one arrival on the current phase.
func (b *MBarrier) Arrive() MBarrierToken { return mbarArrive(unsafe.Pointer(b)) }

// ArriveDrop is mbarrier.arrive_drop: arrive and lower the expected count
// of later phases by one.
func (b *MBarrier) ArriveDrop() MBarrierToken { return mbarArriveDrop(unsafe.Pointer(b)) }

// ArriveExpectTx is mbarrier.arrive.expect_tx (sm_90+): arrive and expect
// `bytes` of asynchronous transactions (a CpAsyncBulkG2S) to complete the
// phase as well.
func (b *MBarrier) ArriveExpectTx(bytes int32) MBarrierToken {
	return mbarArriveExpectTx(unsafe.Pointer(b), bytes)
}

// ExpectTx is mbarrier.expect_tx (sm_90+): add bytes to the transaction
// count without arriving.
func (b *MBarrier) ExpectTx(bytes int32) { mbarExpectTx(unsafe.Pointer(b), bytes) }

// TestWait is mbarrier.test_wait: has the phase of tok completed?
func (b *MBarrier) TestWait(tok MBarrierToken) bool { return mbarTestWait(unsafe.Pointer(b), tok) }

// Wait spins with TestWait until the phase of tok completes.
func (b *MBarrier) Wait(tok MBarrierToken) {
	for !mbarTestWait(unsafe.Pointer(b), tok) {
	}
}

// TryWaitParity is mbarrier.try_wait.parity (sm_90+): has the phase with
// the given parity (0 for the first phase, then alternating) completed?
func (b *MBarrier) TryWaitParity(parity int32) bool {
	return mbarTryWaitParity(unsafe.Pointer(b), parity)
}

// WaitParity spins with TryWaitParity until that phase completes.
func (b *MBarrier) WaitParity(parity int32) {
	for !mbarTryWaitParity(unsafe.Pointer(b), parity) {
	}
}

// PendingCount is mbarrier.pending_count of a token: arrivals still
// missing in that phase.
func PendingCount(tok MBarrierToken) int32 { return mbarPendingCount(tok) }

//go:linkname fenceMBarrierInit cudair.fence.mbarrier_init
func fenceMBarrierInit()

//go:linkname fenceProxyAsyncShared cudair.fence.proxy.async.shared
func fenceProxyAsyncShared()

// FenceMBarrierInit is fence.mbarrier_init.release.cluster: publishes an
// MBarrier.Init to the asynchronous proxy (and the cluster).
func FenceMBarrierInit() { fenceMBarrierInit() }

// FenceProxyAsyncShared is fence.proxy.async.shared::cta: orders ordinary
// shared-memory accesses against the async proxy (TMA) — issue it after
// writing a tile that CpAsyncBulkS2G will read.
func FenceProxyAsyncShared() { fenceProxyAsyncShared() }

//go:linkname cpAsyncBulkG2S cudair.cp.async.bulk.g2s
func cpAsyncBulkG2S(dst, src unsafe.Pointer, bytes int32, bar unsafe.Pointer)

//go:linkname cpAsyncBulkS2G cudair.cp.async.bulk.s2g
func cpAsyncBulkS2G(dst, src unsafe.Pointer, bytes int32)

// CpAsyncBulkG2S is cp.async.bulk.shared::cluster.global (TMA, sm_90+):
// copy `bytes` (a multiple of 16) from global src to shared dst, both
// 16-byte aligned, completing on bar's transaction count (see
// MBarrier.ArriveExpectTx). Issued by one thread.
func CpAsyncBulkG2S(dst, src unsafe.Pointer, bytes int32, bar *MBarrier) {
	cpAsyncBulkG2S(dst, src, bytes, unsafe.Pointer(bar))
}

// CpAsyncBulkS2G is cp.async.bulk.global.shared::cta: copy `bytes` from
// shared src to global dst, tracked by bulk groups (CpAsyncBulkCommit +
// CpAsyncBulkWait0).
func CpAsyncBulkS2G(dst, src unsafe.Pointer, bytes int32) { cpAsyncBulkS2G(dst, src, bytes) }

// CpAsyncBulkCommit is cp.async.bulk.commit_group.
//
//go:linkname CpAsyncBulkCommit llvm.nvvm.cp.async.bulk.commit.group
func CpAsyncBulkCommit()

//go:linkname cpAsyncBulkWaitGroup llvm.nvvm.cp.async.bulk.wait.group
func cpAsyncBulkWaitGroup(n int32)

//go:linkname cpAsyncBulkWaitGroupRead llvm.nvvm.cp.async.bulk.wait.group.read
func cpAsyncBulkWaitGroupRead(n int32)

// CpAsyncBulkWait0 / Wait1 are cp.async.bulk.wait_group 0 / 1: wait until
// at most that many bulk groups are pending. The Read variants only wait
// until the source may be overwritten.
func CpAsyncBulkWait0()     { cpAsyncBulkWaitGroup(0) }
func CpAsyncBulkWait1()     { cpAsyncBulkWaitGroup(1) }
func CpAsyncBulkWaitRead0() { cpAsyncBulkWaitGroupRead(0) }
func CpAsyncBulkWaitRead1() { cpAsyncBulkWaitGroupRead(1) }
