package cuda

import "sync/atomic"

// GridBarrier is a grid-wide barrier (cooperative groups' this_grid().sync())
// implemented with atomics on global memory: no device runtime is needed.
// Allocate one per launch on the host, zeroed, and pass it to the kernel;
// launch cooperatively (gocudrv: Function.LaunchCooperative) so every block
// is resident — otherwise a block that has not been scheduled can never
// arrive and the kernel deadlocks.
//
//	func K(bar *cuda.GridBarrier, ...) {
//		... phase 1 ...
//		bar.Sync()
//		... phase 2 sees every block's phase-1 writes ...
type GridBarrier struct {
	count      uint32 // blocks arrived in the current generation
	generation uint32
}

// Sync blocks until every block of the grid has called Sync. It is a
// __syncthreads for the block as well.
func (b *GridBarrier) Sync() {
	SyncThreads()
	if ThreadIdxX() == 0 && ThreadIdxY() == 0 && ThreadIdxZ() == 0 {
		ThreadFence() // publish this block's writes device-wide
		blocks := uint32(GridDimX() * GridDimY() * GridDimZ())
		gen := atomic.LoadUint32(&b.generation)
		if atomic.AddUint32(&b.count, 1) == blocks {
			atomic.StoreUint32(&b.count, 0)
			atomic.StoreUint32(&b.generation, gen+1) // release everyone
		} else {
			for atomic.LoadUint32(&b.generation) == gen {
				NanoSleep(64)
			}
		}
		ThreadFence()
	}
	SyncThreads()
}
