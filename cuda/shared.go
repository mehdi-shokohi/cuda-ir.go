package cuda

// Shared declares per-block shared memory (__shared__). Use it as a
// package-level variable; `gocuda build` places it in the shared address
// space. Every thread block gets its own uninitialised copy.
//
//	var tile cuda.Shared[[256]float32]
//
//	func Reduce(in cuda.Buf[float32], ...) {
//		t := tile.Get()          // *[256]float32
//		t[cuda.ThreadIdxX()] = ...
//		cuda.SyncThreads()
//
// Only package-level variables are supported; a local Shared value is a
// normal (per-thread) variable.
type Shared[T any] struct{ v T }

// Get returns a pointer to the block's shared instance.
func (s *Shared[T]) Get() *T { return &s.v }
