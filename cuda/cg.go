package cuda

// ---- cooperative groups sugar: thread_block, tiled_partition<N>,
// coalesced_threads(). Plain Go over the warp primitives — no new IR.

// Block is cooperative_groups::thread_block (this_thread_block()).
type Block struct{}

// ThisBlock returns the current thread block.
func ThisBlock() Block { return Block{} }

// Sync is block.sync() (__syncthreads).
func (Block) Sync() { SyncThreads() }

// ThreadRank is block.thread_rank(): the linear thread index in the block.
func (Block) ThreadRank() int32 {
	return (ThreadIdxZ()*BlockDimY()+ThreadIdxY())*BlockDimX() + ThreadIdxX()
}

// Size is block.size() (num_threads).
func (Block) Size() int32 { return BlockDimX() * BlockDimY() * BlockDimZ() }

// GroupIndex is block.group_index(): the block's linear index in the grid.
func (Block) GroupIndex() int32 {
	return (BlockIdxZ()*GridDimY()+BlockIdxY())*GridDimX() + BlockIdxX()
}

// Tile is a tiled_partition<N>(block) of a warp (N a power of two <= 32)
// or a coalesced_group: `mask` names the participating lanes and `width`
// the tile size.
type Tile struct {
	mask  uint32
	width int32
}

// ThisWarp is tiled_partition<32>: the whole warp.
func ThisWarp() Tile { return Tile{FullMask, 32} }

// TiledPartition is tiled_partition<width>(this_thread_block()) restricted
// to a warp: consecutive groups of `width` lanes (1, 2, 4, 8, 16 or 32).
func TiledPartition(width int32) Tile {
	base := LaneID() &^ (width - 1)
	var m uint32 = 0xffffffff
	if width < 32 {
		m = (uint32(1)<<uint32(width) - 1) << uint32(base)
	}
	return Tile{m, width}
}

// CoalescedThreads is coalesced_threads(): the currently active lanes.
func CoalescedThreads() Tile { return Tile{ActiveMask(), 0} }

// Mask is the lane mask the tile passes to the *_sync primitives.
func (t Tile) Mask() uint32 { return t.mask }

// Size is the number of threads in the tile (num_threads()).
func (t Tile) Size() int32 {
	if t.width > 0 {
		return t.width
	}
	return Popc(t.mask)
}

// ThreadRank is thread_rank(): this thread's index within the tile.
func (t Tile) ThreadRank() int32 {
	if t.width > 0 {
		return LaneID() & (t.width - 1)
	}
	return Popc(t.mask & LaneMaskLt())
}

// MetaGroupRank is meta_group_rank(): the tile's index within the warp.
func (t Tile) MetaGroupRank() int32 {
	if t.width > 0 {
		return LaneID() / t.width
	}
	return 0
}

// Sync is tile.sync() (__syncwarp on the tile's lanes).
func (t Tile) Sync() { SyncWarp(t.mask) }

// laneOf maps a tile-relative rank to a warp lane.
func (t Tile) laneOf(rank int32) int32 {
	if t.width > 0 {
		return (LaneID() &^ (t.width - 1)) + rank
	}
	// coalesced group: the rank-th set bit of the mask
	m := t.mask
	for ; rank > 0; rank-- {
		m &= m - 1
	}
	return Ffs(int32(m)) - 1
}

// Shfl is tile.shfl(v, rank): v from the thread with that rank.
func (t Tile) Shfl(v int32, rank int32) int32 { return Shfl(t.mask, v, t.laneOf(rank)) }

// ShflF32 is tile.shfl for float32.
func (t Tile) ShflF32(v float32, rank int32) float32 {
	return ShflF32(t.mask, v, t.laneOf(rank))
}

// ShflUp / ShflDown / ShflXor are the tile-width shuffles (for a
// coalesced group they operate on the whole warp, as in CUDA).
func (t Tile) ShflUp(v int32, delta int32) int32 {
	if t.width > 0 {
		return ShflUpWidth(t.mask, v, delta, t.width)
	}
	return ShflUp(t.mask, v, delta)
}

func (t Tile) ShflDown(v int32, delta int32) int32 {
	if t.width > 0 {
		return ShflDownWidth(t.mask, v, delta, t.width)
	}
	return ShflDown(t.mask, v, delta)
}

func (t Tile) ShflXor(v int32, laneMask int32) int32 {
	if t.width > 0 {
		return ShflXorWidth(t.mask, v, laneMask, t.width)
	}
	return ShflXor(t.mask, v, laneMask)
}

func (t Tile) ShflDownF32(v float32, delta int32) float32 {
	return Float32FromBits(uint32(t.ShflDown(int32(Float32Bits(v)), delta)))
}

func (t Tile) ShflXorF32(v float32, laneMask int32) float32 {
	return Float32FromBits(uint32(t.ShflXor(int32(Float32Bits(v)), laneMask)))
}

// All / Any / Ballot are tile.all/any/ballot(pred). Ballot's bits are
// warp lanes for a coalesced group and tile ranks for a tiled partition.
func (t Tile) All(pred bool) bool { return All(t.mask, pred) }
func (t Tile) Any(pred bool) bool { return Any(t.mask, pred) }
func (t Tile) Ballot(pred bool) uint32 {
	b := Ballot(t.mask, pred)
	if t.width > 0 && t.width < 32 {
		return (b >> uint32(LaneID()&^(t.width-1))) & (uint32(1)<<uint32(t.width) - 1)
	}
	return b
}

const (
	opAdd = iota
	opMax
	opMin
)

func combine(op int, a, b int32) int32 {
	switch op {
	case opMax:
		if b > a {
			return b
		}
		return a
	case opMin:
		if b < a {
			return b
		}
		return a
	}
	return a + b
}

// reduce combines v over the tile into every thread. A tiled partition is
// a power of two: a butterfly. A coalesced group has any size: a
// rank-based tree into rank 0, then a broadcast (as cg::reduce does).
func (t Tile) reduce(op int, v int32) int32 {
	if t.width > 0 {
		for d := t.width / 2; d > 0; d /= 2 {
			v = combine(op, v, t.ShflXor(v, d))
		}
		return v
	}
	n, r := t.Size(), t.ThreadRank()
	for d := int32(16); d > 0; d /= 2 { // 32 >= n, so start at 16
		o := t.Shfl(v, r+d) // every lane shuffles; only in-range results count
		if r+d < n && r < d {
			v = combine(op, v, o)
		}
	}
	return t.Shfl(v, 0)
}

// ReduceAdd is cg::reduce(tile, v, cg::plus<int>()): the sum over the tile
// in every thread; ReduceMax / ReduceMin / ReduceAddF32 likewise. A full
// warp uses redux.sync (sm_80+).
func (t Tile) ReduceAdd(v int32) int32 {
	if t.width == 32 {
		return ReduceAdd(t.mask, v)
	}
	return t.reduce(opAdd, v)
}

func (t Tile) ReduceMax(v int32) int32 {
	if t.width == 32 {
		return ReduceMax(t.mask, v)
	}
	return t.reduce(opMax, v)
}

func (t Tile) ReduceMin(v int32) int32 {
	if t.width == 32 {
		return ReduceMin(t.mask, v)
	}
	return t.reduce(opMin, v)
}

func (t Tile) ReduceAddF32(v float32) float32 {
	if t.width > 0 {
		for d := t.width / 2; d > 0; d /= 2 {
			v += t.ShflXorF32(v, d)
		}
		return v
	}
	n, r := t.Size(), t.ThreadRank()
	for d := int32(16); d > 0; d /= 2 {
		o := t.ShflF32(v, r+d)
		if r+d < n && r < d {
			v += o
		}
	}
	return t.ShflF32(v, 0)
}
