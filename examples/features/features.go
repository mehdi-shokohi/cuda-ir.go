// Package features exercises the device API: shared memory + barriers,
// warp shuffles, atomics, libdevice math.
package features

import (
	"sync/atomic"

	"github.com/mehdi-shokohi/gocuda/cuda"
)

const blockSize = 256

var tile cuda.Shared[[blockSize]float32]

// BlockSum reduces each block of `in` with shared memory + __syncthreads,
// then a warp shuffle for the last 32, and atomically adds to out[0].
func BlockSum(in cuda.Buf[float32], out cuda.Buf[float32], n int32) {
	t := tile.Get()
	tid := cuda.ThreadIdxX()
	i := cuda.GlobalIdX()
	var v float32
	if i < n {
		v = in.At(i)
	}
	t[tid] = v
	cuda.SyncThreads()
	for s := int32(blockSize / 2); s > 32; s >>= 1 {
		if tid < s {
			t[tid] += t[tid+s]
		}
		cuda.SyncThreads()
	}
	if tid < 32 {
		v = t[tid] + t[tid+32]
		for off := int32(16); off > 0; off >>= 1 {
			v += cuda.ShflDownF32(cuda.FullMask, v, off)
		}
		if tid == 0 {
			cuda.AtomicAddFloat32(out.Ptr(0), v)
		}
	}
}

// Count counts elements > 0.5 with Go's sync/atomic, and counts warps
// where every lane agrees, using a warp vote.
func Count(in cuda.Buf[float32], counters cuda.Buf[int32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	big := in.At(i) > 0.5
	if big {
		atomic.AddInt32(counters.Ptr(0), 1)
	}
	if cuda.All(cuda.FullMask, big) && cuda.LaneID() == 0 {
		atomic.AddInt32(counters.Ptr(1), 1)
	}
	if cuda.Ballot(cuda.FullMask, big) != 0 && cuda.LaneID() == 0 {
		atomic.AddInt32(counters.Ptr(2), 1)
	}
}

// Math: out[i] = sin(x)^2 + cos(x)^2 + sqrt(x*x) - |x| + exp(log(x+1)) - (x+1)  == 1
func Math(x cuda.Buf[float32], out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := x.At(i)
	s, c := cuda.Sin(v), cuda.Cos(v)
	r := cuda.FMA(s, s, c*c) + cuda.Sqrt(v*v) - cuda.Abs(v) + cuda.Exp(cuda.Log(v+1)) - (v + 1)
	r += cuda.Pow(v, 2) - v*v + cuda.Floor(v) - cuda.Floor(v)
	out.Set(i, r)
}

// Timing writes elapsed clock64 cycles of a spin into out[block].
func Timing(out cuda.Buf[int64]) {
	start := cuda.Clock64()
	var acc float32
	for k := int32(0); k < 1000; k++ {
		acc = cuda.FMA(acc, 1.0001, 1)
	}
	cuda.SyncThreads()
	if cuda.ThreadIdxX() == 0 {
		out.Set(cuda.BlockIdxX(), cuda.Clock64()-start+int64(acc*0))
	}
}
