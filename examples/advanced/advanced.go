// Package advanced exercises the v0.4 features that run on any sm_80 GPU:
// tensor cores (wmma), textures and surfaces, __restrict__ and
// __grid_constant__ parameters, cache-hint loads/stores, the cooperative
// groups sugar and the %pm counters.
package advanced

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// ---- tensor cores

// Wmma computes C = A * B + Bias for n x n row-major matrices (n a multiple
// of 16) with one warp per 16x16 output tile: launch (n/16)^2 warps.
func Wmma(a, b cuda.Buf[cuda.Half], bias, c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	acc := cuda.WmmaLoadC(bias.Ptr(row*16*n+col*16), n)
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadA(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadB(b.Ptr(k*n+col*16), n)
		acc = cuda.WmmaMma(fa, fb, acc)
	}
	cuda.WmmaStore(c.Ptr(row*16*n+col*16), n, acc)
}

// WmmaTrans is C = A * Bt^T with Bt row-major (so B is read column-major)
// and a zero accumulator, stored column-major into c.
func WmmaTrans(a, bt cuda.Buf[cuda.Half], c cuda.Buf[float32], n int32) {
	warp := cuda.GlobalIdX() / 32
	tiles := n / 16
	row, col := warp/tiles, warp%tiles
	if row >= tiles {
		return
	}
	var acc cuda.FragC
	acc.Fill(0)
	for k := int32(0); k < n; k += 16 {
		fa := cuda.WmmaLoadA(a.Ptr(row*16*n+k), n)
		fb := cuda.WmmaLoadBCol(bt.Ptr(col*16*n+k), n)
		acc = cuda.WmmaMmaRowCol(fa, fb, acc)
	}
	cuda.WmmaStoreCol(c.Ptr(col*16*n+row*16), n, acc)
}

// ---- textures and surfaces

// TexSample writes out[y*w+x] = tex2D(t, x+dx, y+0.5): with dx = 0.5 and
// point filtering that is the element itself, with dx = 1 and linear
// filtering the mean of x and x+1. dims gets the texture's width and
// height.
func TexSample(t cuda.Texture, out cuda.Buf[float32], dims cuda.Buf[int32], w, h int32, dx float32) {
	x, y := cuda.GlobalIdX(), cuda.GlobalIdY()
	if x >= w || y >= h {
		return
	}
	r, _, _, _ := cuda.Tex2D(t, float32(x)+dx, float32(y)+0.5)
	out.Set(y*w+x, r)
	if x == 0 && y == 0 {
		dims.Set(0, cuda.TexWidth(t))
		dims.Set(1, cuda.TexHeight(t))
	}
}

// SurfScale doubles every element of a 2D float surface in place.
func SurfScale(s cuda.Surface, w, h int32) {
	x, y := cuda.GlobalIdX(), cuda.GlobalIdY()
	if x >= w || y >= h {
		return
	}
	cuda.Surf2DWriteFloat32(s, x, y, 2*cuda.Surf2DReadFloat32(s, x, y))
}

// SurfInt: int surface, out[y*w+x] = surf(x, y) + 1 written back.
func SurfInt(s cuda.Surface, w, h int32) {
	x, y := cuda.GlobalIdX(), cuda.GlobalIdY()
	if x >= w || y >= h {
		return
	}
	cuda.Surf2DWriteInt32(s, x, y, cuda.Surf2DReadInt32(s, x, y)+1)
}

// ---- __restrict__: in and out do not alias, so llc may read `in`
// through the read-only cache (ld.global.nc) without LdgF32.

//cuda:restrict
func Restrict(in, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		out.Set(i, in.At(i)*2)
	}
}

// NoRestrict is the same kernel without the directive (plain ld.global).
func NoRestrict(in, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		out.Set(i, in.At(i)*2)
	}
}

// ---- __grid_constant__

// Params is a by-value kernel parameter.
type Params struct {
	Scale, Offset float32
	N             int32
}

//go:noinline
func apply(p *Params, x float32) float32 { return p.Scale*x + p.Offset }

// GridConst takes p's address (a pointer to the parameter in param space:
// cvta.param, no local copy). The parameter must not be modified.
//
//cuda:grid_constant
func GridConst(in, out cuda.Buf[float32], p Params) {
	i := cuda.GlobalIdX()
	if i < p.N {
		out.Set(i, apply(&p, in.At(i)))
	}
}

// LocalCopy is the same kernel without the directive: &p copies the
// parameter to local memory first.
func LocalCopy(in, out cuda.Buf[float32], p Params) {
	i := cuda.GlobalIdX()
	if i < p.N {
		out.Set(i, apply(&p, in.At(i)))
	}
}

// ---- cache hints

// CacheHints: out[i] = 4*in[i] read with every float load hint and written
// with every store hint (the last store wins); iout = 2*iin, lout = 2*lin,
// dout = 2*din likewise. The pointers must be global memory.
func CacheHints(in, out cuda.Buf[float32], iin, iout cuda.Buf[int32], lin, lout cuda.Buf[int64], din, dout cuda.Buf[float64], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	p := in.Ptr(i)
	v := cuda.LoadCAFloat32(p) + cuda.LoadCGFloat32(p) + cuda.LoadCSFloat32(p) + cuda.LoadCVFloat32(p)
	q := out.Ptr(i)
	cuda.StoreWBFloat32(q, 0)
	cuda.StoreCGFloat32(q, 1)
	cuda.StoreWTFloat32(q, 2)
	cuda.StoreCSFloat32(q, v)
	ip := iin.Ptr(i)
	cuda.StoreCSInt32(iout.Ptr(i), cuda.LoadCGInt32(ip)+cuda.LoadLUInt32(ip))
	lp := lin.Ptr(i)
	cuda.StoreWBInt64(lout.Ptr(i), cuda.LoadCSInt64(lp)+cuda.LoadCAInt64(lp))
	dp := din.Ptr(i)
	cuda.StoreCGFloat64(dout.Ptr(i), cuda.LoadCVFloat64(dp)+cuda.LoadLUFloat64(dp))
}

// ---- cooperative groups sugar

var evens int32

// Groups: per tile of 8 threads, tiles[tile] = sum of the tile's 8 values;
// per warp, warps[warp] = max over the warp; count[0] counts the even
// values through coalesced_threads() in a divergent branch, where
// esum[i] = the coalesced group's sum (the warp's even values); ranks[i] =
// block.thread_rank() * 1000 + tile.meta_group_rank() * 10 + tile.thread_rank()
// (+ 100000 * the warp's first even value, for even values).
func Groups(in, tiles, warps, ranks, esum cuda.Buf[int32], count cuda.Buf[int32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	v := in.At(i)
	t := cuda.TiledPartition(8)
	s := t.ReduceAdd(v)
	if t.ThreadRank() == 0 {
		tiles.Set(i/8, s)
	}
	w := cuda.ThisWarp()
	m := w.ReduceMax(v)
	if w.ThreadRank() == 0 {
		warps.Set(i/32, m)
	}
	b := cuda.ThisBlock()
	ranks.Set(i, b.ThreadRank()*1000+t.MetaGroupRank()*10+t.ThreadRank())
	if v&1 == 0 {
		g := cuda.CoalescedThreads()
		if g.ThreadRank() == 0 {
			cuda.AtomicAddInt32Block(count.Ptr(0), g.Size())
		}
		// every lane sees the first even value of its warp
		ranks.Set(i, ranks.At(i)+g.Shfl(v, 0)*100000)
		esum.Set(i, g.ReduceAdd(v))
	}
	b.Sync()
}

// ---- %pm0..3

// PerfCounters stores the four performance-monitor counters.
func PerfCounters(out cuda.Buf[int32]) {
	if cuda.GlobalIdX() == 0 {
		out.Set(0, cuda.PerfCounter0())
		out.Set(1, cuda.PerfCounter1())
		out.Set(2, cuda.PerfCounter2())
		out.Set(3, cuda.PerfCounter3())
	}
}
