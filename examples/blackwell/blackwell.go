// Package blackwell holds the sm_100a-only feature: redux.sync on floats.
// Build with -sm sm_100a -ptx 86.
package blackwell

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// ReduceF32: out[2*warp] = min, out[2*warp+1] = max of the warp's values.
func ReduceF32(in, out cuda.Buf[float32]) {
	i := cuda.GlobalIdX()
	v := in.At(i)
	mn := cuda.ReduceMinF32(cuda.FullMask, v)
	mx := cuda.ReduceMaxF32(cuda.FullMask, v)
	if cuda.LaneID() == 0 {
		out.Set(2*(i/32), mn)
		out.Set(2*(i/32)+1, mx)
	}
}
