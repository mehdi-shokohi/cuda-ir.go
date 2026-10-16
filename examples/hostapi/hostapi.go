// Package hostapi exercises the CUDA array geometries the host package
// added: 1D and 3D arrays, layered and cubemap arrays, multi-channel
// elements and tex2Dgather. Runs on any sm_80 GPU.
package hostapi

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// Tex1DSample writes out[i] = tex1D(t, i+0.5), which with point filtering is
// element i. It also reports txq.width in out[n].
func Tex1DSample(t cuda.Texture, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	r, _, _, _ := cuda.Tex1D(t, float32(i)+0.5)
	out.Set(i, r)
	if i == 0 {
		out.Set(n, float32(cuda.TexWidth(t)))
	}
}

// Tex1DFetchSample writes out[i] = tex1Dfetch(t, i): the unfiltered element.
func Tex1DFetchSample(t cuda.Texture, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i >= n {
		return
	}
	r, _, _, _ := cuda.Tex1DFetch(t, i)
	out.Set(i, r)
}

// Tex3DSample writes out[(z*h+y)*w+x] = tex3D(t, x+0.5, y+0.5, z+0.5): with
// point filtering, the voxel itself. One thread per voxel, launched 1D.
func Tex3DSample(t cuda.Texture, out cuda.Buf[float32], w, h, d int32) {
	i := cuda.GlobalIdX()
	if i >= w*h*d {
		return
	}
	x := i % w
	y := (i / w) % h
	z := i / (w * h)
	r, _, _, _ := cuda.Tex3D(t, float32(x)+0.5, float32(y)+0.5, float32(z)+0.5)
	out.Set(i, r)
}

// Tex2DRGBA writes all four channels of a 4-channel array:
// out[4*(y*w+x)+c]. A fetch always returns four values, so this is the
// kernel side of host.WithChannels(4).
func Tex2DRGBA(t cuda.Texture, out cuda.Buf[float32], w, h int32) {
	i := cuda.GlobalIdX()
	if i >= w*h {
		return
	}
	x, y := i%w, i/w
	r, g, b, a := cuda.Tex2D(t, float32(x)+0.5, float32(y)+0.5)
	out.Set(4*i+0, r)
	out.Set(4*i+1, g)
	out.Set(4*i+2, b)
	out.Set(4*i+3, a)
}

// Tex2DLayeredSample writes out[(l*h+y)*w+x] = tex2DLayered(t, x, y, l).
func Tex2DLayeredSample(t cuda.Texture, out cuda.Buf[float32], w, h, layers int32) {
	i := cuda.GlobalIdX()
	if i >= w*h*layers {
		return
	}
	x := i % w
	y := (i / w) % h
	l := i / (w * h)
	r, _, _, _ := cuda.Tex2DLayered(t, float32(x)+0.5, float32(y)+0.5, l)
	out.Set(i, r)
}

// Tex1DLayeredSample writes out[l*w+x] = tex1DLayered(t, x, l).
func Tex1DLayeredSample(t cuda.Texture, out cuda.Buf[float32], w, layers int32) {
	i := cuda.GlobalIdX()
	if i >= w*layers {
		return
	}
	x, l := i%w, i/w
	r, _, _, _ := cuda.Tex1DLayered(t, float32(x)+0.5, l)
	out.Set(i, r)
}

// TexCubemapSample fetches the centre of each of the six faces: out[f] is
// the value at face f, which the direction vectors below select in the
// driver's face order (+x, -x, +y, -y, +z, -z).
func TexCubemapSample(t cuda.Texture, out cuda.Buf[float32]) {
	f := cuda.GlobalIdX()
	if f >= 6 {
		return
	}
	var x, y, z float32
	switch f {
	case 0:
		x, y, z = 1, 0, 0
	case 1:
		x, y, z = -1, 0, 0
	case 2:
		x, y, z = 0, 1, 0
	case 3:
		x, y, z = 0, -1, 0
	case 4:
		x, y, z = 0, 0, 1
	default:
		x, y, z = 0, 0, -1
	}
	r, _, _, _ := cuda.TexCubemap(t, x, y, z)
	out.Set(f, r)
}

// Tex2DGatherSample writes the four texels tex2Dgather returns at the corner
// between (x, y) and (x+1, y+1) into out[4*i..4*i+3].
func Tex2DGatherSample(t cuda.Texture, out cuda.Buf[float32], w, h int32) {
	i := cuda.GlobalIdX()
	if i >= w*h {
		return
	}
	x, y := i%w, i/w
	v0, v1, v2, v3 := cuda.Tex2DGatherRed(t, float32(x)+1, float32(y)+1)
	out.Set(4*i+0, v0)
	out.Set(4*i+1, v1)
	out.Set(4*i+2, v2)
	out.Set(4*i+3, v3)
}

// Surf3DDouble doubles every element of a 3D float surface in place, the
// kernel side of a 3D host.AllocArray3D with WithSurfaceStore.
func Surf3DDouble(s cuda.Surface, w, h, d int32) {
	i := cuda.GlobalIdX()
	if i >= w*h*d {
		return
	}
	x := i % w
	y := (i / w) % h
	z := i / (w * h)
	v := cuda.Surf3DReadFloat32(s, x, y, z)
	cuda.Surf3DWriteFloat32(s, x, y, z, v*2)
}
