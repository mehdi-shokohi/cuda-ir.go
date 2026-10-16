package cuda

import _ "unsafe" // for go:linkname

// ---- textures and surfaces (cudaTextureObject_t / cudaSurfaceObject_t)
//
// A Texture kernel parameter is the 64-bit texture object handle; the host
// creates it over a CUDA array and passes it with gocudrv's
// cuda.ArgTexture (Surface: cuda.NewSurface + cuda.ArgSurface).
// Fetches return four channels; a single-channel array fills r.

// Texture is a texture object (tex1D/tex2D/tex3D sample from it).
type Texture uint64

// Surface is a surface object (surf1D/surf2Dread/write access it).
type Surface uint64

//go:linkname tex1DF32 llvm.nvvm.tex.unified.1d.v4f32.f32
func tex1DF32(t Texture, x float32) (r, g, b, a float32)

//go:linkname tex2DF32 llvm.nvvm.tex.unified.2d.v4f32.f32
func tex2DF32(t Texture, x, y float32) (r, g, b, a float32)

//go:linkname tex3DF32 llvm.nvvm.tex.unified.3d.v4f32.f32
func tex3DF32(t Texture, x, y, z float32) (r, g, b, a float32)

//go:linkname tex2DS32 llvm.nvvm.tex.unified.2d.v4s32.f32
func tex2DS32(t Texture, x, y float32) (r, g, b, a int32)

//go:linkname tex2DU32 llvm.nvvm.tex.unified.2d.v4u32.f32
func tex2DU32(t Texture, x, y float32) (r, g, b, a uint32)

//go:linkname tex1DFetchF32 llvm.nvvm.tex.unified.1d.v4f32.s32
func tex1DFetchF32(t Texture, i int32) (r, g, b, a float32)

//go:linkname tex3DS32 llvm.nvvm.tex.unified.3d.v4s32.f32
func tex3DS32(t Texture, x, y, z float32) (r, g, b, a int32)

// layered and cubemap fetches take the layer / face as the first
// coordinate, as the .a1d/.a2d/.acube PTX forms do.

//go:linkname tex1DLayeredF32 llvm.nvvm.tex.unified.1d.array.v4f32.f32
func tex1DLayeredF32(t Texture, layer int32, x float32) (r, g, b, a float32)

//go:linkname tex2DLayeredF32 llvm.nvvm.tex.unified.2d.array.v4f32.f32
func tex2DLayeredF32(t Texture, layer int32, x, y float32) (r, g, b, a float32)

//go:linkname tex2DLayeredS32 llvm.nvvm.tex.unified.2d.array.v4s32.f32
func tex2DLayeredS32(t Texture, layer int32, x, y float32) (r, g, b, a int32)

//go:linkname texCubemapF32 llvm.nvvm.tex.unified.cube.v4f32.f32
func texCubemapF32(t Texture, x, y, z float32) (r, g, b, a float32)

//go:linkname texCubemapLayeredF32 llvm.nvvm.tex.unified.cube.array.v4f32.f32
func texCubemapLayeredF32(t Texture, layer int32, x, y, z float32) (r, g, b, a float32)

//go:linkname tld4RF32 llvm.nvvm.tld4.unified.r.2d.v4f32.f32
func tld4RF32(t Texture, x, y float32) (r0, r1, r2, r3 float32)

//go:linkname tld4GF32 llvm.nvvm.tld4.unified.g.2d.v4f32.f32
func tld4GF32(t Texture, x, y float32) (g0, g1, g2, g3 float32)

//go:linkname tld4BF32 llvm.nvvm.tld4.unified.b.2d.v4f32.f32
func tld4BF32(t Texture, x, y float32) (b0, b1, b2, b3 float32)

//go:linkname tld4AF32 llvm.nvvm.tld4.unified.a.2d.v4f32.f32
func tld4AF32(t Texture, x, y float32) (a0, a1, a2, a3 float32)

//go:linkname txqWidth llvm.nvvm.txq.width
func txqWidth(t Texture) int32

//go:linkname txqHeight llvm.nvvm.txq.height
func txqHeight(t Texture) int32

// Tex1D is tex1D<float4>(t, x): a filtered fetch at coordinate x (in
// elements, or 0..1 with normalized coordinates).
func Tex1D(t Texture, x float32) (r, g, b, a float32) { return tex1DF32(t, x) }

// Tex2D is tex2D<float4>(t, x, y).
func Tex2D(t Texture, x, y float32) (r, g, b, a float32) { return tex2DF32(t, x, y) }

// Tex3D is tex3D<float4>(t, x, y, z).
func Tex3D(t Texture, x, y, z float32) (r, g, b, a float32) { return tex3DF32(t, x, y, z) }

// Tex2DInt / Tex2DUint are tex2D<int4> / tex2D<uint4> for integer arrays
// read without normalization.
func Tex2DInt(t Texture, x, y float32) (r, g, b, a int32)   { return tex2DS32(t, x, y) }
func Tex2DUint(t Texture, x, y float32) (r, g, b, a uint32) { return tex2DU32(t, x, y) }

// Tex1DFetch is tex1Dfetch<float4>(t, i): an unfiltered fetch of element i.
func Tex1DFetch(t Texture, i int32) (r, g, b, a float32) { return tex1DFetchF32(t, i) }

// Tex3DInt is tex3D<int4> for an integer 3D array read without
// normalization.
func Tex3DInt(t Texture, x, y, z float32) (r, g, b, a int32) { return tex3DS32(t, x, y, z) }

// Tex1DLayered is tex1DLayered<float4>(t, x, layer): a fetch from one slice
// of a layered 1D array (host: host.AllocArrayLayered1D).
func Tex1DLayered(t Texture, x float32, layer int32) (r, g, b, a float32) {
	return tex1DLayeredF32(t, layer, x)
}

// Tex2DLayered is tex2DLayered<float4>(t, x, y, layer): a fetch from one
// slice of a layered 2D array (host: host.AllocArrayLayered2D).
func Tex2DLayered(t Texture, x, y float32, layer int32) (r, g, b, a float32) {
	return tex2DLayeredF32(t, layer, x, y)
}

// Tex2DLayeredInt is tex2DLayered<int4>.
func Tex2DLayeredInt(t Texture, x, y float32, layer int32) (r, g, b, a int32) {
	return tex2DLayeredS32(t, layer, x, y)
}

// TexCubemap is texCubemap<float4>(t, x, y, z): the direction vector
// (x, y, z) selects the face and the position on it (host:
// host.AllocArrayCubemap).
func TexCubemap(t Texture, x, y, z float32) (r, g, b, a float32) {
	return texCubemapF32(t, x, y, z)
}

// TexCubemapLayered is texCubemapLayered<float4>(t, x, y, z, layer).
func TexCubemapLayered(t Texture, x, y, z float32, layer int32) (r, g, b, a float32) {
	return texCubemapLayeredF32(t, layer, x, y, z)
}

// Tex2DGatherRed is tex2Dgather<float4>(t, x, y, 0): the red channel of the
// four texels that bilinear filtering at (x, y) would blend, in the order
// (0,1), (1,1), (1,0), (0,0) relative to the lower-left one. The array must
// be allocated host.WithTextureGather. GatherGreen/Blue/Alpha are the other
// channels.
func Tex2DGatherRed(t Texture, x, y float32) (v0, v1, v2, v3 float32) { return tld4RF32(t, x, y) }
func Tex2DGatherGreen(t Texture, x, y float32) (v0, v1, v2, v3 float32) {
	return tld4GF32(t, x, y)
}
func Tex2DGatherBlue(t Texture, x, y float32) (v0, v1, v2, v3 float32) { return tld4BF32(t, x, y) }
func Tex2DGatherAlpha(t Texture, x, y float32) (v0, v1, v2, v3 float32) {
	return tld4AF32(t, x, y)
}

// TexWidth / TexHeight are the texture's dimensions (txq.width / txq.height).
func TexWidth(t Texture) int32  { return txqWidth(t) }
func TexHeight(t Texture) int32 { return txqHeight(t) }

//go:linkname suld1DI32 llvm.nvvm.suld.1d.i32.trap
func suld1DI32(s Surface, xBytes int32) int32

//go:linkname sust1DI32 llvm.nvvm.sust.b.1d.i32.trap
func sust1DI32(s Surface, xBytes int32, v int32)

//go:linkname suld2DI32 llvm.nvvm.suld.2d.i32.trap
func suld2DI32(s Surface, xBytes, y int32) int32

//go:linkname sust2DI32 llvm.nvvm.sust.b.2d.i32.trap
func sust2DI32(s Surface, xBytes, y int32, v int32)

// Surf1DReadInt32 / Surf1DWriteInt32 are surf1Dread/surf1Dwrite<int> at
// element x of a 32-bit surface (boundary mode trap). Surf1DReadFloat32 &
// co reinterpret the 32 bits as a float.
func Surf1DReadInt32(s Surface, x int32) int32     { return suld1DI32(s, x*4) }
func Surf1DWriteInt32(s Surface, x int32, v int32) { sust1DI32(s, x*4, v) }
func Surf1DReadFloat32(s Surface, x int32) float32 {
	return Float32FromBits(uint32(suld1DI32(s, x*4)))
}
func Surf1DWriteFloat32(s Surface, x int32, v float32) { sust1DI32(s, x*4, int32(Float32Bits(v))) }

//go:linkname suld3DI32 llvm.nvvm.suld.3d.i32.trap
func suld3DI32(s Surface, xBytes, y, z int32) int32

//go:linkname sust3DI32 llvm.nvvm.sust.b.3d.i32.trap
func sust3DI32(s Surface, xBytes, y, z int32, v int32)

// Surf2DReadInt32 / Surf2DWriteInt32 are surf2Dread/surf2Dwrite<int> at
// element (x, y) of a 32-bit surface.
func Surf2DReadInt32(s Surface, x, y int32) int32     { return suld2DI32(s, x*4, y) }
func Surf2DWriteInt32(s Surface, x, y int32, v int32) { sust2DI32(s, x*4, y, v) }
func Surf2DReadFloat32(s Surface, x, y int32) float32 {
	return Float32FromBits(uint32(suld2DI32(s, x*4, y)))
}
func Surf2DWriteFloat32(s Surface, x, y int32, v float32) {
	sust2DI32(s, x*4, y, int32(Float32Bits(v)))
}

// Surf3DReadInt32 / Surf3DWriteInt32 are surf3Dread/surf3Dwrite<int> at
// element (x, y, z) of a 32-bit surface over a 3D array
// (host.AllocArray3D with host.WithSurfaceStore).
func Surf3DReadInt32(s Surface, x, y, z int32) int32     { return suld3DI32(s, x*4, y, z) }
func Surf3DWriteInt32(s Surface, x, y, z int32, v int32) { sust3DI32(s, x*4, y, z, v) }
func Surf3DReadFloat32(s Surface, x, y, z int32) float32 {
	return Float32FromBits(uint32(suld3DI32(s, x*4, y, z)))
}
func Surf3DWriteFloat32(s Surface, x, y, z int32, v float32) {
	sust3DI32(s, x*4, y, z, int32(Float32Bits(v)))
}
