package host

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// ---- CUDA arrays of every geometry, and textures over them
//
// gocudrv binds cuArray3DCreate, cuMemcpy3D and cuTexObjectCreate already,
// and exports their descriptors — it just has no typed wrapper above 2D,
// single-channel: AllocArray2D and NewTexture are fixed at Array2D[T]. So
// nothing new is bound here; this is the missing wrapper, which is what
// cuda.Tex1D, cuda.Tex3D and the 4-channel fetches need to be reachable
// from Go.
//
//	arr, _ := host.AllocArray3D[float32](ctx, w, h, d)
//	arr.CopyFrom(voxels)
//	tex, _ := host.NewTexture(arr, cuda.TextureConfig{FilterMode: cuda.FilterLinear})
//	launch(..., tex.Arg(), ...)

// CUDA_ARRAY3D flags. gocudrv exports only the surface one.
const (
	arrayLayered       uint32 = 0x01
	arrayCubemap       uint32 = 0x04
	arrayTextureGather uint32 = 0x08
)

// ArrayDesc describes a CUDA array. Height and Depth are 0 for the lower-rank
// geometries, so a 1D array is {Width: w} and a 2D array {Width: w, Height: h}.
type ArrayDesc struct {
	Width, Height, Depth int

	// Layers makes a layered array of 1D or 2D slices (CUDA's
	// cudaArrayLayered); Depth must be 0. tex1DLayered / tex2DLayered
	// sample it.
	Layers int
	// Cubemap makes a cubemap array: Height must equal Width, and the six
	// faces take the place of Depth. With Layers it is a cubemap-layered
	// array.
	Cubemap bool

	// Channels is 1, 2 or 4 values per element; 0 means 1. A fetch always
	// returns four, with the missing channels reading 0 (alpha 1).
	Channels int

	// SurfaceLoadStore allocates so surfaces may read and write the array.
	// NewSurface requires it.
	SurfaceLoadStore bool
	// TextureGather allocates for tex2Dgather.
	TextureGather bool
}

// CUresourcetype values.
const (
	resourceTypeArray   uint32 = 0
	resourceTypeLinear  uint32 = 2
	resourceTypePitch2D uint32 = 3
)

// resourceDesc is the whole CUDA_RESOURCE_DESC: a resType, the 128-byte
// resource union, then flags. gocudrv's version models only res.array, so
// the linear case tex1Dfetch needs is unreachable through it; the union is
// kept as bytes here and written by the setters below.
type resourceDesc struct {
	resType uint32
	_       uint32
	res     [128]byte
	flags   uint32
	_       uint32
}

// setArray fills res.array = {hArray}.
func (d *resourceDesc) setArray(a cudasys.CUarray) {
	d.resType = resourceTypeArray
	*(*cudasys.CUarray)(unsafe.Pointer(&d.res[0])) = a
}

// setLinear fills res.linear = {devPtr, format, numChannels, sizeInBytes}:
// a texture over plain device memory, which is what tex1Dfetch reads.
func (d *resourceDesc) setLinear(ptr cudasys.CUdeviceptr, format uint32, channels int, sizeInBytes uint64) {
	d.resType = resourceTypeLinear
	*(*cudasys.CUdeviceptr)(unsafe.Pointer(&d.res[0])) = ptr
	*(*uint32)(unsafe.Pointer(&d.res[8])) = format
	*(*uint32)(unsafe.Pointer(&d.res[12])) = uint32(channels)
	*(*uint64)(unsafe.Pointer(&d.res[16])) = sizeInBytes
}

// setPitch2D fills res.pitch2D = {devPtr, format, numChannels, width,
// height, pitchInBytes}: a 2D texture over pitched device memory, which
// unlike an array the host can also write through its device pointer.
func (d *resourceDesc) setPitch2D(ptr cudasys.CUdeviceptr, format uint32, channels, width, height int, pitch uint64) {
	d.resType = resourceTypePitch2D
	*(*cudasys.CUdeviceptr)(unsafe.Pointer(&d.res[0])) = ptr
	*(*uint32)(unsafe.Pointer(&d.res[8])) = format
	*(*uint32)(unsafe.Pointer(&d.res[12])) = uint32(channels)
	*(*uint64)(unsafe.Pointer(&d.res[16])) = uint64(width)
	*(*uint64)(unsafe.Pointer(&d.res[24])) = uint64(height)
	*(*uint64)(unsafe.Pointer(&d.res[32])) = pitch
}

// Array is a CUDA array: device memory in an opaque, texture-optimised layout
// with no device pointer. Close it after any Texture or Surface over it, and
// before the owning Context.
type Array[T cuda.Supported] struct {
	ctx      *cuda.Context
	raw      cudasys.CUarray
	desc     ArrayDesc
	channels int
}

// arrayFormat maps T to a CUarray_format, mirroring gocudrv's mapping so an
// array allocated here behaves like one of its own. 8-byte elements have no
// array format and are rejected.
func arrayFormat[T cuda.Supported]() (format uint32, integer, ok bool) {
	switch reflect.TypeOf(*new(T)).Kind() {
	case reflect.Uint8:
		return cudasys.AdFormatUnsignedInt8, true, true
	case reflect.Uint16:
		return cudasys.AdFormatUnsignedInt16, true, true
	case reflect.Uint32:
		return cudasys.AdFormatUnsignedInt32, true, true
	case reflect.Int8:
		return cudasys.AdFormatSignedInt8, true, true
	case reflect.Int16:
		return cudasys.AdFormatSignedInt16, true, true
	case reflect.Int32:
		return cudasys.AdFormatSignedInt32, true, true
	case reflect.Float32:
		return cudasys.AdFormatFloat, false, true
	}
	return 0, false, false
}

// AllocArray creates an array of any geometry with cuArray3DCreate.
func AllocArray[T cuda.Supported](ctx *cuda.Context, d ArrayDesc) (*Array[T], error) {
	if ctx == nil {
		return nil, errors.New("host: AllocArray: nil context")
	}
	format, _, ok := arrayFormat[T]()
	if !ok {
		var zero T
		return nil, fmt.Errorf("host: AllocArray: %T has no CUDA array format", zero)
	}
	if d.Channels == 0 {
		d.Channels = 1
	}
	if err := d.validate(); err != nil {
		return nil, err
	}

	// Layers and cubemap faces both ride in Depth.
	depth := uint64(d.Depth)
	var flags uint32
	if d.Cubemap {
		depth = 6
		flags |= arrayCubemap
	}
	if d.Layers > 0 {
		if d.Cubemap {
			depth = 6 * uint64(d.Layers)
		} else {
			depth = uint64(d.Layers)
		}
		flags |= arrayLayered
	}
	if d.SurfaceLoadStore {
		flags |= cudasys.ArraySurfaceLoadStore
	}
	if d.TextureGather {
		flags |= arrayTextureGather
	}

	cd := cudasys.CUDA_ARRAY3D_DESCRIPTOR{
		Width:       uint64(d.Width),
		Height:      uint64(d.Height),
		Depth:       depth,
		Format:      format,
		NumChannels: uint32(d.Channels),
		Flags:       flags,
	}

	a := &Array[T]{ctx: ctx, desc: d, channels: d.Channels}
	err := OnContext(ctx, func() error {
		raw, err := cudaresult.Array3DCreate(ctx.Driver(), &cd)
		if err != nil {
			return err
		}
		a.raw = raw
		return nil
	})
	if err != nil {
		return nil, err
	}
	return a, nil
}

func (d ArrayDesc) validate() error {
	switch {
	case d.Width <= 0:
		return fmt.Errorf("host: AllocArray: width %d", d.Width)
	case d.Height < 0 || d.Depth < 0 || d.Layers < 0:
		return errors.New("host: AllocArray: negative extent")
	case d.Depth > 0 && d.Layers > 0:
		return errors.New("host: AllocArray: Depth and Layers are exclusive")
	case d.Depth > 0 && d.Height == 0:
		return errors.New("host: AllocArray: a 3D array needs a Height")
	case d.Cubemap && d.Height != d.Width:
		return errors.New("host: AllocArray: a cubemap needs Height == Width")
	case d.Channels != 1 && d.Channels != 2 && d.Channels != 4:
		return fmt.Errorf("host: AllocArray: %d channels, want 1, 2 or 4", d.Channels)
	}
	return nil
}

// AllocArray1D creates a width-element 1D array, which cuda.Tex1D and
// cuda.Tex1DFetch sample.
func AllocArray1D[T cuda.Supported](ctx *cuda.Context, width int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: width}, opts))
}

// AllocArray2D creates a 2D array. Unlike gocudrv's it can be multi-channel.
func AllocArray2D[T cuda.Supported](ctx *cuda.Context, width, height int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: width, Height: height}, opts))
}

// AllocArray3D creates a 3D array, which cuda.Tex3D samples.
func AllocArray3D[T cuda.Supported](ctx *cuda.Context, width, height, depth int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: width, Height: height, Depth: depth}, opts))
}

// AllocArrayLayered1D creates layers 1D slices (tex1DLayered).
func AllocArrayLayered1D[T cuda.Supported](ctx *cuda.Context, width, layers int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: width, Layers: layers}, opts))
}

// AllocArrayLayered2D creates layers 2D slices (tex2DLayered).
func AllocArrayLayered2D[T cuda.Supported](ctx *cuda.Context, width, height, layers int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: width, Height: height, Layers: layers}, opts))
}

// AllocArrayCubemap creates a size-by-size cubemap (texCubemap).
func AllocArrayCubemap[T cuda.Supported](ctx *cuda.Context, size int, opts ...ArrayOpt) (*Array[T], error) {
	return AllocArray[T](ctx, apply(ArrayDesc{Width: size, Height: size, Cubemap: true}, opts))
}

// ArrayOpt adjusts the descriptor the AllocArray* helpers build.
type ArrayOpt func(*ArrayDesc)

// WithChannels sets 1, 2 or 4 values per element.
func WithChannels(n int) ArrayOpt { return func(d *ArrayDesc) { d.Channels = n } }

// WithSurfaceStore allocates so surfaces may read and write the array.
func WithSurfaceStore() ArrayOpt { return func(d *ArrayDesc) { d.SurfaceLoadStore = true } }

// WithTextureGather allocates for tex2Dgather.
func WithTextureGather() ArrayOpt { return func(d *ArrayDesc) { d.TextureGather = true } }

func apply(d ArrayDesc, opts []ArrayOpt) ArrayDesc {
	for _, o := range opts {
		o(&d)
	}
	return d
}

// Width, Height, Depth, Layers and Channels report the array's shape. Height
// and Depth are 0 for the lower-rank geometries, as they were on allocation.
func (a *Array[T]) Width() int    { return a.desc.Width }
func (a *Array[T]) Height() int   { return a.desc.Height }
func (a *Array[T]) Depth() int    { return a.desc.Depth }
func (a *Array[T]) Layers() int   { return a.desc.Layers }
func (a *Array[T]) Channels() int { return a.channels }

// Raw is the underlying CUarray, valid only while the Array is open.
func (a *Array[T]) Raw() cudasys.CUarray {
	if a == nil {
		return 0
	}
	return a.raw
}

// Elements is the number of T the array holds, channels included: what
// CopyFrom and CopyTo expect.
func (a *Array[T]) Elements() int {
	h, d := a.desc.Height, a.desc.Depth
	if h == 0 {
		h = 1
	}
	switch {
	case a.desc.Cubemap && a.desc.Layers > 0:
		d = 6 * a.desc.Layers
	case a.desc.Cubemap:
		d = 6
	case a.desc.Layers > 0:
		d = a.desc.Layers
	case d == 0:
		d = 1
	}
	return a.desc.Width * h * d * a.channels
}

// copyDesc builds the cuMemcpy3D descriptor for a whole-array copy. A 1D or
// 2D array copies as a 3D one with the missing extents set to 1.
func (a *Array[T]) copyDesc(toArray bool, host unsafe.Pointer) cudasys.Memcpy3D {
	var zero T
	elem := uint64(unsafe.Sizeof(zero)) * uint64(a.channels)
	rowBytes := uint64(a.desc.Width) * elem

	h, d := uint64(a.desc.Height), uint64(a.desc.Depth)
	if h == 0 {
		h = 1
	}
	switch {
	case a.desc.Cubemap && a.desc.Layers > 0:
		d = 6 * uint64(a.desc.Layers)
	case a.desc.Cubemap:
		d = 6
	case a.desc.Layers > 0:
		d = uint64(a.desc.Layers)
	case d == 0:
		d = 1
	}

	c := cudasys.Memcpy3D{WidthInBytes: rowBytes, Height: h, Depth: d}
	if toArray {
		c.SrcMemoryType = cudasys.MemoryTypeHost
		c.SrcHost = host
		c.SrcPitch = rowBytes
		c.SrcHeight = h
		c.DstMemoryType = cudasys.MemoryTypeArray
		c.DstArray = uintptr(a.raw)
	} else {
		c.SrcMemoryType = cudasys.MemoryTypeArray
		c.SrcArray = uintptr(a.raw)
		c.DstMemoryType = cudasys.MemoryTypeHost
		c.DstHost = host
		c.DstPitch = rowBytes
		c.DstHeight = h
	}
	return c
}

// CopyFrom fills the whole array from src, which must hold exactly Elements
// values: innermost dimension first, channels interleaved.
func (a *Array[T]) CopyFrom(src []T) error {
	if a == nil || a.raw == 0 {
		return errors.New("host: CopyFrom: array is closed")
	}
	if len(src) != a.Elements() {
		return fmt.Errorf("host: CopyFrom: %d elements, want %d", len(src), a.Elements())
	}
	c := a.copyDesc(true, unsafe.Pointer(&src[0]))
	err := OnContext(a.ctx, func() error { return cudaresult.Memcpy3D(a.ctx.Driver(), &c) })
	runtime.KeepAlive(src)
	return err
}

// CopyTo reads the whole array into dst, which must hold exactly Elements
// values.
func (a *Array[T]) CopyTo(dst []T) error {
	if a == nil || a.raw == 0 {
		return errors.New("host: CopyTo: array is closed")
	}
	if len(dst) != a.Elements() {
		return fmt.Errorf("host: CopyTo: %d elements, want %d", len(dst), a.Elements())
	}
	c := a.copyDesc(false, unsafe.Pointer(&dst[0]))
	err := OnContext(a.ctx, func() error { return cudaresult.Memcpy3D(a.ctx.Driver(), &c) })
	runtime.KeepAlive(dst)
	return err
}

// Close frees the array.
func (a *Array[T]) Close() error {
	if a == nil || a.raw == 0 {
		return nil
	}
	raw := a.raw
	a.raw = 0
	return OnContext(a.ctx, func() error { return cudaresult.ArrayDestroy(a.ctx.Driver(), raw) })
}

// Texture is a texture object over an Array of any geometry.
type Texture struct {
	ctx *cuda.Context
	raw cudasys.CUtexObject
}

// NewTexture creates a texture object sampling a with cfg. Integer element
// types are read as integers, as in gocudrv. The array must stay open for the
// texture's lifetime.
func NewTexture[T cuda.Supported](a *Array[T], cfg cuda.TextureConfig) (*Texture, error) {
	if a == nil || a.raw == 0 {
		return nil, errors.New("host: NewTexture: array is closed")
	}
	_, integer, ok := arrayFormat[T]()
	if !ok {
		return nil, errors.New("host: NewTexture: unsupported element type")
	}

	var res resourceDesc
	res.setArray(a.raw)
	return newTexture(a.ctx, &res, texDesc(cfg, integer))
}

// NewTextureLinear creates a texture over plain device memory rather than a
// CUDA array, which is what cuda.Tex1DFetch reads: an unfiltered,
// unnormalized fetch by element index. The buffer must belong to ctx, and its
// device pointer must meet the device's texture alignment, which cuMemAlloc
// satisfies. channels is 1, 2 or 4; 0 means 1.
func NewTextureLinear[T cuda.Supported](ctx *cuda.Context, b *cuda.Buffer[T], channels int) (*Texture, error) {
	if b == nil {
		return nil, errors.New("host: NewTextureLinear: nil buffer")
	}
	format, integer, ok := arrayFormat[T]()
	if !ok {
		var zero T
		return nil, fmt.Errorf("host: NewTextureLinear: %T has no CUDA array format", zero)
	}
	if channels == 0 {
		channels = 1
	}
	if channels != 1 && channels != 2 && channels != 4 {
		return nil, fmt.Errorf("host: NewTextureLinear: %d channels, want 1, 2 or 4", channels)
	}
	if b.Len()%channels != 0 {
		return nil, fmt.Errorf("host: NewTextureLinear: %d elements is not a multiple of %d channels", b.Len(), channels)
	}

	var zero T
	var res resourceDesc
	res.setLinear(b.DevicePtr(), format, channels, uint64(b.Len())*uint64(unsafe.Sizeof(zero)))
	// A linear texture supports neither filtering nor normalized
	// coordinates, so the config is fixed rather than taken from the caller.
	return newTexture(ctx, &res, texDesc(cuda.TextureConfig{}, integer))
}

// texDesc builds the sampling descriptor. Integer element types are read as
// integers, as in gocudrv.
func texDesc(cfg cuda.TextureConfig, integer bool) cudasys.CUDA_TEXTURE_DESC {
	td := cudasys.CUDA_TEXTURE_DESC{
		AddressMode: [3]uint32{uint32(cfg.AddressMode), uint32(cfg.AddressMode), uint32(cfg.AddressMode)},
		FilterMode:  uint32(cfg.FilterMode),
	}
	if cfg.NormalizedCoordinates {
		td.Flags |= cudasys.TexNormalizedCoordinate
	}
	if integer {
		td.Flags |= cudasys.TexReadAsInteger
	}
	return td
}

func newTexture(ctx *cuda.Context, res *resourceDesc, td cudasys.CUDA_TEXTURE_DESC) (*Texture, error) {
	if err := loadDriver(); err != nil {
		return nil, err
	}
	t := &Texture{ctx: ctx}
	err := OnContext(ctx, func() error {
		return check("cuTexObjectCreate", cuTexObjectCreate(&t.raw, res, &td, nil))
	})
	if err != nil {
		return nil, err
	}
	return t, nil
}

// Raw is the underlying CUtexObject: the handle a kernel receives as a
// cuda.Texture.
func (t *Texture) Raw() cudasys.CUtexObject {
	if t == nil {
		return 0
	}
	return t.raw
}

// Arg passes the texture to a cuda.Texture kernel parameter.
func (t *Texture) Arg() cuda.KernelArg { return cuda.ArgValue(uint64(t.Raw())) }

// ArgEx passes the texture to a LaunchEx kernel.
func (t *Texture) ArgEx() Arg { return ArgValue(uint64(t.Raw())) }

// Close destroys the texture object.
func (t *Texture) Close() error {
	if t == nil || t.raw == 0 {
		return nil
	}
	raw := t.raw
	t.raw = 0
	return OnContext(t.ctx, func() error { return cudaresult.TexObjectDestroy(t.ctx.Driver(), raw) })
}

// Surface is a surface object over an Array of any geometry. The array must
// have been allocated WithSurfaceStore.
type Surface struct {
	ctx *cuda.Context
	raw cudasys.CUsurfObject
}

// NewSurface creates a surface object over a.
func NewSurface[T cuda.Supported](a *Array[T]) (*Surface, error) {
	if a == nil || a.raw == 0 {
		return nil, errors.New("host: NewSurface: array is closed")
	}
	if !a.desc.SurfaceLoadStore {
		return nil, errors.New("host: NewSurface: array was not allocated WithSurfaceStore")
	}
	res := cudasys.CUDA_RESOURCE_DESC{ResType: cudasys.ResourceTypeArray, Handle: a.raw}
	s := &Surface{ctx: a.ctx}
	err := OnContext(a.ctx, func() error {
		raw, err := cudaresult.SurfObjectCreate(a.ctx.Driver(), &res)
		if err != nil {
			return err
		}
		s.raw = raw
		return nil
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

// Raw is the underlying CUsurfObject.
func (s *Surface) Raw() cudasys.CUsurfObject {
	if s == nil {
		return 0
	}
	return s.raw
}

// Arg passes the surface to a cuda.Surface kernel parameter.
func (s *Surface) Arg() cuda.KernelArg { return cuda.ArgValue(uint64(s.Raw())) }

// ArgEx passes the surface to a LaunchEx kernel.
func (s *Surface) ArgEx() Arg { return ArgValue(uint64(s.Raw())) }

// Close destroys the surface object.
func (s *Surface) Close() error {
	if s == nil || s.raw == 0 {
		return nil
	}
	raw := s.raw
	s.raw = 0
	return OnContext(s.ctx, func() error { return cudaresult.SurfObjectDestroy(s.ctx.Driver(), raw) })
}
