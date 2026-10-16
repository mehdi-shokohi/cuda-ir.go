package host

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// ---- cuLaunchKernelEx: launch attributes
//
// gocudrv launches with cuLaunchKernel, which takes no launch attributes, and
// keeps its CUfunction private — so a launch that needs attributes needs its
// own module. LoadModule here loads PTX into the same context gocudrv owns and
// keeps the handle, which is all cuLaunchKernelEx needs.
//
// What this buys over gocudrv's Launch:
//
//   - a cluster shape chosen at launch time, rather than baked into the PTX
//     by //cuda:cluster_dims
//   - programmatic dependent launch (griddepcontrol), whose wait and
//     launch_dependents are no-ops without the attribute
//
// Everything else should keep using gocudrv's Module and Function: this path
// has none of its lifetime tracking or executor serialisation.

// Module is a PTX module loaded into a gocudrv Context by this package.
type Module struct {
	ctx *cuda.Context
	raw cudasys.CUmodule
}

// Function is an entry point of a Module, launchable with attributes.
type Function struct {
	mod  *Module
	raw  cudasys.CUfunction
	name string
}

// LoadModule loads PTX into ctx. Close it when done.
func LoadModule(ctx *cuda.Context, ptx []byte) (*Module, error) {
	if err := loadDriver(); err != nil {
		return nil, err
	}
	if len(ptx) == 0 {
		return nil, errors.New("host: LoadModule: empty PTX")
	}
	img := ptx
	if img[len(img)-1] != 0 { // cuModuleLoadData wants a C string
		img = append(append([]byte(nil), img...), 0)
	}

	m := &Module{ctx: ctx}
	err := OnContext(ctx, func() error {
		raw, err := cudaresult.ModuleLoadData(ctx.Driver(), &img[0])
		if err != nil {
			return err
		}
		m.raw = raw
		return nil
	})
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Close unloads the module.
func (m *Module) Close() error {
	if m == nil || m.raw == 0 {
		return nil
	}
	raw := m.raw
	m.raw = 0
	return OnContext(m.ctx, func() error { return cudaresult.ModuleUnload(m.ctx.Driver(), raw) })
}

// Function looks up an entry point by its PTX name.
func (m *Module) Function(name string) (*Function, error) {
	if m == nil || m.raw == 0 {
		return nil, errors.New("host: Function: module is closed")
	}
	cname := append([]byte(name), 0)
	f := &Function{mod: m, name: name}
	err := OnContext(m.ctx, func() error {
		raw, err := cudaresult.ModuleGetFunction(m.ctx.Driver(), m.raw, &cname[0])
		if err != nil {
			return err
		}
		f.raw = raw
		return nil
	})
	if err != nil {
		return nil, err
	}
	return f, nil
}

// LaunchAttrs are the cuLaunchKernelEx launch attributes. A zero LaunchAttrs
// adds none, making LaunchEx equivalent to an ordinary launch.
type LaunchAttrs struct {
	// ClusterDim is the thread-block cluster shape, in blocks. The zero
	// value adds no attribute; the kernel must not already carry
	// //cuda:cluster_dims, whose .reqnctapercluster the driver would
	// then find in conflict. The grid must be a multiple of the cluster.
	ClusterDim [3]uint32

	// Cooperative makes the launch cooperative (what
	// Function.LaunchCooperative does, available here as an attribute).
	Cooperative bool

	// ProgrammaticStreamSerialization lets the next kernel on the stream
	// start before this one finishes, which is what makes
	// cuda.GridDepWait and cuda.GridDepLaunchDependents do anything.
	ProgrammaticStreamSerialization bool
}

// CUlaunchConfig: seven uint32, then the stream, the attribute array and its
// length. The padding matches the C layout on a 64-bit ABI.
type launchConfig struct {
	gridX, gridY, gridZ    uint32
	blockX, blockY, blockZ uint32
	sharedMemBytes         uint32
	_                      uint32
	stream                 cudasys.CUstream
	attrs                  *launchAttribute
	numAttrs               uint32
	_                      uint32
}

// CUlaunchAttribute: the id, four bytes of padding, then the 64-byte
// CUlaunchAttributeValue union.
type launchAttribute struct {
	id    uint32
	_     uint32
	value [64]byte
}

// CUlaunchAttributeID values.
const (
	attrCooperative        = 2
	attrClusterDimension   = 4
	attrProgrammaticStream = 6
)

// attributes builds the driver's attribute array. A kernel with no attributes
// gets a nil array, which cuLaunchKernelEx accepts.
func (a LaunchAttrs) attributes() []launchAttribute {
	var out []launchAttribute
	if a.ClusterDim != [3]uint32{} {
		var at launchAttribute
		at.id = attrClusterDimension
		*(*[3]uint32)(unsafe.Pointer(&at.value[0])) = a.ClusterDim
		out = append(out, at)
	}
	if a.Cooperative {
		var at launchAttribute
		at.id = attrCooperative
		*(*int32)(unsafe.Pointer(&at.value[0])) = 1
		out = append(out, at)
	}
	if a.ProgrammaticStreamSerialization {
		var at launchAttribute
		at.id = attrProgrammaticStream
		*(*int32)(unsafe.Pointer(&at.value[0])) = 1
		out = append(out, at)
	}
	return out
}

// LaunchEx launches f on the default stream with attrs, then waits for the
// context so the kernel has finished when it returns.
func (f *Function) LaunchEx(cfg cuda.LaunchConfig, attrs LaunchAttrs, args ...Arg) error {
	return f.LaunchExOn(nil, cfg, attrs, args...)
}

// LaunchExOn launches f on stream, which must belong to f's context. A nil
// stream is the legacy default stream; LaunchExOn does not synchronize, so
// the caller waits on the stream itself.
func (f *Function) LaunchExOn(stream *cuda.Stream, cfg cuda.LaunchConfig, attrs LaunchAttrs, args ...Arg) error {
	if f == nil || f.raw == 0 {
		return errors.New("host: LaunchEx: nil function")
	}
	if cfg.GridX == 0 || cfg.BlockX == 0 {
		return fmt.Errorf("host: LaunchEx %s: empty launch config", f.name)
	}

	p := pack(args)
	at := attrs.attributes()
	lc := launchConfig{
		gridX: cfg.GridX, gridY: max1(cfg.GridY), gridZ: max1(cfg.GridZ),
		blockX: cfg.BlockX, blockY: max1(cfg.BlockY), blockZ: max1(cfg.BlockZ),
		sharedMemBytes: cfg.SharedMemBytes,
		stream:         stream.Raw(),
		numAttrs:       uint32(len(at)),
	}
	if len(at) > 0 {
		lc.attrs = &at[0]
	}

	err := OnContext(f.mod.ctx, func() error {
		if err := check("cuLaunchKernelEx", cuLaunchKernelEx(&lc, f.raw, p.params(), nil)); err != nil {
			return err
		}
		if stream != nil {
			return nil
		}
		return cudaresult.CtxSynchronize(f.mod.ctx.Driver())
	})
	p.keepAlive()
	if err != nil {
		return fmt.Errorf("host: LaunchEx %s: %w", f.name, err)
	}
	return nil
}

func max1(n uint32) uint32 {
	if n == 0 {
		return 1
	}
	return n
}

// ---- kernel arguments for LaunchEx
//
// gocudrv's cuda.KernelArg cannot be used here: the interface's only method is
// unexported, so this package can neither build the parameter array from one
// nor read its bytes. These are the equivalent constructors for LaunchEx;
// ordinary launches keep using cuda.Arg and friends.

// Arg is one kernel parameter: the bytes the driver copies into parameter
// space.
type Arg struct{ data []byte }

// argBytes returns an 8-byte-aligned buffer of n bytes. make([]byte, n)
// promises no alignment, so the storage is a []uint64 reinterpreted.
func argBytes(n int) []byte {
	w := make([]uint64, (n+7)/8)
	return unsafe.Slice((*byte)(unsafe.Pointer(&w[0])), n)
}

// ArgValue passes v by value: a scalar, or a struct a kernel takes by value
// (as //cuda:grid_constant parameters are).
func ArgValue[T any](v T) Arg {
	b := argBytes(int(unsafe.Sizeof(v)))
	copy(b, unsafe.Slice((*byte)(unsafe.Pointer(&v)), len(b)))
	return Arg{data: b}
}

// ArgBuffer passes b's device pointer, for a kernel parameter of type
// *T or cuda.Buf[T].
func ArgBuffer[T cuda.Supported](b *cuda.Buffer[T]) Arg {
	return ArgValue(b.DevicePtr())
}

// ArgSliceEx passes b to a parameter of type []T, as ArgSlice does for
// gocudrv's own launches.
func ArgSliceEx[T cuda.Supported](b *cuda.Buffer[T]) Arg {
	return ArgValue(goSlice{ptr: uintptr(b.DevicePtr()), len: int64(b.Len()), cap: int64(b.Len())})
}

// ArgBytes passes raw bytes, for a parameter this package has no constructor
// for.
func ArgBytes(p []byte) Arg {
	b := argBytes(len(p))
	copy(b, p)
	return Arg{data: b}
}

// ArgEx passes the descriptor to a LaunchEx kernel, as Arg does for
// gocudrv's launches.
func (m *TensorMap) ArgEx() Arg { return ArgBytes(m.Bytes()) }

// packedArgs is the void** the driver reads parameter values through.
type packedArgs struct {
	ptrs []unsafe.Pointer
	keep []Arg
}

func pack(args []Arg) *packedArgs {
	p := &packedArgs{keep: args}
	if len(args) == 0 {
		return p
	}
	p.ptrs = make([]unsafe.Pointer, len(args))
	for i, a := range args {
		p.ptrs[i] = unsafe.Pointer(&a.data[0])
	}
	return p
}

func (p *packedArgs) params() *unsafe.Pointer {
	if len(p.ptrs) == 0 {
		return nil
	}
	return &p.ptrs[0]
}

// keepAlive holds the argument storage until the driver has copied it, which
// cuLaunchKernelEx does before it returns.
func (p *packedArgs) keepAlive() {
	runtime.KeepAlive(p.keep)
	runtime.KeepAlive(p.ptrs)
}
