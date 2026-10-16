// driver.go binds the CUDA driver entry points gocudrv v0.3.2 does not.
//
// gocudrv exposes its loaded driver (Context.Driver), its raw context
// (Context.Raw) and its device pointers (Buffer.DevicePtr) precisely so a
// sibling module can issue driver calls against the context it already
// owns. Everything here takes that route: the context, device, streams and
// buffers stay gocudrv's, and only the handful of symbols missing from its
// Driver struct are bound here. dlopen refcounts, so opening libcuda a
// second time hands back the same handle gocudrv holds — no second mapping,
// and no second CUDA context.
//
// Still pure Go: the binding is purego.RegisterLibFunc, so the host builds
// with CGO_ENABLED=0 like the rest of cuda-ir.go. (llgo's `//go:linkname F
// C.f` would be equivalent at the source level but needs the host compiled
// by llgo, which the device side alone requires today.)
package host

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/eitamring/gocudrv/cuda"
	"github.com/eitamring/gocudrv/cudaresult"
	"github.com/eitamring/gocudrv/cudasys"
)

// The entry points gocudrv does not bind. Their signatures mirror the C
// prototypes; purego marshals them, so no uintptr casts at the call sites.
var (
	cuLaunchKernelEx func(cfg *launchConfig, fn cudasys.CUfunction, params, extra *unsafe.Pointer) int32

	// cuTexObjectCreate is bound here rather than used through gocudrv
	// because gocudrv's CUDA_RESOURCE_DESC models only the array case of
	// the resource union: the linear case tex1Dfetch needs falls in the
	// 120 bytes it keeps as reserved padding. See resourceDesc.
	cuTexObjectCreate func(tex *cudasys.CUtexObject, res *resourceDesc, td *cudasys.CUDA_TEXTURE_DESC, viewDesc unsafe.Pointer) int32

	cuTensorMapEncodeTiled func(
		tensorMap *[16]uint64,
		dataType uint32,
		rank uint32,
		globalAddress cudasys.CUdeviceptr,
		globalDim *uint64,
		globalStrides *uint64,
		boxDim *uint32,
		elementStrides *uint32,
		interleave uint32,
		swizzle uint32,
		l2Promotion uint32,
		oobFill uint32,
	) int32
)

var (
	driverOnce sync.Once
	driverErr  error
)

// ErrUnsupportedPlatform is returned where the CUDA driver library cannot be
// opened by this package.
var ErrUnsupportedPlatform = errors.New("host: loading the CUDA driver is not supported on this platform")

// loadDriver opens the CUDA driver library and binds the extra entry points
// once. It is safe to call from any goroutine and needs no CUDA context.
func loadDriver() error {
	driverOnce.Do(func() {
		h, err := openDriver()
		if err != nil {
			driverErr = fmt.Errorf("host: open CUDA driver: %w", err)
			return
		}
		for _, s := range []struct {
			fptr any
			name string
		}{
			{&cuLaunchKernelEx, "cuLaunchKernelEx"},
			{&cuTexObjectCreate, "cuTexObjectCreate"},
			{&cuTensorMapEncodeTiled, "cuTensorMapEncodeTiled"},
		} {
			if err := register(s.fptr, h, s.name); err != nil {
				driverErr = err
				return
			}
		}
	})
	return driverErr
}

// register binds one entry point, turning purego's panic on a missing symbol
// into an error: a driver too old for one of these calls must not take down
// the process.
func register(fptr any, handle uintptr, name string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("host: bind %s: %v (CUDA driver too old?)", name, r)
		}
	}()
	purego.RegisterLibFunc(fptr, handle, name)
	return nil
}

// check turns a CUresult from one of the raw calls above into an error,
// reusing gocudrv's result type so the message matches the rest of the API.
func check(op string, r int32) error {
	if r == 0 {
		return nil
	}
	return &cudaresult.Error{Code: cudasys.CUresult(r), Op: op}
}

// OnContext runs fn with c current on a pinned OS thread.
//
// CUDA's current context is per thread, and gocudrv makes c current only on
// its own executor thread: a raw driver call from any other goroutine would
// see no context at all. OnContext pins the calling goroutine, makes c
// current, runs fn and restores the thread's previous context.
//
// fn issues work outside gocudrv's executor, so it is not serialised against
// concurrent gocudrv calls on the same context. Keep fn short, and give work
// launched inside it its own stream (or synchronize before returning).
func OnContext(c *cuda.Context, fn func() error) error {
	if c == nil {
		return errors.New("host: nil context")
	}
	drv := c.Driver()
	if drv == nil {
		return errors.New("host: context has no driver")
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	prev, err := cudaresult.CtxGetCurrent(drv)
	if err != nil {
		return err
	}
	if err := cudaresult.CtxSetCurrent(drv, c.Raw()); err != nil {
		return err
	}
	defer cudaresult.CtxSetCurrent(drv, prev)

	return fn()
}
