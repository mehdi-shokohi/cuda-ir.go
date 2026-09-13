// Package host has launch-side helpers for kernels compiled by cuda-ir.go,
// on top of gocudrv.
package host

import (
	"unsafe"

	"github.com/eitamring/gocudrv/cuda"
)

// goSlice is the memory layout of a Go slice header (what a []T kernel
// parameter is: a 24-byte .param).
type goSlice struct {
	ptr      uintptr
	len, cap int64
}

// ArgSlice passes a device buffer to a kernel parameter of type []T: the
// kernel sees a slice of len(b) elements, indexed with Go bounds checks.
func ArgSlice[T cuda.Supported](b *cuda.Buffer[T]) cuda.KernelArg {
	s := &goSlice{ptr: uintptr(b.DevicePtr()), len: int64(b.Len()), cap: int64(b.Len())}
	return cuda.ArgRaw(unsafe.Pointer(s), int(unsafe.Sizeof(*s)))
}
