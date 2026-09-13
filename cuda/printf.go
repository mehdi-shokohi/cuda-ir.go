package cuda

import "unsafe"

// Device printf. The output is buffered by the driver and written to the
// process's stdout when the host synchronises (cuCtxSynchronize & co).
//
//	cuda.Printf("thread %d: x=%f\n", cuda.Args().Int(i).Float(x))
//
// The C varargs layout is built by Args: ints take 4 bytes, int64 and
// floats 8 (float32 is promoted to double, as C does), so use %d, %u, %x
// for Int/Uint, %lld/%llu for Int64/Uint64, %f/%g/%e for Float, %c, %p.

//go:linkname vprintf vprintf
func vprintf(format *byte, args unsafe.Pointer) int32

const maxPrintfArgs = 16

// PrintfArgs packs printf arguments. The zero value is empty.
type PrintfArgs struct {
	buf [maxPrintfArgs * 8]byte
	n   int32
}

// Args starts an argument list.
func Args() *PrintfArgs { return &PrintfArgs{} }

func (a *PrintfArgs) put4(v uint32) *PrintfArgs {
	if a.n+4 <= int32(len(a.buf)) {
		*(*uint32)(unsafe.Pointer(&a.buf[a.n])) = v
		a.n += 4
	}
	return a
}

func (a *PrintfArgs) put8(v uint64) *PrintfArgs {
	a.n = (a.n + 7) &^ 7
	if a.n+8 <= int32(len(a.buf)) {
		*(*uint64)(unsafe.Pointer(&a.buf[a.n])) = v
		a.n += 8
	}
	return a
}

// Int adds an int (%d). Uint adds an unsigned (%u, %x).
func (a *PrintfArgs) Int(v int32) *PrintfArgs   { return a.put4(uint32(v)) }
func (a *PrintfArgs) Uint(v uint32) *PrintfArgs { return a.put4(v) }

// Int64 / Uint64 add a long long (%lld / %llu).
func (a *PrintfArgs) Int64(v int64) *PrintfArgs   { return a.put8(uint64(v)) }
func (a *PrintfArgs) Uint64(v uint64) *PrintfArgs { return a.put8(v) }

// Float adds a float32 as a double (%f, %g, %e). Float64 likewise.
func (a *PrintfArgs) Float(v float32) *PrintfArgs   { return a.put8(Float64Bits(float64(v))) }
func (a *PrintfArgs) Float64(v float64) *PrintfArgs { return a.put8(Float64Bits(v)) }

// Ptr adds a pointer (%p).
func (a *PrintfArgs) Ptr(p unsafe.Pointer) *PrintfArgs { return a.put8(uint64(uintptr(p))) }

// Printf is the device printf. format is a Go string (no NUL needed) of at
// most 255 bytes; args may be nil when there are none.
func Printf(format string, args *PrintfArgs) {
	var f [256]byte
	n := len(format)
	if n > 255 {
		n = 255
	}
	for i := 0; i < n; i++ {
		f[i] = format[i]
	}
	f[n] = 0
	var p unsafe.Pointer
	if args != nil {
		p = unsafe.Pointer(&args.buf[0])
	}
	vprintf(&f[0], p)
}
