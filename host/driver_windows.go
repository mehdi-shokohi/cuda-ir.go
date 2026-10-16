//go:build windows

package host

import (
	"runtime"
	"syscall"
	"unsafe"
)

// loadLibrarySearchSystem32 is LOAD_LIBRARY_SEARCH_SYSTEM32: a bare library
// name resolves from %windir%\System32 only, never the application directory
// or PATH, which blocks DLL preloading. The driver installs nvcuda.dll there.
// gocudrv opens the driver the same way.
const loadLibrarySearchSystem32 = 0x00000800

var (
	kernel32           = syscall.NewLazyDLL("kernel32.dll")
	procLoadLibraryExW = kernel32.NewProc("LoadLibraryExW")
)

// openDriver returns a handle to nvcuda.dll. LoadLibraryEx refcounts, so this
// is the handle gocudrv already holds, not a second mapping.
func openDriver() (uintptr, error) {
	p, err := syscall.UTF16PtrFromString("nvcuda.dll")
	if err != nil {
		return 0, err
	}
	h, _, callErr := procLoadLibraryExW.Call(uintptr(unsafe.Pointer(p)), 0, loadLibrarySearchSystem32)
	runtime.KeepAlive(p)
	if h == 0 {
		return 0, callErr
	}
	return h, nil
}
