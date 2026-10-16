//go:build !windows

package host

import (
	"errors"
	"fmt"

	"github.com/ebitengine/purego"
)

// driverCandidates are the paths tried when opening the CUDA driver, matching
// gocudrv's own list (including the WSL2 path the Windows driver projects into
// the WSL filesystem) so this package opens the same library it did.
var driverCandidates = []string{
	"libcuda.so.1",
	"/usr/lib/x86_64-linux-gnu/libcuda.so.1",
	"/usr/lib/wsl/lib/libcuda.so.1",
}

// openDriver returns a handle to the CUDA driver library. dlopen refcounts,
// so this is the handle gocudrv already holds, not a second mapping.
func openDriver() (uintptr, error) {
	var errs []error
	for _, p := range driverCandidates {
		h, err := purego.Dlopen(p, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return h, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", p, err))
	}
	return 0, errors.Join(errs...)
}
