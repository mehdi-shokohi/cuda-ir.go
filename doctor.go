package cudair

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

// Doctor writes the state of every external dependency to w, with a fix
// hint for each missing one, and returns an error if anything is missing.
func Doctor(w io.Writer) error {
	ok := true
	report := func(good bool, name, detail, hint string) {
		mark := "ok  "
		if !good {
			mark = "MISSING"
			ok = false
		}
		fmt.Fprintf(w, "%-8s %-14s %s\n", mark, name, detail)
		if !good && hint != "" {
			fmt.Fprintf(w, "         -> %s\n", hint)
		}
	}
	if p, err := exec.LookPath("llgen"); err == nil {
		report(true, "llgen", p, "")
	} else {
		report(false, "llgen", "not in PATH", "git clone https://github.com/xgo-dev/llgo && cd llgo && go install ./chore/llgen")
	}
	if r := os.Getenv("LLGO_ROOT"); r != "" {
		_, err := os.Stat(filepath.Join(r, "runtime", "go.mod"))
		report(err == nil, "LLGO_ROOT", r, "LLGO_ROOT must point at the llgo source checkout")
	} else {
		report(false, "LLGO_ROOT", "not set", "export LLGO_ROOT=/path/to/llgo (the checkout llgen was built from)")
	}
	llc := ""
	for _, t := range []string{"llvm-link", "opt", "llc"} {
		found := llvmTool(t)
		report(found != "", t, found, "install LLVM 22 (https://apt.llvm.org) or set LLVM_SUFFIX, e.g. LLVM_SUFFIX=-22")
		if t == "llc" {
			llc = found
		}
	}
	if llc != "" {
		out, _ := exec.Command(llc, "--version").Output()
		report(bytes.Contains(out, []byte("nvptx")), "nvptx backend", llc+" has the nvptx64 target", "this LLVM build lacks the NVPTX backend")
	}
	if p, err := exec.LookPath("ptxas"); err == nil {
		report(true, "ptxas", p+" (optional: build-time PTX check)", "")
	} else {
		report(true, "ptxas", "not found (optional; -check=false)", "")
	}
	lib := libdevicePath()
	report(lib != "", "libdevice", lib, "install the CUDA toolkit or set LIBDEVICE=/path/to/libdevice.10.bc (needed for cuda.Sin/Exp/...)")
	drv := false
	for _, p := range []string{"/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1", "/usr/lib/libcuda.so.1", "/usr/lib/aarch64-linux-gnu/libcuda.so.1"} {
		if _, err := os.Stat(p); err == nil {
			drv = true
			break
		}
	}
	report(drv, "libcuda.so.1", "NVIDIA driver (needed at run time only)", "install the NVIDIA driver")
	if !ok {
		return errors.New("some dependencies are missing")
	}
	return nil
}
