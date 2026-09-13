// gocuda build compiles a Go package of kernels to PTX.
//
//	gocuda build [-o out.ptx] [-sm sm_80] [-kernel A,B] [-O 2] [-keep] ./pkg
//	gocuda doctor            # check that every tool gocuda needs is installed
//
// Pipeline: llgen (llgo) for the package and its in-module dependencies ->
// llvm-link -> IR fixups for NVPTX -> opt -> llc -> [ptxas check].
//
// Fixups applied to the IR (llgo has no nvptx target, so its output is not
// valid for llc/ptxas as-is):
//  1. target triple / datalayout -> nvptx64-nvidia-cuda
//  2. global symbols mangled to PTX's charset [A-Za-z0-9_$]
//     ("pkg.Func" -> "pkg_Func"; llc aborts on '/', ptxas on '.')
//  3. kernels marked `ptx_kernel` so they become `.entry` (launchable).
//     Default: every exported function of the root package returning nothing.
//  4. `declare` of void llgo runtime helpers (PanicIndex, AssertNilDeref...)
//     replaced by bodies that `trap` — Go panics become device traps.
//     A runtime helper that returns a value is reported as unsupported.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

const nvptxDL = `target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"`
const nvptxTriple = `target triple = "nvptx64-nvidia-cuda"`
const runtimePrefix = "github.com/xgo-dev/llgo/runtime/"

var (
	reQuoted = regexp.MustCompile(`@"([^"]*)"`)
	reBare   = regexp.MustCompile(`@([A-Za-z0-9_.$]+)`)
	reDefine = regexp.MustCompile(`^define (.*?)(void|[^ ]+) @("[^"]+"|[^(]+)\(`)
	reDecl   = regexp.MustCompile(`^declare (void|[^ ]+) @("[^"]+"|[^(]+)\(([^)]*)\)(.*)$`)
	reGlobal = regexp.MustCompile(`^@("[^"]+"|[^ ]+) = (.*?)global (%"github.com/mehdi-shokohi/gocuda/cuda\.Shared\[.*?\]") `)
)

const sharedTypePrefix = `%"github.com/mehdi-shokohi/gocuda/cuda.Shared[`
const libdevicePrefix = "__nv_"

// temps are intermediate files removed on exit unless -keep is given;
// tempDirs are always removed.
var temps, tempDirs []string
var keepTemps bool

func cleanup() {
	for _, d := range tempDirs {
		os.RemoveAll(d)
	}
	if keepTemps {
		return
	}
	for _, f := range temps {
		os.Remove(f)
	}
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "gocuda: "+format+"\n", a...)
	cleanup()
	os.Exit(1)
}

// llvmTool finds e.g. "llc" as $LLVM_SUFFIX-suffixed, "-22", or bare.
func llvmTool(name string) string {
	for _, suf := range []string{os.Getenv("LLVM_SUFFIX"), "-22", ""} {
		if p, err := exec.LookPath(name + suf); err == nil {
			return p
		}
	}
	fatal("%s not found in PATH (install llvm-22 or set LLVM_SUFFIX)", name)
	return ""
}

func run(verbose bool, name string, args ...string) []byte {
	return runIn("", verbose, name, args...)
}

// runIn runs a command in dir ("" = current directory) and returns its stdout.
func runIn(dir string, verbose bool, name string, args ...string) []byte {
	if verbose {
		if dir != "" {
			fmt.Fprintf(os.Stderr, "+ (cd %s) ", dir)
		} else {
			fmt.Fprint(os.Stderr, "+ ")
		}
		fmt.Fprintln(os.Stderr, name, strings.Join(args, " "))
	}
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fatal("%s failed: %v", name, err)
	}
	return out.Bytes()
}

type pkgInfo struct{ ImportPath, Dir, ModDir string }

// modulePackages lists the root package and its non-stdlib dependencies.
func modulePackages(pkg string) (root string, pkgs []pkgInfo) {
	out := run(false, "go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}\t{{.Dir}}\t{{if .Module}}{{.Module.Dir}}{{end}}{{end}}", pkg)
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		f := strings.SplitN(line, "\t", 3)
		pkgs = append(pkgs, pkgInfo{f[0], f[1], f[2]})
	}
	if len(pkgs) == 0 {
		fatal("no packages for %s", pkg)
	}
	return pkgs[len(pkgs)-1].ImportPath, pkgs // -deps lists the root last
}

// writable reports whether files can be created in dir.
func writable(dir string) bool {
	f, err := os.CreateTemp(dir, ".gocuda-*")
	if err != nil {
		return false
	}
	f.Close()
	os.Remove(f.Name())
	return true
}

// genIR runs llgen on a package and returns its LLVM IR text. llgen always
// writes <pkgdir>/llgo_autogen.ll, so a package in the read-only module
// cache is compiled from a temporary copy of its module.
func genIR(llgen string, p pkgInfo, verbose bool) []byte {
	dir := p.Dir
	if !writable(dir) {
		if p.ModDir == "" {
			fatal("%s: directory is not writable and not in a module", p.ImportPath)
		}
		rel, err := filepath.Rel(p.ModDir, p.Dir)
		if err != nil {
			fatal("%v", err)
		}
		tmp, err := os.MkdirTemp("", "gocuda-mod-*")
		if err != nil {
			fatal("%v", err)
		}
		tempDirs = append(tempDirs, tmp)
		if err := os.CopyFS(tmp, os.DirFS(p.ModDir)); err != nil {
			fatal("copy module %s: %v", p.ModDir, err)
		}
		dir = filepath.Join(tmp, rel)
	}
	// run inside the package so llgen resolves it within its own module
	runIn(dir, verbose, llgen, "-phase=pre-abi", ".")
	gen := filepath.Join(dir, "llgo_autogen.ll")
	ir, err := os.ReadFile(gen)
	if err != nil {
		fatal("%v", err)
	}
	os.Remove(gen)
	return ir
}

// splitName splits "pkg/path.Name" into package path and short name.
func splitName(name string) (pkg, short string) {
	if i := strings.LastIndex(name, "."); i >= 0 {
		return name[:i], name[i+1:]
	}
	return "", name
}

// mangler maps llgo symbol names to PTX identifiers ([A-Za-z0-9_$]).
// Kernels keep their short Go name so the host can look them up by it.
type mangler struct{ kernels map[string]string }

func (m mangler) mangle(name string) string {
	if strings.HasPrefix(name, "llvm.") { // intrinsics are not symbols
		return name
	}
	if short, ok := m.kernels[name]; ok {
		return short
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '$':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func isExported(name string) bool { return name != "" && name[0] >= 'A' && name[0] <= 'Z' }

func scanLines(src []byte, fn func(line string)) {
	sc := bufio.NewScanner(bytes.NewReader(src))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		fn(sc.Text())
	}
}

// findKernels selects kernel functions: the named ones, or if none are
// named, every exported function of rootPkg that returns nothing.
func findKernels(src []byte, rootPkg string, want map[string]bool) map[string]string {
	kernels := map[string]string{} // full llgo name -> short name
	scanLines(src, func(line string) {
		m := reDefine.FindStringSubmatch(line)
		if m == nil {
			return
		}
		name := strings.Trim(m[3], `"`)
		pkg, short := splitName(name)
		if len(want) > 0 {
			if !want[short] && !want[name] {
				return
			}
			if m[2] != "void" {
				fatal("kernel %s must not return a value (returns %s)", name, m[2])
			}
			delete(want, short)
			delete(want, name)
		} else if pkg != rootPkg || !isExported(short) || m[2] != "void" {
			return
		}
		if prev, dup := kernels[short]; dup {
			fatal("kernel name %s is ambiguous: %s and %s", short, prev, name)
		}
		kernels[name] = short
	})
	for k := range want {
		fatal("kernel %q not found", k)
	}
	return kernels
}

// findShared returns the package-level variables of type cuda.Shared[T];
// they become addrspace(3) (per-block shared memory) globals.
func findShared(src []byte) map[string]bool {
	shared := map[string]bool{}
	scanLines(src, func(line string) {
		if m := reGlobal.FindStringSubmatch(line); m != nil {
			shared[strings.Trim(m[1], `"`)] = true
		}
	})
	return shared
}

// fixup rewrites linked llgo IR for the NVPTX backend. It returns the
// rewritten IR, the kernel names, and whether libdevice is needed.
func fixup(src []byte, rootPkg string, want map[string]bool) ([]byte, []string, bool) {
	kernels := findKernels(src, rootPkg, want)
	shared := findShared(src)
	m := mangler{kernels}
	var out, found []string
	needTrap, needLibdevice := false, false
	scanLines(src, func(line string) {
		switch {
		case strings.HasPrefix(line, "target datalayout"):
			line = nvptxDL
		case strings.HasPrefix(line, "target triple"):
			line = nvptxTriple
		}
		if d := reDecl.FindStringSubmatch(line); d != nil {
			ret, name, params, attrs := d[1], strings.Trim(d[2], `"`), d[3], d[4]
			_, short := splitName(name)
			switch {
			case strings.HasPrefix(name, runtimePrefix):
				if ret != "void" {
					fatal("unsupported on GPU: code depends on llgo runtime function %q (returns %s)", name, ret)
				}
				needTrap = true
				line = fmt.Sprintf("define void @%s(%s)%s {\n  call void @llvm.trap()\n  unreachable\n}", m.mangle(name), params, attrs)
			case short == "init" && ret == "void" && params == "":
				// package initialisers of imported packages never run on the GPU
				line = fmt.Sprintf("define void @%s()%s {\n  ret void\n}", m.mangle(name), attrs)
			case strings.HasPrefix(name, libdevicePrefix):
				needLibdevice = true
			case strings.HasPrefix(name, "llvm."):
			default:
				fatal("unsupported on GPU: call to %s — the Go standard library is not available in kernels; use package cuda (math -> cuda.Sqrt/Sin/..., sync/atomic is fine)", name)
			}
		}
		if d := reDefine.FindStringSubmatch(line); d != nil {
			// llgo emits generic instantiations as `linkonce`, which LLVM may
			// not inline; every instantiation is identical, so ODR is safe.
			if strings.HasPrefix(d[1], "linkonce ") {
				line = strings.Replace(line, "define linkonce ", "define linkonce_odr ", 1)
			}
			if short, ok := kernels[strings.Trim(d[3], `"`)]; ok {
				line = strings.Replace(line, "define "+d[1], "define "+d[1]+"ptx_kernel ", 1)
				found = append(found, short)
			}
		}
		// cuda.Shared[T] globals: definition moves to addrspace(3); every use
		// goes through an addrspacecast back to a generic pointer.
		if g := reGlobal.FindStringSubmatch(line); g != nil {
			line = strings.Replace(line, "global "+g[3], "addrspace(3) global "+g[3], 1)
			line = strings.Replace(line, "zeroinitializer", "undef", 1)
		} else if len(shared) > 0 {
			line = reQuoted.ReplaceAllStringFunc(line, func(s string) string {
				name := s[2 : len(s)-1]
				if !shared[name] {
					return s
				}
				return fmt.Sprintf("addrspacecast (ptr addrspace(3) @%s to ptr)", m.mangle(name))
			})
			line = reBare.ReplaceAllStringFunc(line, func(s string) string {
				if !shared[s[1:]] {
					return s
				}
				return fmt.Sprintf("addrspacecast (ptr addrspace(3) @%s to ptr)", m.mangle(s[1:]))
			})
		}
		line = reQuoted.ReplaceAllStringFunc(line, func(s string) string { return "@" + m.mangle(s[2:len(s)-1]) })
		line = reBare.ReplaceAllStringFunc(line, func(s string) string { return "@" + m.mangle(s[1:]) })
		out = append(out, line)
	})
	if needTrap {
		out = append(out, "", "declare void @llvm.trap()")
	}
	return []byte(strings.Join(out, "\n") + "\n"), found, needLibdevice
}

// libdevicePath finds NVIDIA's libdevice bitcode (math functions).
func libdevicePath() string {
	if p := libdevicePathOrEmpty(); p != "" {
		return p
	}
	fatal("kernel uses libdevice math (__nv_*) but libdevice.10.bc was not found; set LIBDEVICE or CUDA_HOME")
	return ""
}

func libdevicePathOrEmpty() string {
	var roots []string
	if p := os.Getenv("LIBDEVICE"); p != "" {
		return p
	}
	for _, env := range []string{"CUDA_HOME", "CUDA_PATH"} {
		if p := os.Getenv(env); p != "" {
			roots = append(roots, p)
		}
	}
	roots = append(roots, "/usr/local/cuda", "/opt/cuda")
	if matches, _ := filepath.Glob("/usr/local/cuda-*"); matches != nil {
		roots = append(roots, matches...)
	}
	for _, r := range roots {
		p := filepath.Join(r, "nvvm", "libdevice", "libdevice.10.bc")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

// doctor reports the state of every external dependency.
func doctor() {
	ok := true
	report := func(good bool, name, detail, hint string) {
		mark := "ok  "
		if !good {
			mark = "MISSING"
			ok = false
		}
		fmt.Printf("%-8s %-14s %s\n", mark, name, detail)
		if !good && hint != "" {
			fmt.Printf("         -> %s\n", hint)
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
		found := ""
		for _, suf := range []string{os.Getenv("LLVM_SUFFIX"), "-22", ""} {
			if p, err := exec.LookPath(t + suf); err == nil {
				found = p
				break
			}
		}
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
	lib := ""
	func() {
		defer func() { recover() }()
		lib = libdevicePathOrEmpty()
	}()
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
		os.Exit(1)
	}
}

func main() {
	if len(os.Args) >= 2 && os.Args[1] == "doctor" {
		doctor()
		return
	}
	if len(os.Args) < 2 || os.Args[1] != "build" {
		fmt.Fprintln(os.Stderr, "usage: gocuda build [-o out.ptx] [-sm sm_80] [-kernel A,B] [-O 2] [-keep] [-v] ./pkg\n       gocuda doctor")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	outFile := fs.String("o", "", "output .ptx (default <pkg>.ptx in the current directory)")
	sm := fs.String("sm", "sm_80", "target compute capability for llc/ptxas")
	kernelList := fs.String("kernel", "", "comma-separated kernel names (default: all exported void funcs of the package)")
	optLevel := fs.String("O", "2", "opt level passed to opt (0 to skip)")
	keep := fs.Bool("keep", false, "keep intermediate .ll files next to the output")
	check := fs.Bool("check", true, "run ptxas on the result if available")
	verbose := fs.Bool("v", false, "print commands")
	fs.Parse(os.Args[2:])
	keepTemps = *keep
	if fs.NArg() != 1 {
		fatal("exactly one package argument is required")
	}
	if os.Getenv("LLGO_ROOT") == "" {
		fmt.Fprintln(os.Stderr, "gocuda: warning: LLGO_ROOT is not set; llgen may fail")
	}
	llgen, err := exec.LookPath("llgen")
	if err != nil {
		fatal("llgen not found in PATH (go install ./chore/llgen in the llgo repo)")
	}

	rootPkg, pkgs := modulePackages(fs.Arg(0))
	base := filepath.Base(rootPkg)
	if *outFile == "" {
		*outFile = base + ".ptx"
	}
	work := filepath.Dir(*outFile)
	prefix := filepath.Join(work, base)

	// 1. llgen each package
	var lls []string
	for _, p := range pkgs {
		dst := fmt.Sprintf("%s.%s.ll", prefix, mangler{}.mangle(p.ImportPath))
		if err := os.WriteFile(dst, genIR(llgen, p, *verbose), 0o644); err != nil {
			fatal("%v", err)
		}
		lls = append(lls, dst)
		temps = append(temps, dst)
	}
	// 2. link
	linked := prefix + ".linked.ll"
	temps = append(temps, linked)
	run(*verbose, llvmTool("llvm-link"), append(lls, "-S", "-o", linked)...)
	// 3. fixups
	src, err := os.ReadFile(linked)
	if err != nil {
		fatal("%v", err)
	}
	kernels := map[string]bool{}
	for _, k := range strings.Split(*kernelList, ",") {
		if k != "" {
			kernels[k] = true
		}
	}
	fixed, found, needLibdevice := fixup(src, rootPkg, kernels)
	if len(found) == 0 {
		fatal("no kernels: export a func that returns nothing, or pass -kernel")
	}
	fixedFile := prefix + ".nvptx.ll"
	temps = append(temps, fixedFile)
	if err := os.WriteFile(fixedFile, fixed, 0o644); err != nil {
		fatal("%v", err)
	}
	// 3b. libdevice: pull in only the __nv_* functions used, as internal symbols
	if needLibdevice {
		withLib := prefix + ".libdevice.ll"
		temps = append(temps, withLib)
		run(*verbose, llvmTool("llvm-link"), fixedFile, "--only-needed", "--internalize", libdevicePath(), "-S", "-o", withLib)
		fixedFile = withLib
	}
	// 4. opt
	final := fixedFile
	if *optLevel != "0" {
		final = prefix + ".opt.ll"
		temps = append(temps, final)
		// internalize everything but the kernels so unused device functions
		// are dropped and the rest can be inlined; after inlining, turn
		// generic loads/stores on shared/global memory into
		// address-space-specific ones (ld.shared, ld.global)
		run(*verbose, llvmTool("opt"),
			"-internalize-public-api-list="+strings.Join(found, ","),
			"-passes=internalize,globaldce,default<O"+*optLevel+">,infer-address-spaces,instcombine,simplifycfg",
			fixedFile, "-S", "-o", final)
	}
	// 5. llc
	run(*verbose, llvmTool("llc"), "-march=nvptx64", "-mcpu="+*sm, final, "-o", *outFile)
	// 6. ptxas check
	if *check {
		if ptxas, err := exec.LookPath("ptxas"); err == nil {
			run(*verbose, ptxas, "-arch="+*sm, *outFile, "-o", os.DevNull)
		}
	}
	cleanup()
	fmt.Printf("%s: kernels %s\n", *outFile, strings.Join(found, ", "))
}
