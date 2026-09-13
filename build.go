// Package cudair compiles a Go package of kernels to PTX.
//
// The same pipeline is available from Go ([Build], [BuildFile]) and from the
// gocuda command (`gocuda build`), so a program can compile its kernels when
// it starts instead of shipping a pre-built .ptx:
//
//	res, err := cudair.Build("example.com/myapp/kernels", nil)
//	mod, err := ctx.LoadModule(res.PTX)   // gocudrv
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
package cudair

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
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
	reGlobal = regexp.MustCompile(`^@("[^"]+"|[^ ]+) = (.*?)global (%"github.com/mehdi-shokohi/cuda-ir.go/cuda\.Shared\[.*?\]") `)
)

const libdevicePrefix = "__nv_"

// Options control [Build]. The zero value matches `gocuda build` defaults.
type Options struct {
	// Dir is the directory the package argument is resolved in; it must be
	// inside the Go module that contains the kernels. Default: the current
	// directory.
	Dir string
	// SM is the compute capability passed to llc and ptxas (default sm_80).
	// The CUDA driver JIT adapts the PTX to the GPU that is present.
	SM string
	// Kernels names the functions that become kernels. Default: every
	// exported function of the package that returns nothing.
	Kernels []string
	// Opt is the LLVM optimisation level (default "2"; "0" skips opt).
	Opt string
	// NoCheck skips verifying the PTX with ptxas (when it is installed).
	NoCheck bool
	// WorkDir, when set, receives the intermediate .ll files and they are
	// kept. Default: a temporary directory that is removed afterwards.
	WorkDir string
	// Log, when set, receives every command that is run.
	Log io.Writer
}

// Result is a compiled kernel package.
type Result struct {
	Package string   // import path of the compiled package
	PTX     []byte   // PTX text, ready for cuModuleLoadData / ctx.LoadModule
	Kernels []string // .entry names, in definition order
}

// Build compiles the Go package pkg (an import path or a ./relative path)
// to PTX. Every exported function of the package that returns nothing
// becomes a kernel unless opts.Kernels says otherwise. nil opts means
// defaults.
func Build(pkg string, opts *Options) (*Result, error) {
	b, err := newBuilder(opts)
	if err != nil {
		return nil, err
	}
	defer b.cleanup()
	return b.build(pkg)
}

// BuildFile is [Build] followed by writing the PTX to out. An empty out
// means <pkg>.ptx in the current directory.
func BuildFile(pkg, out string, opts *Options) (*Result, error) {
	b, err := newBuilder(opts)
	if err != nil {
		return nil, err
	}
	defer b.cleanup()
	res, err := b.build(pkg)
	if err != nil {
		return nil, err
	}
	if out == "" {
		out = path.Base(res.Package) + ".ptx"
	}
	if err := os.WriteFile(out, res.PTX, 0o644); err != nil {
		return nil, err
	}
	return res, nil
}

type builder struct {
	Options
	llgen    string
	tempDirs []string // module copies made for read-only packages
	workTemp bool     // WorkDir is ours to delete
}

func newBuilder(opts *Options) (*builder, error) {
	b := &builder{}
	if opts != nil {
		b.Options = *opts
	}
	if b.SM == "" {
		b.SM = "sm_80"
	}
	if b.Opt == "" {
		b.Opt = "2"
	}
	var err error
	if b.llgen, err = exec.LookPath("llgen"); err != nil {
		return nil, errors.New("llgen not found in PATH (go install ./chore/llgen in the llgo repo)")
	}
	if b.WorkDir == "" {
		if b.WorkDir, err = os.MkdirTemp("", "cudair-*"); err != nil {
			return nil, err
		}
		b.workTemp = true
	}
	return b, nil
}

func (b *builder) cleanup() {
	for _, d := range b.tempDirs {
		os.RemoveAll(d)
	}
	if b.workTemp {
		os.RemoveAll(b.WorkDir)
	}
}

func (b *builder) build(pkg string) (*Result, error) {
	rootPkg, pkgs, err := b.modulePackages(pkg)
	if err != nil {
		return nil, err
	}
	prefix := filepath.Join(b.WorkDir, path.Base(rootPkg))

	// 1. llgen each package
	var lls []string
	for _, p := range pkgs {
		ir, err := b.genIR(p)
		if err != nil {
			return nil, err
		}
		dst := fmt.Sprintf("%s.%s.ll", prefix, mangler{}.mangle(p.ImportPath))
		if err := os.WriteFile(dst, ir, 0o644); err != nil {
			return nil, err
		}
		lls = append(lls, dst)
	}
	// 2. link
	linked := prefix + ".linked.ll"
	if err := b.llvm("llvm-link", append(lls, "-S", "-o", linked)...); err != nil {
		return nil, err
	}
	// 3. fixups
	src, err := os.ReadFile(linked)
	if err != nil {
		return nil, err
	}
	want := map[string]bool{}
	for _, k := range b.Kernels {
		if k != "" {
			want[k] = true
		}
	}
	fixed, found, needLibdevice, err := fixup(src, rootPkg, want)
	if err != nil {
		return nil, err
	}
	if len(found) == 0 {
		return nil, errors.New("no kernels: export a func that returns nothing, or name the kernels")
	}
	fixedFile := prefix + ".nvptx.ll"
	if err := os.WriteFile(fixedFile, fixed, 0o644); err != nil {
		return nil, err
	}
	// 3b. libdevice: pull in only the __nv_* functions used, as internal symbols
	if needLibdevice {
		lib := libdevicePath()
		if lib == "" {
			return nil, errors.New("kernel uses libdevice math (__nv_*) but libdevice.10.bc was not found; set LIBDEVICE or CUDA_HOME")
		}
		withLib := prefix + ".libdevice.ll"
		if err := b.llvm("llvm-link", fixedFile, "--only-needed", "--internalize", lib, "-S", "-o", withLib); err != nil {
			return nil, err
		}
		fixedFile = withLib
	}
	// 4. opt
	final := fixedFile
	if b.Opt != "0" {
		final = prefix + ".opt.ll"
		// internalize everything but the kernels so unused device functions
		// are dropped and the rest can be inlined; after inlining, turn
		// generic loads/stores on shared/global memory into
		// address-space-specific ones (ld.shared, ld.global)
		if err := b.llvm("opt",
			"-internalize-public-api-list="+strings.Join(found, ","),
			"-passes=internalize,globaldce,default<O"+b.Opt+">,infer-address-spaces,instcombine,simplifycfg",
			fixedFile, "-S", "-o", final); err != nil {
			return nil, err
		}
	}
	// 5. llc
	ptxFile := prefix + ".ptx"
	if err := b.llvm("llc", "-march=nvptx64", "-mcpu="+b.SM, final, "-o", ptxFile); err != nil {
		return nil, err
	}
	// 6. ptxas check
	if !b.NoCheck {
		if ptxas, err := exec.LookPath("ptxas"); err == nil {
			if err := b.run("", ptxas, "-arch="+b.SM, ptxFile, "-o", os.DevNull); err != nil {
				return nil, err
			}
		}
	}
	ptx, err := os.ReadFile(ptxFile)
	if err != nil {
		return nil, err
	}
	return &Result{Package: rootPkg, PTX: ptx, Kernels: found}, nil
}

// llvmTool finds e.g. "llc" as $LLVM_SUFFIX-suffixed, "-22", or bare.
func llvmTool(name string) string {
	for _, suf := range []string{os.Getenv("LLVM_SUFFIX"), "-22", ""} {
		if p, err := exec.LookPath(name + suf); err == nil {
			return p
		}
	}
	return ""
}

func (b *builder) llvm(tool string, args ...string) error {
	p := llvmTool(tool)
	if p == "" {
		return fmt.Errorf("%s not found in PATH (install llvm-22 or set LLVM_SUFFIX)", tool)
	}
	return b.run("", p, args...)
}

// run runs a command in dir ("" = current directory); its stderr is part
// of the returned error.
func (b *builder) run(dir, name string, args ...string) error {
	_, err := b.output(dir, name, args...)
	return err
}

func (b *builder) output(dir, name string, args ...string) ([]byte, error) {
	if b.Log != nil {
		if dir != "" {
			fmt.Fprintf(b.Log, "+ (cd %s) ", dir)
		} else {
			fmt.Fprint(b.Log, "+ ")
		}
		fmt.Fprintln(b.Log, name, strings.Join(args, " "))
	}
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	var out, stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			return nil, fmt.Errorf("%s failed: %v", filepath.Base(name), err)
		}
		return nil, fmt.Errorf("%s failed: %v\n%s", filepath.Base(name), err, msg)
	}
	return out.Bytes(), nil
}

type pkgInfo struct{ ImportPath, Dir, ModDir string }

// modulePackages lists the root package and its non-stdlib dependencies.
func (b *builder) modulePackages(pkg string) (root string, pkgs []pkgInfo, err error) {
	out, err := b.output(b.Dir, "go", "list", "-deps", "-f", "{{if not .Standard}}{{.ImportPath}}\t{{.Dir}}\t{{if .Module}}{{.Module.Dir}}{{end}}{{end}}", pkg)
	if err != nil {
		return "", nil, err
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		f := strings.SplitN(line, "\t", 3)
		pkgs = append(pkgs, pkgInfo{f[0], f[1], f[2]})
	}
	if len(pkgs) == 0 {
		return "", nil, fmt.Errorf("no packages for %s", pkg)
	}
	return pkgs[len(pkgs)-1].ImportPath, pkgs, nil // -deps lists the root last
}

// writable reports whether files can be created in dir.
func writable(dir string) bool {
	f, err := os.CreateTemp(dir, ".cudair-*")
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
func (b *builder) genIR(p pkgInfo) ([]byte, error) {
	dir := p.Dir
	if !writable(dir) {
		if p.ModDir == "" {
			return nil, fmt.Errorf("%s: directory is not writable and not in a module", p.ImportPath)
		}
		rel, err := filepath.Rel(p.ModDir, p.Dir)
		if err != nil {
			return nil, err
		}
		tmp, err := os.MkdirTemp("", "cudair-mod-*")
		if err != nil {
			return nil, err
		}
		b.tempDirs = append(b.tempDirs, tmp)
		if err := os.CopyFS(tmp, os.DirFS(p.ModDir)); err != nil {
			return nil, fmt.Errorf("copy module %s: %v", p.ModDir, err)
		}
		dir = filepath.Join(tmp, rel)
	}
	// run inside the package so llgen resolves it within its own module
	if err := b.run(dir, b.llgen, "-phase=pre-abi", "."); err != nil {
		return nil, err
	}
	gen := filepath.Join(dir, "llgo_autogen.ll")
	ir, err := os.ReadFile(gen)
	if err != nil {
		return nil, err
	}
	os.Remove(gen)
	return ir, nil
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

// scanLines calls fn for each line of src until fn returns an error.
func scanLines(src []byte, fn func(line string) error) error {
	sc := bufio.NewScanner(bytes.NewReader(src))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		if err := fn(sc.Text()); err != nil {
			return err
		}
	}
	return sc.Err()
}

// findKernels selects kernel functions: the named ones, or if none are
// named, every exported function of rootPkg that returns nothing.
func findKernels(src []byte, rootPkg string, want map[string]bool) (map[string]string, error) {
	kernels := map[string]string{} // full llgo name -> short name
	err := scanLines(src, func(line string) error {
		m := reDefine.FindStringSubmatch(line)
		if m == nil {
			return nil
		}
		name := strings.Trim(m[3], `"`)
		pkg, short := splitName(name)
		if len(want) > 0 {
			if !want[short] && !want[name] {
				return nil
			}
			if m[2] != "void" {
				return fmt.Errorf("kernel %s must not return a value (returns %s)", name, m[2])
			}
			delete(want, short)
			delete(want, name)
		} else if pkg != rootPkg || !isExported(short) || m[2] != "void" {
			return nil
		}
		if prev, dup := kernels[short]; dup {
			return fmt.Errorf("kernel name %s is ambiguous: %s and %s", short, prev, name)
		}
		kernels[name] = short
		return nil
	})
	if err != nil {
		return nil, err
	}
	for k := range want {
		return nil, fmt.Errorf("kernel %q not found", k)
	}
	return kernels, nil
}

// findShared returns the package-level variables of type cuda.Shared[T];
// they become addrspace(3) (per-block shared memory) globals.
func findShared(src []byte) map[string]bool {
	shared := map[string]bool{}
	scanLines(src, func(line string) error {
		if m := reGlobal.FindStringSubmatch(line); m != nil {
			shared[strings.Trim(m[1], `"`)] = true
		}
		return nil
	})
	return shared
}

// fixup rewrites linked llgo IR for the NVPTX backend. It returns the
// rewritten IR, the kernel names, and whether libdevice is needed.
func fixup(src []byte, rootPkg string, want map[string]bool) ([]byte, []string, bool, error) {
	kernels, err := findKernels(src, rootPkg, want)
	if err != nil {
		return nil, nil, false, err
	}
	shared := findShared(src)
	m := mangler{kernels}
	var out, found []string
	needTrap, needLibdevice := false, false
	err = scanLines(src, func(line string) error {
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
					return fmt.Errorf("unsupported on GPU: code depends on llgo runtime function %q (returns %s)", name, ret)
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
				return fmt.Errorf("unsupported on GPU: call to %s — the Go standard library is not available in kernels; use package cuda (math -> cuda.Sqrt/Sin/..., sync/atomic is fine)", name)
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
		return nil
	})
	if err != nil {
		return nil, nil, false, err
	}
	if needTrap {
		out = append(out, "", "declare void @llvm.trap()")
	}
	return []byte(strings.Join(out, "\n") + "\n"), found, needLibdevice, nil
}

// libdevicePath finds NVIDIA's libdevice bitcode (math functions), or "".
func libdevicePath() string {
	if p := os.Getenv("LIBDEVICE"); p != "" {
		return p
	}
	var roots []string
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
