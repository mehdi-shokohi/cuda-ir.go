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
//  4. `declare` of llgo runtime helpers replaced: panics (PanicIndex,
//     AssertNilDeref, Panic...) by bodies that `trap`, heap allocation
//     (AllocU/AllocZ) and copy() (SliceCopy) by stack-based device
//     implementations (helpers.go). A call to any other runtime helper is
//     reported as unsupported.
//  5. cuda.Shared / DynShared / Constant globals move to their address
//     space; exported package-level variables keep their Go name.
//  6. `cudair.*` IR helpers, math/bits and math.Float32bits & co get bodies.
//  7. `//cuda:launch_bounds N [M]` / `//cuda:maxnreg N` / `//cuda:cluster_dims`
//     doc-comment directives become NVPTX function attributes;
//     `//cuda:restrict` marks the kernel's pointer parameters `noalias`,
//     `//cuda:grid_constant` passes its struct parameters `byval` in
//     parameter space (no local copy when their address is taken).
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
	"sort"
	"strings"
)

const nvptxDL = `target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"`
const nvptxTriple = `target triple = "nvptx64-nvidia-cuda"`
const runtimePrefix = "github.com/xgo-dev/llgo/runtime/"

var (
	reQuoted = regexp.MustCompile(`@"([^"]*)"`)
	reBare   = regexp.MustCompile(`@([A-Za-z0-9_.$]+)`)
	reDefine = regexp.MustCompile(`^define (.*?)(void|[^ ]+) @("[^"]+"|[^(]+)\(`)
	reGlobal = regexp.MustCompile(`^@("[^"]+"|[^ ]+) = (.*?)global %"github.com/mehdi-shokohi/cuda-ir.go/cuda\.(Shared|DynShared|Constant)\[.*?\]" `)
	// any global variable definition (not a function, not a type)
	reAnyGlobal = regexp.MustCompile(`^@("[^"]+"|[^ ]+) = (?:[a-z_]+ )*(?:addrspace\(\d+\) )?global `)
	reCall      = regexp.MustCompile(`call [^@]*@("[^"]+"|[A-Za-z0-9_.$/]+)\(`)
	// heap allocation: `%3 = call ptr @"...runtime.AllocU"(i64 16)`
	reAlloc = regexp.MustCompile(`^(\s*)(%[^ ]+) = call ptr @"` + regexp.QuoteMeta(runtimePrefix) + `internal/runtime\.(AllocU|AllocZ)"\(i64 ([^)]+)\)`)
	// a cuda.Buf[T] parameter of a kernel: `%"...cuda.Buf[float32]" %3`
	reBufParam = regexp.MustCompile(`(%"github.com/mehdi-shokohi/cuda-ir.go/cuda\.Buf\[[^"]*\]") %([0-9A-Za-z_.]+)`)
	// a struct-typed kernel parameter: `%"pkg.Params" %2` (a Go slice is
	// `%"...runtime.Slice"` and stays a by-value aggregate)
	reStructParam = regexp.MustCompile(`(%"[^"]*") %([0-9A-Za-z_.]+)`)
	// the local copy llgo makes of an address-taken parameter, after the
	// AllocU/AllocZ rewrite: `%3 = alloca i8, i64 12, align 16` (followed
	// by its memset on a second line for AllocZ)
	reAllocaLine = regexp.MustCompile(`^\s*(%[^ ]+) = alloca i8, i64 [^,]+, align \d+(\n\s*call void @llvm\.memset[^\n]*)?$`)
	reDeclName   = regexp.MustCompile(`^declare .*@("[^"]+"|[^(]+)\(`)
	// a global's alignment below 16
	reAlignSmall = regexp.MustCompile(`, align (1|2|4|8)$`)
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
	// PTX is the PTX ISA version llc emits, without the dot (default "78",
	// CUDA 11.8+; dynamic allocation needs at least "73").
	PTX string
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
	// Globals are the exported package-level variables of the package
	// (except cuda.Shared / DynShared), by their Go name; the host reaches
	// them with cuModuleGetGlobal (gocudrv: mod.Global(name)).
	Globals []string
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
	if b.PTX == "" {
		b.PTX = "78"
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

	// 1. llgen each package. The cuda.Shared / DynShared / Constant
	// variables are recognised here, per package, by their Go type name:
	// llvm-link unifies structurally identical named types, so after
	// linking a `cuda.Shared[[8]float32]` may be typed `cuda.FragC`.
	var lls []string
	cudaGlobals := map[string]cudaGlobal{}
	for _, p := range pkgs {
		ir, err := b.genIR(p)
		if err != nil {
			return nil, err
		}
		for name, g := range findCudaGlobals(ir) {
			cudaGlobals[name] = g
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
	dirs, err := readDirectives(pkgs[len(pkgs)-1].Dir)
	if err != nil {
		return nil, err
	}
	fixed, found, globals, needLibdevice, err := fixup(src, rootPkg, want, dirs, cudaGlobals)
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
	// 4. opt, in two steps. First inline everything: a Go function that
	// returns &T{} now holds an alloca (see fixup), and the inliner scopes
	// a callee's allocas with llvm.lifetime / stacksave markers that would
	// make the memory dead as soon as the constructor returns. Those markers
	// are stripped from the fully inlined IR before the real optimisation.
	final := fixedFile
	if b.Opt != "0" {
		inlined := prefix + ".inlined.ll"
		public := "-internalize-public-api-list=" + strings.Join(append(append([]string{}, found...), globals...), ",")
		if err := b.llvm("opt", public, "-inline-threshold=1000000",
			"-passes=internalize,globaldce,inline", fixedFile, "-S", "-o", inlined); err != nil {
			return nil, err
		}
		if err := stripLifetimes(inlined); err != nil {
			return nil, err
		}
		final = prefix + ".opt.ll"
		// internalize everything but the kernels (and host-visible globals)
		// so unused device functions are dropped and the rest can be
		// inlined; after inlining, turn generic loads/stores on
		// shared/global memory into address-space-specific ones
		// (ld.shared, ld.global)
		if err := b.llvm("opt", public,
			"-passes=internalize,globaldce,default<O"+b.Opt+">,infer-address-spaces,instcombine,simplifycfg",
			inlined, "-S", "-o", final); err != nil {
			return nil, err
		}
	}
	// 5. llc
	ptxFile := prefix + ".ptx"
	if err := b.llvm("llc", "-march=nvptx64", "-mcpu="+b.SM, "-mattr=+ptx"+b.PTX, final, "-o", ptxFile); err != nil {
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
	return &Result{Package: rootPkg, PTX: ptx, Kernels: found, Globals: globals}, nil
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

// parseDecl splits `declare [attrs] <ret> @name(<params>)<attrs>` into
// [_, ret, name, params, attrs] (nil if line is not a declaration). Return
// attributes (nonnull, zeroext, ...) are dropped; the type may contain
// spaces ({ i32, i1 }), the parameters may contain parentheses (addrspace(3)).
func parseDecl(line string) []string {
	if !strings.HasPrefix(line, "declare ") {
		return nil
	}
	rest := line[len("declare "):]
	at := strings.Index(rest, " @")
	open := strings.Index(rest, "(")
	if at < 0 || open < at {
		return nil
	}
	ret := strings.TrimSpace(rest[:at])
	for { // strip leading attribute words
		f := strings.Fields(ret)
		if len(f) < 2 || !isAttrWord(f[0]) {
			break
		}
		ret = strings.TrimSpace(ret[len(f[0]):])
	}
	name := rest[at+2 : open]
	depth, end := 0, -1
	for i := open; i < len(rest); i++ {
		switch rest[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				end = i
			}
		}
		if end >= 0 {
			break
		}
	}
	if end < 0 {
		return nil
	}
	return []string{line, ret, name, rest[open+1 : end], rest[end+1:]}
}

// nameParams turns a declaration's parameter type list ("i1, ptr") into a
// definition's ("i1 %0, ptr %1").
func nameParams(params string) string {
	var out []string
	depth, start := 0, 0
	for i := 0; i <= len(params); i++ {
		if i == len(params) || (params[i] == ',' && depth == 0) {
			if t := strings.TrimSpace(params[start:i]); t != "" {
				out = append(out, fmt.Sprintf("%s %%%d", t, len(out)))
			}
			start = i + 1
			continue
		}
		switch params[i] {
		case '{', '[', '(', '<':
			depth++
		case '}', ']', ')', '>':
			depth--
		}
	}
	return strings.Join(out, ", ")
}

// isAttrWord reports whether w is a parameter/return attribute rather than
// the start of a type.
func isAttrWord(w string) bool {
	switch {
	case w == "void", w == "ptr", w == "float", w == "double", w == "half", w == "bfloat":
		return false
	case strings.HasPrefix(w, "i") && len(w) > 1 && w[1] >= '0' && w[1] <= '9':
		return false
	case strings.HasPrefix(w, "%"), strings.HasPrefix(w, "{"), strings.HasPrefix(w, "["), strings.HasPrefix(w, "<"):
		return false
	}
	return true
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
type mangler struct{ short map[string]string }

func (m mangler) mangle(name string) string {
	if strings.HasPrefix(name, "llvm.") { // intrinsics are not symbols
		return name
	}
	if short, ok := m.short[name]; ok {
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

// cudaGlobal is a package-level variable of a cuda.Shared / DynShared /
// Constant type; the kind selects the address space it is placed in.
type cudaGlobal struct{ kind string }

// findCudaGlobals returns the package-level variables whose type is one of
// the cuda memory-space wrappers.
func findCudaGlobals(src []byte) map[string]cudaGlobal {
	globals := map[string]cudaGlobal{}
	scanLines(src, func(line string) error {
		if m := reGlobal.FindStringSubmatch(line); m != nil {
			globals[strings.Trim(m[1], `"`)] = cudaGlobal{kind: m[3]}
		}
		return nil
	})
	return globals
}

// addrSpace of the cuda wrapper kinds: __shared__ = 3, __constant__ = 4.
func addrSpace(kind string) int {
	if kind == "Constant" {
		return 4
	}
	return 3
}

// fixup rewrites linked llgo IR for the NVPTX backend. cudaGlobals are
// the cuda memory-space variables found in the per-package IR. It
// returns the rewritten IR, the kernel names, the host-visible global
// names, and whether libdevice is needed.
func fixup(src []byte, rootPkg string, want map[string]bool, dirs map[string]directive, cudaGlobals map[string]cudaGlobal) ([]byte, []string, []string, bool, error) {
	kernels, err := findKernels(src, rootPkg, want)
	if err != nil {
		return nil, nil, nil, false, err
	}
	if cudaGlobals == nil {
		cudaGlobals = map[string]cudaGlobal{}
	}
	for name, g := range findCudaGlobals(src) {
		cudaGlobals[name] = g
	}
	short := map[string]string{} // full llgo name -> PTX name (kernels and exported globals)
	for k, v := range kernels {
		short[k] = v
	}
	m := mangler{short}
	var out, found, globals, bodies []string
	declared := map[string]bool{}
	needTrap, needLibdevice := false, false
	// pass 1: exported package-level variables of the root package keep
	// their Go name so the host can cuModuleGetGlobal them; which runtime
	// functions are actually called (declarations alone are harmless: type
	// descriptors reference equality helpers that never run).
	called := map[string]bool{}
	srcDeclared := map[string]bool{} // every `declare` in llgo's IR
	scanLines(src, func(line string) error {
		if d := reDeclName.FindStringSubmatch(line); d != nil {
			srcDeclared[strings.Trim(d[1], `"`)] = true
		}
		if g := reAnyGlobal.FindStringSubmatch(line); g != nil {
			name := strings.Trim(g[1], `"`)
			if pkg, s := splitName(name); pkg == rootPkg && isExported(s) {
				if _, isShared := cudaGlobals[name]; !isShared || cudaGlobals[name].kind == "Constant" {
					short[name] = s
					globals = append(globals, s)
				}
			}
		}
		if reAlloc.MatchString(line) {
			return nil // lowered to alloca below
		}
		for _, c := range reCall.FindAllStringSubmatch(line, -1) {
			called[strings.Trim(c[1], `"`)] = true
		}
		return nil
	})
	// the kernel being emitted: lines to insert after the entry label, value
	// renames to apply to its body, and the byval struct params whose local
	// copy (`alloca` + `store`) is elided
	var entry []string
	var renames []*regexp.Regexp
	var renameTo []string
	var gridParams map[string]string // param name -> type of `//cuda:grid_constant` params
	kernelStart, inKernel, inEntry := 0, false, false
	rename := func(from, to string) {
		renames = append(renames, regexp.MustCompile(`%`+regexp.QuoteMeta(from)+`\b`))
		renameTo = append(renameTo, "%"+to)
	}
	err = scanLines(src, func(line string) error {
		if inKernel {
			switch {
			case line == "}":
				inKernel = false
			case inEntry && strings.HasSuffix(line, ":"):
				// the entry label: rebuild the Buf[T] values from the ptr
				// params, load the byval structs
				out = append(out, line)
				out = append(out, entry...)
				entry, inEntry = nil, false
				return nil
			default:
				for i, re := range renames {
					line = re.ReplaceAllString(line, renameTo[i])
				}
				// `store %T %gN, ptr %M, align A` into llgo's local copy of
				// a grid_constant param: the copy (alloca + memset) is
				// dropped and uses of %M read the parameter in place
				if f := strings.Fields(line); len(f) >= 5 && f[0] == "store" && f[3] == "ptr" {
					param := strings.TrimPrefix(strings.TrimSuffix(f[2], ","), "%g")
					dst := strings.TrimSuffix(f[4], ",")
					if ty, ok := gridParams[param]; ok && f[1] == ty {
						for i := len(out) - 1; i >= kernelStart; i-- {
							m := reAllocaLine.FindStringSubmatch(out[i])
							if m == nil || m[1] != dst {
								continue
							}
							out = append(out[:i], out[i+1:]...)
							rename(dst[1:], param)
							return nil
						}
					}
				}
			}
		}
		if a := reAlloc.FindStringSubmatch(line); a != nil {
			// heap allocation -> stack allocation at the call site: nothing
			// allocated in a kernel can outlive it. (Not a helper function:
			// the inliner would end the alloca's lifetime at the return.)
			indent, res, kind, size := a[1], a[2], a[3], a[4]
			line = fmt.Sprintf("%s%s = alloca i8, i64 %s, align 16", indent, res, size)
			if kind == "AllocZ" {
				memset := fmt.Sprintf("call void @llvm.memset.p0.i64(ptr %s, i8 0, i64 %s, i1 false)", res, size)
				bodies = append(bodies, memset)
				line += "\n" + indent + memset
			}
			out = append(out, line)
			return nil
		}
		switch {
		case strings.HasPrefix(line, "target datalayout"):
			line = nvptxDL
		case strings.HasPrefix(line, "target triple"):
			line = nvptxTriple
		}
		if d := parseDecl(line); d != nil {
			ret, name, params, attrs := d[1], strings.Trim(d[2], `"`), d[3], d[4]
			_, shortName := splitName(name)
			define := func(body string) {
				body = strings.Replace(body, "NAME", m.mangle(name), 1)
				body = strings.ReplaceAll(body, "%SLICE", `%"`+runtimePrefix+`internal/runtime.Slice"`)
				// a helper may carry `declare`s of intrinsics whose
				// signature intrinsicDecls cannot derive (aggregate
				// returns); keep each once, and not if llgo declared it
				var keep []string
				for _, l := range strings.Split(body, "\n") {
					if d := reDeclName.FindStringSubmatch(l); d != nil {
						n := strings.Trim(d[1], `"`)
						if declared[n] || srcDeclared[n] {
							continue
						}
						declared[n] = true
					}
					keep = append(keep, l)
				}
				body = strings.Join(keep, "\n")
				bodies = append(bodies, body)
				line = body
			}
			switch {
			case strings.HasPrefix(name, runtimePrefix):
				if impl, ok := runtimeImpls[shortName]; ok {
					define(impl)
					break
				}
				if ret != "void" && called[name] {
					return fmt.Errorf("unsupported on GPU: code depends on llgo runtime function %q (returns %s)", name, ret)
				}
				// panics (PanicIndex, Panic...) trap; Assert*(cond, ...)
				// helpers trap when cond holds; a value-returning helper
				// that is never called gets the trap body too, which is
				// valid for any return type
				needTrap = true
				if strings.HasPrefix(shortName, "Assert") && strings.HasPrefix(params, "i1") {
					line = fmt.Sprintf("define void @%s(%s)%s {\n  br i1 %%0, label %%t, label %%r\nt:\n  call void @llvm.trap()\n  unreachable\nr:\n  ret void\n}", m.mangle(name), nameParams(params), attrs)
					break
				}
				line = fmt.Sprintf("define %s @%s(%s)%s {\n  call void @llvm.trap()\n  unreachable\n}", ret, m.mangle(name), params, attrs)
			case shortName == "init" && ret == "void" && params == "":
				// package initialisers of imported packages never run on the GPU
				line = fmt.Sprintf("define void @%s()%s {\n  ret void\n}", m.mangle(name), attrs)
			case strings.HasPrefix(name, libdevicePrefix):
				needLibdevice = true
			case strings.HasPrefix(name, helperPrefix):
				body, ok := helpers[strings.TrimPrefix(name, helperPrefix)]
				if !ok {
					return fmt.Errorf("unknown cudair IR helper %s", name)
				}
				define(body)
			case strings.HasPrefix(name, "llvm."):
				declared[name] = true
			case syscalls[name]:
				// device-runtime syscalls provided by the driver (vprintf)
			default:
				if body, ok := stdlibImpls[name]; ok {
					define(body)
					break
				}
				return fmt.Errorf("unsupported on GPU: call to %s — the Go standard library is not available in kernels; use package cuda (math -> cuda.Sqrt/Sin/..., sync/atomic and math/bits are fine)", name)
			}
		}
		if d := reDefine.FindStringSubmatch(line); d != nil {
			// llgo emits generic instantiations as `linkonce`, which LLVM may
			// not inline; every instantiation is identical, so ODR is safe.
			if strings.HasPrefix(d[1], "linkonce ") {
				line = strings.Replace(line, "define linkonce ", "define linkonce_odr ", 1)
			}
			if s, ok := kernels[strings.Trim(d[3], `"`)]; ok {
				// Buf[T] parameters become plain `ptr` parameters (the same
				// 8-byte .param for the host); llc then knows they point
				// into global memory and emits ld.global / st.global instead
				// of generic loads and stores. The body sees the struct again
				// from the entry block on.
				inKernel, inEntry, kernelStart = true, true, len(out)+1
				entry, renames, renameTo, gridParams = nil, nil, nil, map[string]string{}
				for _, m := range reBufParam.FindAllStringSubmatch(line, -1) {
					entry = append(entry, fmt.Sprintf("  %%g%[1]s = insertvalue %[2]s undef, ptr %%%[1]s, 0", m[2], m[1]))
					rename(m[2], "g"+m[2])
				}
				line = reBufParam.ReplaceAllString(line, "ptr %$2")
				line = strings.Replace(line, "define "+d[1], "define "+d[1]+"ptx_kernel ", 1)
				dir := dirs[s]
				if dir.restrict {
					// __restrict__: no two pointer parameters alias
					line = strings.ReplaceAll(line, "ptr %", "ptr noalias %")
				}
				if dir.gridConstant {
					// __grid_constant__: struct parameters stay in parameter
					// space (byval); the body reads them there and may take
					// their address without a local copy
					line = reStructParam.ReplaceAllStringFunc(line, func(p string) string {
						m := reStructParam.FindStringSubmatch(p)
						if strings.HasSuffix(m[1], `internal/runtime.Slice"`) {
							return p
						}
						gridParams[m[2]] = m[1]
						entry = append(entry, fmt.Sprintf("  %%g%[1]s = load %[2]s, ptr %%%[1]s, align 16", m[2], m[1]))
						rename(m[2], "g"+m[2])
						return fmt.Sprintf(`ptr byval(%s) align 16 "nvvm.grid_constant" %%%s`, m[1], m[2])
					})
				}
				if a := dir.attrs(); a != "" {
					line = strings.Replace(line, ") #", ") "+a+" #", 1)
				}
				found = append(found, s)
			}
		}
		// cuda.Shared / DynShared / Constant globals: the definition moves to
		// its address space; every use goes through an addrspacecast back
		// to a generic pointer (infer-address-spaces folds them away).
		if g := reAnyGlobal.FindStringSubmatch(line); g != nil && cudaGlobals[strings.Trim(g[1], `"`)].kind != "" {
			name := strings.Trim(g[1], `"`)
			switch cudaGlobals[name].kind {
			case "DynShared":
				// extern __shared__: size comes from the launch (SharedMemBytes)
				line = fmt.Sprintf("@%s = external addrspace(3) global [0 x i8], align 16", m.mangle(name))
			case "Constant":
				line = strings.Replace(line, "global %", "addrspace(4) global %", 1)
			default:
				line = strings.Replace(line, "global %", "addrspace(3) global %", 1)
				line = strings.Replace(line, "zeroinitializer", "undef", 1)
				// 16-byte aligned like the dynamic block: vector accesses,
				// cp.async and TMA copies need it
				line = reAlignSmall.ReplaceAllString(line, ", align 16")
			}
		} else if len(cudaGlobals) > 0 {
			cast := func(name string) string {
				return fmt.Sprintf("addrspacecast (ptr addrspace(%d) @%s to ptr)", addrSpace(cudaGlobals[name].kind), m.mangle(name))
			}
			line = reQuoted.ReplaceAllStringFunc(line, func(s string) string {
				if name := s[2 : len(s)-1]; cudaGlobals[name].kind != "" {
					return cast(name)
				}
				return s
			})
			line = reBare.ReplaceAllStringFunc(line, func(s string) string {
				if cudaGlobals[s[1:]].kind != "" {
					return cast(s[1:])
				}
				return s
			})
		}
		line = reQuoted.ReplaceAllStringFunc(line, func(s string) string { return "@" + m.mangle(s[2:len(s)-1]) })
		line = reBare.ReplaceAllStringFunc(line, func(s string) string { return "@" + m.mangle(s[1:]) })
		out = append(out, line)
		return nil
	})
	if err != nil {
		return nil, nil, nil, false, err
	}
	if needTrap {
		bodies = append(bodies, "call void @llvm.trap()")
	}
	sort.Strings(globals)
	out = append(out, "")
	out = append(out, intrinsicDecls(bodies, declared)...)
	out = append(out, invariantMD+" = !{}")
	return []byte(strings.Join(out, "\n") + "\n"), found, globals, needLibdevice, nil
}

// syscalls are functions the CUDA driver provides to device code.
var syscalls = map[string]bool{"vprintf": true}

// stripLifetimes removes the llvm.lifetime.* and stacksave/stackrestore
// calls the inliner added for callee allocas, in place.
func stripLifetimes(file string) error {
	src, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	var out []string
	scanLines(src, func(line string) error {
		t := strings.TrimSpace(line)
		if strings.HasPrefix(t, "call void @llvm.lifetime.") || strings.HasPrefix(t, "call void @llvm.stackrestore") ||
			strings.Contains(t, "= call ptr @llvm.stacksave") {
			return nil
		}
		out = append(out, line)
		return nil
	})
	return os.WriteFile(file, []byte(strings.Join(out, "\n")+"\n"), 0o644)
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
