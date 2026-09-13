# cuda-ir.go — write CUDA kernels in native Go

`cuda-ir.go` lets you write CUDA kernels in plain Go and run them on NVIDIA GPUs — no C, no nvcc, no cgo.

- **Kernels are Go functions.** `threadIdx`, `__shared__`, `__syncthreads`, warp shuffles, atomics and
  libdevice math are all a `cuda.*` call away; `sync/atomic` just works.
- **Go → PTX.** [llgo](https://github.com/xgo-dev/llgo) lowers the package to LLVM IR; `cudair` rewrites it
  for the NVPTX backend and hands it to `llc`. The CUDA driver JITs the PTX for whatever GPU is present.
- **Compile in-process or ahead of time.** `cudair.Build("./kernels", nil)` gives you PTX bytes at run time;
  `gocuda build` writes a `.ptx` to `//go:embed` and ship.
- **Pure-Go host side.** Launch through [gocudrv](https://github.com/eitamring/gocudrv) — `libcuda.so.1` is
  dlopen'ed, `CGO_ENABLED=0` builds work.
- **Go semantics on the GPU.** Nil derefs and out-of-range indexes become a PTX `trap`; anything that needs
  the Go runtime is a compile error, not a silent crash.

```go
package kernels

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// Exported + returns nothing  =>  a CUDA kernel named "VecAdd".
func VecAdd(a, b, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		out.Set(i, a.At(i)+b.At(i))
	}
}
```

```go
// host side: pure Go, no cgo (github.com/eitamring/gocudrv)
res, _ := cudair.Build("example.com/myapp/kernels", nil)   // Go -> PTX, in-process
cuda.Init()
dev, _ := cuda.GetDevice(0)
ctx, _ := dev.Primary()
mod, _ := ctx.LoadModule(res.PTX)
k, _ := mod.Function("VecAdd")
da, _ := cuda.Alloc[float32](ctx, n); da.CopyFrom(bg, a)   // db, dout likewise
k.Launch(bg, cuda.LaunchConfig1D(n, 256), cuda.Arg(da), cuda.Arg(db), cuda.Arg(dout), cuda.ArgValue(int32(n)))
ctx.Synchronize(bg)
dout.CopyTo(bg, out)
```

To try it: `make deps` installs everything the compiler needs (LLVM 22, the llgo
checkout + `llgen`, `gocuda`), then `make test` runs the examples on your GPU — see
[Installation](#installation). The full runnable version of the snippet above is
`main_test.go` (`go test -run TestVecAdd -v .`).

The same compiler is a command for build-time use:

```bash
gocuda build -o kernels.ptx ./kernels     # Go -> PTX; ship the file, //go:embed it, ...
```

How it works: llgo compiles the Go package to LLVM IR, `cudair` rewrites that IR for the
NVPTX backend (symbol names, kernel calling convention, shared memory, panics → `trap`,
libdevice math) and runs LLVM's `llc` to produce PTX.

## Installation

### Quick: `install.sh`

```bash
git clone https://github.com/mehdi-shokohi/cuda-ir.go.git
cd cuda-ir.go && ./install.sh          # -y to skip questions
source ~/.bashrc                   # (or ~/.zshrc) picks up LLGO_ROOT and PATH
make test                          # compiles the examples and runs them on your GPU
```

The script is idempotent and asks before anything that needs `sudo` or edits your
shell profile. It:

1. checks Go 1.24+
2. installs LLVM 22 — Debian/Ubuntu/Mint via [apt.llvm.org](https://apt.llvm.org)'s
   `llvm.sh 22` (`llvm-22-dev clang-22 lld-22 pkg-config libffi-dev zlib1g-dev`),
   macOS via `brew install llvm@22`
3. clones [llgo](https://github.com/xgo-dev/llgo) to `$LLGO_ROOT` (default `~/llgo`)
   at the tested commit and builds `llgen`
4. installs `gocuda`
5. checks the NVIDIA driver and CUDA toolkit (reports, does not install)
6. adds `LLGO_ROOT` and `$GOPATH/bin` to your shell profile
7. runs `gocuda doctor`

Overrides: `LLGO_ROOT`, `LLGO_REF` (`main` for latest llgo), `LLVM_VER`, `LLGO_REPO`.

### Manual

#### 1. Go 1.24+

#### 2. LLVM 22 (with the NVPTX backend — the standard packages have it)

Ubuntu/Debian:
```bash
wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 22
sudo apt-get install -y llvm-22-dev clang-22 lld-22 pkg-config libffi-dev zlib1g-dev
```
macOS: `brew install llvm@22 lld@22 && brew link --force llvm@22`.
Other distros: any LLVM 22 providing `llvm-link`, `opt`, `llc`. If the tools are not
suffixed `-22`, set `LLVM_SUFFIX` (e.g. `LLVM_SUFFIX=` for bare names).

#### 3. llgo's `llgen` (Go → LLVM IR)

`llgen` is a development tool inside the llgo repository and needs the checkout at run time:
```bash
git clone https://github.com/xgo-dev/llgo.git ~/llgo
cd ~/llgo && go install ./chore/llgen        # needs LLVM 22 dev packages from step 2
export LLGO_ROOT=~/llgo                      # put this in your shell profile
```
(`llgen` only needs libLLVM and libffi; the full llgo compiler wants more, see llgo's README.)

#### 4. CUDA

- NVIDIA driver (`libcuda.so.1`) — required at run time.
- CUDA toolkit — for `libdevice.10.bc` (used by `cuda.Sin/Exp/Pow/...`) and the
  optional `ptxas` build-time check. Any 12.x/13.x toolkit works; `gocuda` looks in
  `$CUDA_HOME`, `/usr/local/cuda*`, `/opt/cuda`, or `$LIBDEVICE`.

#### 5. cuda-ir.go (the `gocuda` command)

```bash
go install github.com/mehdi-shokohi/cuda-ir.go/cmd/gocuda@latest
gocuda doctor
```
`doctor` prints every dependency with a fix hint:
```
ok       llgen          /home/you/go/bin/llgen
ok       LLGO_ROOT      /home/you/llgo
ok       llvm-link      /usr/bin/llvm-link-22
ok       opt            /usr/bin/opt-22
ok       llc            /usr/bin/llc-22
ok       nvptx backend  /usr/bin/llc-22 has the nvptx64 target
ok       ptxas          /usr/bin/ptxas (optional: build-time PTX check)
ok       libdevice      /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
ok       libcuda.so.1   NVIDIA driver (needed at run time only)
```

## Using it in your project

```bash
go mod init example.com/myapp
go get github.com/mehdi-shokohi/cuda-ir.go            # cudair.Build + cuda.* device API
go get github.com/eitamring/gocudrv               # host API (no cgo)
```

```
myapp/
  kernels/kernels.go      # package kernels: your GPU code, imports cuda-ir.go/cuda
  main.go                 # compiles + launches the kernels with gocudrv
```

There are two ways to get from `kernels/` to PTX; both run the same pipeline
and need llgen/LLVM on the machine that does the compiling.

### From Go: `cudair.Build`

```go
import "github.com/mehdi-shokohi/cuda-ir.go" // package cudair

res, err := cudair.Build("example.com/myapp/kernels", nil)      // or "./kernels"
// res.PTX     []byte   -> ctx.LoadModule(res.PTX)
// res.Kernels []string -> "VecAdd", ...
```

`Build` resolves the package with `go list` from the current directory
(`Options.Dir` to change that), so the program must run inside the module that
holds the kernels — it is the "compile on start" / dev-loop mode; every run
recompiles. `Options` mirrors the command flags:

```go
cudair.Build("./kernels", &cudair.Options{
	SM:      "sm_80",            // llc/ptxas target (driver JIT adapts it)
	Kernels: []string{"VecAdd"}, // default: all exported void funcs
	Opt:     "2",                // "0" skips opt
	NoCheck: true,               // skip the ptxas check
	WorkDir: "build",            // keep the intermediate .ll files here
	Log:     os.Stderr,          // print the commands
})
```

`cudair.BuildFile(pkg, "kernels.ptx", opts)` additionally writes the file.
`cudair.Doctor(os.Stdout)` is the `doctor` command.

### From the command line: `gocuda build`

For shipping a binary without the LLVM toolchain: generate the PTX at build
time and embed it.

```
  kernels/gen.go          # //go:generate gocuda build -o kernels.ptx .
  main.go                 # //go:embed kernels/kernels.ptx
```

```bash
go generate ./...     # -> kernels/kernels.ptx
go run .
```

Options:

```
-o file.ptx     output (default <pkg>.ptx)
-sm sm_80       target for llc/ptxas (the driver JIT adapts PTX to the actual GPU)
-kernel A,B     only these functions become kernels (default: all exported void funcs)
-O 2            LLVM optimisation level (0 disables)
-keep           keep intermediate .ll files
-check=false    skip the ptxas check
-v              print the commands
```

### Kernel rules

| Go | GPU |
|---|---|
| exported func returning nothing | kernel `.entry`, PTX name = Go name |
| unexported func | device function (inlined when possible) |
| `cuda.Buf[T]`, `*T` parameter | device pointer — pass `cuda.Arg(buffer)` |
| `int32/uint32/int64/uint64/int/float32/float64` parameter | `cuda.ArgValue(v)` with the **same** Go type (`int` is 64-bit) |
| `var t cuda.Shared[[256]float32]` at package level | `__shared__` memory (`t.Get()` → `*[256]float32`) |
| `sync/atomic` on device pointers | native `atom.*` instructions |
| `cuda.Sqrt/Sin/Exp/...` | PTX instruction or libdevice |
| `math.*`, `fmt`, any other stdlib | **compile error** (not available on the GPU) |
| nil deref / index out of range | PTX `trap` → the launch fails with an error |
| `make`, `append`, maps, strings, interfaces, goroutines, `defer`, closures | **compile error** (need the Go runtime) |

`Buf[T].At/Set` are unchecked like C; indexing Go arrays (e.g. shared memory) is bounds-checked.

## Device API (CUDA C ↔ package cuda)

All verified on hardware by `examples/features`.

| CUDA C | Go |
|---|---|
| `threadIdx/blockIdx/blockDim/gridDim .x/.y/.z` | `ThreadIdxX()` … `GridDimZ()` |
| `blockIdx.x*blockDim.x+threadIdx.x` | `GlobalIdX/Y/Z()`, `GridStrideX()` |
| `warpSize`, lane id | `WarpSize()`, `LaneID()` |
| `__syncthreads()`, `_and/_or/_count` | `SyncThreads()`, `SyncThreadsAnd/Or/Count()` |
| `__syncwarp(mask)` | `SyncWarp(mask)` |
| `__threadfence[_block/_system]()` | `ThreadFence[Block/System]()` |
| `__shared__ T x[N]` | `var x cuda.Shared[[N]T]`; `x.Get()` |
| `__shfl_sync/_up/_down/_xor` | `Shfl/ShflUp/ShflDown/ShflXor` (+ `F32` variants) |
| `__all_sync/__any_sync/__ballot_sync/__activemask` | `All/Any/Ballot/ActiveMask` |
| integer `atomicAdd/Sub/Exch/CAS/And/Or/Xor` | Go `sync/atomic` (`AddInt32`, `CompareAndSwapInt32`, `SwapInt32`, …) |
| `atomicAdd(float*/double*)` | `AtomicAddFloat32/64` |
| `atomicInc/atomicDec` | `AtomicInc/AtomicDec` |
| `sqrtf fabsf floorf ceilf truncf roundf fmaf fminf fmaxf` | `Sqrt Abs Floor Ceil Trunc Round FMA Min Max` |
| `sinf cosf tanf expf exp2f logf log2f powf tanhf erff rsqrtf` (+ double) | `Sin Cos Tan Exp Exp2 Log Log2 Pow Tanh Erf Rsqrt` (+ `…64`) |
| `__sinf __cosf __expf __logf __powf` (fast) | `FastSin FastCos FastExp FastLog FastPow` |
| `clock() clock64() globaltimer` | `Clock() Clock64() GlobalTimer()` |

Not yet covered: `atomicMin/Max` on ints, dynamic shared memory (`extern __shared__`),
device `printf`, half precision / tensor cores (`wmma`), textures, cooperative-groups grid
sync, `__popc/__clz/__brev`, `__nanosleep`. Each is one `//go:linkname` to an
`llvm.nvvm.*` / `llvm.*` intrinsic or a libdevice `__nv_*` function; see `cuda/sync.go`
for the pattern.

## Examples

The examples compile the kernels at run time, so the machine needs the full
toolchain first: run `make deps` once (it is `./install.sh`, see
[Installation](#installation)) and make sure `LLGO_ROOT` points at the llgo checkout
it created (default `~/llgo`; `make` sets it if the variable is empty).

```bash
make deps        # once: LLVM 22, llgo + llgen, gocuda, checks the NVIDIA driver / CUDA toolkit
make doctor      # verify every dependency
make test        # runs examples/vecadd and examples/features on the GPU (PTX compiled in-process)
make test-ptx    # same through `gocuda build` + `-ptx file`
go test -v .     # the README VecAdd sample as a Go test (main_test.go)
```

- `examples/vecadd` — VecAdd, Saxpy (grid-stride loop), Square (calls a device func)
- `examples/features` — block reduction with shared memory + `__syncthreads` + warp
  shuffles + float atomics, `sync/atomic` counters, warp votes, libdevice math, `clock64`

Each `run/main.go` calls `cudair.Build` by default; `-ptx file` loads a pre-built
PTX instead, `-v` prints the compiler commands.

## What the compiler does to llgo's IR

llgo has no NVPTX target; its output needs these rewrites (`cudair.Build` / `gocuda build`) before `llc`/`ptxas` accept it:

1. target triple / datalayout → `nvptx64-nvidia-cuda`
2. symbol names: PTX identifiers allow only `[A-Za-z0-9_$]`; llgo's `pkg.Func` and
   `github.com/.../runtime.X` are mangled (kernels keep their short Go name)
3. kernels get the `ptx_kernel` calling convention (`.entry`)
4. llgo's nil/bounds-check calls (`runtime.AssertNilDeref`, `runtime.PanicIndex`) become
   `llvm.trap`; other runtime dependencies are reported as unsupported
5. `init()` of imported packages is stubbed; other unresolved Go symbols are errors
6. `cuda.Shared[T]` package globals move to `addrspace(3)`
7. `__nv_*` uses pull just the needed functions out of libdevice
8. generic instantiations become `linkonce_odr` (inlinable), everything but the kernels is
   internalized, and `infer-address-spaces` yields `ld.shared` / `ld.global`

## Host bindings compared (2026-09, CUDA 13.1 / driver 580)

| library | cgo | builds vs CUDA 13 | last release |
|---|---|---|---|
| **github.com/eitamring/gocudrv** (used here) | no | yes | 2026-07 |
| gorgonia.org/cu | yes | no (`cuCtxCreate`, graph API changed) | 2024-06 |
| github.com/InternatBlackhole/cudago | yes | no | 2025-04 |
| github.com/l0rem1psum/go-cuda-toolkit | yes | no | 2025-10 |

## Tested on

RTX 5060 Laptop (sm_120), driver 580 / CUDA 13.1, LLVM 22.1.8, llgo v1.0.2+, Go 1.27.
