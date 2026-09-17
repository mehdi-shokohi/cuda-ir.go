# cuda-ir.go — write CUDA kernels in native Go

`cuda-ir.go` lets you write CUDA kernels in plain Go and run them on NVIDIA GPUs — no C, no nvcc, no cgo.

- **Kernels are Go functions.** `threadIdx`, `__shared__`, `__syncthreads`, warp shuffles, atomics,
  libdevice math, tensor cores, textures, clusters and TMA are all a `cuda.*` call away;
  `sync/atomic` just works.
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
	PTX:     "78",               // PTX ISA version to emit (73 minimum)
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
-ptx 78         PTX ISA version to emit (CUDA 11.8+; 73 minimum)
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
| unexported func | device function (inlined when possible; `//go:noinline` is honoured) |
| `cuda.Buf[T]`, `*T` parameter | device pointer — pass `cuda.Arg(buffer)` |
| `[]T` parameter | Go slice (24-byte param), bounds-checked — pass `host.ArgSlice(buffer)` |
| struct parameter | passed by value — pass `cuda.ArgRaw` with the same layout |
| `int32/uint32/int64/uint64/int/float32/float64` parameter | `cuda.ArgValue(v)` with the **same** Go type (`int` is 64-bit) |
| `var t cuda.Shared[[256]float32]` at package level | `__shared__` memory (`t.Get()` → `*[256]float32`) |
| `var d cuda.DynShared[float32]` at package level | `extern __shared__` (`d.Buf()`; size = `LaunchConfig.SharedMemBytes`) |
| `var C = cuda.Constant[[4]float32]{V: ...}` (exported) | `__constant__`; host: `mod.Global("C")` + `cuda.WriteGlobal` |
| exported package-level `var X int32` | `__device__` global; host reads/writes it by name, `res.Globals` lists them |
| `//cuda:launch_bounds 256 2`, `//cuda:maxnreg 32` doc comment | `__launch_bounds__(256, 2)`, `__maxnreg__(32)` |
| `//cuda:cluster_dims 2 [1 [1]]` doc comment (sm_90) | `__cluster_dims__(2, 1, 1)` — launched with the plain grid, a multiple of the cluster |
| `//cuda:restrict` doc comment | `__restrict__` on every pointer parameter (`noalias`; read-only ones become `ld.global.nc`) |
| `//cuda:grid_constant` doc comment | `__grid_constant__` on every struct parameter: `&p` points into parameter space, no local copy; do not write to it |
| `cuda.Texture` / `cuda.Surface` parameter | `cudaTextureObject_t` / `cudaSurfaceObject_t` — pass gocudrv's `cuda.ArgTexture` / `cuda.ArgSurface` |
| `sync/atomic` on device pointers | native `atom.*` instructions |
| `cuda.Sqrt/Sin/Exp/...`, `math.Sqrt/Float32bits/...`, `math/bits.*` | PTX instruction or libdevice |
| `panic(...)`, nil deref, index out of range, integer `/ 0` | PTX `trap` → the launch fails with an error |
| `&T{}`, `copy()`, `for range`, local arrays that escape, plain function values | stack-allocated (nothing outlives the kernel), `memmove`, indirect `call` |
| `fmt`, any other stdlib, `make`, `append`, maps, strings ops, interfaces, goroutines, `defer`, closures | **compile error** (need the Go runtime) |

`Buf[T].At/Set` are unchecked like C; indexing Go arrays and slices is bounds-checked.
`cuda.Printf` is the device `printf`.

## Device API (CUDA C ↔ package cuda)

All verified on hardware by `go test .` (`examples/features`, `examples/intrinsics`, `examples/memory`,
`examples/advanced`, `examples/hopper`; `examples/blackwell` is compile-checked on GPUs below 10.x).

| CUDA C | Go |
|---|---|
| `threadIdx/blockIdx/blockDim/gridDim .x/.y/.z` | `ThreadIdxX()` … `GridDimZ()` |
| `blockIdx.x*blockDim.x+threadIdx.x` | `GlobalIdX/Y/Z()`, `GridStrideX()` |
| `warpSize`, lane id, `%smid %nsmid %warpid %nwarpid %gridid` | `WarpSize()`, `LaneID()`, `SMID() NumSMs() WarpID() NumWarps() GridID()` |
| `%lanemask_eq/lt/le/gt/ge` | `LaneMaskEq/Lt/Le/Gt/Ge()` |
| `__syncthreads()`, `_and/_or/_count` | `SyncThreads()`, `SyncThreadsAnd/Or/Count()` |
| named barriers `bar.sync id, n` / `bar.arrive id, n` | `SyncBarrier(id, n)`, `ArriveBarrier(id, n)` |
| `__syncwarp(mask)` | `SyncWarp(mask)` |
| `__threadfence[_block/_system]()` | `ThreadFence[Block/System]()` |
| `this_grid().sync()` (cooperative groups) | `cuda.GridBarrier` + `LaunchCooperative` |
| `__shared__ T x[N]`, `extern __shared__ T x[]` | `var x cuda.Shared[[N]T]`; `var x cuda.DynShared[T]` |
| `__constant__ T x[N]` | `var X cuda.Constant[[N]T]` |
| `__shfl_sync/_up/_down/_xor` (+ `width`) | `Shfl/ShflUp/ShflDown/ShflXor` (+ `F32`, `64`, `F64`, `…Width` variants) |
| `__all_sync/__any_sync/__ballot_sync/__activemask` | `All/Any/Ballot/ActiveMask` |
| `__match_any_sync/__match_all_sync` (sm_70) | `MatchAny/MatchAll` (+ `64`) |
| `__reduce_add/min/max/and/or/xor_sync` (sm_80) | `ReduceAdd/Min/Max/MinU/MaxU/And/Or/Xor` |
| integer `atomicAdd/Sub/Exch/CAS/And/Or/Xor` | Go `sync/atomic` (`AddInt32`, `CompareAndSwapInt32`, `SwapInt32`, …) |
| `atomicMin/atomicMax` (`int/unsigned/long long/unsigned long long`) | `AtomicMin/MaxInt32/Uint32/Int64/Uint64` |
| `atomicAdd(float*/double*)`, `atomicExch(float*)`, `atomicCAS(float*)` | `AtomicAddFloat32/64`, `AtomicSwapFloat32/64`, `AtomicCASFloat32/64` |
| `atomicInc/atomicDec` | `AtomicInc/AtomicDec` |
| `atomicAdd_block / atomicAdd_system` | `AtomicAddInt32/64Block`, `AtomicAddInt32/64System` |
| `cuda::atomic_ref` `load(acquire)` / `store(release)` | `LoadAcquireInt32/64`, `StoreReleaseInt32/64` |
| `volatile T*` | `VolatileLoad/StoreInt32/Int64/Float32/Float64` |
| `__ldg(p)` | `LdgF32/F64/I32/I64` (→ `ld.global.nc`) |
| `float4/int4` loads and stores | `LoadFloat4/StoreFloat4`, `LoadInt4/StoreInt4` |
| `cp.async` / `__pipeline_memcpy_async` (sm_80) | `CpAsync4/8/16`, `CpAsync16CG`, `CpAsyncCommit`, `CpAsyncWait1/2`, `CpAsyncWaitAll` |
| `__half`, `__nv_bfloat16` (+ `__hadd/__hmul/__hfma` …) | `cuda.Half`, `cuda.BFloat16` (`FloatToHalf`, `.Float32()`, `.Add/Sub/Mul/Div/FMA`) |
| `__popc/__popcll __clz/__clzll __ffs/__ffsll __brev/__brevll` | `Popc/Popc64 Clz/Clz64 Ffs/Ffs64 Brev/Brev64` (or `math/bits`) |
| `__byte_perm __funnelshift_l/r __mulhi/__umulhi/__mul64hi/__umul64hi __mul24/__umul24 __sad/__usad __hadd/__rhadd/__uhadd` | `BytePerm FunnelShiftL/R MulHi/MulHiU/MulHi64/MulHiU64 Mul24/Mul24U Sad/SadU HAdd/RHAdd/HAddU` |
| `sqrtf fabsf floorf ceilf truncf roundf rintf nearbyintf fmaf fminf fmaxf fdimf copysignf` | `Sqrt Abs Floor Ceil Trunc Round Rint Nearbyint FMA Min Max Fdim Copysign` |
| `sinf cosf tanf sincosf asinf acosf atanf atan2f sinhf coshf asinhf acoshf atanhf sinpif cospif` | `Sin Cos Tan Sincos Asin Acos Atan Atan2 Sinh Cosh Asinh Acosh Atanh Sinpi Cospi` |
| `expf exp2f exp10f expm1f logf log2f log10f log1pf powf cbrtf hypotf rsqrtf` | `Exp Exp2 Exp10 Expm1 Log Log2 Log10 Log1p Pow Cbrt Hypot Rsqrt` |
| `erff erfcf erfinvf tgammaf lgammaf normcdff j0f j1f fmodf remainderf ldexpf frexpf modff isnan isinf isfinite` | `Erf Erfc Erfinv Tgamma Lgamma NormCdf J0 J1 Fmod Remainder Ldexp Frexp Modf IsNaN IsInf IsFinite` |
| double versions of the above | `Sqrt64 Sin64 … Tan64 Floor64 Rsqrt64 Exp2_64 Log2_64 Tanh64 Erf64 Atan2_64 Cbrt64 Hypot64 …` |
| `__sinf __cosf __tanf __expf __exp10f __logf __log2f __log10f __powf __sincosf __fdividef` (fast) | `FastSin FastCos FastTan FastExp FastExp10 FastLog FastLog2 FastLog10 FastPow FastSincos FastDiv` |
| `__fadd_rn/_rz __fmul_rn __frcp_rn __fsqrt_rn __frsqrt_rn __saturatef __float2int_rn __float2uint_rn` | `AddRN AddRZ MulRN RcpRN SqrtRN RsqrtRN Saturate Float32ToInt32RN Float32ToUint32RN` |
| `__float_as_uint / __uint_as_float / __double_as_longlong` | `Float32Bits Float32FromBits Float64Bits Float64FromBits` (or `math.Float32bits`) |
| `clock() clock64() globaltimer`, `__nanosleep(ns)` | `Clock() Clock64() GlobalTimer()`, `NanoSleep(ns)` |
| `__trap() __brkpt() __builtin_assume(c)` | `Trap() Breakpoint() Assume(c)` |
| `__isGlobal/__isShared/__isConstant/__isLocal` | `IsGlobal/IsShared/IsConstant/IsLocal(p)` |
| `printf(fmt, ...)` | `Printf(fmt, Args().Int(i).Float(x)...)` |
| `%pm0..%pm3` | `PerfCounter0..3()` |
| `wmma::load_matrix_sync / mma_sync / store_matrix_sync` (m16n16k16, half → float) | `WmmaLoadA[Col]/WmmaLoadB[Col]/WmmaLoadC[Col]`, `WmmaMma[RowCol/ColRow/ColCol]`, `WmmaStore[Col]`, `FragA/FragB/FragC` (`Fill`, `Elems`) |
| `tex1D/tex2D/tex3D<float4>`, `tex2D<int4/uint4>`, `tex1Dfetch`, `txq` | `Tex1D Tex2D Tex3D Tex2DInt Tex2DUint Tex1DFetch TexWidth TexHeight` on a `cuda.Texture` |
| `surf1Dread/write`, `surf2Dread/write` (32-bit) | `Surf1D/Surf2D{Read,Write}{Int32,Float32}` on a `cuda.Surface` |
| `__ldca/__ldcg/__ldcs/__ldlu/__ldcv`, `__stwb/__stcg/__stcs/__stwt` | `LoadCA/CG/CS/LU/CV{Int32,Int64,Float32,Float64}`, `StoreWB/CG/CS/WT{…}` |
| `this_thread_block()`, `tiled_partition<N>`, `coalesced_threads()`, `thread_rank/size/sync/shfl/ballot`, `cg::reduce` | `ThisBlock()`, `ThisWarp()`, `TiledPartition(n)`, `CoalescedThreads()` → `Block` / `Tile` methods |
| `cluster.sync()`, `barrier.cluster.arrive/wait`, `%clusterid %cluster_ctarank %cluster_nctarank …`, `cluster.map_shared_rank(p, r)` (sm_90) | `ClusterSync/Arrive/Wait`, `ClusterIDX… ClusterCtaRank ClusterSize NumClustersX… ClusterBlockIdxX… ClusterDimX…`, `Shared[T].InCluster(rank)`, `MapShared`, `IsSharedCluster`, `ThreadFenceCluster` |
| `mbarrier.init/arrive/arrive_drop/test_wait/inval` (sm_80), `arrive.expect_tx/expect_tx/try_wait.parity` (sm_90) | `cuda.MBarrier` in shared memory: `Init Arrive ArriveDrop Wait TestWait ArriveExpectTx ExpectTx WaitParity TryWaitParity Inval`, `PendingCount`, `FenceMBarrierInit` |
| TMA `cp.async.bulk` global ↔ shared, `commit_group / wait_group[.read]`, `fence.proxy.async` (sm_90) | `CpAsyncBulkG2S(dst, src, bytes, bar)`, `CpAsyncBulkS2G`, `CpAsyncBulkCommit`, `CpAsyncBulkWait0/1`, `CpAsyncBulkWaitRead0/1`, `FenceProxyAsyncShared` |
| `elect.sync` (sm_90) | `ElectSync(mask) (leader, elected)` |
| `griddepcontrol.wait / launch_dependents` (sm_90) | `GridDepWait()`, `GridDepLaunchDependents()` |
| `__reduce_min/max_sync(float)` (sm_100a) | `ReduceMinF32/MaxF32` (build with `-sm sm_100a -ptx 86`) |

sm_90 features need `-sm sm_90 -ptx 80` (`cudair.Options{SM: "sm_90", PTX: "80"}`); the
driver runs that PTX on any newer GPU. Next: `wgmma`, TMA tensor maps, runtime cluster /
PDL launch attributes — see [ROADMAP.md](ROADMAP.md).

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
- `examples/intrinsics` (`go test -run TestIntrinsics`) — special registers, integer
  intrinsics and `math/bits`, `__match_*`/`__reduce_*`/64-bit shuffles, named barriers,
  `atomicMin/Max` & scoped atomics, the full libdevice math (float32 and float64)
- `examples/memory` (`go test -run 'TestMemory|TestTraps'`) — dynamic shared memory,
  `__constant__`/`__device__` globals, `__launch_bounds__`, device `printf`, `panic`/`copy`/
  slice parameters, `__ldg`/`volatile`/`float4`, half precision, `cp.async`, grid barrier
- `examples/advanced` (`go test -run TestAdvanced`) — tensor cores (`wmma`), textures and
  surfaces, `//cuda:restrict`, `//cuda:grid_constant`, cache-hint loads/stores, cooperative
  groups sugar, `%pm` counters
- `examples/hopper` (`go test -run TestHopper`, sm_90, needs a compute capability ≥ 9.0 GPU) —
  thread block clusters + distributed shared memory, `mbarrier`, TMA bulk copies,
  `elect.sync`, `griddepcontrol`
- `examples/blackwell` (`go test -run TestBlackwell`, sm_100a) — `redux.sync` on floats;
  compiled everywhere, executed on a 10.x GPU

Each `run/main.go` calls `cudair.Build` by default; `-ptx file` loads a pre-built
PTX instead, `-v` prints the compiler commands.

## What the compiler does to llgo's IR

llgo has no NVPTX target; its output needs these rewrites (`cudair.Build` / `gocuda build`) before `llc`/`ptxas` accept it:

1. target triple / datalayout → `nvptx64-nvidia-cuda`
2. symbol names: PTX identifiers allow only `[A-Za-z0-9_$]`; llgo's `pkg.Func` and
   `github.com/.../runtime.X` are mangled (kernels keep their short Go name)
3. kernels get the `ptx_kernel` calling convention (`.entry`)
4. llgo's panic calls (`runtime.PanicIndex`, `runtime.Panic`, `Assert*`) become `llvm.trap`;
   heap allocation (`runtime.AllocU/AllocZ`) becomes an `alloca` at the call site and
   `copy()` (`runtime.SliceCopy`) a `memmove`; a call to any other runtime function is an error
5. `init()` of imported packages is stubbed; `math/bits.*`, `math.Float32bits` & co get
   intrinsic bodies; other unresolved Go symbols are errors
6. `cuda.Shared` / `DynShared` / `Constant` package globals move to `addrspace(3)` / `(4)`;
   exported globals keep their Go name for `cuModuleGetGlobal`
7. `cuda.Buf[T]` kernel parameters become plain pointer parameters, so llc knows they are
   global memory (`ld.global` instead of generic loads; `__ldg` → `ld.global.nc`)
8. `cudair.*` IR helpers (`helpers.go`) supply what Go cannot spell: `atomicrmw min/max`,
   scoped/acquire-release atomics, volatile and invariant loads, `half`/`bfloat` arithmetic,
   `<4 x float>` accesses, `cp.async`, `wmma` fragments, `mbarrier`/TMA address spaces and
   inline-asm PTX (`ld.global.cg`, `mbarrier.try_wait.parity`); `//cuda:launch_bounds` /
   `cluster_dims` directives become attributes, `//cuda:restrict` adds `noalias`,
   `//cuda:grid_constant` turns struct parameters into `byval` ones and drops llgo's local copy
9. `__nv_*` uses pull just the needed functions out of libdevice
10. everything is inlined and the inliner's lifetime markers stripped (an `alloca` behind a
    `&T{}` constructor must live as long as the kernel), then `-O2` with everything but the
    kernels internalized, and `infer-address-spaces` yields `ld.shared` / `ld.global` / `ld.const`

## Host bindings compared (2026-09, CUDA 13.1 / driver 580)

| library | cgo | builds vs CUDA 13 | last release |
|---|---|---|---|
| **github.com/eitamring/gocudrv** (used here) | no | yes | 2026-07 |
| gorgonia.org/cu | yes | no (`cuCtxCreate`, graph API changed) | 2024-06 |
| github.com/InternatBlackhole/cudago | yes | no | 2025-04 |
| github.com/l0rem1psum/go-cuda-toolkit | yes | no | 2025-10 |

## Tested on

RTX 5060 Laptop (sm_120), driver 580 / CUDA 13.1, LLVM 22.1.8, llgo v1.0.2+, Go 1.27.

## License

Apache License 2.0 — see [LICENSE](LICENSE). The pipeline builds on
[llgo](https://github.com/xgo-dev/llgo) and LLVM (Apache-2.0) and
[gocudrv](https://github.com/eitamring/gocudrv) (MIT); `libdevice.10.bc` is read from
the user's CUDA toolkit and is not redistributed.
