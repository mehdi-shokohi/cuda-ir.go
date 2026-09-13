# Roadmap — CUDA device-API coverage

What `package cuda` covers is the table in [README → Device API](README.md#device-api-cuda-c--package-cuda).
This file tracks the CUDA C device-side keywords / intrinsics by release. Every "done"
entry is proven on hardware by a Go test (`go test -v .`): the subtest named in the last
column checks the result against a CPU model and, where it matters, the PTX for the expected
instruction. Names were checked against LLVM 22's `IntrinsicsNVVM.td` and CUDA 13.1's
`libdevice.10.bc`.

**how**: `intrinsic` = one `//go:linkname` to `llvm.*`/`llvm.nvvm.*`; `libdevice` = one
`//go:linkname` to `__nv_*`; `helper` = an IR body in `helpers.go` (for what Go cannot
spell: `atomicrmw min`, `syncscope`, `volatile`, `half`, `<4 x float>`, address-space-typed
intrinsics); `build.go` = compiler/IR rewrite work; `host` = also needs the launch side.

## v0.2 — done: one `//go:linkname` each

| CUDA C / PTX | Go | how | proven by |
|---|---|---|---|
| `%smid %nsmid %warpid %nwarpid %gridid` | `SMID NumSMs WarpID NumWarps GridID` | intrinsic | `TestIntrinsics/Sregs` |
| `%lanemask_eq/lt/le/gt/ge` | `LaneMaskEq/Lt/Le/Gt/Ge` | intrinsic | `TestIntrinsics/Sregs` |
| `__popc/__popcll __clz/__clzll __brev/__brevll` | `Popc/Popc64 Clz/Clz64 Brev/Brev64` | intrinsic (`ctpop ctlz bitreverse`) | `TestIntrinsics/IntMath` |
| `__ffs/__ffsll __byte_perm __mulhi/__umulhi/__mul64hi/__umul64hi __mul24/__umul24 __sad/__usad __hadd/__rhadd/__uhadd` | `Ffs/Ffs64 BytePerm MulHi… Mul24… Sad/SadU HAdd/RHAdd/HAddU` | libdevice | `TestIntrinsics/IntMath` |
| `__funnelshift_l/r` | `FunnelShiftL/R` | intrinsic (`fshl fshr`) | `TestIntrinsics/IntMath` |
| `math/bits.OnesCount/LeadingZeros/TrailingZeros/Len/Reverse/ReverseBytes/RotateLeft` (8/16/32/64) | Go's `math/bits` | build.go (`stdlibImpls`) | `TestIntrinsics/MathBits` |
| `__match_any_sync / __match_all_sync` (sm_70) | `MatchAny/MatchAll` (+`64`) | intrinsic | `TestIntrinsics/WarpOps` |
| `__reduce_add/min/max/and/or/xor_sync` (sm_80) | `ReduceAdd/Min/Max/MinU/MaxU/And/Or/Xor` | intrinsic (`redux.sync`) | `TestIntrinsics/WarpOps` |
| `__shfl_*` on 64-bit / double, `width` argument | `Shfl64… ShflF64… ShflWidth…` | two 32-bit shuffles / packed `c` operand | `TestIntrinsics/WarpOps` |
| named barriers `bar.sync id, n` / `bar.arrive id, n` | `SyncBarrier ArriveBarrier` | intrinsic (`barrier.cta.sync/arrive.aligned.count`) | `TestIntrinsics/Barriers` |
| `__syncthreads*` modern spelling | unchanged API | `llvm.nvvm.barrier.cta.sync.aligned.all` & `red.*` (was the auto-upgraded `barrier0`) | `TestFeatures` |
| `__nanosleep` (sm_70), `__trap __brkpt __builtin_assume` | `NanoSleep Trap Breakpoint Assume` | intrinsic | `TestIntrinsics/Barriers`, `TestTraps` |
| `__isGlobal/__isShared/__isConstant/__isLocal` | `IsGlobal/IsShared/IsConstant/IsLocal` | intrinsic (`isspacep`) | `TestIntrinsics/Barriers` |
| `atomicMin/atomicMax` (s32 u32 s64 u64) | `AtomicMin/Max{Int32,Uint32,Int64,Uint64}` | helper (`atomicrmw min/max/umin/umax`) | `TestIntrinsics/Atomics` |
| `atomicExch/atomicCAS(float*/double*)` | `AtomicSwapFloat32/64 AtomicCASFloat32/64` | `sync/atomic` on the bits | `TestIntrinsics/Atomics` |
| `atomicAdd(float*)` modern spelling | unchanged API | helper (`atomicrmw fadd`, was `llvm.nvvm.atomic.load.add`) | `TestFeatures/BlockSum`, `TestIntrinsics/Atomics` |
| `atomicAdd_block / atomicAdd_system` | `AtomicAddInt32/64Block/System` | helper (`syncscope("block"/"system")`) | `TestIntrinsics/Atomics` |
| `cuda::atomic_ref` `load(acquire)`/`store(release)` | `LoadAcquireInt32/64 StoreReleaseInt32/64` | helper (`load atomic … acquire`) | `TestIntrinsics/Atomics` |
| `asinf acosf atanf atan2f sinhf coshf asinhf acoshf atanhf cbrtf hypotf log10f log1pf expm1f exp10f erfcf erfinvf lgammaf tgammaf fmodf remainderf copysignf rintf nearbyintf ldexpf frexpf modff fdimf sincosf sinpif cospif normcdff j0f j1f isnan isinf isfinite` | same names, Go-cased | libdevice | `TestIntrinsics/MathExt` |
| `__fdividef __tanf __exp10f __log2f __log10f __sincosf __frcp_rn __fsqrt_rn __frsqrt_rn __fadd_rn/rz __fmul_rn __saturatef __float2int_rn __float2uint_rn __float_as_uint …` | `FastDiv FastTan … RcpRN SqrtRN RsqrtRN AddRN AddRZ MulRN Saturate Float32ToInt32RN … Float32Bits …` | libdevice / `unsafe` | `TestIntrinsics/MathExt` |
| `math.Sqrt Abs Floor Ceil Trunc Round FMA Copysign IsNaN Float32bits Float32frombits Float64bits Float64frombits` | Go's `math` | build.go (`stdlibImpls`) | `TestIntrinsics/MathExt` |
| float64 twins (`tan floor ceil trunc round fmin fmax rsqrt exp2 log2 tanh erf asin acos atan atan2 sinh cosh cbrt hypot log10 log1p expm1 erfc lgamma tgamma fmod copysign rint ldexp isnan isinf`) | `Tan64 Floor64 … Exp2_64 Log2_64 … Atan2_64 …` | intrinsic / libdevice | `TestIntrinsics/Math64` |

## v0.3 — done: `build.go` work

| CUDA C | Go | what changed | proven by |
|---|---|---|---|
| `extern __shared__ T x[]` | `var x cuda.DynShared[T]`; `x.Buf()` | global → `external addrspace(3) global [0 x i8]`; `LaunchConfig.SharedMemBytes` on the host | `TestMemory/DynShared` |
| `__constant__ T x[N]` | `var X = cuda.Constant[[N]T]{V: …}` | global → `addrspace(4)`, keeps its initialiser; excluded from `internalize` so the host may rewrite it | `TestMemory/Globals` |
| `__device__` globals visible to the host | exported package `var` | exported globals keep their Go name (`res.Globals`), `mod.Global(name)` + `Read/WriteGlobal` | `TestMemory/Globals` |
| `__launch_bounds__(t, b)`, `__maxnreg__(n)` | `//cuda:launch_bounds t b`, `//cuda:maxnreg n` doc comments | `directives.go` parses the package, adds `"nvvm.maxntid"/"nvvm.minctasm"/"nvvm.maxnreg"` attributes | `TestMemory/LaunchBounds` |
| `__noinline__` | `//go:noinline` | already honoured by llgo (the earlier probe was constant-folded by IPSCCP) | — |
| device `printf` | `cuda.Printf(fmt, cuda.Args().Int(i).Float(x)…)` | `vprintf` syscall allowed; C varargs layout built in Go | `TestMemory/Printf` |
| `assert()` / `panic(...)` | `panic(v)` | `runtime.Panic` → trap; `Assert*(cond, …)` helpers get conditional trap bodies | `TestTraps/panic`, `TestMemory/PanicCopy` |
| heap allocation (`&T{}`, escaping locals) | plain Go | `runtime.AllocU/AllocZ` call sites → `alloca` (+`memset`); the IR is fully inlined first and the inliner's lifetime markers stripped, so a constructor's alloca lives for the whole kernel (PTX ISA ≥ 7.3 for dynamic `alloca`, default now 7.8) | `TestMemory/Printf`, `TestMemory/PanicCopy` |
| `memcpy` in a kernel | `copy(dst, src)` | `runtime.SliceCopy` → `llvm.memmove` | `TestMemory/PanicCopy` |
| pointer + length parameters | `func K(out []float32, in []float32)` | works as a 24-byte `.param`; `host.ArgSlice(buf)` builds it; indexing is bounds-checked | `TestMemory/Slices`, `TestTraps/oob` |
| `__ldg(p)` | `LdgF32/F64/I32/I64` | helper: `addrspacecast` to global + `!invariant.load` → `ld.global.nc` | `TestMemory/Ldg` |
| `volatile T*` | `VolatileLoad/Store{Int32,Int64,Float32,Float64}` | helper (`load/store volatile`) | `TestMemory/Volatile` |
| `float4/int4` accesses | `LoadFloat4/StoreFloat4 LoadInt4/StoreInt4` | helper (`<4 x float>` load/store) → `ld.global.v4` | `TestMemory/Vec4` |
| `__half / __nv_bfloat16` conversions and `__hadd/__hsub/__hmul/__hdiv/__hfma` | `cuda.Half`, `cuda.BFloat16` | helper (`half`/`bfloat` typed IR) → `cvt.rn.f16.f32`, `fma.rn.f16`, `fma.rn.bf16` | `TestMemory/Half` |
| `cp.async` / `__pipeline_memcpy_async` (sm_80) | `CpAsync4/8/16 CpAsync16CG CpAsyncCommit CpAsyncWait1/2 CpAsyncWaitAll` | helper casts to `addrspace(3)/(1)` for `llvm.nvvm.cp.async.*` | `TestMemory/CpAsync` |
| `this_grid().sync()` (cooperative groups) | `cuda.GridBarrier.Sync()` + `Function.LaunchCooperative` | atomics + `__threadfence` software barrier, no device runtime | `TestMemory/GridSync` |
| kernel `Buf[T]` parameters | unchanged API | become `ptr` parameters so llc's argument lowering marks them global: `ld.global`/`st.global` instead of generic loads everywhere | `TestVecAdd`, `TestFeatures` (and every PTX check above) |

Also fixed on the way: `declare`s with return attributes (`nonnull ptr …`) or struct returns
were silently skipped by the old regex; the "unsupported runtime function" check now looks at
call sites, not declarations (type descriptors reference `strequal` without calling it).

## v0.4 — next

| CUDA C | proposed Go | notes |
|---|---|---|
| Tensor cores: `wmma::load_matrix_sync / mma_sync / store_matrix_sync`, `mma.sync` (sm_70+), `wgmma` (sm_90) | opaque `FragA/FragB/FragC` + `WmmaLoadA/…/WmmaMma/WmmaStore` | `llvm.nvvm.wmma.m16n16k16.*`; fragments are `<8 x half>`/`<8 x float>` aggregates → helpers; `cuda.Half` exists now |
| Textures / surfaces: `tex1D/tex2D/tex3D<T>(texObj,…)`, `surf2Dread/write` | `cuda.Texture2D` (u64 handle param), `Tex2DF32(t, x, y) (r,g,b,a)` | `llvm.nvvm.tex.unified.2d.v4f32.f32` (22 families in LLVM 22), `llvm.nvvm.suld/sust.*`; gocudrv creates the objects |
| Thread block clusters (sm_90): `__cluster_dims__`, `cluster.sync()`, `%clusterid`, `%cluster_ctarank`, distributed shared memory (`mapa`) | `ClusterSync() ClusterID() ClusterCtaRank()`, `Shared[T].InCluster(rank)` | `llvm.nvvm.barrier.cluster.arrive/wait`, `…sreg.clusterid.*`, `llvm.nvvm.mapa`; `//cuda:cluster_dims` → `"nvvm.cluster_dim"`; host launch attributes |
| TMA `cp.async.bulk`, `mbarrier` (sm_90) | after `cp.async` | `llvm.nvvm.cp.async.bulk.*`, `llvm.nvvm.mbarrier.*` |
| `__reduce_min/max_sync(float)` (sm_100), `elect.sync` (sm_90) | `ReduceMinF32/MaxF32`, `ElectSync` | `llvm.nvvm.redux.sync.fmin/fmax`, `llvm.nvvm.elect.sync`; need `-sm sm_90/sm_100` |
| cache-hint loads/stores `__ldca/__ldcg/__ldcs/__ldlu/__ldcv`, `__stwb/__stcg/__stcs/__stwt` | `LoadCG/… StoreCS/…` | `!nontemporal` for `.cs`; the others need inline-asm-style helpers |
| `__restrict__` on `Buf` parameters | `//cuda:restrict` | `noalias` on the (now plain `ptr`) kernel params |
| `__grid_constant__`, `griddepcontrol` / programmatic dependent launch (sm_90) | attribute; `GridDepLaunchDependents() GridDepWait()` | `!nvvm.annotations`; `llvm.nvvm.griddepcontrol.*` |
| Cooperative groups sugar: `thread_block`, `tiled_partition<32>`, `coalesced_threads()` | `cuda.Warp` / `cuda.Tile[N]` helpers | over `Shfl/Ballot/SyncWarp/ActiveMask`; no new IR |
| `%pm0..%pm3` | `PerfCounter0..3` | intrinsic `…sreg.pm0..3` |

## Not planned (needs a device runtime or is host-side)

- Dynamic parallelism (kernel launching kernels) — needs `cudadevrt`.
- Device `malloc/free` with a heap that outlives the kernel — allocation inside a kernel is
  stack-based (`alloca`) and dies with the kernel; use pre-allocated buffers for anything else.
- Go `make/append/map/string operations/interface/goroutine/defer/closure` on the device —
  stay compile errors.
- Streams, events, graphs, unified memory, `cudaMemcpyAsync`, `cudaOccupancy*` — host API,
  covered by [gocudrv](https://github.com/eitamring/gocudrv).
