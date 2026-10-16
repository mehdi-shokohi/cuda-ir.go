# Roadmap — CUDA device-API coverage

What `package cuda` covers is the table in [README → Device API](README.md#device-api-cuda-c--package-cuda).
This file tracks the CUDA C device-side keywords / intrinsics by release. Every "done"
entry is proven on hardware by a Go test (`go test -v .`): the subtest named in the last
column checks the result against a CPU model and, where it matters, the PTX for the expected
instruction (the few exceptions — features the test GPU cannot run — say so in that
column). Names were checked against LLVM 22's `IntrinsicsNVVM.td` and CUDA 13.1's
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

## v0.4 — done: tensor cores, textures, sm_90

Proven by `TestAdvanced` (`examples/advanced`, any sm_80 GPU), `TestHopper`
(`examples/hopper`, built with `-sm sm_90 -ptx 80`; runs on compute capability ≥ 9.0 —
tested on sm_120) and `TestBlackwell` (`examples/blackwell`, `-sm sm_100a -ptx 86`).

| CUDA C | Go | how | proven by |
|---|---|---|---|
| `wmma::load_matrix_sync / mma_sync / store_matrix_sync` m16n16k16 half → float, row/col layouts, `fill_fragment` | `FragA/FragB/FragC`, `WmmaLoadA[Col]/WmmaLoadB[Col]/WmmaLoadC[Col]`, `WmmaMma[RowCol/ColRow/ColCol]`, `WmmaStore[Col]`, `FragC.Fill/Elems` | helper (fragments are `{8 x <2 x half>}` / `{8 x float}` aggregates behind a pointer; the helper carries the intrinsic `declare`) | `TestAdvanced/Wmma` (64×64 matmul + bias vs CPU, both B layouts, both store layouts) |
| `tex2D<float4>(texObj, x, y)`, `tex1D`, `tex3D`, `tex2D<int4/uint4>`, `tex1Dfetch`, `txq.width/height` | `cuda.Texture` (u64 param, host: gocudrv `NewTexture` + `ArgTexture`), `Tex2D Tex1D Tex3D Tex2DInt Tex2DUint Tex1DFetch TexWidth TexHeight` | intrinsic (`llvm.nvvm.tex.unified.*`; a 4-value Go return is the `{float x4}` the intrinsic returns) | `TestAdvanced/Texture` (point + linear filtering, clamp, txq); `Tex1D/Tex3D/Tex1DFetch/Tex2DInt` compile-checked only — gocudrv has 2D arrays |
| `surf2Dread/surf2Dwrite`, `surf1Dread/write` (32-bit elements) | `cuda.Surface` (host: `NewSurface` + `ArgSurface`), `Surf2DRead/WriteInt32/Float32`, `Surf1D…` | intrinsic (`llvm.nvvm.suld/sust.b.*.i32.trap`; x in bytes, floats through the bits) | `TestAdvanced/Surface` (float and int surfaces) |
| `__cluster_dims__(x,y,z)`, `cluster.sync()`, `barrier.cluster.arrive/wait`, `%clusterid %nclusterid %cluster_ctarank %cluster_nctarank %cluster_ctaid %cluster_nctaid`, `cluster.map_shared_rank` (distributed shared memory), `__isClusterShared`, `fence.acq_rel/sc.cluster` (sm_90) | `//cuda:cluster_dims 2 [1 [1]]` → `.reqnctapercluster` so a plain launch works; `ClusterSync/Arrive/Wait`, `ClusterIDX… NumClustersX… ClusterCtaRank ClusterSize ClusterBlockIdxX… ClusterDimX…`, `Shared[T].InCluster(rank)`, `MapShared`, `IsSharedCluster`, `ThreadFenceCluster FenceSCCluster` | directive → `"nvvm.cluster_dim"` attribute; intrinsics (`llvm.nvvm.mapa` on the generic pointer); asm helper for the acq_rel fence | `TestHopper/Cluster` (8 blocks in clusters of 2 read each other's tile; rank/size/id checked) |
| `mbarrier.init/arrive/arrive_drop/test_wait/pending_count/inval` (sm_80), `arrive.expect_tx / expect_tx / try_wait.parity` (sm_90), `fence.mbarrier_init` | `cuda.MBarrier` in `Shared[…]`: `Init Arrive ArriveDrop Wait/TestWait ArriveExpectTx ExpectTx WaitParity/TryWaitParity Inval`, `PendingCount`, `FenceMBarrierInit` | helpers (`addrspacecast` to `addrspace(3)` for `llvm.nvvm.mbarrier.*.shared`; inline asm for the sm_90 forms) | `TestHopper/MBarrier`, `TestHopper/TMA` |
| TMA: `cp.async.bulk.shared::cluster.global` (mbarrier completion), `cp.async.bulk.global.shared::cta`, `commit_group / wait_group[.read]`, `fence.proxy.async.shared::cta` (sm_90) | `CpAsyncBulkG2S(dst, src, bytes, bar)`, `CpAsyncBulkS2G`, `CpAsyncBulkCommit`, `CpAsyncBulkWait0/1`, `CpAsyncBulkWaitRead0/1`, `FenceProxyAsyncShared` | helpers (`llvm.nvvm.cp.async.bulk.*` with `addrspace(7)/(3)/(1)` casts); `cuda.Shared` globals are now 16-byte aligned | `TestHopper/TMA` (1 KiB tiles global → shared → global, doubled in between) |
| `elect.sync` (sm_90) | `ElectSync(mask) (leader, elected)` | intrinsic (`{i32, i1}` return) | `TestHopper/Elect` (one lane per warp, `leader` is its lane id) |
| `griddepcontrol.wait / launch_dependents` (programmatic dependent launch, sm_90) | `GridDepWait GridDepLaunchDependents` | intrinsic | `TestHopper/GridDep` (executes as no-ops without the launch attribute, which gocudrv does not expose) |
| `__reduce_min/max_sync(float)` (`redux.sync.min/max.f32`, sm_100a) | `ReduceMinF32/MaxF32` | intrinsic; LLVM 22 selects it for `sm_100a` only | `TestBlackwell`: compiled and checked in the PTX; executed only on a 10.x GPU (not the sm_120 test machine) |
| `__ldca/__ldcg/__ldcs/__ldlu/__ldcv`, `__stwb/__stcg/__stcs/__stwt` (int32/int64/float32/float64) | `LoadCA/CG/CS/LU/CV{Int32,Int64,Float32,Float64}`, `StoreWB/CG/CS/WT{…}` | helper (inline asm `ld.global.<hint>` / `st.global.<hint>` on the pointer cast to global) | `TestAdvanced/CacheHints` (every variant in the PTX, values checked) |
| `__restrict__` | `//cuda:restrict` doc comment | `noalias` on the kernel's pointer params → llc emits `ld.global.nc` for read-only ones | `TestAdvanced/Restrict` (`Restrict` has `ld.global.nc`, the same kernel without the directive does not) |
| `__grid_constant__` | `//cuda:grid_constant` doc comment (struct params must not be written) | struct params become `ptr byval(T) "nvvm.grid_constant"`; llgo's local copy for `&p` is elided → `cvta.param`, no `.local` | `TestAdvanced/GridConstant` (`&p` passed to a `//go:noinline` function; no `.local` depot, the undirected twin has one) |
| Cooperative groups: `this_thread_block()`, `tiled_partition<N>`, `coalesced_threads()`, `thread_rank/size/meta_group_rank/sync/shfl*/all/any/ballot`, `cg::reduce` | `ThisBlock()`, `ThisWarp()`, `TiledPartition(n)`, `CoalescedThreads()` → `Block` / `Tile` methods | plain Go over `Shfl*/Ballot/SyncWarp/ActiveMask/redux` | `TestAdvanced/Groups` |
| `%pm0..%pm3` | `PerfCounter0..3` | intrinsic | `TestAdvanced/PerfCounters` |

Also on the way: `parser.ParseDir` (deprecated in Go 1.25) replaced by per-file parsing in
`directives.go`; IR helpers may carry their own `declare` lines (needed for aggregate-returning
intrinsics, deduplicated against llgo's declarations); no `unsafe` in kernels any more — every
pointer-taking API is generic over `*T` (`CpAsync`, `CpAsyncBulkG2S/S2G`, `IsGlobal…`,
`MapShared`, `Ldg`, `VolatileLoad/Store`, `LoadHint/StoreHint`) with `Buf[T]` methods
(`Ldg`, `Volatile/SetVolatile`, `Load/Store(i, hint)`, `View`) and `cuda.As[U](p)` for
reinterpreting an element.

## v0.5 — done: raw mma.sync shapes, ldmatrix/stmatrix, the other wmma types

Proven by `TestTensor` (`examples/tensor`, any sm_80 GPU) and `TestHopper/StMatrix`
(`examples/hopper`, `-sm sm_90 -ptx 80`). Every matmul subtest checks the result against a
CPU model with `==`: the operands are small integers, so half, bfloat16 and tf32 hold them
exactly. They are also deliberately asymmetric — an operand built from `i+j` or `i*j` makes
`A[i+8][j] == A[i][j+8]`, and the mma A fragment holds exactly those two elements in
different registers, so a matmul over such data still comes out right when two registers of
a fragment are swapped. Every layout below was checked by mutating one index and confirming
the test fails.

| CUDA C / PTX | Go | how | proven by |
|---|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and `.f16.f16.f16.f16` | `MmaF16`, `MmaF16Half`; fragments `MmaA MmaB MmaAccF32 MmaAccF16` (`Regs`, `Elems`, `Fill`) | helper (the intrinsic takes 4+2 `<2 x half>` and 4 `float` registers, so the fragment is a `[N]uint32`/`[N]float32` behind a pointer; the helper carries its own `declare`) | `TestTensor/MmaF16` (64×64 matmul, row-major B, B^T and a loaded accumulator), `TestTensor/MmaF16Half` |
| the same shape with `bf16` operands | `MmaBF16` (same fragment layout, so the same loads) | helper | `TestTensor/MmaBF16` |
| `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`, `cvt.rna.tf32.f32` | `MmaTF32`, `FloatToTF32` | helper; `llvm.nvvm.f2tf32.rna` for the conversion | `TestTensor/MmaTF32` (B row-major and B^T) |
| `mma.sync.aligned.m16n8k32.row.col.s32.{s8,u8}.{s8,u8}.s32`, `.satfinite` | `MmaS8 MmaU8 MmaS8Sat MmaU8Sat`, `MmaAccI32` | helper | `TestTensor/MmaInt8` |
| the per-lane fragment layouts of all three shapes | `MmaLoadA16/MmaLoadB16[Col]`, `MmaLoadATF32/MmaLoadBTF32[Col]`, `MmaLoadA8/MmaLoadB8[Col]`, `MmaLoadAcc/MmaStoreAcc[I32,F16]` | plain Go: `laneID>>2` is the row group, `laneID&3` the column pair, as in the ISA tables | every matmul subtest above |
| `ldmatrix.sync.aligned.m8n8.x1/x2/x4[.trans].b16` (sm_75) | `LdMatrix1/2/4[Trans](p *T)`, `MmaAFromRegs/MmaBFromRegs` | helper (`addrspacecast` to `addrspace(3)`; x1 returns a bare `i32`, x2/x4 an aggregate) | `TestTensor/LdMatrix` (every lane's registers against the ISA layout, element by element), `TestTensor/MmaLdMatrix` (a 16×16 · 16×8 product fed by `x4` and `x2.trans` — the fragments `ldmatrix` builds must equal `MmaLoadA16`'s and `MmaLoadB16`'s) |
| `stmatrix.sync.aligned.m8n8.x1/x2/x4[.trans].b16` (sm_90) | `StMatrix1/2/4[Trans](p *T, v)` | helper | `TestHopper/StMatrix` (a round trip through `ldmatrix` leaves the tile unchanged, and `.trans` transposes each 8×8 matrix) |
| `wmma` with `nv_bfloat16` (m16n16k16) | `WmmaLoadABF16[Col]/WmmaLoadBBF16[Col]`, `WmmaMmaBF16[RowCol/ColRow/ColCol]`, `FragABF16/FragBBF16` (the accumulator is the existing `FragC`) | helper | `TestTensor/WmmaBF16` (64×64 matmul + bias, row.row and row.col) |
| `wmma` with `precision::tf32` (m16n16k8) | `WmmaLoadATF32[Col]/WmmaLoadBTF32[Col]/WmmaLoadCTF32[Col]/WmmaStoreTF32[Col]`, `WmmaMmaTF32[…]`, `FragATF32/FragBTF32` | helper (the m16n16k8 geometry has its own accumulator load/store intrinsics) | `TestTensor/WmmaTF32` |
| `wmma` with `signed/unsigned char` → `int` (m16n16k16) | `WmmaLoadA{S8,U8}[Col]/WmmaLoadB{S8,U8}[Col]/WmmaLoadCI32[Col]/WmmaStoreI32[Col]`, `WmmaMma{S8,U8}[…]`, `FragAI8/FragBI8/FragCI32` | helper | `TestTensor/WmmaInt8` |

## v0.6 — done: the host-side gaps (package `host`)

Three v0.6 rows were blocked not on the device side but on the host: gocudrv
v0.3.2 does not bind `cuTensorMapEncodeTiled` or `cuLaunchKernelEx`, and its
typed array wrapper is fixed at 2D single-channel. None of that needed an
upstream change. gocudrv exposes its loaded driver (`Context.Driver`), its raw
context (`Context.Raw`) and its device pointers (`Buffer.DevicePtr`) so a
sibling module can work against the context it already owns, so package `host`
takes that route: the context, device, streams and buffers stay gocudrv's, and
only the missing entry points are bound here, with `purego.RegisterLibFunc`
against the same `libcuda.so.1` (dlopen refcounts — one mapping, one context).
The host still builds with `CGO_ENABLED=0`.

Two things to know when reading `host`:

- **Thread affinity.** gocudrv makes its context current only on its own
  executor thread, so a raw driver call from any other goroutine sees no
  context at all. `host.OnContext` pins the calling goroutine, makes the
  context current, runs the call and restores the thread's previous context.
  Work issued inside it is not serialised against concurrent gocudrv calls.
- **`cuLaunchKernelEx` needs a `CUfunction`**, and gocudrv's `Function` has no
  `Raw` accessor — the one thing an upstream patch would improve. So
  `host.LoadModule` loads the PTX itself (through gocudrv's exported
  `cudaresult.ModuleLoadData`) and keeps its own handle. A two-line
  `Function.Raw()` upstream would let `host/launchex.go` go away; it is not a
  prerequisite for anything here.

Proven by `TestHostAPI` (`examples/hostapi`, any sm_80 GPU), `TestLaunchEx`
and `TestTensorMap` (`examples/hopper`, `-sm sm_90 -ptx 80`, compute
capability ≥ 9.0 — tested on sm_120).

| CUDA C | Go | how | proven by |
|---|---|---|---|
| `cuTensorMapEncodeTiled`, `cp.async.bulk.tensor.{1..5}d.shared::cluster.global.tile` / `.global.shared::cta.tile` (sm_90) | host: `host.TiledTensor`, `host.EncodeTiled[T]`/`EncodeTiledPtr` → `host.TensorMap` (`Arg`, `Bytes`); device: `cuda.TensorMap`, `CpAsyncBulkTensor1D..5D`, `CpAsyncBulkTensorS2G1D..5D` | `cuTensorMapEncodeTiled` bound with `purego.RegisterLibFunc` (12 args, typed); the descriptor is kept 64-byte aligned inside an oversized heap allocation. Device side: one helper per rank casting the tile to `addrspace(7)` and the mbarrier to `addrspace(3)` for `llvm.nvvm.cp.async.bulk.tensor.{g2s,s2g}.tile.Nd` | `TestTensorMap/TensorTile2D` (six 32×32 tiles global → shared → global through two maps, doubled in between), `TestTensorMap/OutOfBounds` (a 40×40 tensor in 32×32 tiles, clamped at the edges), `TestTensorMap/Encode` (a valid descriptor accepted, five invalid ones rejected) |
| `cuLaunchKernelEx` launch attributes: `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` (runtime cluster dims), `…PROGRAMMATIC_STREAM_SERIALIZATION` (makes `griddepcontrol` live), `…COOPERATIVE` | `host.LoadModule` → `Module.Function` → `Function.LaunchEx(cfg, host.LaunchAttrs{…}, args…)`, with `host.Arg`/`ArgValue`/`ArgBuffer`/`ArgSliceEx`/`ArgBytes` | `cuLaunchKernelEx` bound here, plus a `CUlaunchConfig`/`CUlaunchAttribute` layout; the module is loaded by `host` because gocudrv keeps its `CUfunction` private. gocudrv's `cuda.KernelArg` cannot be reused — its only method is unexported — so `LaunchEx` has its own argument constructors | `TestLaunchEx/RuntimeClusterDims` (clusters of 1, 2 and 4 over 8 blocks; `%cluster_ctarank` and `%cluster_nctarank` checked against each shape, on a kernel whose PTX deliberately has **no** `.reqnctapercluster`), `TestLaunchEx/ProgrammaticDependentLaunch`, `TestLaunchEx/Cooperative` |
| `tex1D`, `tex3D`, `tex1DLayered`/`tex2DLayered`, `texCubemap`/`texCubemapLayered`, `tex2Dgather`, `tex1Dfetch` over linear memory, 2- and 4-channel arrays, `surf3Dread`/`surf3Dwrite` | host: `host.AllocArray[T]` + `AllocArray1D/2D/3D/Layered1D/Layered2D/Cubemap`, `WithChannels/WithSurfaceStore/WithTextureGather`, `host.NewTexture`/`NewTextureLinear`/`NewSurface`; device: `cuda.Tex1DLayered/Tex2DLayered[Int]/TexCubemap[Layered]/Tex3DInt/Tex2DGather{Red,Green,Blue,Alpha}/Surf3D{Read,Write}{Int32,Float32}` | **no new binding**: `cuArray3DCreate`, `cuMemcpy3D`, `cuTexObjectCreate`, `cuSurfObjectCreate` and their descriptors are already bound and exported by gocudrv — only the typed wrapper above 2D single-channel was missing. The exception is `tex1Dfetch`, which reads *linear* memory: its `res.linear` case falls in the 120 bytes gocudrv's `CUDA_RESOURCE_DESC` keeps as reserved padding, so `host` carries the full `CUDA_RESOURCE_DESC` (array, linear and pitch2D) and binds `cuTexObjectCreate` itself | `TestHostAPI/Tex1D` (`tex1D` over a 1D array, `tex1Dfetch` over linear memory, `txq.width`, array round trip), `/Tex3D`, `/Channels4` (a 4-channel array fills r,g,b,a), `/Layered` (3 layers of 8×4 and of 8), `/Cubemap` (all six faces selected by direction), `/Gather` (the four bilinear texels, against a CPU model with clamping), `/Surface3D` |

Also on the way: `cuda.Shared` globals are now 128-byte aligned rather than 16.
A `cp.async.bulk.tensor` destination must be 128-byte aligned or the copy
faults with `CUDA_ERROR_MISALIGNED_ADDRESS`; shared memory is plentiful, so
every `Shared` global gets the strictest alignment any shared access needs.
`cp.async.bulk.tensor`'s `.shared::cta` destination form only arrived in PTX
ISA 8.6, so the g2s helpers target `shared::cluster` (`addrspace(7)`) — with no
cluster, a CTA's cluster-shared window is its own shared memory.

## v0.6 — still to do

| CUDA C | proposed Go | notes |
|---|---|---|
| `wgmma` (sm_90a), `__nv_fp8` mma (`m16n8k16` e4m3/e5m2, `m16n8k32` f8f6f4), sparse `mma.sp`, `mma.m8n8k4`/`m16n8k8` f16, f64 and sub-byte shapes | `Wgmma…`, `MmaFP8…`, `MmaSp…` | `llvm.nvvm.wgmma.*` needs `-sm sm_90a` (not runnable on sm_120 hardware; compile-check like `TestBlackwell`). The fp8 and sparse shapes are the same helper pattern as v0.5, so they are mechanical; fp8 needs `sm_89`+ and host-side e4m3/e5m2 conversion in the tests |
| TMA `im2col` mode, `cp.async.bulk.tensor` multicast to the cluster, `prefetch`, and the `reduce_{add,min,max,…}` copies | `CpAsyncBulkTensorIm2col…`, a `cta_mask` argument, `CpAsyncBulkTensorPrefetch…`, `CpAsyncBulkTensorReduce…` | the tile mode is done (above); these are the remaining modes of the same intrinsic family, and `host.TiledTensor` already has the `Interleave`/`Swizzle` knobs a multicast matmul wants. `cuTensorMapEncodeIm2col` is in the driver but not yet bound |
| `__nanosleep`-based `cuda::barrier` phases, `cuda::pipeline` sugar over `cp.async` / TMA | `Pipeline` helper | pure Go over the existing primitives |
| `redux.sync.f32` on hardware | — | needs a 10.x GPU |

## Not planned (needs a device runtime or is host-side)

- Dynamic parallelism (kernel launching kernels) — needs `cudadevrt`.
- Device `malloc/free` with a heap that outlives the kernel — allocation inside a kernel is
  stack-based (`alloca`) and dies with the kernel; use pre-allocated buffers for anything else.
- Go `make/append/map/string operations/interface/goroutine/defer/closure` on the device —
  stay compile errors.
- Streams, events, graphs, unified memory, `cudaMemcpyAsync`, `cudaOccupancy*` — host API,
  covered by [gocudrv](https://github.com/eitamring/gocudrv). Texture and surface objects,
  CUDA arrays, launch attributes and TMA tensor maps are host API too, but the parts gocudrv
  does not reach are in package `host` (see v0.6): it shares gocudrv's context rather than
  opening its own.
