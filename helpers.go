package cudair

import (
	"fmt"
	"regexp"
	"strings"
)

// IR helpers: device functions that have no Go spelling (atomicrmw min/max,
// scoped atomics, volatile / invariant loads, half arithmetic, vector loads,
// cp.async ...). Package cuda declares them with
//
//	//go:linkname atomicMinI32 cudair.atomic.min.i32
//	func atomicMinI32(p *int32, v int32) int32
//
// and fixup replaces the resulting `declare` with the definition below. The
// bodies are tiny and get inlined by opt, so after `infer-address-spaces`
// they become single PTX instructions.
//
// Signatures must match what llgo emits for the Go declaration:
// int32/uint32 -> i32, int64/uint64 -> i64, uint16 -> i16, float32 -> float,
// float64 -> double, bool -> i1, any pointer -> ptr.
const helperPrefix = "cudair."

// invariantMD is the metadata node id used for !invariant.load (ld.global.nc).
// Numbered metadata ids need not be contiguous in textual IR.
const invariantMD = "!987001"

var helpers = map[string]string{}

func init() {
	// atomicMin / atomicMax on integers (atomicrmw min|max|umin|umax)
	for _, ty := range []string{"i32", "i64"} {
		for _, op := range []string{"min", "max", "umin", "umax"} {
			helpers[fmt.Sprintf("atomic.%s.%s", op, ty)] = fmt.Sprintf(
				"define %[1]s @NAME(ptr %%p, %[1]s %%v) {\n  %%r = atomicrmw %[2]s ptr %%p, %[1]s %%v seq_cst, align %[3]d\n  ret %[1]s %%r\n}", ty, op, size(ty))
		}
		// scoped atomicAdd (atomicAdd_block / atomicAdd_system); default scope is device
		for scope, name := range map[string]string{"block": "block", "system": "system"} {
			helpers[fmt.Sprintf("atomic.add.%s.%s", ty, name)] = fmt.Sprintf(
				"define %[1]s @NAME(ptr %%p, %[1]s %%v) {\n  %%r = atomicrmw add ptr %%p, %[1]s %%v syncscope(\"%[2]s\") seq_cst, align %[3]d\n  ret %[1]s %%r\n}", ty, scope, size(ty))
		}
		// acquire load / release store at device scope (cuda::atomic_ref<T, thread_scope_device>)
		helpers["load.acquire."+ty] = fmt.Sprintf(
			"define %[1]s @NAME(ptr %%p) {\n  %%r = load atomic %[1]s, ptr %%p syncscope(\"device\") acquire, align %[2]d\n  ret %[1]s %%r\n}", ty, size(ty))
		helpers["store.release."+ty] = fmt.Sprintf(
			"define void @NAME(ptr %%p, %[1]s %%v) {\n  store atomic %[1]s %%v, ptr %%p syncscope(\"device\") release, align %[2]d\n  ret void\n}", ty, size(ty))
	}
	// atomicAdd(float*/double*): atomicrmw fadd (the old llvm.nvvm.atomic.load.add.f32 is deprecated)
	for _, ty := range []string{"float", "double"} {
		helpers["atomic.fadd."+ty] = fmt.Sprintf(
			"define %[1]s @NAME(ptr %%p, %[1]s %%v) {\n  %%r = atomicrmw fadd ptr %%p, %[1]s %%v seq_cst, align %[2]d\n  ret %[1]s %%r\n}", ty, size(ty))
	}
	// volatile and read-only (__ldg, ld.global.nc) loads, volatile stores
	for _, ty := range []string{"i8", "i16", "i32", "i64", "float", "double"} {
		helpers["load.volatile."+ty] = fmt.Sprintf(
			"define %[1]s @NAME(ptr %%p) {\n  %%r = load volatile %[1]s, ptr %%p, align %[2]d\n  ret %[1]s %%r\n}", ty, size(ty))
		helpers["store.volatile."+ty] = fmt.Sprintf(
			"define void @NAME(ptr %%p, %[1]s %%v) {\n  store volatile %[1]s %%v, ptr %%p, align %[2]d\n  ret void\n}", ty, size(ty))
		// __ldg: the pointer is promised to be global memory (ld.global.nc)
		helpers["ldg."+ty] = fmt.Sprintf(
			"define %[1]s @NAME(ptr %%p) {\n  %%g = addrspacecast ptr %%p to ptr addrspace(1)\n  %%r = load %[1]s, ptr addrspace(1) %%g, align %[2]d, !invariant.load %[3]s\n  ret %[1]s %%r\n}", ty, size(ty), invariantMD)
	}
	// half / bfloat16: payload is a uint16 on the Go side
	for _, h := range []string{"half", "bfloat"} {
		helpers[h+".to.f32"] = fmt.Sprintf("define float @NAME(i16 %%v) {\n  %%h = bitcast i16 %%v to %[1]s\n  %%r = fpext %[1]s %%h to float\n  ret float %%r\n}", h)
		helpers["f32.to."+h] = fmt.Sprintf("define i16 @NAME(float %%v) {\n  %%h = fptrunc float %%v to %[1]s\n  %%r = bitcast %[1]s %%h to i16\n  ret i16 %%r\n}", h)
		for op, ins := range map[string]string{"add": "fadd", "sub": "fsub", "mul": "fmul", "div": "fdiv"} {
			helpers[h+"."+op] = fmt.Sprintf("define i16 @NAME(i16 %%a, i16 %%b) {\n  %%x = bitcast i16 %%a to %[1]s\n  %%y = bitcast i16 %%b to %[1]s\n  %%z = %[2]s %[1]s %%x, %%y\n  %%r = bitcast %[1]s %%z to i16\n  ret i16 %%r\n}", h, ins)
		}
		helpers[h+".fma"] = fmt.Sprintf("define i16 @NAME(i16 %%a, i16 %%b, i16 %%c) {\n  %%x = bitcast i16 %%a to %[1]s\n  %%y = bitcast i16 %%b to %[1]s\n  %%w = bitcast i16 %%c to %[1]s\n  %%z = call %[1]s @llvm.fma.f16(%[1]s %%x, %[1]s %%y, %[1]s %%w)\n  %%r = bitcast %[1]s %%z to i16\n  ret i16 %%r\n}", h)
	}
	helpers["bfloat.fma"] = strings.Replace(helpers["bfloat.fma"], "llvm.fma.f16", "llvm.fma.bf16", 1)
	// 128-bit vector load/store (float4 / int4): dst and src are 16-byte aligned
	for _, v := range []string{"float", "i32"} {
		helpers["ld.v4."+v] = fmt.Sprintf("define void @NAME(ptr %%dst, ptr %%src) {\n  %%v = load <4 x %[1]s>, ptr %%src, align 16\n  store <4 x %[1]s> %%v, ptr %%dst, align 16\n  ret void\n}", v)
		helpers["st.v4."+v] = fmt.Sprintf("define void @NAME(ptr %%dst, ptr %%src) {\n  %%v = load <4 x %[1]s>, ptr %%src, align 16\n  store <4 x %[1]s> %%v, ptr %%dst, align 16\n  ret void\n}", v)
	}
	// cp.async (sm_80): dst must be shared memory, src global; the intrinsics
	// want typed address spaces, the Go side has generic pointers.
	for _, n := range []string{"4", "8", "16"} {
		helpers["cp.async.ca."+n] = fmt.Sprintf("define void @NAME(ptr %%dst, ptr %%src) {\n  %%d = addrspacecast ptr %%dst to ptr addrspace(3)\n  %%s = addrspacecast ptr %%src to ptr addrspace(1)\n  call void @llvm.nvvm.cp.async.ca.shared.global.%[1]s(ptr addrspace(3) %%d, ptr addrspace(1) %%s)\n  ret void\n}", n)
	}
	helpers["cp.async.cg.16"] = "define void @NAME(ptr %dst, ptr %src) {\n  %d = addrspacecast ptr %dst to ptr addrspace(3)\n  %s = addrspacecast ptr %src to ptr addrspace(1)\n  call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) %d, ptr addrspace(1) %s)\n  ret void\n}"

	// cache-hint loads and stores (__ldcg & co): no LLVM spelling, so
	// inline asm on the pointer cast to global memory
	for ty, tc := range map[string][2]string{"i32": {"b32", "r"}, "i64": {"b64", "l"}, "float": {"f32", "f"}, "double": {"f64", "d"}} {
		ptxTy, con := tc[0], tc[1] // PTX type suffix, inline-asm register constraint
		for _, hint := range []string{"ca", "cg", "cs", "lu", "cv"} {
			helpers[fmt.Sprintf("ld.%s.%s", hint, ty)] = fmt.Sprintf(
				"define %[1]s @NAME(ptr %%p) {\n  %%g = addrspacecast ptr %%p to ptr addrspace(1)\n  %%r = call %[1]s asm sideeffect \"ld.global.%[2]s.%[3]s $0, [$1];\", \"=%[4]s,l\"(ptr addrspace(1) %%g)\n  ret %[1]s %%r\n}", ty, hint, ptxTy, con)
		}
		for _, hint := range []string{"wb", "cg", "cs", "wt"} {
			helpers[fmt.Sprintf("st.%s.%s", hint, ty)] = fmt.Sprintf(
				"define void @NAME(ptr %%p, %[1]s %%v) {\n  %%g = addrspacecast ptr %%p to ptr addrspace(1)\n  call void asm sideeffect \"st.global.%[2]s.%[3]s [$0], $1;\", \"l,%[4]s\"(ptr addrspace(1) %%g, %[1]s %%v)\n  ret void\n}", ty, hint, ptxTy, con)
		}
	}
	// tensor cores: wmma m16n16k16, half in, float accumulate. A fragment is
	// an aggregate of 8 <2 x half> (A/B) or 8 floats (C/D) that the Go side
	// holds as [8]uint32 / [8]float32 behind a pointer.
	const fragAB = "{ <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> }"
	const fragC = "{ float, float, float, float, float, float, float, float }"
	for _, layout := range []string{"row", "col"} {
		for _, m := range []string{"a", "b"} {
			helpers[fmt.Sprintf("wmma.load.%s.%s", m, layout)] = fmt.Sprintf(
				"declare %[1]s @llvm.nvvm.wmma.m16n16k16.load.%[2]s.%[3]s.stride.f16.p0(ptr, i32)\ndefine void @NAME(ptr %%f, ptr %%p, i32 %%ldm) {\n  %%v = call %[1]s @llvm.nvvm.wmma.m16n16k16.load.%[2]s.%[3]s.stride.f16.p0(ptr %%p, i32 %%ldm)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}", fragAB, m, layout)
		}
		helpers["wmma.load.c."+layout] = fmt.Sprintf(
			"declare %[1]s @llvm.nvvm.wmma.m16n16k16.load.c.%[2]s.stride.f32.p0(ptr, i32)\ndefine void @NAME(ptr %%f, ptr %%p, i32 %%ldm) {\n  %%v = call %[1]s @llvm.nvvm.wmma.m16n16k16.load.c.%[2]s.stride.f32.p0(ptr %%p, i32 %%ldm)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}", fragC, layout)
		helpers["wmma.store.d."+layout] = fmt.Sprintf(
			"declare void @llvm.nvvm.wmma.m16n16k16.store.d.%[2]s.stride.f32.p0(ptr, float, float, float, float, float, float, float, float, i32)\ndefine void @NAME(ptr %%p, ptr %%f, i32 %%ldm) {\n  %%v = load %[1]s, ptr %%f, align 4\n%[3]s  call void @llvm.nvvm.wmma.m16n16k16.store.d.%[2]s.stride.f32.p0(ptr %%p, %[4]s, i32 %%ldm)\n  ret void\n}", fragC, layout, extractAll("v", "d", fragC, 8), args("float", "d", 8))
		for _, lb := range []string{"row", "col"} {
			helpers[fmt.Sprintf("wmma.mma.%s.%s", layout, lb)] = fmt.Sprintf(
				"declare %[1]s @llvm.nvvm.wmma.m16n16k16.mma.%[3]s.%[4]s.f32.f32(%[5]s)\ndefine void @NAME(ptr %%d, ptr %%a, ptr %%b, ptr %%c) {\n  %%va = load %[2]s, ptr %%a, align 4\n  %%vb = load %[2]s, ptr %%b, align 4\n  %%vc = load %[1]s, ptr %%c, align 4\n%[6]s%[7]s%[8]s  %%r = call %[1]s @llvm.nvvm.wmma.m16n16k16.mma.%[3]s.%[4]s.f32.f32(%[9]s)\n  store %[1]s %%r, ptr %%d, align 4\n  ret void\n}",
				fragC, fragAB, layout, lb,
				strings.Repeat("<2 x half>, ", 16)+strings.TrimSuffix(strings.Repeat("float, ", 8), ", "),
				extractAll("va", "a", fragAB, 8), extractAll("vb", "b", fragAB, 8), extractAll("vc", "c", fragC, 8),
				args("<2 x half>", "a", 8)+", "+args("<2 x half>", "b", 8)+", "+args("float", "c", 8))
		}
	}
	// mbarrier (sm_80+) and bulk async copies (sm_90+): the intrinsics want
	// shared / global typed pointers; arrive.expect_tx and try_wait.parity
	// have no LLVM 22 intrinsic and use inline asm on the 32-bit shared
	// address.
	const toShared = "  %s = addrspacecast ptr %p to ptr addrspace(3)\n"
	const toShared32 = toShared + "  %a = ptrtoint ptr addrspace(3) %s to i32\n"
	helpers["mbarrier.init"] = "define void @NAME(ptr %p, i32 %n) {\n" + toShared + "  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) %s, i32 %n)\n  ret void\n}"
	helpers["mbarrier.inval"] = "define void @NAME(ptr %p) {\n" + toShared + "  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) %s)\n  ret void\n}"
	helpers["mbarrier.arrive"] = "define i64 @NAME(ptr %p) {\n" + toShared + "  %r = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %s)\n  ret i64 %r\n}"
	helpers["mbarrier.arrive.drop"] = "define i64 @NAME(ptr %p) {\n" + toShared + "  %r = call i64 @llvm.nvvm.mbarrier.arrive.drop.shared(ptr addrspace(3) %s)\n  ret i64 %r\n}"
	helpers["mbarrier.test.wait"] = "define i1 @NAME(ptr %p, i64 %t) {\n" + toShared + "  %r = call i1 @llvm.nvvm.mbarrier.test.wait.shared(ptr addrspace(3) %s, i64 %t)\n  ret i1 %r\n}"
	helpers["mbarrier.arrive.expect_tx"] = "define i64 @NAME(ptr %p, i32 %n) {\n" + toShared32 + "  %r = call i64 asm sideeffect \"mbarrier.arrive.expect_tx.shared::cta.b64 $0, [$1], $2;\", \"=l,r,r\"(i32 %a, i32 %n)\n  ret i64 %r\n}"
	helpers["mbarrier.expect_tx"] = "define void @NAME(ptr %p, i32 %n) {\n" + toShared32 + "  call void asm sideeffect \"mbarrier.expect_tx.shared::cta.b64 [$0], $1;\", \"r,r\"(i32 %a, i32 %n)\n  ret void\n}"
	helpers["mbarrier.try_wait.parity"] = "define i1 @NAME(ptr %p, i32 %parity) {\n" + toShared32 + "  %r = call i32 asm sideeffect \"{\\0A\\09.reg .pred p;\\0A\\09mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\\0A\\09selp.u32 $0, 1, 0, p;\\0A\\09}\", \"=r,r,r\"(i32 %a, i32 %parity)\n  %b = icmp ne i32 %r, 0\n  ret i1 %b\n}"
	helpers["fence.mbarrier_init"] = "define void @NAME() {\n  call void asm sideeffect \"fence.mbarrier_init.release.cluster;\", \"\"()\n  ret void\n}"
	helpers["fence.proxy.async.shared"] = "define void @NAME() {\n  call void asm sideeffect \"fence.proxy.async.shared::cta;\", \"\"()\n  ret void\n}"
	helpers["fence.acq_rel.cluster"] = "define void @NAME() {\n  call void asm sideeffect \"fence.acq_rel.cluster;\", \"\"()\n  ret void\n}"
	helpers["cp.async.bulk.g2s"] = "define void @NAME(ptr %dst, ptr %src, i32 %n, ptr %bar) {\n  %d = addrspacecast ptr %dst to ptr addrspace(7)\n  %s = addrspacecast ptr %src to ptr addrspace(1)\n  %b = addrspacecast ptr %bar to ptr addrspace(3)\n  call void @llvm.nvvm.cp.async.bulk.global.to.shared.cluster(ptr addrspace(7) %d, ptr addrspace(3) %b, ptr addrspace(1) %s, i32 %n, i16 0, i64 0, i1 false, i1 false)\n  ret void\n}"
	helpers["cp.async.bulk.s2g"] = "define void @NAME(ptr %dst, ptr %src, i32 %n) {\n  %d = addrspacecast ptr %dst to ptr addrspace(1)\n  %s = addrspacecast ptr %src to ptr addrspace(3)\n  call void @llvm.nvvm.cp.async.bulk.shared.cta.to.global(ptr addrspace(1) %d, ptr addrspace(3) %s, i32 %n, i64 0, i1 false)\n  ret void\n}"
}

// extractAll returns IR lines that extract the n fields of the aggregate
// %v (of type ty) into %<prefix>0..n-1.
func extractAll(v, prefix, ty string, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "  %%%s%d = extractvalue %s %%%s, %d\n", prefix, i, ty, v, i)
	}
	return b.String()
}

// args returns "ty %prefix0, ty %prefix1, ...".
func args(ty, prefix string, n int) string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("%s %%%s%d", ty, prefix, i)
	}
	return strings.Join(out, ", ")
}

func size(ty string) int {
	switch ty {
	case "i64", "double":
		return 8
	case "i16", "half", "bfloat":
		return 2
	case "i8":
		return 1
	}
	return 4
}

// runtimeImpls are llgo runtime functions that have a device implementation
// instead of trapping. (Heap allocation, AllocU/AllocZ, is rewritten at the
// call site into an alloca — see fixup.)
var runtimeImpls = map[string]string{
	// copy(dst, src): n = min(len(dst), len(src)); memmove n*elem bytes
	"SliceCopy": "define i64 @NAME(%SLICE %dst, ptr %src, i64 %n, i64 %sz) {\n  %dp = extractvalue %SLICE %dst, 0\n  %dl = extractvalue %SLICE %dst, 1\n  %m = call i64 @llvm.umin.i64(i64 %dl, i64 %n)\n  %bytes = mul i64 %m, %sz\n  call void @llvm.memmove.p0.p0.i64(ptr %dp, ptr %src, i64 %bytes, i1 false)\n  ret i64 %m\n}",
}

// stdlibImpls maps the few standard-library functions that are pure
// bit-twiddling to LLVM intrinsics, so math/bits and math.Float32bits & co
// work in kernels. Key: llgo symbol name; value: define body with NAME.
// Go int/uint are i64.
var stdlibImpls = map[string]string{
	"math.Float32bits":     "define i32 @NAME(float %v) {\n  %r = bitcast float %v to i32\n  ret i32 %r\n}",
	"math.Float32frombits": "define float @NAME(i32 %v) {\n  %r = bitcast i32 %v to float\n  ret float %r\n}",
	"math.Float64bits":     "define i64 @NAME(double %v) {\n  %r = bitcast double %v to i64\n  ret i64 %r\n}",
	"math.Float64frombits": "define double @NAME(i64 %v) {\n  %r = bitcast i64 %v to double\n  ret double %r\n}",
	"math.Sqrt":            "define double @NAME(double %v) {\n  %r = call double @llvm.sqrt.f64(double %v)\n  ret double %r\n}",
	"math.Abs":             "define double @NAME(double %v) {\n  %r = call double @llvm.fabs.f64(double %v)\n  ret double %r\n}",
	"math.Floor":           "define double @NAME(double %v) {\n  %r = call double @llvm.floor.f64(double %v)\n  ret double %r\n}",
	"math.Ceil":            "define double @NAME(double %v) {\n  %r = call double @llvm.ceil.f64(double %v)\n  ret double %r\n}",
	"math.Trunc":           "define double @NAME(double %v) {\n  %r = call double @llvm.trunc.f64(double %v)\n  ret double %r\n}",
	"math.Round":           "define double @NAME(double %v) {\n  %r = call double @llvm.round.f64(double %v)\n  ret double %r\n}",
	"math.FMA":             "define double @NAME(double %a, double %b, double %c) {\n  %r = call double @llvm.fma.f64(double %a, double %b, double %c)\n  ret double %r\n}",
	"math.Copysign":        "define double @NAME(double %a, double %b) {\n  %r = call double @llvm.copysign.f64(double %a, double %b)\n  ret double %r\n}",
	"math.IsNaN":           "define i1 @NAME(double %v) {\n  %r = fcmp uno double %v, %v\n  ret i1 %r\n}",
}

func init() {
	for _, w := range []struct {
		n    int
		name string
	}{{8, "8"}, {16, "16"}, {32, "32"}, {64, "64"}, {64, ""}} { // "" = uint-sized
		ty := fmt.Sprintf("i%d", w.n)
		stdlibImpls["math/bits.OnesCount"+w.name] = fmt.Sprintf("define i64 @NAME(%[1]s %%v) {\n  %%c = call %[1]s @llvm.ctpop.%[1]s(%[1]s %%v)\n%[2]s  ret i64 %%r\n}", ty, widen(ty, "%c"))
		stdlibImpls["math/bits.LeadingZeros"+w.name] = fmt.Sprintf("define i64 @NAME(%[1]s %%v) {\n  %%c = call %[1]s @llvm.ctlz.%[1]s(%[1]s %%v, i1 false)\n%[2]s  ret i64 %%r\n}", ty, widen(ty, "%c"))
		stdlibImpls["math/bits.TrailingZeros"+w.name] = fmt.Sprintf("define i64 @NAME(%[1]s %%v) {\n  %%c = call %[1]s @llvm.cttz.%[1]s(%[1]s %%v, i1 false)\n%[2]s  ret i64 %%r\n}", ty, widen(ty, "%c"))
		stdlibImpls["math/bits.Len"+w.name] = fmt.Sprintf("define i64 @NAME(%[1]s %%v) {\n  %%c = call %[1]s @llvm.ctlz.%[1]s(%[1]s %%v, i1 false)\n  %%l = sub %[1]s %[3]d, %%c\n%[2]s  ret i64 %%r\n}", ty, widen(ty, "%l"), w.n)
		stdlibImpls["math/bits.Reverse"+w.name] = fmt.Sprintf("define %[1]s @NAME(%[1]s %%v) {\n  %%r = call %[1]s @llvm.bitreverse.%[1]s(%[1]s %%v)\n  ret %[1]s %%r\n}", ty)
		if w.n >= 16 {
			stdlibImpls["math/bits.ReverseBytes"+w.name] = fmt.Sprintf("define %[1]s @NAME(%[1]s %%v) {\n  %%r = call %[1]s @llvm.bswap.%[1]s(%[1]s %%v)\n  ret %[1]s %%r\n}", ty)
		}
		// RotateLeft(x, k int): fshl masks the shift amount, so negative k
		// (rotate right) behaves as Go specifies.
		stdlibImpls["math/bits.RotateLeft"+w.name] = fmt.Sprintf("define %[1]s @NAME(%[1]s %%v, i64 %%k) {\n  %%s = trunc i64 %%k to %[1]s\n  %%r = call %[1]s @llvm.fshl.%[1]s(%[1]s %%v, %[1]s %%v, %[1]s %%s)\n  ret %[1]s %%r\n}", ty)
	}
}

// widen returns IR that sets %r to expr widened to i64.
func widen(ty, expr string) string {
	if ty == "i64" {
		return fmt.Sprintf("  %%r = add i64 %s, 0\n", expr)
	}
	return fmt.Sprintf("  %%r = zext %s %s to i64\n", ty, expr)
}

var reCallIntrinsic = regexp.MustCompile(`@(llvm\.[A-Za-z0-9_.]+)\(`)

// intrinsicDecls returns `declare` lines for every llvm.* intrinsic used in
// bodies that is not already declared in the module. Intrinsic signatures
// are derived from the call: `call <ret> @llvm.x(<args>)`.
func intrinsicDecls(bodies []string, declared map[string]bool) []string {
	reCall := regexp.MustCompile(`call (\S+) @(llvm\.[A-Za-z0-9_.]+)\(((?:[^()]|\([^()]*\))*)\)`)
	var out []string
	seen := map[string]bool{}
	for _, b := range bodies {
		for _, m := range reCall.FindAllStringSubmatch(b, -1) {
			ret, name, args := m[1], m[2], m[3]
			if declared[name] || seen[name] {
				continue
			}
			seen[name] = true
			var tys []string
			for _, a := range strings.Split(args, ",") {
				f := strings.Fields(strings.TrimSpace(a))
				if len(f) == 0 {
					continue
				}
				if f[0] == "ptr" && len(f) > 1 && strings.HasPrefix(f[1], "addrspace") {
					tys = append(tys, f[0]+" "+f[1])
				} else {
					tys = append(tys, f[0])
				}
			}
			out = append(out, fmt.Sprintf("declare %s @%s(%s)", ret, name, strings.Join(tys, ", ")))
		}
	}
	return out
}
