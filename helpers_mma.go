package cudair

import (
	"fmt"
	"strings"
)

// IR helpers for the raw tensor-core instructions: mma.sync (m16n8k16 /
// m16n8k8 / m16n8k32), ldmatrix / stmatrix, and the wmma element types the
// m16n16k16 f16 path in helpers.go does not cover (bf16, tf32, s8/u8).
//
// Every one of these intrinsics returns or takes an aggregate of registers,
// which Go cannot spell, so package cuda holds a fragment as a [N]uint32 /
// [N]float32 / [N]int32 array and the helper loads or stores the aggregate
// through a pointer to it. Each body carries its own `declare`, because
// intrinsicDecls cannot derive a signature with a struct return.

// agg is the IR aggregate type of n registers of type ty.
func agg(ty string, n int) string {
	if n == 1 {
		return ty // a 1-register intrinsic returns the scalar, not a struct
	}
	out := make([]string, n)
	for i := range out {
		out[i] = ty
	}
	return "{ " + strings.Join(out, ", ") + " }"
}

// loadFrag returns IR that loads the n registers of %<name> (type ty) from
// ptr %<ptr> into %<name>0..n-1.
func loadFrag(name, ptr, ty string, n int) string {
	if n == 1 { // a 1-register fragment is the bare scalar
		return fmt.Sprintf("  %%%s0 = load %s, ptr %%%s, align 4\n", name, ty, ptr)
	}
	return fmt.Sprintf("  %%v%s = load %s, ptr %%%s, align 4\n", name, agg(ty, n), ptr) +
		extractAll("v"+name, name, agg(ty, n), n)
}

// mmaShape describes one mma.sync variant: the register counts and IR types
// of its A, B and C/D fragments.
type mmaShape struct {
	geom   string // m16n8k16, m16n8k8, m16n8k32
	sig    string // the intrinsic's type signature (f32.f32, bf16, s8, satfinite.s8 ...)
	abTy   string // IR type of an A/B register
	na, nb int    // A / B register counts
	cdTy   string // IR type of a C/D register
	ncd    int    // C/D register count
}

func init() {
	// ---- mma.sync: one warp, D = A*B + C with the fragment layout fixed by
	// the ISA (unlike wmma, which keeps it opaque). Only row:col is legal.
	const h2 = "<2 x half>"
	for key, s := range map[string]mmaShape{
		"mma.m16n8k16.f16.f32": {"m16n8k16", "f32.f32", h2, 4, 2, "float", 4},
		"mma.m16n8k16.f16.f16": {"m16n8k16", "f16.f16", h2, 4, 2, h2, 2},
		"mma.m16n8k16.bf16":    {"m16n8k16", "bf16", "i32", 4, 2, "float", 4},
		"mma.m16n8k8.tf32":     {"m16n8k8", "tf32", "i32", 4, 2, "float", 4},
		"mma.m16n8k32.s8":      {"m16n8k32", "s8", "i32", 4, 2, "i32", 4},
		"mma.m16n8k32.u8":      {"m16n8k32", "u8", "i32", 4, 2, "i32", 4},
		"mma.m16n8k32.s8.sat":  {"m16n8k32", "satfinite.s8", "i32", 4, 2, "i32", 4},
		"mma.m16n8k32.u8.sat":  {"m16n8k32", "satfinite.u8", "i32", 4, 2, "i32", 4},
	} {
		intr := fmt.Sprintf("llvm.nvvm.mma.%s.row.col.%s", s.geom, s.sig)
		params := strings.Join(append(append(
			repeat(s.abTy, s.na), repeat(s.abTy, s.nb)...), repeat(s.cdTy, s.ncd)...), ", ")
		call := args(s.abTy, "a", s.na) + ", " + args(s.abTy, "b", s.nb) + ", " + args(s.cdTy, "c", s.ncd)
		helpers[key] = fmt.Sprintf(
			"declare %[1]s @%[2]s(%[3]s)\ndefine void @NAME(ptr %%d, ptr %%a, ptr %%b, ptr %%c) {\n%[4]s%[5]s%[6]s  %%r = call %[1]s @%[2]s(%[7]s)\n  store %[1]s %%r, ptr %%d, align 4\n  ret void\n}",
			agg(s.cdTy, s.ncd), intr, params,
			loadFrag("a", "a", s.abTy, s.na), loadFrag("b", "b", s.abTy, s.nb), loadFrag("c", "c", s.cdTy, s.ncd),
			call)
	}

	// ---- ldmatrix / stmatrix (m8n8.b16): the whole warp moves one to four
	// 8x8 matrices between shared memory and registers. Each lane passes the
	// address of the row it contributes; the intrinsic wants addrspace(3).
	for _, n := range []int{1, 2, 4} {
		for _, tr := range []string{"", ".trans"} {
			suffix := fmt.Sprintf("x%d%s", n, tr)
			key := strings.ReplaceAll(suffix, ".", "")
			ld := fmt.Sprintf("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x%d%s.b16.p3", n, tr)
			helpers["ldmatrix."+key] = fmt.Sprintf(
				"declare %[1]s @%[2]s(ptr addrspace(3))\ndefine void @NAME(ptr %%f, ptr %%p) {\n  %%s = addrspacecast ptr %%p to ptr addrspace(3)\n  %%v = call %[1]s @%[2]s(ptr addrspace(3) %%s)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}",
				agg("i32", n), ld)
			st := fmt.Sprintf("llvm.nvvm.stmatrix.sync.aligned.m8n8.x%d%s.b16.p3", n, tr)
			helpers["stmatrix."+key] = fmt.Sprintf(
				"declare void @%[1]s(ptr addrspace(3), %[2]s)\ndefine void @NAME(ptr %%p, ptr %%f) {\n  %%s = addrspacecast ptr %%p to ptr addrspace(3)\n%[3]s  call void @%[1]s(ptr addrspace(3) %%s, %[4]s)\n  ret void\n}",
				st, strings.Join(repeat("i32", n), ", "), loadFrag("f", "f", "i32", n), args("i32", "f", n))
		}
	}

	// ---- wmma m16n16k16 / m16n16k8 for the element types helpers.go's f16
	// path does not cover. The accumulator is 8 registers as there, so the
	// f32 loads/stores are already in place; bf16 and tf32 reuse them and
	// only s8/u8 need the s32 accumulator.
	for _, w := range []struct {
		name, geom, elt string // Go helper key infix, wmma geometry, PTX element type
		nab             int    // A/B register count
		accTy, accElt   string // accumulator IR register type and PTX type
		nacc            int
	}{
		{"bf16", "m16n16k16", "bf16", 4, "float", "f32", 8},
		{"tf32", "m16n16k8", "tf32", 4, "float", "f32", 8},
		{"s8", "m16n16k16", "s8", 2, "i32", "s32", 8},
		{"u8", "m16n16k16", "u8", 2, "i32", "s32", 8},
	} {
		for _, layout := range []string{"row", "col"} {
			for _, m := range []string{"a", "b"} {
				intr := fmt.Sprintf("llvm.nvvm.wmma.%s.load.%s.%s.stride.%s.p0", w.geom, m, layout, w.elt)
				helpers[fmt.Sprintf("wmma.load.%s.%s.%s", w.name, m, layout)] = fmt.Sprintf(
					"declare %[1]s @%[2]s(ptr, i32)\ndefine void @NAME(ptr %%f, ptr %%p, i32 %%ldm) {\n  %%v = call %[1]s @%[2]s(ptr %%p, i32 %%ldm)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}",
					agg("i32", w.nab), intr)
			}
			// the s32 accumulator (s8/u8 only; f32 is already in helpers.go)
			if w.accElt == "s32" {
				ldc := fmt.Sprintf("llvm.nvvm.wmma.%s.load.c.%s.stride.s32.p0", w.geom, layout)
				helpers["wmma.load.s32.c."+layout] = fmt.Sprintf(
					"declare %[1]s @%[2]s(ptr, i32)\ndefine void @NAME(ptr %%f, ptr %%p, i32 %%ldm) {\n  %%v = call %[1]s @%[2]s(ptr %%p, i32 %%ldm)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}",
					agg("i32", w.nacc), ldc)
				std := fmt.Sprintf("llvm.nvvm.wmma.%s.store.d.%s.stride.s32.p0", w.geom, layout)
				helpers["wmma.store.s32.d."+layout] = fmt.Sprintf(
					"declare void @%[1]s(ptr, %[2]s, i32)\ndefine void @NAME(ptr %%p, ptr %%f, i32 %%ldm) {\n%[3]s  call void @%[1]s(ptr %%p, %[4]s, i32 %%ldm)\n  ret void\n}",
					std, strings.Join(repeat("i32", w.nacc), ", "), loadFrag("f", "f", "i32", w.nacc), args("i32", "f", w.nacc))
			}
			// tf32 stores f32 through the m16n16k8 geometry's own intrinsic
			if w.name == "tf32" {
				ldc := fmt.Sprintf("llvm.nvvm.wmma.%s.load.c.%s.stride.f32.p0", w.geom, layout)
				helpers["wmma.load.tf32.c."+layout] = fmt.Sprintf(
					"declare %[1]s @%[2]s(ptr, i32)\ndefine void @NAME(ptr %%f, ptr %%p, i32 %%ldm) {\n  %%v = call %[1]s @%[2]s(ptr %%p, i32 %%ldm)\n  store %[1]s %%v, ptr %%f, align 4\n  ret void\n}",
					agg("float", w.nacc), ldc)
				std := fmt.Sprintf("llvm.nvvm.wmma.%s.store.d.%s.stride.f32.p0", w.geom, layout)
				helpers["wmma.store.tf32.d."+layout] = fmt.Sprintf(
					"declare void @%[1]s(ptr, %[2]s, i32)\ndefine void @NAME(ptr %%p, ptr %%f, i32 %%ldm) {\n%[3]s  call void @%[1]s(ptr %%p, %[4]s, i32 %%ldm)\n  ret void\n}",
					std, strings.Join(repeat("float", w.nacc), ", "), loadFrag("f", "f", "float", w.nacc), args("float", "f", w.nacc))
			}
			for _, lb := range []string{"row", "col"} {
				intr := fmt.Sprintf("llvm.nvvm.wmma.%s.mma.%s.%s.%s", w.geom, layout, lb, w.elt)
				params := strings.Join(append(append(
					repeat("i32", w.nab), repeat("i32", w.nab)...), repeat(w.accTy, w.nacc)...), ", ")
				call := args("i32", "a", w.nab) + ", " + args("i32", "b", w.nab) + ", " + args(w.accTy, "c", w.nacc)
				helpers[fmt.Sprintf("wmma.mma.%s.%s.%s", w.name, layout, lb)] = fmt.Sprintf(
					"declare %[1]s @%[2]s(%[3]s)\ndefine void @NAME(ptr %%d, ptr %%a, ptr %%b, ptr %%c) {\n%[4]s%[5]s%[6]s  %%r = call %[1]s @%[2]s(%[7]s)\n  store %[1]s %%r, ptr %%d, align 4\n  ret void\n}",
					agg(w.accTy, w.nacc), intr, params,
					loadFrag("a", "a", "i32", w.nab), loadFrag("b", "b", "i32", w.nab), loadFrag("c", "c", w.accTy, w.nacc),
					call)
			}
		}
	}
}

// repeat returns n copies of ty.
func repeat(ty string, n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = ty
	}
	return out
}
