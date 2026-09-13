package cudair_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/mehdi-shokohi/cuda-ir.go/host"
)

const memPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/memory"

// TestMemory runs examples/memory: dynamic shared memory, __constant__ /
// __device__ globals, launch bounds, printf, Go copy/panic/slices,
// __ldg / volatile / float4, half precision, cp.async, grid barrier.
func TestMemory(t *testing.T) {
	ctx, mod, res := loadKernels(t, memPkg)
	bg := context.Background()
	ptx := string(res.PTX)
	const n = 4096
	cfg := cuda.LaunchConfig1D(n, 256)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i) / n
	}
	dx := upload(t, ctx, x)

	t.Run("DynShared", func(t *testing.T) { // extern __shared__ + SharedMemBytes
		if !strings.Contains(ptx, ".extern .shared .align 16 .b8") {
			t.Fatal("no .extern .shared in the PTX")
		}
		dout := upload(t, ctx, make([]float32, n))
		c := cfg
		c.SharedMemBytes = 256 * 4
		launch(t, ctx, mod, "DynReverse", c, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i := range out {
			if want := x[i&^255+255-i&255]; out[i] != want {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
		fmt.Printf("DynShared: %d elements reversed per block through %d bytes of dynamic shared memory\n", n, c.SharedMemBytes)
	})

	t.Run("Globals", func(t *testing.T) { // __constant__ Coef, __device__ Scale (host writes) and Counter (host reads)
		if fmt.Sprint(res.Globals) != "[Coef Counter Scale]" {
			t.Fatalf("res.Globals = %v", res.Globals)
		}
		for _, want := range []string{".const .align 4 .b8 Coef[16]", ".global .align 4 .u32 Counter", "ld.const"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Poly", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i, v := range x {
			if want := 1 + v*(2+v*(3+v*4)); !near(float64(out[i]), float64(want), 1e-5) {
				t.Fatalf("Poly(%v) = %v, want %v (initialiser)", v, out[i], want)
			}
		}
		counter, err := mod.Global("Counter")
		if err != nil {
			t.Fatal(err)
		}
		cnt := make([]int32, 1)
		if err := cuda.ReadGlobal(bg, cnt, counter); err != nil {
			t.Fatal(err)
		}
		if cnt[0] != n {
			t.Fatalf("Counter = %d, want %d", cnt[0], n)
		}
		// host rewrites the constant bank and a device global, kernel sees it
		coef, err := mod.Global("Coef")
		if err != nil {
			t.Fatal(err)
		}
		if err := cuda.WriteGlobal(bg, coef, []float32{0, 1, 0, 0}); err != nil {
			t.Fatal(err)
		}
		scale, err := mod.Global("Scale")
		if err != nil {
			t.Fatal(err)
		}
		if err := cuda.WriteGlobal(bg, scale, []float32{3}); err != nil {
			t.Fatal(err)
		}
		launch(t, ctx, mod, "Poly", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out = download(t, dout)
		for i, v := range x {
			if want := 3 * v; !near(float64(out[i]), float64(want), 1e-5) {
				t.Fatalf("Poly(%v) = %v, want %v (after WriteGlobal)", v, out[i], want)
			}
		}
		fmt.Printf("Globals: %v; Counter read back = %d; Coef/Scale rewritten from the host\n", res.Globals, cnt[0])
	})

	t.Run("LaunchBounds", func(t *testing.T) { // //cuda:launch_bounds 128 2, //cuda:maxnreg 32
		for _, want := range []string{".maxntid 128", ".minnctapersm 2", ".maxnreg 32"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout := upload(t, ctx, make([]int32, n))
		launch(t, ctx, mod, "Bounded", cuda.LaunchConfig1D(n, 128), cuda.Arg(dout))
		k, _ := mod.Function("Bounded")
		if err := k.Launch(bg, cuda.LaunchConfig1D(n, 256), cuda.Arg(dout)); err == nil {
			t.Fatal("launch with 256 threads > maxntid 128 succeeded")
		} else {
			fmt.Printf("LaunchBounds: .maxntid 128 / .minnctapersm 2 / .maxnreg 32 in PTX; 256-thread launch rejected: %v\n", err)
		}
	})

	t.Run("Printf", func(t *testing.T) { // device printf (vprintf)
		got := captureStdout(t, func() {
			launch(t, ctx, mod, "Print", cuda.LaunchConfig1D(3*32, 32), cuda.Arg(dx))
		})
		for b := 0; b < 3; b++ {
			want := fmt.Sprintf("block %d: x=%.2f n=%d hex=beef\n", b, x[b], int64(b)*1000000000000)
			if !strings.Contains(got, want) {
				t.Fatalf("printf output %q lacks %q", got, want)
			}
		}
		fmt.Printf("Printf: captured %q\n", got)
	})

	t.Run("PanicCopy", func(t *testing.T) { // copy() into a local array, range, panic -> trap
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "PanicCopy", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)), cuda.ArgValue(int32(0)))
		out := download(t, dout)
		for i := range out {
			w := x[i&^3 : i&^3+4]
			if want := w[0] + w[1] + w[2] + w[3]; out[i] != want {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
		fmt.Printf("PanicCopy: copy() + range over a local array OK (bad=0)\n")
	})

	t.Run("Slices", func(t *testing.T) { // []T kernel parameters via host.ArgSlice
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Slices", cfg, host.ArgSlice(dout), host.ArgSlice(dx))
		out := download(t, dout)
		for i := range out {
			if out[i] != 2*x[i] {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], 2*x[i])
			}
		}
		fmt.Printf("Slices: []float32 parameters (24-byte .param) with bounds checks OK\n")
	})

	t.Run("Vec4", func(t *testing.T) { // 128-bit ld.global.v4 / st.global.v4
		if !strings.Contains(ptx, "ld.global.v4.b32") || !strings.Contains(ptx, "st.global.v4.b32") {
			t.Fatal("no ld.global.v4 / st.global.v4 in the PTX")
		}
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Vec4", cuda.LaunchConfig1D(n/4, 256), cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n/4)))
		out := download(t, dout)
		for i := range out {
			if out[i] != 2*x[i] {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], 2*x[i])
			}
		}
		fmt.Printf("Vec4: LoadFloat4/StoreFloat4 -> ld.global.v4 / st.global.v4\n")
	})

	t.Run("Ldg", func(t *testing.T) { // __ldg -> ld.global.nc
		if !strings.Contains(ptx, "ld.global.nc.b32") {
			t.Fatal("no ld.global.nc in the PTX")
		}
		idx := make([]int32, n)
		for i := range idx {
			idx[i] = int32((i * 37) % n)
		}
		didx := upload(t, ctx, idx)
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Ldg", cfg, cuda.Arg(dx), cuda.Arg(didx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i := range out {
			if want := x[i] + x[idx[i]]; out[i] != want {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
		fmt.Printf("Ldg: LdgF32/LdgI32 -> ld.global.nc\n")
	})

	t.Run("Volatile", func(t *testing.T) { // ld.volatile / st.volatile
		if !strings.Contains(ptx, "ld.volatile.global") || !strings.Contains(ptx, "st.volatile.global") {
			t.Fatal("no ld.volatile / st.volatile in the PTX")
		}
		dflag := upload(t, ctx, make([]int32, n/256))
		dout := upload(t, ctx, make([]int32, n))
		launch(t, ctx, mod, "Volatile", cfg, cuda.Arg(dflag), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i := range out {
			if out[i] != int32(i/256+1) {
				t.Fatalf("out[%d] = %d, want %d", i, out[i], i/256+1)
			}
		}
		fmt.Printf("Volatile: flag polling with VolatileLoad/StoreInt32 OK\n")
	})

	t.Run("Half", func(t *testing.T) { // __half / __nv_bfloat16 conversions + fma
		for _, want := range []string{"cvt.rn.f16.f32", "fma.rn.f16", "cvt.rn.bf16.f32", "fma.rn.bf16"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout := upload(t, ctx, make([]float32, 2*n))
		launch(t, ctx, mod, "HalfOps", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i, v := range x {
			f := float64(v)
			if !near(float64(out[2*i]), f*f+1, 2e-3) { // binary16: 11-bit significand
				t.Fatalf("half x*x+1 (%v) = %v", v, out[2*i])
			}
			if !near(float64(out[2*i+1]), 3*f, 1.6e-2) { // bfloat16: 8-bit significand
				t.Fatalf("bf16 3x (%v) = %v", v, out[2*i+1])
			}
		}
		fmt.Printf("Half: FloatToHalf/FMA and bfloat16 Mul/Add within precision (x=0.5: %v %v)\n", out[n], out[n+1])
	})

	t.Run("CpAsync", func(t *testing.T) { // cp.async.ca.shared.global + commit/wait
		if !strings.Contains(ptx, "cp.async.ca.shared.global") || !strings.Contains(ptx, "cp.async.wait_all") {
			t.Fatal("no cp.async in the PTX")
		}
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "CpAsyncReverse", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		out := download(t, dout)
		for i := range out {
			if want := x[i&^255+255-i&255]; out[i] != want {
				t.Fatalf("out[%d] = %v, want %v", i, out[i], want)
			}
		}
		fmt.Printf("CpAsync: global -> shared tile via cp.async, reversed back OK\n")
	})

	t.Run("GridSync", func(t *testing.T) { // grid-wide barrier with a cooperative launch
		const blocks = 8
		k, err := mod.Function("GridSum")
		if err != nil {
			t.Fatal(err)
		}
		if max, err := k.MaxCooperativeGridBlocks(256, 0); err != nil {
			t.Skipf("cooperative launch unavailable: %v", err)
		} else if max < blocks {
			t.Skipf("only %d blocks can be co-resident", max)
		}
		dbar := upload(t, ctx, []uint32{0, 0})
		ddata := upload(t, ctx, make([]int32, blocks))
		dout := upload(t, ctx, make([]int32, blocks))
		if err := k.LaunchCooperative(bg, cuda.LaunchConfig1D(blocks*256, 256), cuda.Arg(dbar), cuda.Arg(ddata), cuda.Arg(dout)); err != nil {
			t.Fatal(err)
		}
		if err := ctx.Synchronize(bg); err != nil {
			t.Fatal(err)
		}
		out := download(t, dout)
		for b, v := range out {
			if v != blocks*(blocks+1)/2 {
				t.Fatalf("block %d saw sum %d, want %d", b, v, blocks*(blocks+1)/2)
			}
		}
		fmt.Printf("GridSync: %d blocks synchronised through cuda.GridBarrier: %v\n", blocks, out)
	})
}

// TestTraps runs the kernels that must fail (Go panic, slice index out of
// range) in a child process: a device trap poisons the CUDA context.
func TestTraps(t *testing.T) {
	if os.Getenv("CUDAIR_TRAP_CHILD") != "" {
		ctx, mod, _ := loadKernels(t, memPkg)
		bg := context.Background()
		dout := upload(t, ctx, make([]float32, 64))
		var k *cuda.Function
		var args []cuda.KernelArg
		switch os.Getenv("CUDAIR_TRAP_CHILD") {
		case "panic":
			k, _ = mod.Function("PanicCopy")
			args = []cuda.KernelArg{cuda.Arg(dout), cuda.Arg(dout), cuda.ArgValue(int32(64)), cuda.ArgValue(int32(1))}
		case "oob":
			k, _ = mod.Function("SliceOOB")
			args = []cuda.KernelArg{host.ArgSlice(dout)}
		}
		err := k.Launch(bg, cuda.LaunchConfig1D(64, 64), args...)
		if err == nil {
			err = ctx.Synchronize(bg)
		}
		if err == nil {
			t.Fatal("kernel did not trap")
		}
		fmt.Printf("trapped: %v\n", err)
		return
	}
	for _, mode := range []string{"panic", "oob"} {
		t.Run(mode, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestTraps$", "-test.v")
			cmd.Env = append(os.Environ(), "CUDAIR_TRAP_CHILD="+mode, "LLGO_ROOT="+findLLGoRoot())
			out, err := cmd.CombinedOutput()
			if err != nil || !bytes.Contains(out, []byte("trapped:")) {
				t.Fatalf("child failed: %v\n%s", err, out)
			}
			line := out[bytes.Index(out, []byte("trapped:")):]
			fmt.Printf("Traps/%s: %s", mode, line[:bytes.IndexByte(line, '\n')+1])
		})
	}
}

// captureStdout runs fn with fd 1 redirected to a pipe (the CUDA driver
// writes device printf output to the process's stdout) and returns it.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	saved, err := syscall.Dup(1)
	if err != nil {
		t.Fatal(err)
	}
	if err := syscall.Dup2(int(w.Fd()), 1); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	done := make(chan struct{})
	go func() { io.Copy(&buf, r); close(done) }()
	fn()
	syscall.Dup2(saved, 1)
	syscall.Close(saved)
	w.Close()
	<-done
	r.Close()
	return buf.String()
}
