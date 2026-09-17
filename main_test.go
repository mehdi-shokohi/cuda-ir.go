package cudair_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
	cudair "github.com/mehdi-shokohi/cuda-ir.go"
)

// TestVecAdd is the README sample: compile examples/vecadd to PTX in-process,
// launch VecAdd through gocudrv and check out = a + b.
func TestVecAdd(t *testing.T) {
	ctx, mod, _ := loadKernels(t, "github.com/mehdi-shokohi/cuda-ir.go/examples/vecadd")
	k, err := mod.Function("VecAdd")
	if err != nil {
		t.Fatal(err)
	}

	const n = 1 << 20
	a, b, out := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = 1000 - float32(i)
	}
	bg := context.Background()
	da := upload(t, ctx, a)
	db := upload(t, ctx, b)
	dout := upload(t, ctx, out)

	if err := k.Launch(bg, cuda.LaunchConfig1D(n, 256), cuda.Arg(da), cuda.Arg(db), cuda.Arg(dout), cuda.ArgValue(int32(n))); err != nil {
		t.Fatal(err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatal(err)
	}
	if err := dout.CopyTo(bg, out); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := range out {
		if i < 10 { // print a sample; n is 1M
			fmt.Printf("out[%d] = %v + %v = %v\n", i, a[i], b[i], out[i])
		}
		if want := a[i] + b[i]; out[i] != want {
			if bad < 3 {
				t.Errorf("out[%d] = %v, want %v", i, out[i], want)
			}
			bad++
		}
	}
	if bad > 0 {
		t.Fatalf("%d mismatches", bad)
	}
}

// TestFeatures runs examples/features (the same checks as examples/features/run):
// shared memory + __syncthreads + warp shuffle + float atomics (BlockSum),
// sync/atomic + warp votes (Count), libdevice math (Math), clock64 (Timing).
func TestFeatures(t *testing.T) {
	ctx, mod, _ := loadKernels(t, "github.com/mehdi-shokohi/cuda-ir.go/examples/features")
	bg := context.Background()

	const n = 1 << 16
	x := make([]float32, n)
	var sum float64
	nbig := 0
	for i := range x {
		x[i] = float32(i%1000) / 1000
		sum += float64(x[i])
		if x[i] > 0.5 {
			nbig++
		}
	}
	dx := upload(t, ctx, x)
	cfg := cuda.LaunchConfig1D(n, 256)

	t.Run("BlockSum", func(t *testing.T) { // shared memory + syncthreads + ShflDownF32 + AtomicAddFloat32
		dout := upload(t, ctx, []float32{0})
		launch(t, ctx, mod, "BlockSum", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		got := make([]float32, 1)
		if err := dout.CopyTo(bg, got); err != nil {
			t.Fatal(err)
		}
		fmt.Printf("BlockSum: gpu=%v cpu=%v\n", got[0], sum)
		if math.Abs(float64(got[0])-sum) > sum*1e-5 {
			t.Fatalf("BlockSum = %v, want %v", got[0], sum)
		}
	})

	t.Run("Count", func(t *testing.T) { // sync/atomic.AddInt32 + warp votes All/Ballot
		dc := upload(t, ctx, []int32{0, 0, 0})
		launch(t, ctx, mod, "Count", cfg, cuda.Arg(dx), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		cnt := make([]int32, 3)
		if err := dc.CopyTo(bg, cnt); err != nil {
			t.Fatal(err)
		}
		// CPU model of the warp votes
		allW, anyW := 0, 0
		for w := 0; w < n/32; w++ {
			all, any := true, false
			for l := 0; l < 32; l++ {
				b := x[w*32+l] > 0.5
				all, any = all && b, any || b
			}
			if all {
				allW++
			}
			if any {
				anyW++
			}
		}
		fmt.Printf("Count: big gpu=%d cpu=%d; warps all gpu=%d cpu=%d; warps any gpu=%d cpu=%d\n",
			cnt[0], nbig, cnt[1], allW, cnt[2], anyW)
		if int(cnt[0]) != nbig {
			t.Errorf("atomic count = %d, want %d", cnt[0], nbig)
		}
		if int(cnt[1]) != allW {
			t.Errorf("All() warps = %d, want %d", cnt[1], allW)
		}
		if int(cnt[2]) != anyW {
			t.Errorf("Ballot() warps = %d, want %d", cnt[2], anyW)
		}
	})

	t.Run("Math", func(t *testing.T) { // libdevice sin/cos/exp/log/pow + llvm sqrt/fma/fabs/floor
		dm := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Math", cfg, cuda.Arg(dx), cuda.Arg(dm), cuda.ArgValue(int32(n)))
		m := make([]float32, n)
		if err := dm.CopyTo(bg, m); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i := range m {
			if math.Abs(float64(m[i])-1) > 1e-4 {
				if bad < 3 {
					t.Errorf("m[%d] = %v, want 1", i, m[i])
				}
				bad++
			}
		}
		fmt.Printf("Math: %d of %d off (m[123]=%v)\n", bad, n, m[123])
		if bad > 0 {
			t.Fatalf("%d of %d off", bad, n)
		}
	})

	t.Run("Timing", func(t *testing.T) { // Clock64
		dt := upload(t, ctx, make([]int64, 4))
		launch(t, ctx, mod, "Timing", cuda.LaunchConfig1D(4*256, 256), cuda.Arg(dt))
		cycles := make([]int64, 4)
		if err := dt.CopyTo(bg, cycles); err != nil {
			t.Fatal(err)
		}
		fmt.Printf("Timing: cycles per block: %v\n", cycles)
		for i, c := range cycles {
			if c <= 0 {
				t.Errorf("block %d: %d cycles, want > 0", i, c)
			}
		}
	})
}

// loadKernels compiles pkg to PTX in-process and loads it on device 0.
// Skips the test when no CUDA driver / device is present.
func loadKernels(t *testing.T, pkg string) (*cuda.Context, *cuda.Module, *cudair.Result) {
	t.Helper()
	return loadKernelsOpts(t, pkg, nil)
}

// loadKernelsOpts is loadKernels with build options (e.g. SM: "sm_90").
func loadKernelsOpts(t *testing.T, pkg string, opts *cudair.Options) (*cuda.Context, *cuda.Module, *cudair.Result) {
	t.Helper()
	res := buildKernels(t, pkg, opts)
	dev := device(t)
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ctx.Close() })
	mod, err := ctx.LoadModule(res.PTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mod.Close() })
	return ctx, mod, res
}

// buildKernels compiles pkg to PTX in-process.
func buildKernels(t *testing.T, pkg string, opts *cudair.Options) *cudair.Result {
	t.Helper()
	if os.Getenv("LLGO_ROOT") == "" {
		os.Setenv("LLGO_ROOT", findLLGoRoot())
	}
	res, err := cudair.Build(pkg, opts) // Go -> PTX, in-process
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Log("kernels:", res.Kernels)
	return res
}

// device returns GPU 0, skipping the test without a driver or device.
func device(t *testing.T) *cuda.Device {
	t.Helper()
	if err := cuda.Init(); err != nil {
		t.Skipf("no CUDA driver: %v", err)
	}
	dev, err := cuda.GetDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	return dev
}

// computeCapability of GPU 0 (skips without a device).
func computeCapability(t *testing.T) (major, minor int) {
	t.Helper()
	major, minor, err := device(t).ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	return major, minor
}

// download copies a device buffer into a new slice.
func download[T cuda.Supported](t *testing.T, d *cuda.Buffer[T]) []T {
	t.Helper()
	out := make([]T, d.Len())
	if err := d.CopyTo(context.Background(), out); err != nil {
		t.Fatal(err)
	}
	return out
}

// near reports whether got is within tol (absolute + relative) of want.
func near(got, want, tol float64) bool {
	d := math.Abs(got - want)
	return d <= tol || d <= tol*math.Abs(want)
}

// upload allocates a device buffer of len(src) and copies src into it.
func upload[T cuda.Supported](t *testing.T, ctx *cuda.Context, src []T) *cuda.Buffer[T] {
	t.Helper()
	d, err := cuda.Alloc[T](ctx, len(src))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.CopyFrom(context.Background(), src); err != nil {
		t.Fatal(err)
	}
	return d
}

// launch runs kernel name with cfg and waits for it.
func launch(t *testing.T, ctx *cuda.Context, mod *cuda.Module, name string, cfg cuda.LaunchConfig, args ...cuda.KernelArg) {
	t.Helper()
	k, err := mod.Function(name)
	if err != nil {
		t.Fatal(err)
	}
	bg := context.Background()
	if err := k.Launch(bg, cfg, args...); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := ctx.Synchronize(bg); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}

// findLLGoRoot guesses the llgo checkout llgen was built from: a sibling
// ../llgo, else ~/llgo (install.sh's default). llgen runs inside the kernel
// package directory, so the path must be absolute.
func findLLGoRoot() string {
	if p, err := filepath.Abs("../llgo"); err == nil {
		if _, err := os.Stat(filepath.Join(p, "runtime", "go.mod")); err == nil {
			return p
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "llgo")
}

// TestSharedTypeUnification: a cuda.Shared whose layout equals another
// struct's (FragC) is still placed in __shared__ after llvm-link unified
// the type names, and every block gets its own copy.
func TestSharedTypeUnification(t *testing.T) {
	res := buildKernels(t, "github.com/mehdi-shokohi/cuda-ir.go/examples/features", nil)
	ptx := string(res.PTX)
	if !strings.Contains(ptx, ".shared .align 16 .b8 github_com_mehdi_shokohi_cuda_ir_go_examples_features_scratch[32]") {
		t.Fatalf("scratch is not __shared__:\n%s", grepLines(ptx, "features_scratch"))
	}
	ctx, mod, _ := loadKernels(t, "github.com/mehdi-shokohi/cuda-ir.go/examples/features")
	bg := context.Background()
	const blocks, n = 512, 512 * 256
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i / 32 % 7) // constant within a warp: warp w of block b sums to 32*((b*8+w)%7)
	}
	in, out := upload(t, ctx, x), upload(t, ctx, make([]float32, blocks*8))
	fn, err := mod.Function("WarpSums")
	if err != nil {
		t.Fatal(err)
	}
	if err := fn.Launch(bg, cuda.LaunchConfig{GridX: blocks, GridY: 1, GridZ: 1, BlockX: 256, BlockY: 1, BlockZ: 1}, cuda.Arg(in), cuda.Arg(out), cuda.ArgValue(int32(n))); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, blocks*8)
	if err := out.CopyTo(bg, got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if want := float32(32 * (i % 7)); v != want {
			t.Fatalf("out[%d] = %v, want %v (blocks share scratch?)", i, v, want)
		}
	}
}

func grepLines(s, sub string) string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.Contains(l, sub) {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}
