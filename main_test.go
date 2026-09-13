package cudair_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
	cudair "github.com/mehdi-shokohi/cuda-ir.go"
)

// TestVecAdd is the README sample: compile examples/vecadd to PTX in-process,
// launch VecAdd through gocudrv and check out = a + b.
func TestVecAdd(t *testing.T) {
	if os.Getenv("LLGO_ROOT") == "" {
		os.Setenv("LLGO_ROOT", findLLGoRoot())
	}
	res, err := cudair.Build("github.com/mehdi-shokohi/cuda-ir.go/examples/vecadd", nil) // Go -> PTX, in-process
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	t.Log("kernels:", res.Kernels)

	if err := cuda.Init(); err != nil {
		t.Skipf("no CUDA driver: %v", err)
	}
	dev, err := cuda.GetDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Close()
	mod, err := ctx.LoadModule(res.PTX)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()
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
	da, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		t.Fatal(err)
	}
	db, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		t.Fatal(err)
	}
	dout, err := cuda.Alloc[float32](ctx, n)
	if err != nil {
		t.Fatal(err)
	}
	if err := da.CopyFrom(bg, a); err != nil {
		t.Fatal(err)
	}
	if err := db.CopyFrom(bg, b); err != nil {
		t.Fatal(err)
	}

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
