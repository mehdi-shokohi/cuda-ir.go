// Runs the kernels from ../vecadd.go on the GPU through gocudrv
// (pure-Go CUDA driver bindings, no cgo: libcuda.so.1 is dlopen'ed at run time).
//
// The PTX is compiled in-process by cudair.Build (needs llgen + LLVM, like
// the gocuda command). To use a pre-built file instead:
//
//	gocuda build -o vecadd.ptx ./examples/vecadd && go run ./examples/vecadd/run -ptx vecadd.ptx
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/eitamring/gocudrv/cuda"
	"github.com/mehdi-shokohi/cuda-ir.go"
)

func main() {
	ptxFile := flag.String("ptx", "", "use this pre-built PTX instead of compiling the kernels")
	verbose := flag.Bool("v", false, "print the compiler commands")
	flag.Parse()
	must(cuda.Init())
	dev, err := cuda.GetDevice(0)
	must(err)
	name, _ := dev.Name()
	fmt.Println("device:", name)
	ctx, err := dev.Primary()
	must(err)
	defer ctx.Close()
	mod, err := ctx.LoadModule(loadPTX(*ptxFile, *verbose))
	must(err)
	defer mod.Close()

	const n = 1 << 20
	a, b, out := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = 1000 - float32(i)
	}
	bg := context.Background()
	da, err := cuda.Alloc[float32](ctx, n)
	must(err)
	db, err := cuda.Alloc[float32](ctx, n)
	must(err)
	dout, err := cuda.Alloc[float32](ctx, n)
	must(err)
	must(da.CopyFrom(bg, a))
	must(db.CopyFrom(bg, b))
	cfg := cuda.LaunchConfig1D(n, 256)

	// VecAdd: out = a + b.  cuda.Buf[T] params take cuda.Arg(buffer),
	// scalars take cuda.ArgValue(v) with the exact Go type of the parameter.
	k, err := mod.Function("VecAdd")
	must(err)
	must(k.Launch(bg, cfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dout), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	must(dout.CopyTo(bg, out))
	report("VecAdd", out, func(i int) float32 { return a[i] + b[i] })

	// Saxpy (grid-stride loop, 64 blocks): out = 2*a + out
	k, err = mod.Function("Saxpy")
	must(err)
	must(k.Launch(bg, cuda.LaunchConfig1D(64*256, 256), cuda.ArgValue(float32(2)), cuda.Arg(da), cuda.Arg(dout), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	must(dout.CopyTo(bg, out))
	report("Saxpy", out, func(i int) float32 { return 2*a[i] + (a[i] + b[i]) })

	// Square (in place on a)
	k, err = mod.Function("Square")
	must(err)
	must(k.Launch(bg, cfg, cuda.Arg(da), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	must(da.CopyTo(bg, out))
	report("Square", out, func(i int) float32 { return a[i] * a[i] })
}

// loadPTX compiles the kernel package to PTX, or reads file if given.
func loadPTX(file string, verbose bool) []byte {
	if file != "" {
		src, err := os.ReadFile(file)
		must(err)
		return src
	}
	opts := &cudair.Options{}
	if verbose {
		opts.Log = os.Stderr
	}
	res, err := cudair.Build("github.com/mehdi-shokohi/cuda-ir.go/examples/vecadd", opts)
	must(err)
	fmt.Println("kernels:", res.Kernels)
	return res.PTX
}

func report(name string, got []float32, want func(i int) float32) {
	bad := 0
	for i := range got {
		if got[i] != want(i) {
			if bad < 3 {
				fmt.Printf("  %s[%d] = %v, want %v\n", name, i, got[i], want(i))
			}
			bad++
		}
	}
	status := "PASS"
	if bad > 0 {
		status = "FAIL"
	}
	fmt.Printf("%-7s n=%d -> %s (%d mismatches)\n", name, len(got), status, bad)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
