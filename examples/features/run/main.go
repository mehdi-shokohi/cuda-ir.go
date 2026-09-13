// Runs the kernels from ../features.go on the GPU and checks each result
// against a CPU model. The PTX is compiled in-process by cudair.Build;
// -ptx file uses a pre-built one instead (see examples/vecadd/run).
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
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
	ctx, err := dev.Primary()
	must(err)
	defer ctx.Close()
	mod, err := ctx.LoadModule(loadPTX(*ptxFile, *verbose))
	must(err)
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
	dx, _ := cuda.Alloc[float32](ctx, n)
	must(dx.CopyFrom(bg, x))
	cfg := cuda.LaunchConfig1D(n, 256)

	// BlockSum: shared memory + syncthreads + shfl + float atomic
	dout, _ := cuda.Alloc[float32](ctx, 1)
	must(dout.CopyFrom(bg, []float32{0}))
	k, err := mod.Function("BlockSum")
	must(err)
	must(k.Launch(bg, cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	got := make([]float32, 1)
	must(dout.CopyTo(bg, got))
	check("BlockSum (shared+sync+shfl+atomicAdd f32)", math.Abs(float64(got[0])-sum) < sum*1e-5, fmt.Sprintf("gpu=%v cpu=%v", got[0], sum))

	// Count: sync/atomic + warp votes
	dc, _ := cuda.Alloc[int32](ctx, 3)
	must(dc.CopyFrom(bg, []int32{0, 0, 0}))
	k, err = mod.Function("Count")
	must(err)
	must(k.Launch(bg, cfg, cuda.Arg(dx), cuda.Arg(dc), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	cnt := make([]int32, 3)
	must(dc.CopyTo(bg, cnt))
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
	check("Count (sync/atomic.AddInt32)", int(cnt[0]) == nbig, fmt.Sprintf("gpu=%d cpu=%d", cnt[0], nbig))
	check("All/Ballot (warp vote)", int(cnt[1]) == allW && int(cnt[2]) == anyW, fmt.Sprintf("gpu=%v cpu=%d,%d", cnt[1:], allW, anyW))

	// Math via libdevice + llvm intrinsics
	dm, _ := cuda.Alloc[float32](ctx, n)
	k, err = mod.Function("Math")
	must(err)
	must(k.Launch(bg, cfg, cuda.Arg(dx), cuda.Arg(dm), cuda.ArgValue(int32(n))))
	must(ctx.Synchronize(bg))
	m := make([]float32, n)
	must(dm.CopyTo(bg, m))
	bad := 0
	for i := range m {
		if math.Abs(float64(m[i])-1) > 1e-4 {
			bad++
		}
	}
	check("Math (libdevice sin/cos/exp/log/pow + llvm sqrt/fma/fabs/floor)", bad == 0, fmt.Sprintf("%d of %d off (m[123]=%v)", bad, n, m[123]))

	// Timing: clock64
	dt, _ := cuda.Alloc[int64](ctx, 4)
	k, err = mod.Function("Timing")
	must(err)
	must(k.Launch(bg, cuda.LaunchConfig1D(4*256, 256), cuda.Arg(dt)))
	must(ctx.Synchronize(bg))
	t := make([]int64, 4)
	must(dt.CopyTo(bg, t))
	check("Clock64", t[0] > 0 && t[1] > 0, fmt.Sprintf("cycles per block: %v", t))
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
	res, err := cudair.Build("github.com/mehdi-shokohi/cuda-ir.go/examples/features", opts)
	must(err)
	return res.PTX
}

func check(name string, ok bool, detail string) {
	s := "PASS"
	if !ok {
		s = "FAIL"
	}
	fmt.Printf("%-4s %-70s %s\n", s, name, detail)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
