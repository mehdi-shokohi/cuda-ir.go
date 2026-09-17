package cudair_test

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
	"unsafe"

	"github.com/eitamring/gocudrv/cuda"
	cudair "github.com/mehdi-shokohi/cuda-ir.go"
)

const advPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/advanced"

// TestAdvanced runs examples/advanced: tensor cores, textures and
// surfaces, __restrict__, __grid_constant__, cache-hint loads/stores, the
// cooperative-groups sugar and the %pm counters.
func TestAdvanced(t *testing.T) {
	ctx, mod, res := loadKernels(t, advPkg)
	bg := context.Background()
	ptx := string(res.PTX)

	t.Run("Wmma", func(t *testing.T) { // wmma.load/mma/store m16n16k16 f16 -> f32
		for _, want := range []string{"wmma.load.a.sync.aligned.row.m16n16k16.f16", "wmma.load.b.sync.aligned.col.m16n16k16.f16",
			"wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32", "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32",
			"wmma.store.d.sync.aligned.row.m16n16k16.f32", "wmma.store.d.sync.aligned.col.m16n16k16.f32"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		const n = 64
		a, b, bt := make([]uint16, n*n), make([]uint16, n*n), make([]uint16, n*n)
		af, bf, bias := make([]float32, n*n), make([]float32, n*n), make([]float32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				af[i*n+j] = float32((i+j)%5 - 2)
				bf[i*n+j] = float32((i * j) % 3)
				bias[i*n+j] = float32(i - j)
				a[i*n+j] = float32ToHalf(af[i*n+j])
				b[i*n+j] = float32ToHalf(bf[i*n+j])
				bt[j*n+i] = b[i*n+j]
			}
		}
		want := make([]float32, n*n) // C = A*B + bias on the CPU
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				s := bias[i*n+j]
				for k := 0; k < n; k++ {
					s += af[i*n+k] * bf[k*n+j]
				}
				want[i*n+j] = s
			}
		}
		da, db, dbt := upload(t, ctx, a), upload(t, ctx, b), upload(t, ctx, bt)
		dbias := upload(t, ctx, bias)
		dc := upload(t, ctx, make([]float32, n*n))
		warps := (n / 16) * (n / 16)
		cfg := cuda.LaunchConfig1D(warps*32, 128)
		launch(t, ctx, mod, "Wmma", cfg, cuda.Arg(da), cuda.Arg(db), cuda.Arg(dbias), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		out := download(t, dc)
		for i := range out {
			if out[i] != want[i] {
				t.Fatalf("Wmma: C[%d][%d] = %v, want %v", i/n, i%n, out[i], want[i])
			}
		}
		// WmmaTrans: B given transposed (read column-major), no bias, result stored column-major
		dc2 := upload(t, ctx, make([]float32, n*n))
		launch(t, ctx, mod, "WmmaTrans", cfg, cuda.Arg(da), cuda.Arg(dbt), cuda.Arg(dc2), cuda.ArgValue(int32(n)))
		out = download(t, dc2)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if got := out[j*n+i]; got != want[i*n+j]-bias[i*n+j] {
					t.Fatalf("WmmaTrans: C[%d][%d] = %v, want %v", i, j, got, want[i*n+j]-bias[i*n+j])
				}
			}
		}
		fmt.Printf("Wmma: %dx%d half matmul + bias on tensor cores matches the CPU (row.row and row.col, row/col stores)\n", n, n)
	})

	t.Run("Texture", func(t *testing.T) { // tex.2d.v4.f32.f32 with point and linear filtering, txq
		for _, want := range []string{"tex.2d.v4.f32.f32", "txq.width", "txq.height"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		const w, h = 64, 32
		data := make([]float32, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				data[y*w+x] = float32(x + 100*y)
			}
		}
		arr, err := cuda.AllocArray2D[float32](ctx, w, h)
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(bg, data); err != nil {
			t.Fatal(err)
		}
		cfg := cuda.LaunchConfig{GridX: w / 16, GridY: h / 16, GridZ: 1, BlockX: 16, BlockY: 16, BlockZ: 1}
		dout := upload(t, ctx, make([]float32, w*h))
		ddims := upload(t, ctx, make([]int32, 2))
		for _, tc := range []struct {
			name string
			cfg  cuda.TextureConfig
			dx   float32
			want func(x, y int) float32
		}{
			{"point", cuda.TextureConfig{FilterMode: cuda.FilterPoint, AddressMode: cuda.AddressClamp}, 0.5, func(x, y int) float32 { return data[y*w+x] }},
			{"linear", cuda.TextureConfig{FilterMode: cuda.FilterLinear, AddressMode: cuda.AddressClamp}, 1, func(x, y int) float32 {
				if x == w-1 {
					return data[y*w+x] // clamped
				}
				return (data[y*w+x] + data[y*w+x+1]) / 2
			}},
		} {
			tex, err := cuda.NewTexture(arr, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			launch(t, ctx, mod, "TexSample", cfg, cuda.ArgTexture(tex), cuda.Arg(dout), cuda.Arg(ddims), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)), cuda.ArgValue(tc.dx))
			tex.Close()
			out := download(t, dout)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if want := tc.want(x, y); !near(float64(out[y*w+x]), float64(want), 1e-2) {
						t.Fatalf("%s: tex(%d, %d) = %v, want %v", tc.name, x, y, out[y*w+x], want)
					}
				}
			}
		}
		if dims := download(t, ddims); dims[0] != w || dims[1] != h {
			t.Fatalf("TexWidth/TexHeight = %v, want [%d %d]", dims, w, h)
		}
		fmt.Printf("Texture: tex2D point and linear filtering over a %dx%d CUDA array OK, txq.width/height = %v\n", w, h, download(t, ddims))
	})

	t.Run("Surface", func(t *testing.T) { // suld/sust.b.2d.b32
		if !strings.Contains(ptx, "suld.b.2d.b32.trap") || !strings.Contains(ptx, "sust.b.2d.b32.trap") {
			t.Fatal("no suld/sust in the PTX")
		}
		const w, h = 32, 16
		cfg := cuda.LaunchConfig{GridX: w / 16, GridY: h / 16, GridZ: 1, BlockX: 16, BlockY: 16, BlockZ: 1}
		fdata := make([]float32, w*h)
		idata := make([]int32, w*h)
		for i := range fdata {
			fdata[i] = float32(i) * 0.25
			idata[i] = int32(i * 3)
		}
		farr, err := cuda.AllocArray2D[float32](ctx, w, h, cuda.WithSurfaceStore())
		if err != nil {
			t.Fatal(err)
		}
		defer farr.Close()
		if err := farr.CopyFrom(bg, fdata); err != nil {
			t.Fatal(err)
		}
		fs, err := cuda.NewSurface(farr)
		if err != nil {
			t.Fatal(err)
		}
		launch(t, ctx, mod, "SurfScale", cfg, cuda.ArgSurface(fs), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)))
		fs.Close()
		fout := make([]float32, w*h)
		if err := farr.CopyTo(bg, fout); err != nil {
			t.Fatal(err)
		}
		for i := range fout {
			if fout[i] != 2*fdata[i] {
				t.Fatalf("SurfScale[%d] = %v, want %v", i, fout[i], 2*fdata[i])
			}
		}
		iarr, err := cuda.AllocArray2D[int32](ctx, w, h, cuda.WithSurfaceStore())
		if err != nil {
			t.Fatal(err)
		}
		defer iarr.Close()
		if err := iarr.CopyFrom(bg, idata); err != nil {
			t.Fatal(err)
		}
		is, err := cuda.NewSurface(iarr)
		if err != nil {
			t.Fatal(err)
		}
		launch(t, ctx, mod, "SurfInt", cfg, cuda.ArgSurface(is), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)))
		is.Close()
		iout := make([]int32, w*h)
		if err := iarr.CopyTo(bg, iout); err != nil {
			t.Fatal(err)
		}
		for i := range iout {
			if iout[i] != idata[i]+1 {
				t.Fatalf("SurfInt[%d] = %v, want %v", i, iout[i], idata[i]+1)
			}
		}
		fmt.Printf("Surface: surf2Dread/write on float and int %dx%d surfaces OK\n", w, h)
	})

	const n = 4096
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i) / 64
	}
	dx := upload(t, ctx, x)
	cfg := cuda.LaunchConfig1D(n, 256)

	t.Run("Restrict", func(t *testing.T) { // //cuda:restrict -> noalias -> ld.global.nc without LdgF32
		if !strings.Contains(entry(ptx, "Restrict"), "ld.global.nc") {
			t.Fatal("Restrict: no ld.global.nc in the PTX")
		}
		if strings.Contains(entry(ptx, "NoRestrict"), "ld.global.nc") {
			t.Fatal("NoRestrict: unexpected ld.global.nc")
		}
		dout := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Restrict", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		for i, v := range download(t, dout) {
			if v != 2*x[i] {
				t.Fatalf("out[%d] = %v, want %v", i, v, 2*x[i])
			}
		}
		fmt.Printf("Restrict: noalias parameters -> ld.global.nc (NoRestrict: plain ld.global)\n")
	})

	t.Run("GridConstant", func(t *testing.T) { // //cuda:grid_constant -> byval param, cvta.param, no local copy
		if e := entry(ptx, "GridConst"); !strings.Contains(e, "cvta.param") || strings.Contains(e, ".local") {
			t.Fatal("GridConst: expected cvta.param and no .local in the PTX")
		}
		if !strings.Contains(entry(ptx, "LocalCopy"), ".local") {
			t.Fatal("LocalCopy: expected a .local copy of the parameter")
		}
		type params struct {
			Scale, Offset float32
			N             int32
		}
		p := params{Scale: 3, Offset: 0.5, N: n}
		for _, k := range []string{"GridConst", "LocalCopy"} {
			dout := upload(t, ctx, make([]float32, n))
			launch(t, ctx, mod, k, cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.ArgRaw(unsafe.Pointer(&p), int(unsafe.Sizeof(p))))
			for i, v := range download(t, dout) {
				if want := p.Scale*x[i] + p.Offset; v != want {
					t.Fatalf("%s: out[%d] = %v, want %v", k, i, v, want)
				}
			}
		}
		fmt.Printf("GridConstant: &param in GridConst is cvta.param with no .local depot (LocalCopy has one)\n")
	})

	t.Run("CacheHints", func(t *testing.T) { // ld.global.{ca,cg,cs,lu,cv} / st.global.{wb,cg,cs,wt}
		// Buf.Load/Store move the bits (b32); the typed *float64 functions use f64
		for _, want := range []string{"ld.global.ca.b32", "ld.global.cg.b32", "ld.global.cs.b32", "ld.global.lu.b32", "ld.global.cv.b32",
			"st.global.wb.b32", "st.global.cg.b32", "st.global.wt.b32", "st.global.cs.b32",
			"ld.global.cs.b64", "ld.global.ca.b64", "ld.global.cv.f64", "ld.global.lu.f64", "st.global.wb.b64", "st.global.cg.f64"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		iin, lin, din := make([]int32, n), make([]int64, n), make([]float64, n)
		for i := range iin {
			iin[i], lin[i], din[i] = int32(i-n/2), int64(i)<<33, float64(i)*1e-3
		}
		diin, dlin, ddin := upload(t, ctx, iin), upload(t, ctx, lin), upload(t, ctx, din)
		dout, diout := upload(t, ctx, make([]float32, n)), upload(t, ctx, make([]int32, n))
		dlout, ddout := upload(t, ctx, make([]int64, n)), upload(t, ctx, make([]float64, n))
		launch(t, ctx, mod, "CacheHints", cfg, cuda.Arg(dx), cuda.Arg(dout), cuda.Arg(diin), cuda.Arg(diout),
			cuda.Arg(dlin), cuda.Arg(dlout), cuda.Arg(ddin), cuda.Arg(ddout), cuda.ArgValue(int32(n)))
		out, iout, lout, dout2 := download(t, dout), download(t, diout), download(t, dlout), download(t, ddout)
		for i := range out {
			if out[i] != 4*x[i] || iout[i] != 2*iin[i] || lout[i] != 2*lin[i] || dout2[i] != 2*din[i] {
				t.Fatalf("[%d] = %v / %d / %d / %v, want %v / %d / %d / %v", i, out[i], iout[i], lout[i], dout2[i], 4*x[i], 2*iin[i], 2*lin[i], 2*din[i])
			}
		}
		fmt.Printf("CacheHints: every ld.global.{ca,cg,cs,lu,cv} / st.global.{wb,cg,cs,wt} variant present and correct\n")
	})

	t.Run("Groups", func(t *testing.T) { // thread_block / tiled_partition<8> / this warp / coalesced_threads
		in := make([]int32, n)
		for i := range in {
			in[i] = int32((i * 7) % 13)
		}
		din := upload(t, ctx, in)
		dtiles, dwarps, dranks := upload(t, ctx, make([]int32, n/8)), upload(t, ctx, make([]int32, n/32)), upload(t, ctx, make([]int32, n))
		dcount, desum := upload(t, ctx, make([]int32, 1)), upload(t, ctx, make([]int32, n))
		launch(t, ctx, mod, "Groups", cfg, cuda.Arg(din), cuda.Arg(dtiles), cuda.Arg(dwarps), cuda.Arg(dranks), cuda.Arg(desum), cuda.Arg(dcount), cuda.ArgValue(int32(n)))
		tiles, warps, ranks, esum, count := download(t, dtiles), download(t, dwarps), download(t, dranks), download(t, desum), download(t, dcount)
		evens := int32(0)
		for i := 0; i < n; i++ {
			if i%8 == 0 {
				var s int32
				for k := 0; k < 8; k++ {
					s += in[i+k]
				}
				if tiles[i/8] != s {
					t.Fatalf("tile %d sum = %d, want %d", i/8, tiles[i/8], s)
				}
			}
			if i%32 == 0 {
				m, first, esumWant := int32(-1), int32(-1), int32(0)
				for k := 0; k < 32; k++ {
					if in[i+k] > m {
						m = in[i+k]
					}
					if in[i+k]%2 == 0 {
						esumWant += in[i+k]
						if first < 0 {
							first = in[i+k]
						}
					}
				}
				if warps[i/32] != m {
					t.Fatalf("warp %d max = %d, want %d", i/32, warps[i/32], m)
				}
				for k := 0; k < 32; k++ {
					want := int32(i%256+k)*1000 + int32(k/8)*10 + int32(k%8)
					if in[i+k]%2 == 0 {
						want += first * 100000
						if esum[i+k] != esumWant {
							t.Fatalf("coalesced sum at %d = %d, want %d", i+k, esum[i+k], esumWant)
						}
					}
					if ranks[i+k] != want {
						t.Fatalf("ranks[%d] = %d, want %d", i+k, ranks[i+k], want)
					}
				}
			}
			if in[i]%2 == 0 {
				evens++
			}
		}
		if count[0] != evens {
			t.Fatalf("coalesced count = %d, want %d", count[0], evens)
		}
		fmt.Printf("Groups: tiled_partition<8> sums, warp max, block/tile ranks, coalesced_threads() count (%d evens) and reduce OK\n", evens)
	})

	t.Run("PerfCounters", func(t *testing.T) { // %pm0..3
		for _, want := range []string{"%pm0", "%pm1", "%pm2", "%pm3"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout := upload(t, ctx, make([]int32, 4))
		launch(t, ctx, mod, "PerfCounters", cuda.LaunchConfig1D(32, 32), cuda.Arg(dout))
		fmt.Printf("PerfCounters: %%pm0..3 = %v\n", download(t, dout))
	})
}

const hopperPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/hopper"

// TestHopper runs examples/hopper (sm_90): thread block clusters with
// distributed shared memory, mbarrier, TMA bulk copies, elect.sync and
// griddepcontrol. Skipped on GPUs below compute capability 9.0.
func TestHopper(t *testing.T) {
	if major, minor := computeCapability(t); major < 9 {
		t.Skipf("compute capability %d.%d < 9.0", major, minor)
	}
	ctx, mod, res := loadKernelsOpts(t, hopperPkg, &cudair.Options{SM: "sm_90", PTX: "80"})
	ptx := string(res.PTX)
	const blocks, bs = 8, 256
	cfg := cuda.LaunchConfig1D(blocks*bs, bs)

	t.Run("Cluster", func(t *testing.T) { // //cuda:cluster_dims 2, barrier.cluster, mapa
		for _, want := range []string{".explicitcluster", ".reqnctapercluster 2, 1, 1", "barrier.cluster.arrive", "barrier.cluster.wait", "mapa.u64", "%cluster_ctarank"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout, dinfo := upload(t, ctx, make([]int32, blocks*bs)), upload(t, ctx, make([]int32, blocks))
		launch(t, ctx, mod, "ClusterExchange", cfg, cuda.Arg(dout), cuda.Arg(dinfo))
		out, info := download(t, dout), download(t, dinfo)
		for i, v := range out {
			b, tid := int32(i/bs), int32(i%bs)
			if want := (b^1)*1000 + tid; v != want {
				t.Fatalf("out[%d] = %d, want %d (partner block's shared memory)", i, v, want)
			}
		}
		for b, v := range info {
			if want := int32(b%2) + 10*2 + 100*int32(b/2) + 1000*blocks/2; v != want {
				t.Fatalf("info[%d] = %d, want %d", b, v, want)
			}
		}
		fmt.Printf("Cluster: %d blocks in clusters of 2 exchanged tiles through distributed shared memory; info = %v\n", blocks, info)
	})

	t.Run("MBarrier", func(t *testing.T) { // mbarrier.init/arrive/test_wait as a block barrier
		for _, want := range []string{"mbarrier.init.shared.b64", "mbarrier.arrive.shared.b64", "mbarrier.test_wait.shared.b64", "mbarrier.inval.shared.b64"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		in := make([]int32, blocks*bs)
		for i := range in {
			in[i] = int32(i % 7)
		}
		din, dout := upload(t, ctx, in), upload(t, ctx, make([]int32, blocks))
		launch(t, ctx, mod, "MBarrierSum", cfg, cuda.Arg(din), cuda.Arg(dout))
		for b, v := range download(t, dout) {
			var s int32
			for _, w := range in[b*bs : (b+1)*bs] {
				s += w
			}
			if v != s {
				t.Fatalf("block %d sum = %d, want %d", b, v, s)
			}
		}
		fmt.Printf("MBarrier: %d blocks synchronised through an mbarrier in shared memory\n", blocks)
	})

	t.Run("TMA", func(t *testing.T) { // cp.async.bulk global<->shared with mbarrier completion
		for _, want := range []string{"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes", "cp.async.bulk.global.shared::cta.bulk_group",
			"mbarrier.arrive.expect_tx.shared::cta.b64", "mbarrier.try_wait.parity.shared::cta.b64", "fence.proxy.async.shared::cta", "cp.async.bulk.wait_group"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		in := make([]int32, blocks*bs)
		for i := range in {
			in[i] = int32(i*3 - 100)
		}
		din, dout := upload(t, ctx, in), upload(t, ctx, make([]int32, blocks*bs))
		launch(t, ctx, mod, "TmaDouble", cfg, cuda.Arg(din), cuda.Arg(dout), cuda.ArgValue(int32(blocks*bs)))
		for i, v := range download(t, dout) {
			if v != 2*in[i] {
				t.Fatalf("out[%d] = %d, want %d", i, v, 2*in[i])
			}
		}
		fmt.Printf("TMA: %d x 1 KiB tiles bulk-copied global -> shared -> global OK\n", blocks)
	})

	t.Run("Elect", func(t *testing.T) { // elect.sync
		if !strings.Contains(ptx, "elect.sync") {
			t.Fatal("no elect.sync in the PTX")
		}
		dleader, delected := upload(t, ctx, make([]int32, blocks*bs)), upload(t, ctx, make([]int32, blocks*bs))
		launch(t, ctx, mod, "Elect", cfg, cuda.Arg(dleader), cuda.Arg(delected))
		leader, elected := download(t, dleader), download(t, delected)
		for w := 0; w < blocks*bs/32; w++ {
			cnt, lane := 0, -1
			for k := 0; k < 32; k++ {
				if elected[w*32+k] != 0 {
					cnt, lane = cnt+1, k
				}
				if leader[w*32+k] != leader[w*32] {
					t.Fatalf("warp %d: leader differs across lanes: %v", w, leader[w*32:w*32+32])
				}
			}
			if cnt != 1 || int(leader[w*32]) != lane {
				t.Fatalf("warp %d: %d lanes elected, leader %d (elected lane %d)", w, cnt, leader[w*32], lane)
			}
		}
		fmt.Printf("Elect: one lane elected per warp, leader lane = %d\n", leader[0])
	})

	t.Run("MatMul16", func(t *testing.T) { // the README sample: TMA-fed wmma matmul
		const n = 64
		a, b := make([]uint16, n*n), make([]uint16, n*n)
		af, bf, c := make([]float32, n*n), make([]float32, n*n), make([]float32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				af[i*n+j], bf[i*n+j], c[i*n+j] = float32((i+2*j)%7-3), float32((3*i+j)%5-2), float32(j-i)
				a[i*n+j], b[i*n+j] = float32ToHalf(af[i*n+j]), float32ToHalf(bf[i*n+j])
			}
		}
		want := make([]float32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				s := c[i*n+j]
				for k := 0; k < n; k++ {
					s += af[i*n+k] * bf[k*n+j]
				}
				want[i*n+j] = s
			}
		}
		da, db, dc := upload(t, ctx, a), upload(t, ctx, b), upload(t, ctx, c)
		launch(t, ctx, mod, "MatMul16", cuda.LaunchConfig{GridX: n / 16, GridY: n / 16, GridZ: 1, BlockX: 32, BlockY: 1, BlockZ: 1},
			cuda.Arg(da), cuda.Arg(db), cuda.Arg(dc), cuda.ArgValue(int32(n)))
		for i, v := range download(t, dc) {
			if v != want[i] {
				t.Fatalf("C[%d][%d] = %v, want %v", i/n, i%n, v, want[i])
			}
		}
		fmt.Printf("MatMul16: %dx%d TMA-fed tensor-core matmul (the README sample) matches the CPU\n", n, n)
	})

	t.Run("GridDep", func(t *testing.T) { // griddepcontrol.wait / launch_dependents (no-ops without PDL)
		for _, want := range []string{"griddepcontrol.wait", "griddepcontrol.launch_dependents", "fence.acq_rel.cluster"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		dout := upload(t, ctx, make([]int32, blocks*bs))
		launch(t, ctx, mod, "GridDep", cfg, cuda.Arg(dout))
		for i, v := range download(t, dout) {
			if v != int32(i) {
				t.Fatalf("out[%d] = %d", i, v)
			}
		}
		fmt.Printf("GridDep: griddepcontrol.wait / launch_dependents executed\n")
	})
}

const blackwellPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/blackwell"

// TestBlackwell compiles examples/blackwell for sm_100a (redux.sync on
// floats) and runs it only on a compute capability 10.x GPU; ptxas may be
// too old for sm_100a, so the check is skipped.
func TestBlackwell(t *testing.T) {
	res := buildKernels(t, blackwellPkg, &cudair.Options{SM: "sm_100a", PTX: "86", NoCheck: true})
	ptx := string(res.PTX)
	for _, want := range []string{".target sm_100a", "redux.sync.min.f32", "redux.sync.max.f32"} {
		if !strings.Contains(ptx, want) {
			t.Fatalf("PTX lacks %q", want)
		}
	}
	major, minor := computeCapability(t)
	if major != 10 {
		fmt.Printf("Blackwell: redux.sync.min/max.f32 compiled for sm_100a; not run on this %d.%d GPU\n", major, minor)
		return
	}
	ctx, mod, _ := loadKernelsOpts(t, blackwellPkg, &cudair.Options{SM: "sm_100a", PTX: "86", NoCheck: true})
	const n = 1024
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(math.Sin(float64(i)))
	}
	din, dout := upload(t, ctx, in), upload(t, ctx, make([]float32, n/16))
	launch(t, ctx, mod, "ReduceF32", cuda.LaunchConfig1D(n, 256), cuda.Arg(din), cuda.Arg(dout))
	out := download(t, dout)
	for w := 0; w < n/32; w++ {
		mn, mx := in[w*32], in[w*32]
		for _, v := range in[w*32 : w*32+32] {
			mn, mx = float32(math.Min(float64(mn), float64(v))), float32(math.Max(float64(mx), float64(v)))
		}
		if out[2*w] != mn || out[2*w+1] != mx {
			t.Fatalf("warp %d: min/max = %v/%v, want %v/%v", w, out[2*w], out[2*w+1], mn, mx)
		}
	}
	fmt.Printf("Blackwell: redux.sync.min/max.f32 OK\n")
}

// entry returns the PTX of the kernel named name.
func entry(ptx, name string) string {
	i := strings.Index(ptx, ".entry "+name+"(")
	if i < 0 {
		return ""
	}
	rest := ptx[i:]
	if j := strings.Index(rest, "\n}\n"); j >= 0 {
		rest = rest[:j]
	}
	return rest
}

// float32ToHalf converts to IEEE binary16 (round to nearest even).
func float32ToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case exp >= 31:
		return sign | 0x7c00 // inf (no NaN in the tests)
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		h := uint16(mant >> shift)
		if rem := mant & (1<<shift - 1); rem > 1<<(shift-1) || rem == 1<<(shift-1) && h&1 == 1 {
			h++
		}
		return sign | h
	}
	h := sign | uint16(exp)<<10 | uint16(mant>>13)
	if rem := mant & 0x1fff; rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++
	}
	return h
}
