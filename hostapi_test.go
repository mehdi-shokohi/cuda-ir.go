package cudair_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/eitamring/gocudrv/cuda"
	cudair "github.com/mehdi-shokohi/cuda-ir.go"
	chost "github.com/mehdi-shokohi/cuda-ir.go/host"
)

const hostapiPkg = "github.com/mehdi-shokohi/cuda-ir.go/examples/hostapi"

// TestHostAPI runs examples/hostapi: the CUDA array geometries gocudrv has no
// wrapper for (1D, 3D, layered, cubemap, multi-channel) reached through
// package host, plus tex2Dgather and a 3D surface.
func TestHostAPI(t *testing.T) {
	ctx, mod, res := loadKernels(t, hostapiPkg)
	ptx := string(res.PTX)

	t.Run("Tex1D", func(t *testing.T) { // tex.1d.v4.f32.f32 over a 1D CUDA array
		for _, want := range []string{"tex.1d.v4.f32.f32", "tex.1d.v4.f32.s32", "txq.width"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		const n = 64
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i)*1.5 - 7
		}
		arr, err := chost.AllocArray1D[float32](ctx, n)
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, n+1))
		launch(t, ctx, mod, "Tex1DSample", cuda.LaunchConfig1D(n, 64),
			tex.Arg(), cuda.Arg(dout), cuda.ArgValue(int32(n)))
		got := download(t, dout)
		for i := range data {
			if got[i] != data[i] {
				t.Fatalf("tex1D[%d] = %v, want %v", i, got[i], data[i])
			}
		}
		if got[n] != n {
			t.Fatalf("txq.width = %v, want %d", got[n], n)
		}

		// tex1Dfetch reads linear memory, not an array, so it needs a
		// texture over a plain device buffer (res.linear).
		dlin := upload(t, ctx, data)
		ltex, err := chost.NewTextureLinear(ctx, dlin, 1)
		if err != nil {
			t.Fatal(err)
		}
		defer ltex.Close()
		dfetch := upload(t, ctx, make([]float32, n))
		launch(t, ctx, mod, "Tex1DFetchSample", cuda.LaunchConfig1D(n, 64),
			ltex.Arg(), cuda.Arg(dfetch), cuda.ArgValue(int32(n)))
		for i, v := range download(t, dfetch) {
			if v != data[i] {
				t.Fatalf("tex1Dfetch[%d] = %v, want %v", i, v, data[i])
			}
		}
		// round trip back out of the array
		back := make([]float32, n)
		if err := arr.CopyTo(back); err != nil {
			t.Fatal(err)
		}
		for i := range data {
			if back[i] != data[i] {
				t.Fatalf("CopyTo[%d] = %v, want %v", i, back[i], data[i])
			}
		}
		fmt.Printf("Tex1D: tex1D over a %d-element 1D CUDA array, tex1Dfetch over linear memory OK\n", n)
	})

	t.Run("Tex3D", func(t *testing.T) { // tex.3d.v4.f32.f32 over a 3D CUDA array
		if !strings.Contains(ptx, "tex.3d.v4.f32.f32") {
			t.Fatal("PTX lacks tex.3d.v4.f32.f32")
		}
		const w, h, d = 8, 4, 3
		data := make([]float32, w*h*d)
		for z := 0; z < d; z++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					data[(z*h+y)*w+x] = float32(x + 10*y + 100*z)
				}
			}
		}
		arr, err := chost.AllocArray3D[float32](ctx, w, h, d)
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, w*h*d))
		launch(t, ctx, mod, "Tex3DSample", cuda.LaunchConfig1D(w*h*d, 64),
			tex.Arg(), cuda.Arg(dout), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)), cuda.ArgValue(int32(d)))
		for i, v := range download(t, dout) {
			if v != data[i] {
				t.Fatalf("tex3D[%d,%d,%d] = %v, want %v", i%w, (i/w)%h, i/(w*h), v, data[i])
			}
		}
		fmt.Printf("Tex3D: tex3D over a %dx%dx%d 3D CUDA array OK\n", w, h, d)
	})

	t.Run("Channels4", func(t *testing.T) { // a 4-channel array: all of r,g,b,a carry data
		const w, h = 6, 5
		data := make([]float32, w*h*4) // interleaved rgba
		for i := 0; i < w*h; i++ {
			for c := 0; c < 4; c++ {
				data[4*i+c] = float32(i*10 + c)
			}
		}
		arr, err := chost.AllocArray2D[float32](ctx, w, h, chost.WithChannels(4))
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if got, want := arr.Elements(), w*h*4; got != want {
			t.Fatalf("Elements = %d, want %d", got, want)
		}
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, w*h*4))
		launch(t, ctx, mod, "Tex2DRGBA", cuda.LaunchConfig1D(w*h, 64),
			tex.Arg(), cuda.Arg(dout), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)))
		for i, v := range download(t, dout) {
			if v != data[i] {
				t.Fatalf("tex2D rgba[%d] channel %d = %v, want %v", i/4, i%4, v, data[i])
			}
		}
		fmt.Printf("Channels4: a 4-channel %dx%d array fills r,g,b,a OK\n", w, h)
	})

	t.Run("Layered", func(t *testing.T) { // tex.a2d / tex.a1d
		for _, want := range []string{"tex.a2d.v4.f32.f32", "tex.a1d.v4.f32.f32"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		const w, h, layers = 8, 4, 3
		data := make([]float32, w*h*layers)
		for l := 0; l < layers; l++ {
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					data[(l*h+y)*w+x] = float32(x + 10*y + 1000*l)
				}
			}
		}
		arr, err := chost.AllocArrayLayered2D[float32](ctx, w, h, layers)
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, w*h*layers))
		launch(t, ctx, mod, "Tex2DLayeredSample", cuda.LaunchConfig1D(w*h*layers, 64),
			tex.Arg(), cuda.Arg(dout), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)), cuda.ArgValue(int32(layers)))
		for i, v := range download(t, dout) {
			if v != data[i] {
				t.Fatalf("tex2DLayered layer %d (%d,%d) = %v, want %v", i/(w*h), i%w, (i/w)%h, v, data[i])
			}
		}

		// the 1D layered geometry too
		d1 := make([]float32, w*layers)
		for i := range d1 {
			d1[i] = float32(i%w) + 1000*float32(i/w)
		}
		arr1, err := chost.AllocArrayLayered1D[float32](ctx, w, layers)
		if err != nil {
			t.Fatal(err)
		}
		defer arr1.Close()
		if err := arr1.CopyFrom(d1); err != nil {
			t.Fatal(err)
		}
		tex1, err := chost.NewTexture(arr1, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex1.Close()
		dout1 := upload(t, ctx, make([]float32, w*layers))
		launch(t, ctx, mod, "Tex1DLayeredSample", cuda.LaunchConfig1D(w*layers, 64),
			tex1.Arg(), cuda.Arg(dout1), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(layers)))
		for i, v := range download(t, dout1) {
			if v != d1[i] {
				t.Fatalf("tex1DLayered layer %d x %d = %v, want %v", i/w, i%w, v, d1[i])
			}
		}
		fmt.Printf("Layered: tex2DLayered over %d layers of %dx%d and tex1DLayered over %d of %d OK\n",
			layers, w, h, layers, w)
	})

	t.Run("Cubemap", func(t *testing.T) { // tex.cube.v4.f32.f32
		if !strings.Contains(ptx, "tex.cube.v4.f32.f32") {
			t.Fatal("PTX lacks tex.cube.v4.f32.f32")
		}
		const size = 4
		// every element of face f holds f, so a fetch anywhere on a face
		// identifies it whatever the filtering does.
		data := make([]float32, size*size*6)
		for f := 0; f < 6; f++ {
			for i := 0; i < size*size; i++ {
				data[f*size*size+i] = float32(f)
			}
		}
		arr, err := chost.AllocArrayCubemap[float32](ctx, size)
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, 6))
		launch(t, ctx, mod, "TexCubemapSample", cuda.LaunchConfig1D(6, 32), tex.Arg(), cuda.Arg(dout))
		got := download(t, dout)
		for f := 0; f < 6; f++ {
			if got[f] != float32(f) {
				t.Fatalf("texCubemap face %d = %v, want %d (faces: %v)", f, got[f], f, got)
			}
		}
		fmt.Printf("Cubemap: texCubemap selects all six faces of a %dx%d cubemap OK\n", size, size)
	})

	t.Run("Gather", func(t *testing.T) { // tld4.r.2d.v4.f32.f32
		if !strings.Contains(ptx, "tld4.r.2d.v4.f32.f32") {
			t.Fatal("PTX lacks tld4.r.2d.v4.f32.f32")
		}
		const w, h = 4, 4
		data := make([]float32, w*h)
		for i := range data {
			data[i] = float32(i)
		}
		arr, err := chost.AllocArray2D[float32](ctx, w, h, chost.WithTextureGather())
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		tex, err := chost.NewTexture(arr, cuda.TextureConfig{AddressMode: cuda.AddressClamp})
		if err != nil {
			t.Fatal(err)
		}
		defer tex.Close()

		dout := upload(t, ctx, make([]float32, w*h*4))
		launch(t, ctx, mod, "Tex2DGatherSample", cuda.LaunchConfig1D(w*h, 32),
			tex.Arg(), cuda.Arg(dout), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)))
		got := download(t, dout)
		// at the corner (x+1, y+1) the four texels are (x,y), (x+1,y),
		// (x,y+1), (x+1,y+1); tld4 returns them counter-clockwise from the
		// lower left: (x, y+1), (x+1, y+1), (x+1, y), (x, y).
		at := func(x, y int) float32 {
			x = min(max(x, 0), w-1)
			y = min(max(y, 0), h-1)
			return data[y*w+x]
		}
		for i := 0; i < w*h; i++ {
			x, y := i%w, i/w
			want := [4]float32{at(x, y+1), at(x+1, y+1), at(x+1, y), at(x, y)}
			for c := 0; c < 4; c++ {
				if got[4*i+c] != want[c] {
					t.Fatalf("tex2Dgather at (%d,%d): got %v, want %v", x, y, got[4*i:4*i+4], want)
				}
			}
		}
		fmt.Printf("Gather: tex2Dgather returns the four bilinear texels of a %dx%d array OK\n", w, h)
	})

	t.Run("Surface3D", func(t *testing.T) { // suld/sust.b.3d.b32
		for _, want := range []string{"suld.b.3d.b32", "sust.b.3d.b32"} {
			if !strings.Contains(ptx, want) {
				t.Fatalf("PTX lacks %q", want)
			}
		}
		const w, h, d = 8, 4, 2
		data := make([]float32, w*h*d)
		for i := range data {
			data[i] = float32(i) + 0.5
		}
		arr, err := chost.AllocArray3D[float32](ctx, w, h, d, chost.WithSurfaceStore())
		if err != nil {
			t.Fatal(err)
		}
		defer arr.Close()
		if err := arr.CopyFrom(data); err != nil {
			t.Fatal(err)
		}
		surf, err := chost.NewSurface(arr)
		if err != nil {
			t.Fatal(err)
		}
		defer surf.Close()

		launch(t, ctx, mod, "Surf3DDouble", cuda.LaunchConfig1D(w*h*d, 64),
			surf.Arg(), cuda.ArgValue(int32(w)), cuda.ArgValue(int32(h)), cuda.ArgValue(int32(d)))
		got := make([]float32, w*h*d)
		if err := arr.CopyTo(got); err != nil {
			t.Fatal(err)
		}
		for i := range data {
			if got[i] != data[i]*2 {
				t.Fatalf("surf3D[%d] = %v, want %v", i, got[i], data[i]*2)
			}
		}
		fmt.Printf("Surface3D: surf3Dread/write doubled a %dx%dx%d surface in place OK\n", w, h, d)
	})
}

// TestLaunchEx runs the cuLaunchKernelEx path: a cluster shape chosen at
// launch time (which //cuda:cluster_dims bakes into the PTX instead) and the
// programmatic-dependent-launch attribute, neither of which gocudrv's
// cuLaunchKernel can express. Needs compute capability >= 9.0.
func TestLaunchEx(t *testing.T) {
	res := buildKernels(t, hopperPkg, &cudair.Options{SM: "sm_90", PTX: "80"})
	dev := device(t)
	major, _, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major < 9 {
		t.Skipf("clusters need compute capability >= 9.0, have %d.x", major)
	}
	ctx, err := dev.Primary()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ctx.Close() })

	// our own module, because gocudrv keeps its CUfunction private
	mod, err := chost.LoadModule(ctx, res.PTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mod.Close() })

	t.Run("RuntimeClusterDims", func(t *testing.T) {
		ptx := string(res.PTX)
		// the proof that the shape is not compiled in: ClusterExchange
		// carries the directive and ClusterRankRuntime does not.
		if !strings.Contains(ptx, ".reqnctapercluster") {
			t.Fatal("PTX lacks .reqnctapercluster for the directed kernel")
		}
		i := strings.Index(ptx, "ClusterRankRuntime")
		if i < 0 {
			t.Fatal("PTX lacks ClusterRankRuntime")
		}
		// .reqnctapercluster, if present for this kernel, would sit between
		// the entry name and the body.
		if head := ptx[i:min(i+400, len(ptx))]; strings.Contains(head, ".reqnctapercluster") {
			t.Fatal("ClusterRankRuntime has .reqnctapercluster; the shape would not be the launch attribute's")
		}

		fn, err := mod.Function("ClusterRankRuntime")
		if err != nil {
			t.Fatal(err)
		}
		const blocks = 8
		for _, cluster := range []uint32{1, 2, 4} {
			dout := upload(t, ctx, make([]int32, blocks))
			cfg := cuda.LaunchConfig{GridX: blocks, BlockX: 32}
			err := fn.LaunchEx(cfg, chost.LaunchAttrs{ClusterDim: [3]uint32{cluster, 1, 1}},
				chost.ArgBuffer(dout))
			if err != nil {
				t.Fatal(err)
			}
			got := download(t, dout)
			for b := 0; b < blocks; b++ {
				want := int32(uint32(b)%cluster) + 100*int32(cluster)
				if got[b] != want {
					t.Fatalf("cluster %d: block %d reported %d, want %d (all: %v)", cluster, b, got[b], want, got)
				}
			}
			fmt.Printf("RuntimeClusterDims: cluster %d -> ranks %v OK\n", cluster, got)
		}
	})

	t.Run("ProgrammaticDependentLaunch", func(t *testing.T) {
		fn, err := mod.Function("GridDep")
		if err != nil {
			t.Fatal(err)
		}
		const n = 256
		dout := upload(t, ctx, make([]int32, n))
		cfg := cuda.LaunchConfig1D(n, 64)
		// With the attribute set, griddepcontrol.wait and
		// launch_dependents are live rather than no-ops.
		if err := fn.LaunchEx(cfg, chost.LaunchAttrs{ProgrammaticStreamSerialization: true},
			chost.ArgBuffer(dout)); err != nil {
			t.Fatal(err)
		}
		for i, v := range download(t, dout) {
			if v != int32(i) {
				t.Fatalf("GridDep out[%d] = %d, want %d", i, v, i)
			}
		}
		fmt.Printf("ProgrammaticDependentLaunch: griddepcontrol with the launch attribute over %d threads OK\n", n)
	})

	t.Run("Cooperative", func(t *testing.T) {
		// the cooperative attribute is the Ex spelling of
		// cuLaunchCooperativeKernel.
		fn, err := mod.Function("GridDep")
		if err != nil {
			t.Fatal(err)
		}
		const n = 64
		dout := upload(t, ctx, make([]int32, n))
		if err := fn.LaunchEx(cuda.LaunchConfig1D(n, 64), chost.LaunchAttrs{Cooperative: true},
			chost.ArgBuffer(dout)); err != nil {
			t.Fatal(err)
		}
		for i, v := range download(t, dout) {
			if v != int32(i) {
				t.Fatalf("cooperative out[%d] = %d, want %d", i, v, i)
			}
		}
		fmt.Println("Cooperative: the cooperative launch attribute OK")
	})
}

// TestTensorMap runs the TMA tensor-copy path: the host encodes a
// CUtensorMap with cuTensorMapEncodeTiled (which gocudrv does not bind) and
// the kernel moves tiles through it with cp.async.bulk.tensor. Needs compute
// capability >= 9.0.
func TestTensorMap(t *testing.T) {
	res := buildKernels(t, hopperPkg, &cudair.Options{SM: "sm_90", PTX: "80"})
	dev := device(t)
	major, _, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major < 9 {
		t.Skipf("TMA tensor copies need compute capability >= 9.0, have %d.x", major)
	}
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

	ptx := string(res.PTX)
	for _, want := range []string{
		"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
		"cp.async.bulk.tensor.2d.global.shared::cta.tile",
	} {
		if !strings.Contains(ptx, want) {
			t.Fatalf("PTX lacks %q", want)
		}
	}

	t.Run("Encode", func(t *testing.T) {
		// the constraints EncodeTiled checks before the driver does
		d := upload(t, ctx, make([]float32, 64*64))
		base := chost.TiledTensor{
			DataType:      chost.TensorMapFloat32,
			GlobalDim:     []uint64{64, 64},
			GlobalStrides: []uint64{64 * 4},
			BoxDim:        []uint32{32, 32},
		}
		if _, err := chost.EncodeTiled(ctx, d, base); err != nil {
			t.Fatalf("valid descriptor rejected: %v", err)
		}
		for name, mutate := range map[string]func(*chost.TiledTensor){
			"stride not a multiple of 16": func(x *chost.TiledTensor) { x.GlobalStrides = []uint64{64*4 + 4} },
			"box too large":               func(x *chost.TiledTensor) { x.BoxDim = []uint32{512, 32} },
			"wrong stride count":          func(x *chost.TiledTensor) { x.GlobalStrides = []uint64{256, 256} },
			"wrong box count":             func(x *chost.TiledTensor) { x.BoxDim = []uint32{32} },
		} {
			bad := base
			mutate(&bad)
			if _, err := chost.EncodeTiled(ctx, d, bad); err == nil {
				t.Fatalf("%s: accepted", name)
			}
		}
		// an element size that disagrees with DataType
		di := upload(t, ctx, make([]int32, 64*64))
		half := base
		half.DataType = chost.TensorMapFloat16
		if _, err := chost.EncodeTiled(ctx, di, half); err == nil {
			t.Fatal("4-byte buffer accepted as float16")
		}
		fmt.Println("Encode: cuTensorMapEncodeTiled accepts a valid 2D descriptor and rejects five bad ones")
	})

	t.Run("TensorTile2D", func(t *testing.T) {
		// 96x64: three 32-wide tiles across, two 32-high down, so the grid
		// is exactly covered and every tile is interior.
		const w, h = 96, 64
		src := make([]float32, w*h)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				src[y*w+x] = float32(x) + 1000*float32(y)
			}
		}
		dsrc := upload(t, ctx, src)
		ddst := upload(t, ctx, make([]float32, w*h))

		desc := chost.TiledTensor{
			DataType:      chost.TensorMapFloat32,
			GlobalDim:     []uint64{w, h}, // innermost (x) first
			GlobalStrides: []uint64{w * 4},
			BoxDim:        []uint32{32, 32},
		}
		smap, err := chost.EncodeTiled(ctx, dsrc, desc)
		if err != nil {
			t.Fatal(err)
		}
		dmap, err := chost.EncodeTiled(ctx, ddst, desc)
		if err != nil {
			t.Fatal(err)
		}

		// the descriptors live in global memory, 128 bytes each
		dsmap := upload(t, ctx, smap.Bytes())
		ddmap := upload(t, ctx, dmap.Bytes())

		cfg := cuda.LaunchConfig{GridX: w / 32, GridY: h / 32, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1}
		launch(t, ctx, mod, "TensorTileDouble", cfg, cuda.Arg(dsmap), cuda.Arg(ddmap))

		got := download(t, ddst)
		for i := range src {
			if got[i] != src[i]*2 {
				t.Fatalf("tile at (%d,%d): got %v, want %v", i%w, i/w, got[i], src[i]*2)
			}
		}
		fmt.Printf("TensorTile2D: %d 32x32 tiles moved global->shared->global through a tensor map OK\n",
			(w/32)*(h/32))
	})

	t.Run("OutOfBounds", func(t *testing.T) {
		// 40x40 with 32x32 tiles: the second tile in each direction hangs
		// off the edge, which the map must clamp rather than fault.
		const w, h = 40, 40
		src := make([]float32, w*h)
		for i := range src {
			src[i] = float32(i) + 1
		}
		dsrc := upload(t, ctx, src)
		ddst := upload(t, ctx, make([]float32, w*h))
		desc := chost.TiledTensor{
			DataType:      chost.TensorMapFloat32,
			GlobalDim:     []uint64{w, h},
			GlobalStrides: []uint64{w * 4},
			BoxDim:        []uint32{32, 32},
			OOBFill:       chost.OOBFillNaN,
		}
		smap, err := chost.EncodeTiled(ctx, dsrc, desc)
		if err != nil {
			t.Fatal(err)
		}
		dmap, err := chost.EncodeTiled(ctx, ddst, desc)
		if err != nil {
			t.Fatal(err)
		}
		dsmap := upload(t, ctx, smap.Bytes())
		ddmap := upload(t, ctx, dmap.Bytes())

		cfg := cuda.LaunchConfig{GridX: 2, GridY: 2, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1}
		launch(t, ctx, mod, "TensorTileDouble", cfg, cuda.Arg(dsmap), cuda.Arg(ddmap))

		// every in-bounds element is covered by exactly one tile, and the
		// out-of-bounds parts were dropped on the way back rather than
		// writing outside the tensor.
		for i, v := range download(t, ddst) {
			if v != src[i]*2 {
				t.Fatalf("clamped tile at (%d,%d): got %v, want %v", i%w, i/w, v, src[i]*2)
			}
		}
		fmt.Printf("OutOfBounds: a %dx%d tensor in 32x32 tiles clamps at the edges OK\n", w, h)
	})
}
