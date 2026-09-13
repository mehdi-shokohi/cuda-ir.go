package vecadd

import "github.com/mehdi-shokohi/cuda-ir.go/cuda"

// VecAdd: out[i] = a[i] + b[i]. Exported + returns nothing => a kernel.
func VecAdd(a, b, out cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		out.Set(i, a.At(i)+b.At(i))
	}
}

// Saxpy with a grid-stride loop: y = alpha*x + y.
func Saxpy(alpha float32, x, y cuda.Buf[float32], n int32) {
	for i := cuda.GlobalIdX(); i < n; i += cuda.GridStrideX() {
		y.Set(i, alpha*x.At(i)+y.At(i))
	}
}

// helper with a lowercase name: compiled as a device .func, not a kernel.
func square(v float32) float32 { return v * v }

func Square(x cuda.Buf[float32], n int32) {
	i := cuda.GlobalIdX()
	if i < n {
		x.Set(i, square(x.At(i)))
	}
}
