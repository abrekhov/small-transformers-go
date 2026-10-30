//go:build !darwin || !cgo

package mat

import (
	"math"
	"runtime"
	"sync"
)

const Backend = "pure-go"

func MM(c, a, b []float32, m, k, n int) {
	zeroed(c)
	parallel(m, k, n, func(s, e int) { mmRange(c, a, b, m, k, n, s, e) })
}

func MMT(c, a, b []float32, m, k, n int) {
	zeroed(c)
	parallel(m, k, n, func(s, e int) { mmtRange(c, a, b, m, k, n, s, e) })
}

func MBT(c, a, b []float32, m, k, n int) {
	zeroed(c)
	parallel(m, k, n, func(s, e int) { mbtRange(c, a, b, m, k, n, s, e) })
}

func Gemm(c, a, b []float32, m, k, n, as, bs, cs int, ta, tb bool, scale float32) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				ai, bi := i*as+p, p*bs+j
				if ta {
					ai = p*as + i
				}
				if tb {
					bi = j*bs + p
				}
				sum += a[ai] * b[bi]
			}
			c[i*cs+j] = scale * sum
		}
	}
}

func Gelu(out, x []float32) {
	for i, v := range x {
		u := 0.7978845608028654 * (v + 0.044715*v*v*v)
		out[i] = 0.5 * v * (1 + float32(math.Tanh(float64(u))))
	}
}

func GeluBack(out, x, dout []float32) {
	for i, v := range x {
		u := 0.7978845608028654 * (v + 0.044715*v*v*v)
		th := float32(math.Tanh(float64(u)))
		out[i] = dout[i] * (0.5*(1+th) + 0.5*v*(1-th*th)*0.7978845608028654*(1+3*0.044715*v*v))
	}
}

func parallel(m, k, n int, fn func(s, e int)) {
	if m*k*n < 1<<16 {
		fn(0, m)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > m {
		workers = m
	}
	chunk := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > m {
			end = m
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}
