package mat

func zeroed(c []float32) []float32 {
	for i := range c {
		c[i] = 0
	}
	return c
}

func ColSum(c, a []float32, m, n int) {
	zeroed(c)
	for i := 0; i < m; i++ {
		arow := a[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			c[j] += arow[j]
		}
	}
}

func Zero(x []float32) {
	for i := range x {
		x[i] = 0
	}
}
