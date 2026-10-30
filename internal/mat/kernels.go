package mat

func mmRange(c, a, b []float32, m, k, n, s, e int) {
	for i := s; i < e; i++ {
		arow := a[i*k : (i+1)*k]
		crow := c[i*n : (i+1)*n]
		for p := 0; p < k; p++ {
			av := arow[p]
			if av == 0 {
				continue
			}
			brow := b[p*n : (p+1)*n]
			for j := 0; j < n; j++ {
				crow[j] += av * brow[j]
			}
		}
	}
}

func mmtRange(c, a, b []float32, m, k, n, s, e int) {
	for i := s; i < e; i++ {
		crow := c[i*n : (i+1)*n]
		for p := 0; p < k; p++ {
			av := a[p*m+i]
			if av == 0 {
				continue
			}
			brow := b[p*n : (p+1)*n]
			for j := 0; j < n; j++ {
				crow[j] += av * brow[j]
			}
		}
	}
}

func mbtRange(c, a, b []float32, m, k, n, s, e int) {
	for i := s; i < e; i++ {
		arow := a[i*k : (i+1)*k]
		crow := c[i*n : (i+1)*n]
		for j := 0; j < n; j++ {
			brow := b[j*k : (j+1)*k]
			var sum float32
			for p := 0; p < k; p++ {
				sum += arow[p] * brow[p]
			}
			crow[j] = sum
		}
	}
}
