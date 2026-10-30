package gpt

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"small-transformers-go/internal/mat"
)

type Config struct {
	Vocab  int
	Dim    int
	Heads  int
	Layers int
	Ctx    int
	Chat   bool
}

func (c Config) Validate() error {
	if c.Vocab < 1 || c.Dim < 1 || c.Heads < 1 || c.Layers < 1 || c.Ctx < 1 || c.Dim%c.Heads != 0 {
		return fmt.Errorf("vocab, dim, heads, layers and seq must be positive; dim must be divisible by heads")
	}
	return nil
}

func (c Config) sizes() []int {
	D, F := c.Dim, 4*c.Dim
	s := []int{c.Vocab * D, c.Ctx * D}
	for l := 0; l < c.Layers; l++ {
		s = append(s, D, D, 3*D*D, 3*D, D*D, D, D, D, D*F, F, F*D, D)
	}
	return append(s, D, D, D*c.Vocab, c.Vocab)
}

const (
	layerSlots = 12
	headSlots  = 4
)

const (
	ln1g = 0
	ln1b = 1
	wqkv = 2
	bqkv = 3
	wo   = 4
	bo   = 5
	ln2g = 6
	ln2b = 7
	w1   = 8
	b1   = 9
	w2   = 10
	b2   = 11
)

type Model struct {
	Cfg Config
	W   [][]float32
	G   [][]float32
}

func (m *Model) layerBase(l int) int { return 2 + l*layerSlots }
func (m *Model) headBase() int       { return 2 + m.Cfg.Layers*layerSlots }

func (m *Model) p(l, slot int) []float32 {
	return m.W[m.layerBase(l)+slot]
}
func (m *Model) g(l, slot int) []float32 {
	return m.G[m.layerBase(l)+slot]
}
func (m *Model) headG(slot int) []float32 {
	return m.G[m.headBase()+slot]
}

func zeros(n int) []float32 { return make([]float32, n) }
func ones(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = 1
	}
	return s
}

func normal(std float32, n int, rng *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64()) * std
	}
	return s
}

func New(cfg Config, seed uint64) *Model {
	rng := rand.New(rand.NewPCG(seed, seed))
	D, V, L, C := cfg.Dim, cfg.Vocab, cfg.Layers, cfg.Ctx
	F := 4 * D
	resStd := float32(0.02) / float32(math.Sqrt(float64(2*L)))
	m := &Model{Cfg: cfg}
	m.W = make([][]float32, 2+L*layerSlots+headSlots)
	m.W[0] = normal(0.02, V*D, rng)
	m.W[1] = normal(0.02, C*D, rng)
	for l := 0; l < L; l++ {
		b := m.layerBase(l)
		m.W[b+ln1g] = ones(D)
		m.W[b+ln1b] = zeros(D)
		m.W[b+wqkv] = normal(0.02, 3*D*D, rng)
		m.W[b+bqkv] = zeros(3 * D)
		m.W[b+wo] = normal(resStd, D*D, rng)
		m.W[b+bo] = zeros(D)
		m.W[b+ln2g] = ones(D)
		m.W[b+ln2b] = zeros(D)
		m.W[b+w1] = normal(0.02, D*F, rng)
		m.W[b+b1] = zeros(F)
		m.W[b+w2] = normal(resStd, F*D, rng)
		m.W[b+b2] = zeros(D)
	}
	hb := m.headBase()
	m.W[hb+0] = ones(D)
	m.W[hb+1] = zeros(D)
	m.W[hb+2] = normal(0.02, D*V, rng)
	m.W[hb+3] = zeros(V)
	m.G = make([][]float32, len(m.W))
	for i, w := range m.W {
		m.G[i] = make([]float32, len(w))
	}
	return m
}

type layerCache struct {
	A1, XHat1    []float32
	Mean1, Rstd1 []float32
	Qkv          []float32
	AttP         []float32
	AttV         []float32
	A2, XHat2    []float32
	Mean2, Rstd2 []float32
	HPre, H      []float32
}

type cache struct {
	B, T, D, H, hd, F, V, BT int
	Idx, Targets             []int
	Layers                   []layerCache
	XHatF, MeanF, RstdF      []float32
	LnfOut                   []float32
	Logits                   []float32
}

func (m *Model) newCache(B, T int) *cache {
	D := m.Cfg.Dim
	return &cache{
		B: B, T: T, D: D, H: m.Cfg.Heads, hd: D / m.Cfg.Heads,
		F: 4 * D, V: m.Cfg.Vocab, BT: B * T,
		Layers: make([]layerCache, m.Cfg.Layers),
	}
}

func lnForward(out, xhat, x, g, b, mean, rstd []float32, rows, d int) {
	for i := 0; i < rows; i++ {
		xrow := x[i*d : (i+1)*d]
		var mu float32
		for _, v := range xrow {
			mu += v
		}
		mu /= float32(d)
		var va float32
		for _, v := range xrow {
			dv := v - mu
			va += dv * dv
		}
		va /= float32(d)
		r := float32(1 / math.Sqrt(float64(va)+1e-5))
		mean[i] = mu
		rstd[i] = r
		xrowh := xhat[i*d : (i+1)*d]
		orow := out[i*d : (i+1)*d]
		for j, v := range xrow {
			xh := (v - mu) * r
			xrowh[j] = xh
			orow[j] = xh*g[j] + b[j]
		}
	}
}

func lnBackAccum(dx, xhat, dout, g, rstd, dg, db []float32, rows, d int) {
	inv := 1 / float32(d)
	for i := 0; i < rows; i++ {
		xrow := xhat[i*d : (i+1)*d]
		dorow := dout[i*d : (i+1)*d]
		dxrow := dx[i*d : (i+1)*d]
		r := rstd[i]
		var s1, s2 float32
		for j := 0; j < d; j++ {
			dxh := dorow[j] * g[j]
			s1 += dxh
			s2 += dxh * xrow[j]
			db[j] += dorow[j]
			dg[j] += dorow[j] * xrow[j]
		}
		for j := 0; j < d; j++ {
			dxh := dorow[j] * g[j]
			dxrow[j] += r * (dxh - s1*inv - xrow[j]*s2*inv)
		}
	}
}

func (m *Model) forward(idx, targets []int, c *cache) float32 {
	D, T, BT := c.D, c.T, c.BT
	tokEmb, posEmb := m.W[0], m.W[1]
	x := make([]float32, BT*D)
	for i, tok := range idx {
		t := i % T
		trow := tokEmb[tok*D : (tok+1)*D]
		prow := posEmb[t*D : (t+1)*D]
		xrow := x[i*D : (i+1)*D]
		for j := 0; j < D; j++ {
			xrow[j] = trow[j] + prow[j]
		}
	}
	c.Idx = slices.Clone(idx)
	c.Targets = targets

	for l := 0; l < m.Cfg.Layers; l++ {
		lc := &c.Layers[l]
		lc.A1 = make([]float32, BT*D)
		lc.XHat1 = make([]float32, BT*D)
		lc.Mean1 = make([]float32, BT)
		lc.Rstd1 = make([]float32, BT)
		lnForward(lc.A1, lc.XHat1, x, m.p(l, ln1g), m.p(l, ln1b), lc.Mean1, lc.Rstd1, BT, D)

		lc.Qkv = make([]float32, BT*3*D)
		mat.MM(lc.Qkv, lc.A1, m.p(l, wqkv), BT, D, 3*D)
		bias := m.p(l, bqkv)
		for i := 0; i < BT; i++ {
			qrow := lc.Qkv[i*3*D : (i+1)*3*D]
			for j := 0; j < 3*D; j++ {
				qrow[j] += bias[j]
			}
		}

		lc.AttP = make([]float32, c.B*c.H*T*T)
		lc.AttV = make([]float32, BT*D)
		m.attentionForward(c, lc)

		proj := make([]float32, BT*D)
		mat.MM(proj, lc.AttV, m.p(l, wo), BT, D, D)
		for i := 0; i < BT; i++ {
			prow := proj[i*D : (i+1)*D]
			for j := 0; j < D; j++ {
				prow[j] += m.p(l, bo)[j]
			}
		}
		x2 := make([]float32, BT*D)
		for i := range x2 {
			x2[i] = x[i] + proj[i]
		}

		lc.A2 = make([]float32, BT*D)
		lc.XHat2 = make([]float32, BT*D)
		lc.Mean2 = make([]float32, BT)
		lc.Rstd2 = make([]float32, BT)
		lnForward(lc.A2, lc.XHat2, x2, m.p(l, ln2g), m.p(l, ln2b), lc.Mean2, lc.Rstd2, BT, D)

		lc.HPre = make([]float32, BT*c.F)
		mat.MM(lc.HPre, lc.A2, m.p(l, w1), BT, D, c.F)
		for i := 0; i < BT; i++ {
			hrow := lc.HPre[i*c.F : (i+1)*c.F]
			for j := 0; j < c.F; j++ {
				hrow[j] += m.p(l, b1)[j]
			}
		}
		lc.H = make([]float32, BT*c.F)
		mat.Gelu(lc.H, lc.HPre)
		mlp := make([]float32, BT*D)
		mat.MM(mlp, lc.H, m.p(l, w2), BT, c.F, D)
		for i := 0; i < BT; i++ {
			mrow := mlp[i*D : (i+1)*D]
			for j := 0; j < D; j++ {
				mrow[j] += m.p(l, b2)[j]
			}
		}
		x3 := make([]float32, BT*D)
		for i := range x3 {
			x3[i] = x2[i] + mlp[i]
		}
		x = x3
	}

	c.XHatF = make([]float32, BT*D)
	c.MeanF = make([]float32, BT)
	c.RstdF = make([]float32, BT)
	c.LnfOut = make([]float32, BT*D)
	hb := m.headBase()
	lnForward(c.LnfOut, c.XHatF, x, m.W[hb+0], m.W[hb+1], c.MeanF, c.RstdF, BT, D)

	c.Logits = make([]float32, BT*c.V)
	mat.MM(c.Logits, c.LnfOut, m.W[hb+2], BT, D, c.V)
	for i := 0; i < BT; i++ {
		lrow := c.Logits[i*c.V : (i+1)*c.V]
		for j := 0; j < c.V; j++ {
			lrow[j] += m.W[hb+3][j]
		}
	}

	if targets == nil {
		return 0
	}
	var loss float32
	count := 0
	for i := 0; i < BT; i++ {
		if targets[i] < 0 {
			continue
		}
		count++
		lrow := c.Logits[i*c.V : (i+1)*c.V]
		mx := lrow[0]
		for _, v := range lrow {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for _, v := range lrow {
			sum += float32(math.Exp(float64(v - mx)))
		}
		loss += mx - lrow[targets[i]] + float32(math.Log(float64(sum)))
	}
	return loss / float32(count)
}

func (m *Model) attentionForward(c *cache, lc *layerCache) {
	B, T, D, H, hd := c.B, c.T, c.D, c.H, c.hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		wg.Go(func() {
			for h := 0; h < H; h++ {
				base := b*T*3*D + h*hd
				q, k, v := lc.Qkv[base:], lc.Qkv[base+D:], lc.Qkv[base+2*D:]
				p := lc.AttP[(b*H+h)*T*T : (b*H+h+1)*T*T]
				mat.Gemm(p, q, k, T, hd, T, 3*D, 3*D, T, false, true, scale)
				for t := 0; t < T; t++ {
					row := p[t*T : (t+1)*T]
					mx := slices.Max(row[:t+1])
					var sum float32
					for s := 0; s <= t; s++ {
						row[s] = float32(math.Exp(float64(row[s] - mx)))
						sum += row[s]
					}
					for s := 0; s <= t; s++ {
						row[s] /= sum
					}
					clear(row[t+1:])
				}
				mat.Gemm(lc.AttV[b*T*D+h*hd:], p, v, T, T, hd, T, 3*D, D, false, false, 1)
			}
		})
	}
	wg.Wait()
}

func (m *Model) attentionBackward(c *cache, lc *layerCache, dattv, dqkv []float32) {
	B, T, D, H, hd := c.B, c.T, c.D, c.H, c.hd
	scale := float32(1 / math.Sqrt(float64(hd)))
	var wg sync.WaitGroup
	for b := 0; b < B; b++ {
		wg.Go(func() {
			dp := make([]float32, T*T)
			for h := 0; h < H; h++ {
				base := b*T*3*D + h*hd
				q, k, v := lc.Qkv[base:], lc.Qkv[base+D:], lc.Qkv[base+2*D:]
				p := lc.AttP[(b*H+h)*T*T : (b*H+h+1)*T*T]
				da := dattv[b*T*D+h*hd:]
				mat.Gemm(dp, da, v, T, hd, T, D, 3*D, T, false, true, 1)
				mat.Gemm(dqkv[base+2*D:], p, da, T, T, hd, T, D, 3*D, true, false, 1)
				for t := 0; t < T; t++ {
					row, probs := dp[t*T:(t+1)*T], p[t*T:(t+1)*T]
					var sum float32
					for s := 0; s <= t; s++ {
						sum += row[s] * probs[s]
					}
					for s := 0; s <= t; s++ {
						row[s] = probs[s] * (row[s] - sum)
					}
					clear(row[t+1:])
				}
				mat.Gemm(dqkv[base:], dp, k, T, T, hd, T, 3*D, 3*D, false, false, scale)
				mat.Gemm(dqkv[base+D:], dp, q, T, T, hd, T, 3*D, 3*D, true, false, scale)
			}
		})
	}
	wg.Wait()
}

func (m *Model) Backward(c *cache) {
	D, V, BT, T := c.D, c.V, c.BT, c.T
	hb := m.headBase()

	dlogits := make([]float32, BT*V)
	count := 0
	for _, target := range c.Targets {
		if target >= 0 {
			count++
		}
	}
	for i := 0; i < BT; i++ {
		if c.Targets[i] < 0 {
			continue
		}
		lrow := c.Logits[i*V : (i+1)*V]
		mx := lrow[0]
		for _, v := range lrow {
			if v > mx {
				mx = v
			}
		}
		var sum float32
		for _, v := range lrow {
			sum += float32(math.Exp(float64(v - mx)))
		}
		inv := 1 / float32(count)
		for j := 0; j < V; j++ {
			p := float32(math.Exp(float64(lrow[j]-mx))) / sum
			dlogits[i*V+j] = p * inv
		}
		dlogits[i*V+c.Targets[i]] -= inv
	}

	dlnf := make([]float32, BT*D)
	mat.MBT(dlnf, dlogits, m.W[hb+2], BT, V, D)
	mat.MMT(m.headG(2), c.LnfOut, dlogits, D, BT, V)
	mat.ColSum(m.headG(3), dlogits, BT, V)

	dx := make([]float32, BT*D)
	lnBackAccum(dx, c.XHatF, dlnf, m.W[hb+0], c.RstdF, m.headG(0), m.headG(1), BT, D)

	for l := m.Cfg.Layers - 1; l >= 0; l-- {
		lc := &c.Layers[l]
		dx3 := dx

		dxa := slices.Clone(dx3)
		dh := make([]float32, BT*c.F)
		mat.MBT(dh, dx3, m.p(l, w2), BT, D, c.F)
		mat.MMT(m.g(l, w2), lc.H, dx3, c.F, BT, D)
		mat.ColSum(m.g(l, b2), dx3, BT, D)
		dhpre := make([]float32, BT*c.F)
		mat.GeluBack(dhpre, lc.HPre, dh)
		da2 := make([]float32, BT*D)
		mat.MBT(da2, dhpre, m.p(l, w1), BT, c.F, D)
		mat.MMT(m.g(l, w1), lc.A2, dhpre, D, BT, c.F)
		mat.ColSum(m.g(l, b1), dhpre, BT, c.F)
		lnBackAccum(dxa, lc.XHat2, da2, m.p(l, ln2g), lc.Rstd2, m.g(l, ln2g), m.g(l, ln2b), BT, D)

		dproj := dxa
		dattv := make([]float32, BT*D)
		mat.MBT(dattv, dproj, m.p(l, wo), BT, D, D)
		mat.MMT(m.g(l, wo), lc.AttV, dproj, D, BT, D)
		mat.ColSum(m.g(l, bo), dproj, BT, D)

		dqkv := make([]float32, BT*3*D)
		m.attentionBackward(c, lc, dattv, dqkv)

		da1 := make([]float32, BT*D)
		mat.MBT(da1, dqkv, m.p(l, wqkv), BT, 3*D, D)
		mat.MMT(m.g(l, wqkv), lc.A1, dqkv, D, BT, 3*D)
		mat.ColSum(m.g(l, bqkv), dqkv, BT, 3*D)

		dxout := slices.Clone(dxa)
		lnBackAccum(dxout, lc.XHat1, da1, m.p(l, ln1g), lc.Rstd1, m.g(l, ln1g), m.g(l, ln1b), BT, D)
		dx = dxout
	}

	for i := 0; i < BT; i++ {
		tok := c.Idx[i]
		t := i % T
		drow := dx[i*D : (i+1)*D]
		for j := 0; j < D; j++ {
			m.G[0][tok*D+j] += drow[j]
			m.G[1][t*D+j] += drow[j]
		}
	}
}

type Adam struct {
	m, v  [][]float32
	decay []bool
	t     int
}

func isDecayed(i, L int) bool {
	hb := 2 + L*layerSlots
	switch {
	case i == 0 || i == 1 || i == hb || i == hb+1 || i == hb+3:
		return false
	case i >= 2 && i < hb:
		slot := (i - 2) % layerSlots
		return slot == wqkv || slot == wo || slot == w1 || slot == w2
	case i == hb+2:
		return true
	}
	return false
}

func NewAdam(m *Model) *Adam {
	if m.G == nil {
		m.G = make([][]float32, len(m.W))
		for i, w := range m.W {
			m.G[i] = make([]float32, len(w))
		}
	}
	a := &Adam{
		m:     make([][]float32, len(m.W)),
		v:     make([][]float32, len(m.W)),
		decay: make([]bool, len(m.W)),
	}
	for i, w := range m.W {
		a.m[i] = make([]float32, len(w))
		a.v[i] = make([]float32, len(w))
		a.decay[i] = isDecayed(i, m.Cfg.Layers)
	}
	return a
}

const (
	beta1       = 0.9
	beta2       = 0.999
	eps         = 1e-8
	weightDecay = 0.01
)

func (a *Adam) Step(w, g [][]float32, lr float32) {
	a.t++
	bc1 := float32(1 - math.Pow(beta1, float64(a.t)))
	bc2 := float32(1 - math.Pow(beta2, float64(a.t)))
	for i := range w {
		wd := float32(0)
		if a.decay[i] {
			wd = weightDecay
		}
		wi, gi, mi, vi := w[i], g[i], a.m[i], a.v[i]
		for j := range wi {
			gv := gi[j]
			mi[j] = beta1*mi[j] + (1-beta1)*gv
			vi[j] = beta2*vi[j] + (1-beta2)*gv*gv
			mh := mi[j] / bc1
			vh := vi[j] / bc2
			wi[j] -= lr * (mh/(float32(math.Sqrt(float64(vh)))+eps) + wd*wi[j])
		}
	}
}

func (m *Model) Loss(idx, tgt []int) float32 {
	T := min(len(idx), m.Cfg.Ctx)
	return m.forward(idx, tgt, m.newCache(len(idx)/T, T))
}

func (m *Model) TrainStep(idx, tgt []int, a *Adam, lr float32) (float32, error) {
	B := len(idx) / m.Cfg.Ctx
	c := m.newCache(B, m.Cfg.Ctx)
	loss := m.forward(idx, tgt, c)
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		return loss, fmt.Errorf("non-finite loss")
	}
	for i := range m.G {
		mat.Zero(m.G[i])
	}
	m.Backward(c)
	var norm float64
	for _, g := range m.G {
		for _, v := range g {
			norm += float64(v) * float64(v)
		}
	}
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		return loss, fmt.Errorf("non-finite gradients")
	}
	if norm > 1 {
		scale := float32(1 / math.Sqrt(norm))
		for _, g := range m.G {
			for j := range g {
				g[j] *= scale
			}
		}
	}
	a.Step(m.W, m.G, lr)
	return loss, nil
}

type checkpoint struct {
	Cfg   Config
	Vocab []rune
	W     [][]float32
}

func (m *Model) Save(path string, vocab []rune) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".checkpoint-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	if err := gob.NewEncoder(f).Encode(checkpoint{Cfg: m.Cfg, Vocab: vocab, W: m.W}); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func Load(path string) (*Model, []rune, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	var cp checkpoint
	if err := gob.NewDecoder(f).Decode(&cp); err != nil {
		return nil, nil, err
	}
	if err := cp.Cfg.Validate(); err != nil {
		return nil, nil, fmt.Errorf("invalid checkpoint: %w", err)
	}
	sizes := cp.Cfg.sizes()
	if len(cp.W) != len(sizes) || len(cp.Vocab) != cp.Cfg.Vocab || !slices.IsSorted(cp.Vocab) || len(slices.Compact(slices.Clone(cp.Vocab))) != len(cp.Vocab) {
		return nil, nil, fmt.Errorf("invalid checkpoint layout")
	}
	for i, w := range cp.W {
		if len(w) != sizes[i] {
			return nil, nil, fmt.Errorf("invalid checkpoint tensor %d", i)
		}
		for _, v := range w {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, nil, fmt.Errorf("non-finite checkpoint weight")
			}
		}
	}
	m := &Model{Cfg: cp.Cfg, W: cp.W}
	return m, cp.Vocab, nil
}

func (m *Model) WithVocab(old, vocab []rune, seed uint64) *Model {
	cfg := m.Cfg
	cfg.Vocab = len(vocab)
	next := New(cfg, seed)
	hb := m.headBase()
	for i := range m.W {
		if i != 0 && i != hb+2 && i != hb+3 {
			next.W[i] = m.W[i]
		}
	}
	indices := make(map[rune]int, len(vocab))
	for i, r := range vocab {
		indices[r] = i
	}
	for i, r := range old {
		j := indices[r]
		copy(next.W[0][j*cfg.Dim:(j+1)*cfg.Dim], m.W[0][i*cfg.Dim:(i+1)*cfg.Dim])
		for d := 0; d < cfg.Dim; d++ {
			next.W[hb+2][d*cfg.Vocab+j] = m.W[hb+2][d*m.Cfg.Vocab+i]
		}
		next.W[hb+3][j] = m.W[hb+3][i]
	}
	return next
}

func (m *Model) Generate(ids []int, n int, temp float32, topk int, rng *rand.Rand) []int {
	return m.GenerateUntil(ids, n, temp, topk, rng, nil)
}

func (m *Model) GenerateUntil(ids []int, n int, temp float32, topk int, rng *rand.Rand, stop []int) []int {
	out := slices.Clone(ids)
	for i := 0; i < n; i++ {
		start := len(out) - m.Cfg.Ctx
		if start < 0 {
			start = 0
		}
		ctx := out[start:]
		c := m.newCache(1, len(ctx))
		m.forward(ctx, nil, c)
		last := c.Logits[(len(ctx)-1)*c.V : len(ctx)*c.V]
		probs := make([]float32, len(last))
		copy(probs, last)
		order := make([]int, len(probs))
		for j := range order {
			order[j] = j
		}
		slices.SortStableFunc(order, func(a, b int) int {
			if probs[a] > probs[b] {
				return -1
			}
			if probs[a] < probs[b] {
				return 1
			}
			return 0
		})
		if topk > 0 && topk < len(order) {
			order = order[:topk]
		}
		mx := probs[order[0]]
		var sum float32
		for _, j := range order {
			probs[j] = float32(math.Exp((float64(probs[j]) - float64(mx)) / float64(temp)))
			sum += probs[j]
		}
		r := rng.Float64() * float64(sum)
		var acc float32
		pick := order[len(order)-1]
		for _, j := range order {
			acc += probs[j]
			if float64(acc) >= r {
				pick = j
				break
			}
		}
		out = append(out, pick)
		if len(stop) > 0 && len(out)-len(ids) >= len(stop) && slices.Equal(out[len(out)-len(stop):], stop) {
			return out[:len(out)-len(stop)]
		}
	}
	return out
}

func (m *Model) Describe() string {
	c := m.Cfg
	n := 0
	for _, w := range m.W {
		n += len(w)
	}
	return fmt.Sprintf("GPT(vocab=%d dim=%d heads=%d layers=%d ctx=%d params=%d)",
		c.Vocab, c.Dim, c.Heads, c.Layers, c.Ctx, n)
}
