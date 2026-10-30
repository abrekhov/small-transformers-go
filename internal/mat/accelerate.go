//go:build darwin && cgo

package mat

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>

static void gelu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = 0.7978845608028654f * (x[i] + 0.044715f*x[i]*x[i]*x[i]);
    vvtanhf(out, out, &n);
    for (int i = 0; i < n; i++)
        out[i] = 0.5f*x[i]*(1.0f + out[i]);
}

static void gelu_back(float *out, const float *x, const float *dout, int n) {
    for (int i = 0; i < n; i++)
        out[i] = 0.7978845608028654f * (x[i] + 0.044715f*x[i]*x[i]*x[i]);
    vvtanhf(out, out, &n);
    for (int i = 0; i < n; i++) {
        float th = out[i];
        out[i] = dout[i]*(0.5f*(1.0f + th) + 0.5f*x[i]*(1.0f - th*th)*0.7978845608028654f*(1.0f + 3.0f*0.044715f*x[i]*x[i]));
    }
}
*/
import "C"

const Backend = "accelerate"

func MM(c, a, b []float32, m, k, n int) {
	Gemm(c, a, b, m, k, n, k, n, n, false, false, 1)
}

func MMT(c, a, b []float32, m, k, n int) {
	Gemm(c, a, b, m, k, n, m, n, n, true, false, 1)
}

func MBT(c, a, b []float32, m, k, n int) {
	Gemm(c, a, b, m, k, n, k, k, n, false, true, 1)
}

func Gemm(c, a, b []float32, m, k, n, as, bs, cs int, ta, tb bool, scale float32) {
	at, bt := C.enum_CBLAS_TRANSPOSE(C.CblasNoTrans), C.enum_CBLAS_TRANSPOSE(C.CblasNoTrans)
	if ta {
		at = C.CblasTrans
	}
	if tb {
		bt = C.CblasTrans
	}
	C.cblas_sgemm(C.CblasRowMajor, at, bt, C.int(m), C.int(n), C.int(k),
		C.float(scale), (*C.float)(&a[0]), C.int(as), (*C.float)(&b[0]), C.int(bs),
		0, (*C.float)(&c[0]), C.int(cs))
}

func Gelu(out, x []float32) {
	C.gelu((*C.float)(&out[0]), (*C.float)(&x[0]), C.int(len(x)))
}

func GeluBack(out, x, dout []float32) {
	C.gelu_back((*C.float)(&out[0]), (*C.float)(&x[0]), (*C.float)(&dout[0]), C.int(len(x)))
}
