#ifndef __FFT_MAGNITUDE_H__
#define __FFT_MAGNITUDE_H__

#include <cuda_runtime.h>

cudaError_t compute_fft_magnitude(float *h_input, float *h_output, int n);

#endif // __FFT_MAGNITUDE_H__
