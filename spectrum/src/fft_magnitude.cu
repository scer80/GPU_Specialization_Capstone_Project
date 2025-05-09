#include <iostream>
#include <cufft.h>
#include "fft_magnitude.h"


const char *cufftGetErrorString(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";
        default:
            return "Unknown CUFFT error";
    }
}


__global__ void calculate_magnitude(cufftComplex *d_input, float *d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float real = d_input[idx].x;
        float imag = d_input[idx].y;
        d_output[idx] = sqrtf(real * real + imag * imag);
    }
}


cudaError_t compute_fft_magnitude(float *h_input, float *h_output, int n) {
    cufftHandle plan;
    cudaError_t cudaStatus;
    cufftResult cufftStatus;

    // Allocate device memory
    float *d_input = nullptr;
    cufftComplex *d_output = nullptr;
    cudaStatus = cudaMalloc((void **) &d_input, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for input: " << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }
    cudaStatus = cudaMalloc((void **) &d_output, (n/2 + 1) * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for output: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        return cudaStatus;
    }

    // Create cuFFT plan
    cufftStatus = cufftPlan1d(&plan, n, CUFFT_R2C, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        std::cerr << "cufftPlan1d failed: " << cufftGetErrorString(cufftStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return cudaErrorUnknown;
    }

    // Copy input data to device
    cudaStatus = cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed to copy input to device: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cufftDestroy(plan);
        return cudaStatus;
    }

    // Execute FFT
    cufftStatus = cufftExecR2C(plan, d_input, d_output);
    if (cufftStatus != CUFFT_SUCCESS) {
        std::cerr << "cufftExecR2C failed: " << cufftGetErrorString(cufftStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cufftDestroy(plan);
        return cudaErrorUnknown;
    }

    // Calculate magnitude and copy to host
    float *d_magnitude = nullptr;
    cudaStatus = cudaMalloc((void **) &d_magnitude, (n/2 + 1) * sizeof(float));
     if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for magnitude: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cufftDestroy(plan);
        return cudaStatus;
    }

    dim3 dimBlock(256);
    dim3 dimGrid((n/2 + 1 + dimBlock.x - 1) / dimBlock.x);

    calculate_magnitude<<<dimGrid, dimBlock>>>(d_output, d_magnitude, n/2 + 1);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "calculate_magnitude launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_magnitude);
        cufftDestroy(plan);
        return cudaErrorUnknown;
    }

    cudaStatus = cudaMemcpy(h_output, d_magnitude, (n/2 + 1) * sizeof(float), cudaMemcpyDeviceToHost);    
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed to copy output to host: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_magnitude);
        cufftDestroy(plan);
        return cudaStatus;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_magnitude);
    cufftDestroy(plan);

    return cudaSuccess;
}
