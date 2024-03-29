#pragma once
#include <cuda.h>

template <typename T>
__device__ void transpose(const T* A, T* A_T, int num_rows_input, int num_cols_input) {
    #pragma unroll
    for (int row = 0; row < num_rows_input; row++) {
        #pragma unroll
        for (int col = 0; col < num_cols_input; col++) {
            A_T[col * num_rows_input + row] = A[row * num_cols_input + col];
        }
    }
}

template <typename T>
__device__ void
matrix_multiply(const T* A, const T* B, T* C, int num_rows_A, int num_cols_A, int num_cols_B) {
    #pragma unroll
    for (int row_a = 0; row_a < num_rows_A; row_a++) {
        #pragma unroll
        for (int col_b = 0; col_b < num_cols_B; col_b++) {
            T sum = 0;
            #pragma unroll
            for (int cols_A = 0; cols_A < num_cols_A; cols_A++) {
                sum += A[row_a * num_cols_A + cols_A] * B[cols_A * num_cols_B + col_b];
            }
            C[row_a * num_cols_B + col_b] = sum;
        }
    }
}
