#ifndef UTIL__H__
#define UTIL__H__

#include <sys/time.h>
#include "matrix_storage.h"

double timestamp ();
int findPaddedSize(int realSize, int alignment);
void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment);
double distance(float* vec1, float* vec2, int size);
void correctness_check(coo_matrix<int, float>* mat, float* vec, float* res);
void printMatInfo(coo_matrix<int, float>* mat);
void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores);
void two_vec_compare(float* coovec, float* newvec, int size);
void rearrange_bell_4col(bell_matrix<int, float>* mat, int alignment);
unsigned int getRandMax();
unsigned int getRand(unsigned int upper);

#endif

