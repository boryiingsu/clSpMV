#ifndef __SPMV_CSR_VECTOR__H
#define __SPMV_CSR_VECTOR__H

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_csr_vector(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type devType);

#endif
