#ifndef __SPMV_CSR_SCALAR__H
#define __SPMV_CSR_SCALAR__H

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_csr_scalar(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type devType);

#endif
