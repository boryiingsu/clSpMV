#ifndef __SPMV_CSR__H
#define __SPMV_CSR__H

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_csr(coo_matrix<int, float>* mat, int choice, int dim2Size, cl_device_type devType);

#endif
