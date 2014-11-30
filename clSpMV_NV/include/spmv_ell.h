#ifndef __SPMV_ELL_H__
#define __SPMV_ELL_H__

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_ell(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType);

#endif
