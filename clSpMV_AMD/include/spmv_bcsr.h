#ifndef __SPMV_BCSR_H__
#define __SPMV_BCSR_H__

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_bcsr(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType);

#endif
