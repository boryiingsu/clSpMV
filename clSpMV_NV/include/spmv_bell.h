#ifndef __SPMV_BELL_H__
#define __SPMV_BELL_H__

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_bell(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType);

#endif
