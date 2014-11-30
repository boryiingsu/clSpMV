#ifndef __SPMV_DIA_H__
#define __SPMV_DIA_H__

#include "CL/cl.h"
#include "matrix_storage.h"

void spmv_dia(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType);

#endif
