#ifndef __MAT_IO__H__
#define __MAT_IO__H__

#include "matrix_storage.h"

void ReadMMF(char* filename, coo_matrix<int, float>* mat);

#endif
