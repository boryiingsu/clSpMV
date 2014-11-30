#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "matrix_storage.h"
#include "spmv_serial.h"
#include "fileio.h"
#include "spmv_bdia.h"
#include "spmv_dia.h"
#include "spmv_ell.h"
#include "spmv_bell.h"
#include "spmv_sell.h"
#include "spmv_sbell.h"
#include "spmv_coo.h"
#include "spmv_csr_vector.h"
#include "spmv_csr_scalar.h"
#include "spmv_bcsr.h"
#include "mem_bandwidth.h"


void printDense(coo_matrix<int, float>& mat)
{
    int width = mat.matinfo.width;
    int height = mat.matinfo.height;
    float* dense = (float*)malloc(sizeof(float)*width*height);
    memset(dense, 0, sizeof(int)*width*height);
    for (int i = 0; i < mat.matinfo.nnz; i++)
    {
        int row = mat.coo_row_id[i];
        int col = mat.coo_col_id[i];
        float data = mat.coo_data[i];
        dense[row * width + col] = data;
    }
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%f ", dense[i * width + j]);
        }
        printf("\n");
    }
}

void printCOO(coo_matrix<int, float>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    int nnz = mat.matinfo.nnz;
    for (int i =0 ; i < nnz; i++)
    {
        printf("row %d col %d data %f\n", mat.coo_row_id[i], mat.coo_col_id[i], mat.coo_data[i]);
    }
}

void printDIA(dia_matrix<int, int, float>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("dianum %d length %d alignLength %d\n", mat.dia_num, mat.dia_length, mat.dia_length_aligned);
    int num = mat.dia_num;
    printf("Offset: ");
    for (int i = 0; i < num; i++)
        printf("%d ", mat.dia_offsets[i]);
    printf("\n");
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < mat.dia_length_aligned; j++)
            printf("%f ", mat.dia_data[i * mat.dia_length_aligned + j]);
        printf("\n");
    }
}

void printDIAext(dia_ext_matrix<int, int, float>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("dianum %d length %d alignLength %d\n", mat.dia_num, mat.dia_length, mat.dia_length_aligned);
    int num = mat.dia_num;
    printf("Offset: ");
    for (int i = 0; i < num; i++)
        printf("%d ", mat.dia_offsets[i]);
    printf("\n");
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < mat.dia_length_aligned; j++)
            printf("%f ", mat.dia_data[i * mat.dia_length_aligned + j]);
        printf("\n");
    }
}

void printCSR(csr_matrix<int, float>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    int nnz = mat.matinfo.nnz;
    int height = mat.matinfo.height;
    printf("Row ptr: ");
    for (int i = 0; i <= height; i++)
        printf("%d ", mat.csr_row_ptr[i]);
    printf("\n");
    printf("Col: ");
    for (int i = 0; i < nnz; i++)
        printf("%d ", mat.csr_col_id[i]);
    printf("\n");
    printf("Data: ");
    for (int i = 0; i < nnz; i++)
        printf("%f ", mat.csr_data[i]);
    printf("\n");
}

void printBCSR(bcsr_matrix<int, float>& mat)
{
    printf("width %d height %d nnz %d\n", mat.matinfo.width, mat.matinfo.height, mat.matinfo.nnz);
    printf("bwidth %d bheight %d rownum %d alignsize %d blocknum %d\n", mat.bcsr_bwidth, mat.bcsr_bheight, mat.bcsr_row_num, mat.bcsr_aligned_size, mat.bcsr_block_num);
    printf("Row ptr: ");
    for (int i = 0; i <= mat.bcsr_row_num; i++)
	printf("%d ", mat.bcsr_row_ptr[i]);
    printf("\n");
    printf("Col id: ");
    for (int i = 0; i < mat.bcsr_block_num; i++)
	printf("%d ", mat.bcsr_col_id[i]);
    printf("\n");
    printf("Data: \n");
    int blocksize = mat.bcsr_bwidth * mat.bcsr_bheight;
    for (int i = 0; i < blocksize; i++)
    {
	for (int j = 0; j < mat.bcsr_aligned_size; j++)
	    printf("%f ", mat.bcsr_data[j + i * mat.bcsr_aligned_size]);
	printf("\n");
    }
    printf("\n");
    
}

void writeCSR(char* filename, csr_matrix<int, float>& mat)
{
	FILE* f = fopen(filename, "wb");
	unsigned int tmp = mat.matinfo.width;
	fwrite(&tmp, sizeof(unsigned int), 1, f);

	tmp = mat.matinfo.height;
    //read rows (UINT)
    fwrite(&tmp, sizeof(unsigned int), 1, f);

	tmp = mat.matinfo.nnz;
    //read # nonzero values (UINT)
    fwrite(&tmp, sizeof(unsigned int), 1, f);

    //read column indices (UINT *)
	fwrite(mat.csr_col_id, sizeof(unsigned int), mat.matinfo.nnz, f);

    //read row pointer (UINT *)
	fwrite(mat.csr_row_ptr, sizeof(unsigned int), mat.matinfo.height + 1, f);

    //read all nonzero values (float *)
	fwrite(mat.csr_data, sizeof(float), mat.matinfo.nnz, f);

    fclose(f);
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
	printf("\nUsage: spmv_all input_matrix.mtx method execution_times");
	printf("\nThe matrix needs to be in the matrix market format");
	printf("\nThe method is the format you want to use:");
	printf("\n\tMethod 0: mesure the memory bandwidth and kernel launch overhead only");
	printf("\n\tMethod 1: use the csr matrix format, using the scalar implementations");
	printf("\n\tMethod 2: use the csr matrix format, using the vector implementations");
	printf("\n\tMethod 3: use the bdia matrix format");
	printf("\n\tMethod 4: use the dia matrix format");
	printf("\n\tMethod 5: use the ell matrix format");
	printf("\n\tMethod 6: use the coo matrix format");
	printf("\n\tMethod 7: use the bell matrix format");
	printf("\n\tMethod 8: use the bcsr matrix format");
	printf("\n\tMethod 9: use the sell matrix format");
	printf("\n\tMethod 10: use the sbell matrix format");
	printf("\nThe execution_times refers to how many times of SpMV you want to do to benchmark the execution time\n");
	return 0;
    }

    char* filename = argv[1];
    int choice = 1;
    if (argc > 2)
	choice = atoi(argv[2]);
    int dim2Size = 1;
    int ntimes = 20;
    if (argc > 3)
	ntimes = atoi(argv[3]);

    coo_matrix<int, float> mat;
    init_coo_matrix(mat);
    ReadMMF(filename, &mat);

    char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    
    if (choice == 0)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/mem_bandwidth.cl");
	bandwidth_test(clfilename, CONTEXTTYPE, dim2Size);
    }
    else if (choice == 1)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_csr_scalar.cl");
	spmv_csr_scalar(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 2)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_csr_vector.cl");
	spmv_csr_vector(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 3)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bdia.cl");
	spmv_bdia(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 4)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_dia.cl");
	spmv_dia(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 5)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_ell.cl");
	spmv_ell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 6)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_coo.cl");
	spmv_coo(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 7)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bell.cl");
	spmv_bell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 8)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bcsr.cl");
	spmv_bcsr(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 9)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sell.cl");
	spmv_sell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }
    else if (choice == 10)
    {
	sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sbell.cl");
	spmv_sbell(clfilename, &mat, dim2Size, ntimes, CONTEXTTYPE);
    }

    free_coo_matrix(mat);

    return 0;
}

