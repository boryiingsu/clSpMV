#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CL/cl.h"

#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "constant.h"
#include "matrix_storage.h"
#include "fileio.h"
#include "analyze.h"
#include "spmv_cocktail.h"


//Exapmle of how to use the api to perform multiple SpMVs on gpu, y = A^k*x
//iteration: the number of SpMVs performed in the loop
void example(coo_matrix<int, float>& coomat, float* vec, float* res, int iteration)
{
    int res_size = coomat.matinfo.height;
    int vec_size = coomat.matinfo.width;
    float* cpuvec = (float*)malloc(sizeof(float)*vec_size);
    float* cpures = (float*)malloc(sizeof(float)*res_size);
    float* gpures = (float*)malloc(sizeof(float)*res_size);
    memcpy(cpuvec, vec, sizeof(float)*vec_size);
    memcpy(cpures, res, sizeof(float)*res_size);

    //Initialize the cocktailformat
    cocktail<int, int, float> mat;
    init_cocktail(mat);
    //Analyze the matrix and represent it using thc cocktail format on cpu
    analyze_matrix(coomat, mat);

    cocktail_gpu gpumat;
    cocktail_kernels kernels;
    //Initialize the gpu matrices and kernels.
    //The last argument refers to whether to use texture memory to cache the multiplied vector or not
    init_mat_kernels(mat, vec, res, gpumat, kernels, true);
    //Copy the initial res vector to gpu
    //This step is necessary, because the SpMV computes y = A*x+y
    cpy_result_from_cpu(gpumat, kernels.context, kernels.cmdQueue, res, res_size);
    for (int i = 0; i < iteration; i++)
    {
	printf("GPU Iteration Number %d\n", i+1);
	//Do one GPU SpMV
	do_spmv(kernels, kernels.context, kernels.cmdQueue, 1);
	//Copy the gpumat.res to gpumat.vec, for the next iteration
	cpy_vector_from_gpu(gpumat, kernels.context, kernels.cmdQueue, gpumat.res, res_size);
    }
    //Copy gpu results to CPU
    cpy_result_to_cpu(gpumat, kernels.context, kernels.cmdQueue, gpures, res_size);

    for (int i = 0; i < iteration; i++)
    {
	printf("CPU Iteration Number %d\n", i+1);
	//Do one CPU SpMV
	coo_spmv(&coomat, cpuvec, cpures, vec_size);
	//Copy the res vector to vec, for the next iteration
	memcpy(cpuvec, cpures, sizeof(float)*vec_size);
    }

    //Compare the cpu and gpu results
    two_vec_compare(cpures, gpures, res_size);

    free(gpures);
    free(cpures);
    free(cpuvec);
    //Free gpu memory usage
    free_mat_kernels(gpumat, kernels);
    
}

//Evaluate the correctness and performance of a single SpMV kernel
void do_evaluation(coo_matrix<int, float>& coomat, float* vec, float* res)
{
    float* coores = (float*)malloc(sizeof(float)*coomat.matinfo.height);
    double starttime = timestamp();
    spmv_only(&coomat, vec, coores);
    double endtime = timestamp();
    printf("SpMV serial COO time %lf s\n", endtime - starttime);
    cocktail<int, int, float> mat;
    init_cocktail(mat);

    //The most common parameter setting
    analyze_matrix(coomat, mat);
    //To match the spheres and wind benchmark number:
    //the 0.5 factor decreases the priority of dia based format, so the blocked based format will have higher priority
    //analyze_matrix(coomat, mat, 0.5);
    //To match the in-2004 and mip1 benchmark number:
    //the estimate_flat_partial flat is set to false, it will use the best achievable performance to estimate the flat based performance
    //analyze_matrix(coomat, mat, 1.0, false);
    //To match the Si41Ge41H72 number:
    //the estimate_flat_full flat is set to true, it will decompose the remaining matrix using many flat formats, and estimate the total runtime based on the decomposition
    //analyze_matrix(coomat, mat, 1.0, true, true);

    evaluate(mat, vec, res, coores);
    
    free_cocktail(mat);
    free(coores);
}


int main(int argc, char** argv)
{
    if (argc < 2)
    {
	printf("\nUsage: spmv_cocktail input_matrix.mtx");
	printf("\nThe matrix needs to be in the matrix market format\n");
	return 0;
    }
    char* filename = argv[1];
    coo_matrix<int, float> coomat;
    init_coo_matrix<int, float>(coomat);
    ReadMMF(filename, &coomat);
    printMatInfo(&coomat);
    float* vec = (float*)malloc(sizeof(float)*coomat.matinfo.width);
    float* res = (float*)malloc(sizeof(float)*coomat.matinfo.height);
    initVectorOne<int, float>(vec, coomat.matinfo.width);	
    initVectorZero<int, float>(res, coomat.matinfo.height);
   
    do_evaluation(coomat, vec, res);
    //example(coomat, vec, res, 10);
    
    free_coo_matrix(coomat);
    free(vec);
    free(res);

    return 0;
}


