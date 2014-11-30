#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"


#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"

void spmv_ell_ocl(ell_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devColid;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRes;
    cl_mem devTexVec;

    //Initialize values
    int aligned_length = mat->ell_height_aligned;
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int vecsize = mat->matinfo.width;
    int ellnum = mat->ell_num;
    ALLOCATE_GPU_READ(devColid, mat->ell_col_id, sizeof(int)*aligned_length*ellnum);
    ALLOCATE_GPU_READ(devData, mat->ell_data, sizeof(float)*aligned_length*ellnum);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);
    int paddedres = findPaddedSize(rownum, 512);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
    const cl_image_format floatFormat =
    {
	CL_R,
	CL_FLOAT,
    };


    int width = VEC2DWIDTH;
    int height = (vecsize + VEC2DWIDTH - 1)/VEC2DWIDTH;
    float* image2dVec = (float*)malloc(sizeof(float)*width*height);
    memset(image2dVec, 0, sizeof(float)*width*height);
    for (int i = 0; i < vecsize; i++)
    {
	image2dVec[i] = vec[i];
    }
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height, 1};
    devTexVec = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height, 0, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteImage(cmdQueue, devTexVec, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);

    //printf("\nvec length %d padded length %d", mat->matinfo.width, padveclength);

    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = dim2Size;
    {
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int gsize = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_ell", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &rownum); CHECKERROR;

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nELL cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);
	floptable[methodid] = gflops;
	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	    optflop = gflops;
	}

    }
    
    {
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int aligned4 = aligned_length / 4;
	int row4 = rownum / 4;
	if (rownum % 4 != 0)
	    row4++;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_ell_v4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &ellnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &row4); CHECKERROR;

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nELL float4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	floptable[methodid] = gflops;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	    optflop = gflops;
	}

    }


    //Clean up
    if (image2dVec)
	free(image2dVec);

    if (devColid)
	clReleaseMemObject(devColid);
    if (devData)
	clReleaseMemObject(devData);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devTexVec)
	clReleaseMemObject(devTexVec);
    if (devRes)
	clReleaseMemObject(devRes);


    freeObjects(devices, &context, &cmdQueue, &program);
}

void init_ell_mat(ell_matrix<int, float>& mat, unsigned int dimension, unsigned int ellnum)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int nnz = dimension * ellnum;
    mat.matinfo.nnz = nnz;
    mat.ell_num = ellnum;
    mat.ell_height_aligned = dimension;
    mat.ell_col_id = (int*)malloc(sizeof(int)*nnz);
    mat.ell_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i < nnz; i++)
	mat.ell_data[i] = 1.0f;
    for (unsigned int rowid = 0; rowid < dimension; rowid++)
    {
	int start = rowid - ellnum / 2;
	if (start < 0)
	    start = 0;
	int end = start + ellnum;
	if (end > dimension)
	{
	    end = dimension;
	    start = end - ellnum;
	}
	for (int j = start; j < end; j++)
	{
	    mat.ell_col_id[(j - start) * dimension + rowid] = j;
	}
    }
    for (unsigned int i = 0; i < nnz; i++)
	assert(mat.ell_col_id[i] >= 0 && mat.ell_col_id[i] < dimension);
}


void benchmark_ell(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/ell.ben");
    FILE* outfile = fopen(outname, "w");
    int methodnum = 2;
    double floptable[methodnum];
    for (unsigned int size = 1024; size <= 2*1024*1024; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int ellnum = 1; ellnum <= 128; ellnum *= 2)
	{
	    if (size*ellnum > 67108864)
		break;
	    ell_matrix<int, float> bellmat;
	    init_ell_mat(bellmat, size, ellnum);

	    double opttime = 10000.0f;
	    double optflop = 0.0f;
	    int optmethod = 0;

	    spmv_ell_ocl(&bellmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);
	    
	    printf("\n------------------------------------------------------------------------\n");
	    printf("ELL Dim %d BN %d opttime %f ms optflop %f optmethod %d", size, ellnum, opttime*1000.0, optflop,  optmethod);
	    printf("\n------------------------------------------------------------------------\n");
	    fprintf(outfile, "%d %d", size, ellnum);
	    for (unsigned int k = 0; k < methodnum; k++)
		fprintf(outfile, " %f", floptable[k]);
	    fprintf(outfile, "\n");
	    
	    free_ell_matrix(bellmat);
	}
	free(vec);
	free(res);
    }
    fclose(outfile);
}


int main(int argv, char** argc)
{
    char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_ell.cl");
    int ntimes = 1000;
    benchmark_ell(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



