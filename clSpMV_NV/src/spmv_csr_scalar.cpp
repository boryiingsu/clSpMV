#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"

#include "spmv_csr_scalar.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"


void spmv_csr_scalar_ocl(csr_matrix<int, float>* mat, float* vec, float* result, bool ifPad4, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes)
{

    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devTexVec;
    cl_mem devRes;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devData, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);
    int paddedres = findPaddedSize(rownum, 2 * WORK_GROUP_SIZE);
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

    opttime = 10000.0f;
    optmethod = 0;


    int dim2 = dim2Size;

    {
	int methodid = 1;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	cl_uint work_dim = 2;
	//int dim2 = 32;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	size_t globalsize[] = {padrowsize, dim2};
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_sc_pm", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);
	
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
	printf("\nCSR simple padded mat cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
    }

    {
	int methodid = 3;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	cl_uint work_dim = 2;
	//int dim2 = 32;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	size_t globalsize[] = {padrowsize, dim2};
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_sc_pm_u4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;


	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);

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
	printf("\nCSR simple padded mat unroll 4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
    }

    if (ifPad4)
    {

	//for (int poweriter = 0; poweriter < 10000; poweriter++)
	{
	    int methodid = 5;

	    cl_mem devRowPtrPad;
	    int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	    int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	    memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	    for (int i = 0; i <= mat->matinfo.height; i++)
		rowptrpad[i] = mat->csr_row_ptr[i];
	    ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	    cl_uint work_dim = 2;
	    //int dim2 = 32;
	    size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	    size_t globalsize[] = {padrowsize, dim2};
	    cl_kernel csrKernel = NULL;
	    csrKernel = clCreateKernel(program, "gpu_csr_sc_pm_float4_noif", &errorCode); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);


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
	    printf("\nCSR simple padded mat float 4 noif cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    if (devRowPtrPad)
		clReleaseMemObject(devRowPtrPad);
	    if (csrKernel)
		clReleaseKernel(csrKernel);
	    free(rowptrpad);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	}
    }

    {
	int methodid = 101;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	cl_uint work_dim = 2;
	//int dim2 = 32;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	size_t globalsize[] = {padrowsize, dim2};
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_sc_pm_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);
	
	//printf("\nfreq %lld", freq);
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
	double time_in_sec = (double)(testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nCSR simple padded mat tx cpu time %lf ms GFLOPS %lf code %d\n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
    }

    {
	int methodid = 103;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	cl_uint work_dim = 2;
	//int dim2 = 32;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	size_t globalsize[] = {padrowsize, dim2};
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_sc_pm_u4_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);
	
	//printf("\nfreq %lld", freq);
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
	double time_in_sec = (double)(testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nCSR simple padded mat unroll 4 tx cpu time %lf ms GFLOPS %lf code %d\n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
    }
    if (ifPad4)
    {
    
	{
	    int methodid = 105;
	    cl_mem devRowPtrPad;
	    int padrowsize = findPaddedSize(rownum, WORK_GROUP_SIZE);
	    int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	    memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	    for (int i = 0; i <= mat->matinfo.height; i++)
		rowptrpad[i] = mat->csr_row_ptr[i];
	    ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	    cl_uint work_dim = 2;
	    //int dim2 = 32;
	    size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	    size_t globalsize[] = {padrowsize, dim2};
	    cl_kernel csrKernel = NULL;
	    csrKernel = clCreateKernel(program, "gpu_csr_sc_pm_float4_noif_tx", &errorCode); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	    errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	float* tmpresult = (float*)malloc(sizeof(float)*rownum);
	errorCode = clEnqueueReadBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, tmpresult, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	two_vec_compare(coores, tmpresult, rownum);
	free(tmpresult);
	    
	    //printf("\nfreq %lld", freq);
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
	    double time_in_sec = (double)(testend - teststart)/(double)dim2;
	    double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	    printf("\nCSR simple padded mat float 4 noif tx cpu time %lf ms GFLOPS %lf code %d\n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    if (devRowPtrPad)
		clReleaseMemObject(devRowPtrPad);
	    if (csrKernel)
		clReleaseKernel(csrKernel);
	    free(rowptrpad);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	}

    }

    //Clean up
    if (image2dVec)
	free(image2dVec);

    if (devRowPtr)
	clReleaseMemObject(devRowPtr);
    if (devColId)
	clReleaseMemObject(devColId);
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



void scalar_init_pointers(int* rowptr, int* rowpointers, int* itempointers)
{
    for (int i = 0; i < 256; i++)
    {
        int start = rowptr[i];
        int end = rowptr[i+1];
        int size = end - start;
        rowpointers[i] = i;
        if (size > 0)
            itempointers[i] = start;
        else
            itempointers[i] = -1;	
    }
}


void scalar_collect_colids(int* matcols, int* itempointers, int* colids)
{
    for (int i = 0; i < 256; i++)
    {
        if (itempointers[i] >= 0)
        {
            colids[i] = matcols[itempointers[i]];
        }
        else
        {
            colids[i] = -1;
        }
    }
}

void scalar_find_cachelines(int* colids, int& numlines, int& numitems, int startid)
{
    using namespace std;
    numlines = 0; 
    numitems = 0;
    vector<pair<int, int> > histo;
    histo.clear();
    for (int i = startid; i < startid + 16; i++)
    {
        int col = colids[i];
        if (col >= 0)
        {
            numitems++;
            int cacheid = col / 16;
            bool ifexist = false;
            for (int j = 0; j < histo.size(); j++)
            {
                if (cacheid == histo[j].first)
                {
                    histo[j].second++;
                    ifexist = true;
                    break;
                }
            }
            if (!ifexist)
            {
                numlines++;
                histo.push_back(pair<int, int>(cacheid, 1));
            }
        }
    }
    assert(numlines == histo.size());
}

bool scalar_nextchunk(int* rowptr, int* rowpointers, int* itempointers, int rowsize)
{
    bool ifnotdone = false;
    for (int euid = 0; euid < 16; euid++)
    {
        bool ifremain = false;
        for (int laneid = 0; laneid < 16; laneid++)
        {
            int itemid = euid * 16 + laneid;
            int currow = rowpointers[itemid];
            int end = rowptr[currow + 1];
            int curposition = itempointers[itemid];
            if (curposition >= 0 && curposition + 1 < end)
            {
                ifremain = true;
                ifnotdone = true;
                itempointers[itemid] = curposition + 1;
            }
            else
            {
                itempointers[itemid] = -1;
            }
        }
        if (!ifremain)
        {
            int maxrow = 0;
            for (int i = 0; i < 256; i+=16)
            {
                if (rowpointers[i] > maxrow)
                    maxrow = rowpointers[i];
            }
            int newrow = maxrow + 16;
            if (newrow < rowsize)
            {
                ifnotdone = true;
                for (int laneid = 0; laneid < 16; laneid++)
                {
                    int itemid = euid * 16 + laneid;
                    if (newrow + laneid < rowsize)
                    {
                        rowpointers[itemid] = newrow + laneid;
                        int start = rowptr[newrow + laneid];
                        int end = rowptr[newrow + laneid + 1];
                        int size = end - start;
                        if (size > 0)
                            itempointers[itemid] = rowptr[newrow + laneid];
                        else
                            itempointers[itemid] = -1;
                    }
                    else
                    {
                        rowpointers[itemid] = -1;
                        itempointers[itemid] = -1;
                    }
                }
            }
        }
    }
    return ifnotdone;
}

void scalar_cache_behavior(csr_matrix<int, float>* mat)
{
    int rowpointers[256];
    int itempointers[256];
    int colids[256];
    scalar_init_pointers(mat->csr_row_ptr, rowpointers, itempointers);
    double minusage = 10000.0f;
    double maxusage = 0.0f;
    double sum = 0.0f;
    int iteration = 0;
	int totalitem = 0;
	int totalline = 0;
    while (1)
    {
        scalar_collect_colids(mat->csr_col_id, itempointers, colids);
		for (int euid = 0; euid < 16; euid++)
		{
			int numlines = 0;
			int numitems = 0;
			scalar_find_cachelines(colids, numlines, numitems, euid * 16);
			double ratio = 0.0f;
			if (numlines > 0)
			{
				ratio = (double)numitems * 4.0 / ((double)numlines * 64.0);
				//if (ratio > 1.0f)
					//ratio = 1.0f;
				if (ratio > maxusage)
					maxusage = ratio;
				if (ratio < minusage)
					minusage = ratio;
				sum += ratio;
				iteration++;
				totalitem += numitems;
				totalline += numlines;
			}
		}
		if (!scalar_nextchunk(mat->csr_row_ptr, rowpointers, itempointers, mat->matinfo.height))
            break;
    }
    printf("Scalar Cache Behavior Avg %f  Max %f  Min %f total item %d total cacheline %d\n", sum / (double)iteration, maxusage, minusage, totalitem, totalline);
}



void spmv_csr_scalar(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);
    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(mat, &csrmat);
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);

    //if (choice == 1)
    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	//scalar_cache_behavior(&csrmat);

	spmv_csr_scalar_ocl(&csrmat, vec, res, false, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, 4);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
	spmv_csr_scalar_ocl(&padcsr, vec, res, true, dim2Size, opttime2, optmethod2, oclfilename, deviceType, coores, ntimes);
	free_csr_matrix(padcsr);

	printf("\n------------------------------------------------------------------------\n");
	printf("Scalar kernel without mat padding best time %f ms best method %d", opttime1*1000.0, optmethod1);
	printf("\n------------------------------------------------------------------------\n");
	printf("Scalar kernel with mat padding best time %f ms best method %d", opttime2*1000.0, optmethod2);
	printf("\n------------------------------------------------------------------------\n");
    }


    free(vec);
    free(res);
    free_csr_matrix(csrmat);
    free(coores);
}

