#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"


#include "spmv_coo.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"

void spmv_coo_ocl(coo_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes)
{

    for (int i = 0; i < mat->matinfo.height; i++)
	result[i] = 0.0f;
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowid;
    cl_mem devColid;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRes;
    cl_mem devTexVec;
    cl_mem devTmpRow;
    cl_mem devTmpData;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int vecsize = mat->matinfo.width;
    int num_units = nnz / COO_GROUP_SIZE;
    if (nnz % COO_GROUP_SIZE != 0)
	num_units++;
    int group_num = 192;//(num_units < 24) ? num_units : 24;
    int work_size = group_num * COO_GROUP_SIZE;
    int num_iters = nnz / work_size;
    if (nnz % work_size != 0)
	num_iters++;
    int process_size = num_iters * COO_GROUP_SIZE;
    int active_warp = num_units / num_iters;
    if (num_units % num_iters != 0)
	active_warp++;
    int paddedNNZ = findPaddedSize(nnz, COO_ALIGNMENT);
    int* paddedRow = (int*)malloc(sizeof(int)*paddedNNZ);
    int* paddedCol = (int*)malloc(sizeof(int)*paddedNNZ);
    float* paddedData = (float*)malloc(sizeof(float)*paddedNNZ);
    memcpy(paddedRow, mat->coo_row_id, sizeof(int)*nnz); 
    memcpy(paddedCol, mat->coo_col_id, sizeof(int)*nnz); 
    memcpy(paddedData, mat->coo_data, sizeof(float)*nnz); 
    for (int i = nnz; i < paddedNNZ; i++)
    {
	paddedRow[i] = mat->coo_row_id[nnz - 1];
	paddedCol[i] = mat->coo_col_id[nnz - 1];
	paddedData[i] = 0.0f;
    }

    ALLOCATE_GPU_READ(devRowid, paddedRow, sizeof(int)*paddedNNZ);
    ALLOCATE_GPU_READ(devColid, paddedCol, sizeof(int)*paddedNNZ);
    ALLOCATE_GPU_READ(devData, paddedData, sizeof(float)*paddedNNZ);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);
    int paddedres = findPaddedSize(rownum, 512);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
    devTmpRow = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*MAX_WARP_NUM, NULL, &errorCode); CHECKERROR;
    devTmpData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*MAX_WARP_NUM, NULL, &errorCode); CHECKERROR;

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
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {COO_GROUP_SIZE, 1};
	int gsize = group_num * COO_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_coo_s1", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &process_size); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &paddedNNZ); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devTmpRow); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devTmpData); CHECKERROR;

	printf("process size %d nnz %d gsize %d active_warp %d num_units %d group_num %d num_iters %d \n", process_size, paddedNNZ, gsize, active_warp, num_units, group_num, num_iters);

	size_t blocksize2[] = {COO_GROUP_SIZE * 2, 1};
	size_t globalsize2[] = {COO_GROUP_SIZE * 2, dim2};


	cl_kernel csrKernel2 = NULL;
	csrKernel2 = clCreateKernel(program, "gpu_coo_s2", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 0, sizeof(cl_mem), &devTmpRow); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 1, sizeof(cl_mem), &devTmpData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 2, sizeof(int), &active_warp); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 3, sizeof(cl_mem), &devRes); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel2, work_dim, NULL, globalsize2, blocksize2, 0, NULL, NULL); CHECKERROR;
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


	//int* tmpRow = (int*)malloc(sizeof(int)*MAX_WARP_NUM);
	//float* tmpData = (float*)malloc(sizeof(float)*MAX_WARP_NUM);


	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	//}
	//clFinish(cmdQueue);

	//errorCode = clEnqueueReadBuffer(cmdQueue, devTmpRow, CL_TRUE, 0, sizeof(int)*MAX_WARP_NUM, tmpRow, 0, NULL, NULL); CHECKERROR;
	//errorCode = clEnqueueReadBuffer(cmdQueue, devTmpData, CL_TRUE, 0, sizeof(float)*MAX_WARP_NUM, tmpData, 0, NULL, NULL); CHECKERROR;
	//clFinish(cmdQueue);

	//for (int i = 0; i < ntimes; i++)
	//{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel2, work_dim, NULL, globalsize2, blocksize2, 0, NULL, NULL); CHECKERROR;	    
	}
	clFinish(cmdQueue);

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nCOO cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);
	if (csrKernel2)
	    clReleaseKernel(csrKernel2);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
	//for (int i = 0; i < active_warp; i++)
	//printf("Row %d Data %f\n", tmpRow[i], tmpData[i]);
    }
    fflush(stdout);
    {
	int methodid = 100;
	cl_uint work_dim = 2;
	size_t blocksize[] = {COO_GROUP_SIZE, 1};
	int gsize = group_num * COO_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_coo_s1_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &process_size); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &paddedNNZ); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(cl_mem), &devTmpRow); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 8, sizeof(cl_mem), &devTmpData); CHECKERROR;

	printf("process size %d nnz %d gsize %d active_warp %d num_units %d group_num %d num_iters %d \n", process_size, paddedNNZ, gsize, active_warp, num_units, group_num, num_iters);

	size_t blocksize2[] = {COO_GROUP_SIZE * 2, 1};
	size_t globalsize2[] = {COO_GROUP_SIZE * 2, dim2};


	cl_kernel csrKernel2 = NULL;
	csrKernel2 = clCreateKernel(program, "gpu_coo_s2", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 0, sizeof(cl_mem), &devTmpRow); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 1, sizeof(cl_mem), &devTmpData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 2, sizeof(int), &active_warp); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 3, sizeof(cl_mem), &devRes); CHECKERROR;


	errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel2, work_dim, NULL, globalsize2, blocksize2, 0, NULL, NULL); CHECKERROR;
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


	//int* tmpRow = (int*)malloc(sizeof(int)*MAX_WARP_NUM);
	//float* tmpData = (float*)malloc(sizeof(float)*MAX_WARP_NUM);


	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	//}
	//clFinish(cmdQueue);

	//errorCode = clEnqueueReadBuffer(cmdQueue, devTmpRow, CL_TRUE, 0, sizeof(int)*MAX_WARP_NUM, tmpRow, 0, NULL, NULL); CHECKERROR;
	//errorCode = clEnqueueReadBuffer(cmdQueue, devTmpData, CL_TRUE, 0, sizeof(float)*MAX_WARP_NUM, tmpData, 0, NULL, NULL); CHECKERROR;
	//clFinish(cmdQueue);

	//for (int i = 0; i < ntimes; i++)
	//{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel2, work_dim, NULL, globalsize2, blocksize2, 0, NULL, NULL); CHECKERROR;	    
	}
	clFinish(cmdQueue);

	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
	printf("\nCOO cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);
	if (csrKernel2)
	    clReleaseKernel(csrKernel2);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
	//for (int i = 0; i < active_warp; i++)
	//printf("Row %d Data %f\n", tmpRow[i], tmpData[i]);
    }


    //Clean up
    if (paddedRow)
	free(paddedRow);
    if (paddedCol)
	free(paddedCol);
    if (paddedData)
	free(paddedData);
    if (image2dVec)
	free(image2dVec);

    if (devRowid)
	clReleaseMemObject(devRowid);
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
    if (devTmpRow)
	clReleaseMemObject(devTmpRow);
    if (devTmpData)
	clReleaseMemObject(devTmpData);

    freeObjects(devices, &context, &cmdQueue, &program);
}


void spmv_coo(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);

    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	spmv_coo_ocl(mat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);


	printf("\n------------------------------------------------------------------------\n");
	printf("COO best time %f ms best method %d", opttime1*1000.0, optmethod1);
	printf("\n------------------------------------------------------------------------\n");
    }

    free(vec);
    free(res);
    free(coores);
}

