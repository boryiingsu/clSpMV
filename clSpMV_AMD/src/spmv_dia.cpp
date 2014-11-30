#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"


#include "spmv_dia.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"


void spmv_dia_ocl(dia_matrix<int, int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devOffset;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devTexVec;
    cl_mem devTexVec4;
    cl_mem devRes;

    //Initialize values
    int aligned_length = mat->dia_length_aligned;
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int vecsize = mat->matinfo.width;
    int dianum = mat->dia_num;
    assert(dianum <= MAX_DIA_NUM);
    int* padOffset = (int*)malloc(sizeof(int)*(MAX_DIA_NUM));
    for (int i = 0; i < MAX_DIA_NUM; i++)
    {
	padOffset[i] = 0;
    }
    int minoffset = mat->matinfo.width;
    int maxoffset = -mat->matinfo.width;
    for (int i = 0; i < dianum; i++)
    {
	int curOffset = mat->dia_offsets[i];
	padOffset[i] = curOffset;
	if (curOffset < minoffset)
	    minoffset = curOffset;
	if (curOffset > maxoffset)
	    maxoffset = curOffset;
    }
    int padveclength = mat->matinfo.width;
    int leftoffset = (minoffset >= 0) ? 0 : (-minoffset);
    assert(leftoffset >= 0);
    padveclength += leftoffset;
    if (maxoffset > 0)
	padveclength += maxoffset;
    padveclength += 100;
    float* padvec = (float*)malloc(sizeof(float)*padveclength);
    for (int i = 0; i < padveclength; i++)
	padvec[i] = 0.0f;
    memcpy(padvec + leftoffset, vec, sizeof(float)*mat->matinfo.width);
    ALLOCATE_GPU_READ(devOffset, padOffset, sizeof(int)*(MAX_DIA_NUM));
    ALLOCATE_GPU_READ(devData, mat->dia_data, sizeof(float)*mat->dia_length_aligned*dianum);
    ALLOCATE_GPU_READ(devVec, padvec, sizeof(float)*padveclength);
    int paddedres = findPaddedSize(rownum, 4 * WORK_GROUP_SIZE);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
    const cl_image_format floatFormat =
    {
	CL_R,
	CL_FLOAT,
    };
    const cl_image_format float4Format =
    {
	CL_RGBA,
	CL_FLOAT,
    };
    int width = VEC2DWIDTH;
    int height = (vecsize + VEC2DWIDTH - 1)/VEC2DWIDTH;
    if (height % 4 != 0)
	height += (4 - (height % 4));
    float* image2dVec = (float*)malloc(sizeof(float)*width*height);
    memset(image2dVec, 0, sizeof(float)*width*height);
    for (int i = 0; i < vecsize; i++)
    {
	image2dVec[i] = vec[i];
    }
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height, 1};
    size_t vector4Size[] = {width, height/4, 1};
    devTexVec = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height, 0, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteImage(cmdQueue, devTexVec, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    devTexVec4 = clCreateImage2D(context, CL_MEM_READ_ONLY, &float4Format, width, height/4, 0, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteImage(cmdQueue, devTexVec4, CL_TRUE, origin, vector4Size, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;

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
	csrKernel = clCreateKernel(program, "gpu_dia", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &dianum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nDIA cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
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
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_dia_v4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &dianum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nDIA float4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
	
    }

    {
	int methodid = 100;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int gsize = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_dia_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &dianum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nDIA tx cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
	
    }
    
    {
	int methodid = 101;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int aligned4 = aligned_length / 4;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_dia_v4_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &dianum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devTexVec4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nDIA float4 tx cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}
	
    }


    //Clean up
    if (padOffset)
	free(padOffset);

    if (devOffset)
	clReleaseMemObject(devOffset);
    if (devData)
	clReleaseMemObject(devData);
    if (devTexVec)
	clReleaseMemObject(devTexVec);
    if (devTexVec4)
	clReleaseMemObject(devTexVec4);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devRes)
	clReleaseMemObject(devRes);


    freeObjects(devices, &context, &cmdQueue, &program);
}


void spmv_dia(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);
    dia_matrix<int, int, float> diamat;
    coo2dia<int, int, float>(mat, &diamat, GPU_ALIGNMENT);
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);

    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	spmv_dia_ocl(&diamat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);


	printf("\n------------------------------------------------------------------------\n");
	printf("DIA best time %f ms best method %d", opttime1*1000.0, optmethod1);
	printf("\n------------------------------------------------------------------------\n");
    }

    free_dia_matrix(diamat);
    free(vec);
    free(res);
    free(coores);
}

