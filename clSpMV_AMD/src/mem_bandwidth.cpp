#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "CL/cl.h"

#include "mem_bandwidth.h"
#include "util.h"
#include "oclcommon.h"
#include "constant.h"

void bandwidth_test(char* oclfilename, cl_device_type deviceType, int dim2Size)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int  errorCode = CL_SUCCESS;


    //Measuring overhead
    int dim2 = dim2Size;
    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	for (int blockx = 64; blockx <= 256; blockx *= 2)
	{
	    cl_mem devA;
	    cl_mem devB;

	    //int vecsize = 1024*1024; 

	    float* hostA = (float*)malloc(sizeof(float)*vecsize);
	    float* hostB = (float*)malloc(sizeof(float)*vecsize);
	    for (int i = 0; i < vecsize; i++)
	    {
		hostA[i] = 3.0f;
		hostB[i] = 2.0f;
	    }
	    devA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
	    devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	    errorCode = clEnqueueWriteBuffer(cmdQueue, devA, CL_TRUE, 0, sizeof(float)*vecsize, hostA, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	    clFinish(cmdQueue);
	    int ntimes = 100;
	    cl_uint work_dim = 2;

	    cl_kernel memcpyKernel = NULL;
	    memcpyKernel = clCreateKernel(program, "nothing", &errorCode); CHECKERROR;
	    errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	    errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devA); CHECKERROR;

	    //int blockx = 128;
	    size_t blocksize[] = {blockx, 1};
	    size_t globalsize[] = {vecsize, dim2};

	    double teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
		clFinish(cmdQueue);
	    }
	    double testend = timestamp();
	    double time_in_sec = (testend - teststart)/(double)dim2;

	    printf("\nEnqueue and Sync Overhead: iter %d work item per group %d number of work group %d time %f ms", ntimes, blockx, vecsize/blockx, time_in_sec*1000.0);

	    teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    }
	    clFinish(cmdQueue);
	    testend = timestamp();
	    time_in_sec = (testend - teststart)/(double)dim2;

	    printf("\nEnqueue and one Sync Overhead: iter %d work item per group %d number of work group %d time %f ms", ntimes, blockx, vecsize/blockx, time_in_sec*1000.0);

	    teststart = timestamp();
	    for (int i = 0; i < ntimes; i++)
	    {
		errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	    }
	    testend = timestamp();
	    time_in_sec = (testend - teststart)/(double)dim2;
	    clFinish(cmdQueue);

	    printf("\nEnqueue Overhead Only: iter %d work item per group %d number of work group %d time %f ms", ntimes, blockx, vecsize/blockx, time_in_sec*1000.0);


	    if (devA)
		clReleaseMemObject(devA);
	    if (devB)
		clReleaseMemObject(devB);
	    if (memcpyKernel)
		clReleaseKernel(memcpyKernel);

	    free(hostA);
	    free(hostB);
	}
    }

    //DRAM Bandwidth
    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devA;
	cl_mem devB;

	//int vecsize = 16*1024*1024/sizeof(float); //16MB

	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f;
	    hostB[i] = 2.0f;
	}
	devA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devA, CL_TRUE, 0, sizeof(float)*vecsize, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize/16, dim2};

	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);




	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy float4 Vector Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devA)
	    clReleaseMemObject(devA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);
    }

    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devA;
	cl_mem devB;


	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f;
	    hostB[i] = 2.0f;
	}
	devA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devA, CL_TRUE, 0, sizeof(float)*vecsize, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy_2", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize/16, dim2};


	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);



	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy float2 Vector Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devA)
	    clReleaseMemObject(devA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);
    }


    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devA;
	cl_mem devB;

	//int vecsize = 512*1024/sizeof(float); //512kB

	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f;
	    hostB[i] = 2.0f;
	}
	devA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteBuffer(cmdQueue, devA, CL_TRUE, 0, sizeof(float)*vecsize, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy_gather", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize/16, dim2};

	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);

	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy float Vector Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devA)
	    clReleaseMemObject(devA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);
    }


    //Texture RGBA float4
    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devTexA;
	cl_mem devB;

	const cl_image_format floatFormat =
	{
	    CL_RGBA,
	    CL_FLOAT,
	};


	int vec2dwidth = VEC2DWIDTH;
	size_t origin[] = {0, 0, 0};
	size_t vectorSize[] = {vec2dwidth, vecsize/(vec2dwidth*4), 1};

	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f + (float)(i % 100);
	    hostB[i] = 2.0f;
	}
	devTexA = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, vec2dwidth, vecsize/(vec2dwidth*4), 0, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteImage(cmdQueue, devTexA, CL_TRUE, origin, vectorSize, 0, 0, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy_tex_RGBA4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devTexA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize / 4, dim2};



	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);


	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy Texture RGBA float4 Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devTexA)
	    clReleaseMemObject(devTexA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);
    }

    //Texture RGBA, only one out of the four components is used
    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 32 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devTexA;
	cl_mem devB;

	const cl_image_format floatFormat =
	{
	    CL_RGBA,
	    CL_FLOAT,
	};

	int vec2dwidth = VEC2DWIDTH;
	size_t origin[] = {0, 0, 0};
	size_t vectorSize[] = {vec2dwidth, vecsize/(vec2dwidth*4), 1};

	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f + (float)(i % 100);
	    hostB[i] = 2.0f;
	}
	devTexA = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, vec2dwidth, vecsize/(vec2dwidth*4), 0, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteImage(cmdQueue, devTexA, CL_TRUE, origin, vectorSize, 0, 0, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy_tex_RGBA", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devTexA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize, dim2};



	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);


	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy Texture RGBA float Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devTexA)
	    clReleaseMemObject(devTexA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);
    }

    //Gather Texture R, the system will pad the remaining three slots by 0
    for (int vecsize = 32 * 1024 / sizeof(float); vecsize <= 8 * 1024 * 1024 /sizeof(float); vecsize *= 2)
    {
	cl_mem devTexA;
	cl_mem devB;

	const cl_image_format floatFormat =
	{
	    CL_R,
	    CL_FLOAT,
	};


	int vec2dwidth = VEC2DWIDTH;
	size_t origin[] = {0, 0, 0};
	size_t vectorSize[] = {vec2dwidth, vecsize/vec2dwidth, 1};

	float* hostA = (float*)malloc(sizeof(float)*vecsize);
	float* hostB = (float*)malloc(sizeof(float)*vecsize);
	for (int i = 0; i < vecsize; i++)
	{
	    hostA[i] = 3.0f + (float)(i % 100);
	    hostB[i] = 2.0f;
	}

	devTexA = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, vec2dwidth, vecsize/vec2dwidth, 0, NULL, &errorCode); CHECKERROR;
	devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*vecsize, NULL, &errorCode); CHECKERROR;

	errorCode = clEnqueueWriteImage(cmdQueue, devTexA, CL_TRUE, origin, vectorSize, 0, 0, hostA, 0, NULL, NULL); CHECKERROR;
	errorCode = clEnqueueWriteBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	int ntimes = 20;
	cl_uint work_dim = 2;

	cl_kernel memcpyKernel = NULL;
	memcpyKernel = clCreateKernel(program, "gpu_memcpy_tex_R", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devTexA); CHECKERROR;

	size_t blocksize[] = {256, 1};
	size_t globalsize[] = {vecsize, dim2};


	for (int i = 0; i < 3; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);



	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, memcpyKernel, work_dim, NULL, globalsize, NULL, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy Texture R float Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

	errorCode = clEnqueueReadBuffer(cmdQueue, devB, CL_TRUE, 0, sizeof(float)*vecsize, hostB, 0, NULL, NULL); CHECKERROR;
	clFinish(cmdQueue);
	for (int i = 0; i < vecsize; i++)
	{
	    if (hostA[i] != hostB[i])
	    {
		printf("Wrong %d %f %f\n", i, hostA[i], hostB[i]);
		break;
	    }
	}

	if (devTexA)
	    clReleaseMemObject(devTexA);
	if (devB)
	    clReleaseMemObject(devB);
	if (memcpyKernel)
	    clReleaseKernel(memcpyKernel);

	free(hostA);
	free(hostB);

    }

    freeObjects(devices, &context, &cmdQueue, &program);

}

