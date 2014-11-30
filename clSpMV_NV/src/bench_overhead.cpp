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

void benchmark_overhead(char* clspmvpath, char* oclfilename, cl_device_type deviceType, int dim2Size, int ntimes)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int  errorCode = CL_SUCCESS;


    //Measuring overhead
    int dim2 = dim2Size;
    
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
	//END_TIME;
	double testend = timestamp();
	double time_in_sec = (testend - teststart)/(double)dim2;
	double band = (double)vecsize*sizeof(float)*2/(time_in_sec/ntimes)/(double)1e9;
	printf("\nMemcpy Gather4 Vector Size %d Bytes time %lf ms GBytes/s %lf \n\n",  vecsize*sizeof(float), time_in_sec / (double) ntimes * 1000, band);

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

    cl_uint dev_exec_num;
    size_t devicesSize = 0;
    errorCode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize ); CHECKERROR;
    devices = new cl_device_id[devicesSize / sizeof(cl_device_id)]; CHECKERROR;
    errorCode = clGetContextInfo(context, CL_CONTEXT_DEVICES, devicesSize, devices, NULL ); CHECKERROR;
    errorCode = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( dev_exec_num ), &dev_exec_num, NULL ); CHECKERROR;
    printf("\nCompute units %d\n", dev_exec_num);

    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/overhead.ben");
    FILE* outfile = fopen(outname, "w");
    for (unsigned int groupx = dev_exec_num; groupx <= 131702; groupx *= 2)
    {
	for (unsigned int blockx = 64; blockx <= 256; blockx *= 2)
	{
	    
	    cl_mem devA;
	    cl_mem devB;
	    
	    devA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*1024, NULL, &errorCode); CHECKERROR;
	    devB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*1024, NULL, &errorCode); CHECKERROR;

	    clFinish(cmdQueue);
	    cl_uint work_dim = 2;

	    cl_kernel memcpyKernel = NULL;
	    memcpyKernel = clCreateKernel(program, "nothing", &errorCode); CHECKERROR;
	    errorCode = clSetKernelArg(memcpyKernel, 0, sizeof(cl_mem), &devB); CHECKERROR;
	    errorCode = clSetKernelArg(memcpyKernel, 1, sizeof(cl_mem), &devA); CHECKERROR;

	    size_t blocksize[] = {blockx, 1};
	    size_t globalsize[] = {blockx * groupx, dim2};

	    for (int i = 0; i < 10; i++)
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

	    printf("\nEnqueue and one Sync Overhead: iter %d work item per group %d number of work group %d time %f ms\n", ntimes, blockx, groupx, time_in_sec*1000.0);


	    if (devA)
		clReleaseMemObject(devA);
	    if (devB)
		clReleaseMemObject(devB);
	    if (memcpyKernel)
		clReleaseKernel(memcpyKernel);


	    fprintf(outfile, "%d %d %f\n", groupx, blockx, time_in_sec*1000.0);
	    
	}
    }
    fclose(outfile);

    //Clean up
    freeObjects(devices, &context, &cmdQueue, &program);

}

int main(int argv, char** argc)
{
    char* clspmvpath = getenv("CLSPMVPATH");
    char clfilename[1000];
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/mem_bandwidth.cl");
    int ntimes = 1000;
    benchmark_overhead(clspmvpath, clfilename, CL_DEVICE_TYPE_GPU, 1, ntimes);
    return 0;
}

