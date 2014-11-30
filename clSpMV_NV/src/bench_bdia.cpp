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


void spmv_bdia_ocl(bdia_matrix<int, int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devBandPtr;
    cl_mem devOffset;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRes;

    //Initialize values
    int aligned_length = mat->bdia_length_aligned;
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int vecsize = mat->matinfo.width;
    int bandnum = mat->bdia_band_num;
    assert(bandnum <= MAX_BAND_NUM);
    int* padBandPtr = (int*)malloc(sizeof(int)*(MAX_BAND_NUM + 1));
    int* padOffset = (int*)malloc(sizeof(int)*(MAX_BAND_NUM + 1));
    for (int i = 0; i < MAX_BAND_NUM + 1; i++)
    {
	padBandPtr[i] = padOffset[i] = 0;
    }
    int minoffset = mat->matinfo.width;
    int maxoffset = -mat->matinfo.width;
    for (int i = 0; i <= bandnum; i++)
    {
	padBandPtr[i] = mat->bdia_bptr[i];
    }
    for (int i = 0; i < bandnum; i++)
    {
	int curOffset = mat->bdia_offsets[i];
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
    padveclength += MAX_BAND_WIDTH;
    padveclength += 100;
    float* padvec = (float*)malloc(sizeof(float)*padveclength);
    for (int i = 0; i < padveclength; i++)
	padvec[i] = 0.0f;
    memcpy(padvec + leftoffset, vec, sizeof(float)*mat->matinfo.width);
    int dianum = mat->bdia_bptr[bandnum];
    ALLOCATE_GPU_READ(devBandPtr, padBandPtr, sizeof(int)*(MAX_BAND_NUM + 1));
    ALLOCATE_GPU_READ(devOffset, padOffset, sizeof(int)*(MAX_BAND_NUM + 1));
    ALLOCATE_GPU_READ(devData, mat->bdia_data, sizeof(float)*mat->bdia_length_aligned*dianum);
    ALLOCATE_GPU_READ(devVec, padvec, sizeof(float)*padveclength);
    int paddedres = findPaddedSize(rownum, 4 * WORK_GROUP_SIZE);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
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
	csrKernel = clCreateKernel(program, "gpu_bdia_nlvec", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devBandPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &bandnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &leftoffset); CHECKERROR;
	
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
	printf("\nBDIA simple cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    {
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int gsize = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_bdia", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devBandPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &bandnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nBDIA local vec cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    

    {
	int methodid = 2;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_bdia_g4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devBandPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &aligned_length); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &bandnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nBDIA group 4 local vec cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    {
	int methodid = 3;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int aligned4 = aligned_length / 4;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_bdia_v4_nlvec", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devBandPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &bandnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nBDIA float4 cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    
    {
	int methodid = 4;
	cl_uint work_dim = 2;
	size_t blocksize[] = {WORK_GROUP_SIZE, 1};
	int row4num = rownum / 4;
	if (rownum % 4 != 0)
	    row4num++;
	int aligned4 = aligned_length / 4;
	int gsize = ((row4num + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	
	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_bdia_v4", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devBandPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devOffset); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(int),    &aligned4); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(int),    &bandnum); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 7, sizeof(int),    &leftoffset); CHECKERROR;

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
	printf("\nBDIA float4 local vec cpu time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    if (padBandPtr)
	free(padBandPtr);
    if (padOffset)
	free(padOffset);

    if (devBandPtr)
	clReleaseMemObject(devBandPtr);
    if (devOffset)
	clReleaseMemObject(devOffset);
    if (devData)
	clReleaseMemObject(devData);
    if (devVec)
	clReleaseMemObject(devVec);
    if (devRes)
	clReleaseMemObject(devRes);


    freeObjects(devices, &context, &cmdQueue, &program);
}

void init_bdia_mat(bdia_matrix<int, int, float>& mat, unsigned int dimension, unsigned int bandwidth)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int nnz = dimension;
    for (unsigned int i = 2; i < bandwidth; i+=2)
    {
	nnz += 2 * (dimension - (i/2));
    }
    mat.matinfo.nnz = nnz;
    mat.bdia_band_num = 1;
    mat.bdia_length = dimension;
    mat.bdia_length_aligned = dimension;
    mat.bdia_offsets = (int*)malloc(sizeof(int)*1);
    mat.bdia_offsets[0] = -(bandwidth / 2);
    mat.bdia_bptr = (int*)malloc(sizeof(int)*2);
    mat.bdia_bptr[0] = 0;
    mat.bdia_bptr[1] = bandwidth;
    mat.bdia_data = (float*)malloc(sizeof(float)*bandwidth*dimension);
    for (unsigned int i = 0; i < bandwidth*dimension; i++)
	mat.bdia_data[i] = 1.0f;
}


void benchmark_bdia(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/bdia.ben");
    FILE* outfile = fopen(outname, "w");
    //unsigned int size = 1024*1024;
    //unsigned int bandwidth = 101;
    int methodnum = 5;
    double floptable[methodnum];
    for (unsigned int size = 1024; size <= 2*1024*1024; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int bandwidth = 1; bandwidth < 64; )
	{
	    bdia_matrix<int, int, float> bdiamat;
	    init_bdia_mat(bdiamat, size, bandwidth);

	    double opttime = 10000.0f;
	    double optflop = 0.0f;
	    int optmethod = 0;

	    spmv_bdia_ocl(&bdiamat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);

	    printf("\n------------------------------------------------------------------------\n");
	    printf("Banded DIA Dim %d BW %d opttime %f ms optflop %f optmethod %d", size, bandwidth, opttime*1000.0, optflop,  optmethod);
	    printf("\n------------------------------------------------------------------------\n");
	    fprintf(outfile, "%d %d", size, bandwidth);
	    for (unsigned int k = 0; k < methodnum; k++)
		fprintf(outfile, " %f", floptable[k]);
	    fprintf(outfile, "\n");

	    free_bdia_matrix(bdiamat);
	    if (bandwidth < 14)
		bandwidth += 2;
	    else 
		bandwidth += 4;
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bdia.cl");
    int ntimes = 1000;
    benchmark_bdia(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



