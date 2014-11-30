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

void spmv_coo_ocl(coo_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable, int maxwarpnum)
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
    int num_units = nnz / WARPSIZE;
    if (num_units % nnz != 0)
	num_units++;
    int num_warps = (num_units < maxwarpnum) ? num_units : maxwarpnum;
    int warp_per_group = COO_GROUP_SIZE / WARPSIZE;
    int group_num = num_warps / warp_per_group;
    if (num_warps % warp_per_group != 0)
	group_num++;
    int num_iters = num_units / num_warps;
    if (num_units % num_warps != 0)
	num_iters++;
    int process_size = num_iters * WARPSIZE;
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
    devTmpRow = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*maxwarpnum, NULL, &errorCode); CHECKERROR;
    devTmpData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*maxwarpnum, NULL, &errorCode); CHECKERROR;

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

	printf("process size %d nnz %d gsize %d active_warp %d\n", process_size, paddedNNZ, gsize, active_warp);

	size_t blocksize2[] = {COO_GROUP_SIZE * 2, 1};
	size_t globalsize2[] = {COO_GROUP_SIZE * 2, dim2};


	cl_kernel csrKernel2 = NULL;
	csrKernel2 = clCreateKernel(program, "gpu_coo_s2", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 0, sizeof(cl_mem), &devTmpRow); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 1, sizeof(cl_mem), &devTmpData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 2, sizeof(int), &active_warp); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel2, 3, sizeof(cl_mem), &devRes); CHECKERROR;

	for (int k = 0; k < 3; k++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
	}
	clFinish(cmdQueue);


	//int* tmpRow = (int*)malloc(sizeof(int)*maxwarpnum);
	//float* tmpData = (float*)malloc(sizeof(float)*maxwarpnum);


	double teststart = timestamp();
	for (int i = 0; i < ntimes; i++)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, csrKernel, work_dim, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
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
	floptable[methodid] = gflops;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	    optflop = gflops;
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

void init_coo_mat(coo_matrix<int, float>& mat, unsigned int dimension, unsigned int coonum)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int nnz = dimension * coonum;
    mat.matinfo.nnz = nnz;
    mat.coo_row_id = (int*)malloc(sizeof(int)*nnz);
    mat.coo_col_id = (int*)malloc(sizeof(int)*nnz);
    mat.coo_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i < nnz; i++)
	mat.coo_data[i] = 1.0f;
    for (unsigned int i = 0; i < dimension; i++)
	for (unsigned int j = 0; j < coonum; j++)
	    mat.coo_row_id[i * coonum + j] = i;
    for (unsigned int rowid = 0; rowid < dimension; rowid++)
    {
	int start = rowid - coonum / 2;
	if (start < 0)
	    start = 0;
	int end = start + coonum;
	if (end > dimension)
	{
	    end = dimension;
	    start = end - coonum;
	}
	for (int j = start; j < end; j++)
	{
	    mat.coo_col_id[(j - start) + rowid * coonum] = j;
	}
    }
    for (unsigned int i = 0; i < nnz; i++)
	assert(mat.coo_col_id[i] >= 0 && mat.coo_col_id[i] < dimension);
}


void benchmark_coo(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/coo.ben");
    FILE* outfile = fopen(outname, "w");
    int methodnum = 1;
    double floptable[methodnum];

    //Get device info
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;


    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;
    //Assuming GPU is at devices[0]

    cl_uint dev_exec_num;

    size_t devicesSize = 0;
    errorCode = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &devicesSize ); CHECKERROR;
    devices = new cl_device_id[devicesSize / sizeof(cl_device_id)]; CHECKERROR;
    errorCode = clGetContextInfo(context, CL_CONTEXT_DEVICES, devicesSize, devices, NULL ); CHECKERROR;
    errorCode = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( dev_exec_num ), &dev_exec_num, NULL ); CHECKERROR;

    freeObjects(devices, &context, &cmdQueue, &program);
    printf("\nCompute units %d\n", dev_exec_num);
    unsigned int warp_per_group = COO_GROUP_SIZE/WARPSIZE;
    unsigned int max_group_num = dev_exec_num*MAX_WARP_PER_PROC/warp_per_group;

    for (unsigned int size = 1024; size <= 2*1024*1024; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int coonum = 1; coonum <= 128; coonum *= 2)
	{
	    if (size*coonum > 69905067)
		break;
	    if (coonum > size)
		break;
	    coo_matrix<int, float> coomat;
	    init_coo_mat(coomat, size, coonum);

	    for (unsigned int groupnum = dev_exec_num; groupnum <= max_group_num; groupnum += dev_exec_num)
	    {

		double opttime = 10000.0f;
		double optflop = 0.0f;
		int optmethod = 0;

		spmv_coo_ocl(&coomat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable, groupnum*(COO_GROUP_SIZE/WARPSIZE));

		printf("\n------------------------------------------------------------------------\n");
		printf("COO Dim %d BN %d GN %d opttime %f ms optflop %f optmethod %d", size, coonum, groupnum, opttime*1000.0, optflop,  optmethod);
		printf("\n------------------------------------------------------------------------\n");
		fprintf(outfile, "%d %d %d", size, coonum, groupnum);
		for (unsigned int k = 0; k < methodnum; k++)
		    fprintf(outfile, " %f", floptable[k]);
		fprintf(outfile, "\n");
	    }
	    
	    free_coo_matrix(coomat);
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_coo.cl");
    int ntimes = 1000;
    benchmark_coo(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



