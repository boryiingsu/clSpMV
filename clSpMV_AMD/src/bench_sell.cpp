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

void spmv_sell_ocl(sell_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devSlicePtr;
    cl_mem devColid;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRes;
    cl_mem devTexVec;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int vecsize = mat->matinfo.width;
    int sliceheight = mat->sell_slice_height;
    int slicenum = mat->sell_slice_num;
    int datasize = mat->sell_slice_ptr[slicenum];
    ALLOCATE_GPU_READ(devSlicePtr, mat->sell_slice_ptr, sizeof(int)*(slicenum + 1));
    ALLOCATE_GPU_READ(devColid, mat->sell_col_id, sizeof(int)*datasize);
    ALLOCATE_GPU_READ(devData, mat->sell_data, sizeof(float)*datasize);
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

    int dim2 = dim2Size;
    if (sliceheight == WARPSIZE)
    {
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((rownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	//printf("gsize %d rownum %d slicenum %d sliceheight %d datasize %d nnz %d vecsize %d \n", gsize, rownum, slicenum, sliceheight, datasize, nnz, vecsize);
	//int warpnum = SELL_GROUP_SIZE / WARPSIZE;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_sell_warp", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devSlicePtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &slicenum); CHECKERROR;

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
	printf("\nSELL cpu warp time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
    
    if (sliceheight == SELL_GROUP_SIZE)
    {
	int methodid = 1;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = slicenum * SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_sell_group", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devSlicePtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &slicenum); CHECKERROR;

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
	printf("\nSELL cpu group time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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

    if (devSlicePtr)
	clReleaseMemObject(devSlicePtr);
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


void init_sell_mat(sell_matrix<int, float>& mat, unsigned int dimension, unsigned int ellnum, unsigned int sliceheight)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int nnz = dimension * ellnum;
    mat.matinfo.nnz = nnz;
    mat.sell_slice_height = sliceheight;
    unsigned int slicenum = dimension / sliceheight;
    mat.sell_slice_num = slicenum;
    mat.sell_slice_ptr = (int*)malloc(sizeof(int)*(slicenum+1));
    mat.sell_col_id = (int*)malloc(sizeof(int)*nnz);
    mat.sell_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i <= slicenum; i++)
	mat.sell_slice_ptr[i] = i * sliceheight * ellnum;
    for (unsigned int i = 0; i < nnz; i++)
	mat.sell_data[i] = 1.0f;
    unsigned int slicesize = sliceheight * ellnum;
    for (unsigned int s = 0; s < slicenum; s++)
    {
	for (unsigned int h = 0; h < sliceheight; h++)
	{
	    int rowid = s * sliceheight + h;
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
		mat.sell_col_id[s * slicesize + (j - start) * sliceheight + h] = j;
	    }
	}
    }
    for (unsigned int i = 0; i < nnz; i++)
	assert(mat.sell_col_id[i] >= 0 && mat.sell_col_id[i] < dimension);
}

void benchmark_sell(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/sell.ben");
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
	    sell_matrix<int, float> bellmat;
	    init_sell_mat(bellmat, size, ellnum, WARPSIZE);

	    double opttime = 10000.0f;
	    double optflop = 0.0f;
	    int optmethod = 0;

	    spmv_sell_ocl(&bellmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);
	    free_sell_matrix(bellmat);
	    
	    init_sell_mat(bellmat, size, ellnum, SELL_GROUP_SIZE);
	    spmv_sell_ocl(&bellmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);
	    
	    printf("\n------------------------------------------------------------------------\n");
	    printf("Sliced ELL Dim %d BN %d opttime %f ms optflop %f optmethod %d", size, ellnum, opttime*1000.0, optflop,  optmethod);
	    printf("\n------------------------------------------------------------------------\n");
	    fprintf(outfile, "%d %d", size, ellnum);
	    for (unsigned int k = 0; k < methodnum; k++)
		fprintf(outfile, " %f", floptable[k]);
	    fprintf(outfile, "\n");
	    
	    free_sell_matrix(bellmat);
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sell.cl");
    int ntimes = 1000;
    benchmark_sell(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



