#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"


#include "spmv_sell.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"

void spmv_sell_ocl(sell_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes)
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

    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = dim2Size;
    if (sliceheight == WARPSIZE)
    {
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((rownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	printf("gsize %d rownum %d slicenum %d sliceheight %d datasize %d nnz %d vecsize %d \n", gsize, rownum, slicenum, sliceheight, datasize, nnz, vecsize);
	//int warpnum = SELL_GROUP_SIZE / WARPSIZE;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_sell_warp", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devSlicePtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &slicenum); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernel, 6, sizeof(int) * warpnum * 2, NULL); CHECKERROR;

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
	printf("\nSELL cpu warp time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
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
	printf("\nSELL cpu group time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}

    }
    
   
    if (sliceheight == WARPSIZE)
    {
	int methodid = 100;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((rownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	//int warpnum = SELL_GROUP_SIZE / WARPSIZE;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_sell_warp_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devSlicePtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &slicenum); CHECKERROR;
	//errorCode = clSetKernelArg(csrKernel, 6, sizeof(int) * warpnum * 2, NULL); CHECKERROR;

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
	printf("\nSELL cpu warp tx time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	if (csrKernel)
	    clReleaseKernel(csrKernel);

	double onetime = time_in_sec / (double) ntimes;
	if (onetime < opttime)
	{
	    opttime = onetime;
	    optmethod = methodid;
	}

    }
    
    
    if (sliceheight == SELL_GROUP_SIZE)
    {
	int methodid = 101;
	cl_uint work_dim = 2;
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = slicenum*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_sell_group_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devSlicePtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &slicenum); CHECKERROR;

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
	printf("\nSELL cpu group tx time %lf ms GFLOPS %lf code %d \n\n",   time_in_sec / (double) ntimes * 1000, gflops, methodid);

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


void spmv_sell(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);

    int nnz = mat->matinfo.nnz;
    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	sell_matrix<int, float> sellmat;
	unsigned int sliceheight = WARPSIZE;
	coo2sell<int, float>(mat, &sellmat, sliceheight);
	int slicenum = sellmat.sell_slice_num;
	int totalnum = sellmat.sell_slice_ptr[slicenum];
	int maxwidth = 0;
	int minwidth = 100000;
	for (int i = 0; i < slicenum; i++)
	{
	    int size = sellmat.sell_slice_ptr[i + 1] - sellmat.sell_slice_ptr[i];
	    size /= sliceheight;
	    if (size > maxwidth)
		maxwidth = size;
	    if (size < minwidth)
		minwidth = size;
	}
	//sell_spmv(&sellmat, vec, res, mat->matinfo.width);
	//two_vec_compare(coores, res, mat->matinfo.height);
	spmv_sell_ocl(&sellmat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);


	double gflops = (double)nnz*2/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("SELL best time %f ms best method %d GFLOPS %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");

	free_sell_matrix(sellmat);
    }
    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	sell_matrix<int, float> sellmat;
	unsigned int sliceheight = SELL_GROUP_SIZE;
	coo2sell<int, float>(mat, &sellmat, sliceheight);
	int slicenum = sellmat.sell_slice_num;
	int totalnum = sellmat.sell_slice_ptr[slicenum];
	int maxwidth = 0;
	int minwidth = 100000;
	for (int i = 0; i < slicenum; i++)
	{
	    int size = sellmat.sell_slice_ptr[i + 1] - sellmat.sell_slice_ptr[i];
	    size /= sliceheight;
	    if (size > maxwidth)
		maxwidth = size;
	    if (size < minwidth)
		minwidth = size;
	}
	//sell_spmv(&sellmat, vec, res, mat->matinfo.width);
	//two_vec_compare(coores, res, mat->matinfo.height);
	spmv_sell_ocl(&sellmat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);


	double gflops = (double)nnz*2/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("SELL best time %f ms best method %d GFLOPS %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");

	free_sell_matrix(sellmat);
    }
    

    free(vec);
    free(res);
    free(coores);
}

