#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"


#include "spmv_bcsr.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"

void spmv_bcsr_ocl(b4csr_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes, int bw, int bh)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColid;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devRes;
    cl_mem devTexVec;

    //Initialize values
    int data_align = mat->b4csr_aligned_size;
    int nnz = mat->matinfo.nnz;
    int rownum = mat->matinfo.height;
    int blockrownum = mat->b4csr_row_num;
    int blocknum = mat->b4csr_block_num;
    int vecsize = mat->matinfo.width;
    int bwidth = mat->b4csr_bwidth;
    int bheight = mat->b4csr_bheight;
    int width4num = bwidth / 4;
    int padveclen = findPaddedSize(vecsize, 8);
    float* paddedvec = (float*)malloc(sizeof(float)*padveclen);
    memset(paddedvec, 0, sizeof(float)*padveclen);
    memcpy(paddedvec, vec, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ(devRowPtr, mat->b4csr_row_ptr, sizeof(int)*(blockrownum + 1));
    ALLOCATE_GPU_READ(devColid, mat->b4csr_col_id, sizeof(int)*blocknum);
    ALLOCATE_GPU_READ(devData, mat->b4csr_data, sizeof(float)*data_align*width4num*bheight);
    ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(float)*padveclen);
    int paddedres = findPaddedSize(rownum, 512);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;
    const cl_image_format floatFormat =
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
    size_t vectorSize[] = {width, height/4, 1};
    devTexVec = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height/4, 0, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteImage(cmdQueue, devTexVec, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);

    //printf("\nvec length %d padded length %d", mat->matinfo.width, padveclength);

    opttime = 10000.0f;
    optmethod = 0;
    int dim2 = dim2Size;
    {
	int methodid = 0;
	cl_uint work_dim = 2;
	size_t blocksize[] = {BCSR_VEC_GROUP_SIZE, 1};
	int gsize = blockrownum * BCSR_VEC_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bcsr_red_00";
	kernelname[13] += bh;
	kernelname[14] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &data_align4); CHECKERROR;

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
	printf("\nBCSR %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

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
	size_t blocksize[] = {BCSR_VEC_GROUP_SIZE, 1};
	int gsize = blockrownum * BCSR_VEC_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bcsr_red_00_tx";
	kernelname[13] += bh;
	kernelname[14] += bw;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &data_align4); CHECKERROR;

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
	printf("\nBCSR %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bh, bw,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

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

    if (devRowPtr)
	clReleaseMemObject(devRowPtr);
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


void spmv_bcsr(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);


    
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);
    double overallopttime = 10000.0f;
    int bestbw = 0;
    int bestbh = 0;
    int nnz = mat->matinfo.nnz;

    for (int bwidth = 4; bwidth < 9; bwidth += 4)
    for (int bheight = 1; bheight < 9; bheight*=2)
    {
	//int bwidth = 4;
	//int bheight = 8;
	b4csr_matrix<int, float> b4csrmat;
	if (coo2b4csr<int, float>(mat, &b4csrmat, bwidth, bheight, GPU_ALIGNMENT) == false)
	    continue;
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	spmv_bcsr_ocl(&b4csrmat, vec, res, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes, bwidth, bheight);

	if (opttime1 < overallopttime)
	{
	    overallopttime = opttime1;
	    bestbw = bwidth;
	    bestbh = bheight;
	}
	double gflops = (double)nnz*2/opttime1/(double)1e9;

	printf("BCSR info: block row num %d total block num %d \n", b4csrmat.b4csr_row_num, b4csrmat.b4csr_block_num);
	printf("\n------------------------------------------------------------------------\n");
	printf("BCSR best time %f ms best method %d GFLOPS %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");

	free_b4csr_matrix(b4csrmat);
    }
    printf("\n------------------------------------------------------------------------\n");
    printf("Final BCSR best time %f ms GFLOPS %f Block %dx%d", overallopttime*1000.0, (double)nnz*2/overallopttime/(double)1e9, bestbh, bestbw);
    printf("\n------------------------------------------------------------------------\n");
    

    free(vec);
    free(res);
    free(coores);
}

