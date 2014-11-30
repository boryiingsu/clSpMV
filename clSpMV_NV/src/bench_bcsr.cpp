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

void spmv_bcsr_ocl(b4csr_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable)
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
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};
	int gsize = blockrownum * CSR_VEC_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	int data_align4 = data_align / 4;
	char kernelname[100] = "gpu_bcsr_red_00";
	kernelname[13] += bheight;
	kernelname[14] += bwidth;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtr); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColid); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int),    &data_align4); CHECKERROR;

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
	printf("\nBCSR %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bheight, bwidth,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

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


void init_bcsr_mat(b4csr_matrix<int, float>& mat, unsigned int bheight, unsigned int bwidth, unsigned int dimension, unsigned int blocknum)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int brownum = dimension / bheight;
    unsigned int nnz = brownum * bwidth * bheight * blocknum;
    mat.matinfo.nnz = nnz;
    mat.b4csr_bwidth = bwidth;
    mat.b4csr_bheight = bheight;
    mat.b4csr_row_num = brownum;
    mat.b4csr_aligned_size = 4 * blocknum * brownum;
    mat.b4csr_block_num = blocknum * brownum;
    mat.b4csr_row_ptr = (int*)malloc(sizeof(int)*(brownum+1));
    mat.b4csr_col_id = (int*)malloc(sizeof(int)*blocknum*brownum);
    mat.b4csr_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i <= brownum; i++)
	mat.b4csr_row_ptr[i] = i * blocknum;
    for (unsigned int i = 0; i < nnz; i++)
	mat.b4csr_data[i] = 1.0f;
    unsigned int index = 0;
    for (unsigned int i = 0; i < brownum; i++)
    {
	int start = i - blocknum / 2;
	if (start < 0)
	    start = 0;
	int end = start + blocknum;
	if (end > dimension / bwidth)
	{
	    end = dimension / bwidth;
	    start = end - blocknum;
	}
	for (int j = start; j < end; j++)
	{
	    mat.b4csr_col_id[index] = j;
	    index++;
	}
    }
    for (unsigned int i = 0; i < brownum * blocknum; i++)
	assert(mat.b4csr_col_id[i] >= 0 && mat.b4csr_col_id[i] < (dimension / bwidth));
    assert(index == blocknum * brownum);

}

void benchmark_bcsr(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/bcsr.ben");
    FILE* outfile = fopen(outname, "w");
    int methodnum = 1;
    double floptable[methodnum];
    for (unsigned int bw = 4; bw < 9; bw += 4)
    for (unsigned int bh = 1; bh < 9; bh *= 2)
    for (unsigned int size = 1024; size <= 262144; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int blocknum = 2; blocknum <= 512; blocknum *= 2)
	{
	    if (size*blocknum*bw > 209715200)
		break;
	    if (blocknum * bw >= size)
		break;
	    b4csr_matrix<int, float> bcsrmat;
	    init_bcsr_mat(bcsrmat, bh, bw, size, blocknum);

	    double opttime = 10000.0f;
	    double optflop = 0.0f;
	    int optmethod = 0;

	    spmv_bcsr_ocl(&bcsrmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);

	    printf("\n------------------------------------------------------------------------\n");
	    printf("Blocked ELL Dim %d BN %d opttime %f ms optflop %f optmethod %d", size, blocknum, opttime*1000.0, optflop,  optmethod);
	    printf("\n------------------------------------------------------------------------\n");
	    fprintf(outfile, "%d %d %d %d", bh, bw, size, blocknum);
	    for (unsigned int k = 0; k < methodnum; k++)
		fprintf(outfile, " %f", floptable[k]);
	    fprintf(outfile, "\n");

	    free_b4csr_matrix(bcsrmat);
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_bcsr.cl");
    int ntimes = 1000;
    benchmark_bcsr(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



