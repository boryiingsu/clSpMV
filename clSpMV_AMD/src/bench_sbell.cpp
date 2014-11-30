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


void spmv_sbell_ocl(sbell_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable)
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
    int blockrownum = mat->sbell_row_num;
    int vecsize = mat->matinfo.width;
    int slicenum = mat->sbell_slice_num;
    int sliceheight = mat->sbell_slice_height;
    int bwidth = mat->sbell_bwidth;
    int bheight = mat->sbell_bheight;
    int width4num = bwidth / 4;
    int padveclen = findPaddedSize(vecsize, 8);
    int totalsize = mat->sbell_slice_ptr[slicenum];
    float* paddedvec = (float*)malloc(sizeof(float)*padveclen);
    memset(paddedvec, 0, sizeof(float)*padveclen);
    memcpy(paddedvec, vec, sizeof(float)*vecsize);
    ALLOCATE_GPU_READ(devSlicePtr, mat->sbell_slice_ptr, sizeof(int)*(slicenum + 1));
    ALLOCATE_GPU_READ(devColid, mat->sbell_col_id, sizeof(int)*totalsize);
    ALLOCATE_GPU_READ(devData, mat->sbell_data, sizeof(float)*totalsize*bwidth*bheight);
    ALLOCATE_GPU_READ(devVec, paddedvec, sizeof(float)*padveclen);
    int paddedres = findPaddedSize(rownum, SELL_GROUP_SIZE * bheight);
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
	size_t blocksize[] = {SELL_GROUP_SIZE, 1};
	int gsize = ((blockrownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	size_t globalsize[] = {gsize, dim2};
	char kernelname[100] = "gpu_sbell00";
	kernelname[9] += bheight;
	kernelname[10] += bwidth;

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
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
	printf("\nSBELL %dx%d block cpu time %lf ms GFLOPS %lf code %d \n\n", bheight, bwidth,  time_in_sec / (double) ntimes * 1000, gflops, methodid);

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

void init_sbell_mat(sbell_matrix<int, float>& mat, unsigned int bheight, unsigned int bwidth, unsigned int dimension, unsigned int blocknum, unsigned int sliceheight)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int brownum = dimension / bheight;
    unsigned int nnz = brownum * bwidth * bheight * blocknum;
    unsigned int slicenum = brownum / sliceheight;
    mat.matinfo.nnz = nnz;
    mat.sbell_bwidth = bwidth;
    mat.sbell_bheight = bheight;
    mat.sbell_slice_height = sliceheight;
    mat.sbell_slice_num = slicenum;
    mat.sbell_row_num = brownum;
    mat.sbell_slice_ptr = (int*)malloc(sizeof(int)*(slicenum+1));
    mat.sbell_col_id = (int*)malloc(sizeof(int)*blocknum*brownum);
    mat.sbell_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i <= slicenum; i++)
	mat.sbell_slice_ptr[i] = i * sliceheight * blocknum;
    for (unsigned int i = 0; i < nnz; i++)
	mat.sbell_data[i] = 1.0f;
    unsigned int index = 0;
    unsigned int slicesize = sliceheight * blocknum;
    for (unsigned int s = 0; s < slicenum; s++)
    {
	for (unsigned int h = 0; h < sliceheight; h++)
	{
	    int rowid = s * sliceheight + h;
	    int start = rowid - blocknum / 2;
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
		mat.sbell_col_id[s * slicesize + (j - start) * sliceheight + h] = j;
		index++;
	    }
	}
    }
    for (unsigned int i = 0; i < brownum * blocknum; i++)
	assert(mat.sbell_col_id[i] >= 0 && mat.sbell_col_id[i] < (dimension / bwidth));
    assert(index == blocknum * brownum);

}


void benchmark_sbell(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/sbell.ben");
    FILE* outfile = fopen(outname, "w");
    int methodnum = 1;
    double floptable[methodnum];
    for (unsigned int bw = 4; bw < 9; bw += 4)
    for (unsigned int bh = 1; bh < 9; bh *= 2)
    for (unsigned int size = 1024; size <= 2*1024*1024; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int blocknum = 1; blocknum < 32; )
	{
	    if (size*blocknum*bw > 67108864)
		break;
	    sbell_matrix<int, float> bellmat;
	    init_sbell_mat(bellmat, bh, bw, size, blocknum, WARPSIZE);

	    double opttime = 10000.0f;
	    double optflop = 0.0f;
	    int optmethod = 0;

	    spmv_sbell_ocl(&bellmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable);

	    printf("\n------------------------------------------------------------------------\n");
	    printf("Blocked ELL Dim %d BN %d opttime %f ms optflop %f optmethod %d", size, blocknum, opttime*1000.0, optflop,  optmethod);
	    printf("\n------------------------------------------------------------------------\n");
	    fprintf(outfile, "%d %d %d %d", bh, bw, size, blocknum);
	    for (unsigned int k = 0; k < methodnum; k++)
		fprintf(outfile, " %f", floptable[k]);
	    fprintf(outfile, "\n");

	    free_sbell_matrix(bellmat);
	    if (blocknum < 14)
		blocknum += 2;
	    else 
		blocknum += 4;
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_sbell.cl");
    int ntimes = 1000;
    benchmark_sbell(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



