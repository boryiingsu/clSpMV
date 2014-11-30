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


void spmv_csr_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int dim2Size, double& opttime, double& optflop, int& optmethod, char* oclfilename, cl_device_type deviceType, int ntimes, double* floptable, int groupnum)
{
    cl_device_id* devices = NULL;
    cl_context context = NULL;
    cl_command_queue cmdQueue = NULL;
    cl_program program = NULL;

    assert(initialization(deviceType, devices, &context, &cmdQueue, &program, oclfilename) == 1);

    cl_int errorCode = CL_SUCCESS;

    //Create device memory objects
    cl_mem devRowPtr;
    cl_mem devColId;
    cl_mem devData;
    cl_mem devVec;
    cl_mem devTexVec;
    cl_mem devRes;

    //Initialize values
    int nnz = mat->matinfo.nnz;
    int vecsize = mat->matinfo.width;
    int rownum = mat->matinfo.height;
    int rowptrsize = rownum + 1;
    ALLOCATE_GPU_READ(devRowPtr, mat->csr_row_ptr, sizeof(int)*rowptrsize);
    ALLOCATE_GPU_READ(devColId, mat->csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(devData, mat->csr_data, sizeof(float)*nnz);
    ALLOCATE_GPU_READ(devVec, vec, sizeof(float)*vecsize);
    int paddedres = findPaddedSize(rownum, 16);
    devRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*paddedres, NULL, &errorCode); CHECKERROR;
    //errorCode = clEnqueueWriteBuffer(cmdQueue, devRes, CL_TRUE, 0, sizeof(float)*rownum, result, 0, NULL, NULL); CHECKERROR;

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
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, CSR_VEC_GROUP_SIZE/WARPSIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));
	clFinish(cmdQueue);

	printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;


	{
	    size_t globalsize[] = {groupnum * CSR_VEC_GROUP_SIZE, dim2};

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
	    printf("\nCSR vector SLM row ptr padded mat strided rows fixed size:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum * CSR_VEC_GROUP_SIZE,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    floptable[methodid] = gflops;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
		optflop = gflops;
	    }
	}

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


    }

    
    
    

    {
	int methodid = 1;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, CSR_VEC_GROUP_SIZE/WARPSIZE);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));
	clFinish(cmdQueue);

	printf("\nRow Num %d padded size %d\n", rownum, padrowsize);
	cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_reduction_fs", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;


	{
	    size_t globalsize[] = {groupnum * CSR_VEC_GROUP_SIZE, dim2};

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
	    printf("\nCSR vector SLM row ptr padded mat strided rows fixed size:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum * CSR_VEC_GROUP_SIZE,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    floptable[methodid] = gflops;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
		optflop = gflops;
	    }
	}

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


    }
    

    //Clean up
    if (image2dVec)
	free(image2dVec);

    if (devRowPtr)
	clReleaseMemObject(devRowPtr);
    if (devColId)
	clReleaseMemObject(devColId);
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


void init_csr_mat(csr_matrix<int, float>& mat, unsigned int dimension, unsigned int csrnum)
{
    mat.matinfo.height = dimension;
    mat.matinfo.width = dimension;
    unsigned int nnz = dimension * csrnum;
    mat.matinfo.nnz = nnz;
    mat.csr_row_ptr = (int*)malloc(sizeof(int)*(dimension + 1));
    mat.csr_col_id = (int*)malloc(sizeof(int)*nnz);
    mat.csr_data = (float*)malloc(sizeof(float)*nnz);
    for (unsigned int i = 0; i < nnz; i++)
	mat.csr_data[i] = 1.0f;
    for (unsigned int i = 0; i <= dimension; i++)
	mat.csr_row_ptr[i] = i * csrnum;
    for (unsigned int rowid = 0; rowid < dimension; rowid++)
    {
	int start = rowid - csrnum / 2;
	if (start < 0)
	    start = 0;
	int end = start + csrnum;
	if (end > dimension)
	{
	    end = dimension;
	    start = end - csrnum;
	}
	for (int j = start; j < end; j++)
	{
	    mat.csr_col_id[(j - start) + rowid * csrnum] = j;
	}
    }
    for (unsigned int i = 0; i < nnz; i++)
	assert(mat.csr_col_id[i] >= 0 && mat.csr_col_id[i] < dimension);
}


void benchmark_csr(char* clspmvpath, char* oclfilename, int ntimes, cl_device_type deviceType)
{
    char outname[1000];
    sprintf(outname, "%s%s", clspmvpath, "/benchmark/csr.ben");
    FILE* outfile = fopen(outname, "w");
    int methodnum = 2;
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
    unsigned int warp_per_group = CSR_VEC_GROUP_SIZE/WARPSIZE;
    unsigned int max_group_num = dev_exec_num*MAX_WARP_PER_PROC/warp_per_group;

    for (unsigned int size = 1024; size <= 262144; size*=2)
    {
	float* vec = (float*)malloc(sizeof(float)*size);
	float* res = (float*)malloc(sizeof(float)*size);
	initVectorOne<int, float>(vec, size);	
	initVectorZero<int, float>(res, size);

	for (unsigned int csrnum = 2; csrnum <= 2048; csrnum *= 4)
	//for (unsigned int csrnum = 128; csrnum <= 2048; csrnum *= 4)
	{
	    if (size*csrnum > 104857600)
		break;
	    if (csrnum > size)
		break;
	    csr_matrix<int, float> csrmat;
	    init_csr_mat(csrmat, size, csrnum);

	    for (unsigned int groupnum = dev_exec_num; groupnum <= max_group_num; groupnum += dev_exec_num)
	    {

		double opttime = 10000.0f;
		double optflop = 0.0f;
		int optmethod = 0;

		spmv_csr_ocl(&csrmat, vec, res, 1, opttime, optflop, optmethod, oclfilename, deviceType, ntimes, floptable, groupnum);

		printf("\n------------------------------------------------------------------------\n");
		printf("CSR Dim %d BN %d GN %d opttime %f ms optflop %f optmethod %d", size, csrnum, groupnum, opttime*1000.0, optflop,  optmethod);
		printf("\n------------------------------------------------------------------------\n");
		fprintf(outfile, "%d %d %d", size, csrnum, groupnum);
		for (unsigned int k = 0; k < methodnum; k++)
		    fprintf(outfile, " %f", floptable[k]);
		fprintf(outfile, "\n");
	    }
	    
	    free_csr_matrix(csrmat);
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
    sprintf(clfilename, "%s%s", clspmvpath, "/kernels/spmv_csr_vector.cl");
    int ntimes = 1000;
    benchmark_csr(clspmvpath, clfilename, ntimes, CL_DEVICE_TYPE_GPU);
    return 0;
}



