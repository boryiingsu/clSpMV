#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

#include "CL/cl.h"

#include "spmv_csr_vector.h"
#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "constant.h"


void spmv_csr_vector_ocl(csr_matrix<int, float>* mat, float* vec, float* result, int padNum, int dim2Size, double& opttime, int& optmethod, char* oclfilename, cl_device_type deviceType, float* coores, int ntimes)
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
	int methodid = 7;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, 8);
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


	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
	
	//for (int totalsize = CSR_VEC_GROUP_SIZE * 15; totalsize < CSR_VEC_GROUP_SIZE * 15 * 12; totalsize += CSR_VEC_GROUP_SIZE * 15)
	//for (int totalsize = CSR_VEC_MIN_TH_NUM; totalsize < 4194304; totalsize *= 2)
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

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
	    printf("\nCSR vector SLM row ptr groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	    if (onetime < minlooptime)
	    {
		minlooptime = onetime;
		maxloopsize = groupnum;
	    }
	}
	printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


    }

    
    

    {
	int methodid = 10;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, 8);
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


	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;
	
	//for (int totalsize = 2048; totalsize < 4194304; totalsize *= 2)
	//for (int totalsize = CSR_VEC_MIN_TH_NUM; totalsize < 4194304; totalsize *= 2)
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};
	    //size_t globalsize[] = {totalsize, dim2};

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
	    printf("\nCSR vector SLM row ptr reduction groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	    if (onetime < minlooptime)
	    {
		minlooptime = onetime;
		maxloopsize = groupnum;
	    }
	}
	printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


    }
    
    {
	int methodid = 107;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WARPSIZE / 2);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	printf("\nRow Num %d padded size %d", rownum, padrowsize);
	cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;


	//for (int totalsize = CSR_VEC_MIN_TH_NUM; totalsize < 4194304; totalsize *= 2)
	//for (int totalsize = CSR_VEC_GROUP_SIZE * 15; totalsize < CSR_VEC_GROUP_SIZE * 15 * 12 * 2; totalsize += CSR_VEC_GROUP_SIZE * 15)
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

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
	    printf("\nCSR vector SLM row ptr tex groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	    if (onetime < minlooptime)
	    {
		minlooptime = onetime;
		maxloopsize = groupnum;
	    }
	}
	printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);

	if (devRowPtrPad)
	    clReleaseMemObject(devRowPtrPad);
	if (csrKernel)
	    clReleaseKernel(csrKernel);
	free(rowptrpad);


    }


    
    {
	int methodid = 110;
	cl_mem devRowPtrPad;
	int padrowsize = findPaddedSize(rownum, WARPSIZE / 2);
	int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
	memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
	for (int i = 0; i <= mat->matinfo.height; i++)
	    rowptrpad[i] = mat->csr_row_ptr[i];
	ALLOCATE_GPU_READ(devRowPtrPad, rowptrpad, sizeof(int)*(padrowsize+1));

	printf("\nRow Num %d padded size %d", rownum, padrowsize);
	cl_uint work_dim = 2;
	//int dim2 = 16;
	size_t blocksize[] = {CSR_VEC_GROUP_SIZE, 1};

	cl_kernel csrKernel = NULL;
	csrKernel = clCreateKernel(program, "gpu_csr_ve_reduction_fs_tx", &errorCode); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 0, sizeof(cl_mem), &devRowPtrPad); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 1, sizeof(cl_mem), &devColId); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 2, sizeof(cl_mem), &devData); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 3, sizeof(cl_mem), &devTexVec); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 4, sizeof(cl_mem), &devRes); CHECKERROR;
	errorCode = clSetKernelArg(csrKernel, 5, sizeof(int), &rownum); CHECKERROR;

	int maxloopsize = CSR_VEC_MIN_TH_NUM;
	double minlooptime = 1000.0f;


	//for (int totalsize = CSR_VEC_MIN_TH_NUM; totalsize < 4194304; totalsize *= 2)
	for (int groupnum = 24; groupnum <= 288; groupnum+= 24)
	{
	    //size_t globalsize[] = {totalsize, dim2};
	    size_t globalsize[] = {groupnum*CSR_VEC_GROUP_SIZE, dim2};

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
	    printf("\nCSR vector SLM row ptr reduction tex groupnum:%d cpu time %lf ms GFLOPS %lf code %d \n\n", groupnum,   time_in_sec / (double) ntimes * 1000, gflops, methodid);

	    double onetime = time_in_sec / (double) ntimes;
	    if (onetime < opttime)
	    {
		opttime = onetime;
		optmethod = methodid;
	    }
	    if (onetime < minlooptime)
	    {
		minlooptime = onetime;
		maxloopsize = groupnum;
	    }
	}
	printf("******* Min time %f groupnum %d **********", minlooptime, maxloopsize);

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


void vector_init_pointers(int* rowptr, int* rowpointers, int* itempointers)
{
    for (int i = 0; i < 16; i++)
    {
        int start = rowptr[i];
        int end = rowptr[i+1];
		for (int j = 0; j < 16; j++)
		{
			if (start + j < end)
			{
				rowpointers[i * 16 + j] = i;
				itempointers[i * 16 + j] = start + j;
			}
			else
			{
				rowpointers[i * 16 + j] = i;
				itempointers[i * 16 + j] = -1;
			}
		}
    }
}

void vector_collect_colids(int* matcols, int* itempointers, int* colids)
{
    for (int i = 0; i < 256; i++)
    {
        if (itempointers[i] >= 0)
        {
            colids[i] = matcols[itempointers[i]];
        }
        else
        {
            colids[i] = -1;
        }
    }
}

void vector_find_cachelines(int* colids, int& numlines, int& numitems, int startid)
{
    using namespace std;
    numlines = 0; 
    numitems = 0;
    vector<pair<int, int> > histo;
    histo.clear();
    for (int i = startid; i < startid + 16; i++)
    {
        int col = colids[i];
        if (col >= 0)
        {
            numitems++;
            int cacheid = col / 16;
            bool ifexist = false;
            for (int j = 0; j < histo.size(); j++)
            {
                if (cacheid == histo[j].first)
                {
                    histo[j].second++;
                    ifexist = true;
                    break;
                }
            }
            if (!ifexist)
            {
                numlines++;
                histo.push_back(pair<int, int>(cacheid, 1));
            }
        }
    }
    assert(numlines == histo.size());
}


bool vector_nextchunk(int* rowptr, int* rowpointers, int* itempointers, int rowsize)
{
    bool ifnotdone = false;
    for (int euid = 0; euid < 16; euid++)
    {
        bool ifremain = false;
        int currow = rowpointers[euid * 16];
        for (int laneid = 0; laneid < 16; laneid++)
        {
            int itemid = euid * 16 + laneid;
            int end = rowptr[currow + 1];
            int curposition = itempointers[itemid];
            if (curposition >= 0 && curposition + 16 < end)
            {
                ifremain = true;
                ifnotdone = true;
                itempointers[itemid] = curposition + 16;
            }
            else
            {
                itempointers[itemid] = -1;
            }
        }
        if (!ifremain)
        {
            int maxrow = 0;
            for (int i = 0; i < 256; i+=16)
            {
                if (rowpointers[i] > maxrow)
                    maxrow = rowpointers[i];
            }
            int newrow = maxrow + 1;
            if (newrow < rowsize)
            {
				int start = rowptr[newrow];
                int end = rowptr[newrow + 1];
                ifnotdone = true;
                for (int laneid = 0; laneid < 16; laneid++)
                {
                    int itemid = euid * 16 + laneid;
					rowpointers[itemid] = newrow;
                    if (start + laneid < end)
                    {
                        itempointers[itemid] = start + laneid;
                    }
                    else
                    {
                        itempointers[itemid] = -1;
                    }
                }
            }
        }
    }
    return ifnotdone;
}


void vector_cache_behavior(csr_matrix<int, float>* mat)
{
    int rowpointers[256];
    int itempointers[256];
    int colids[256];
    vector_init_pointers(mat->csr_row_ptr, rowpointers, itempointers);
    double minusage = 10000.0f;
    double maxusage = 0.0f;
    double sum = 0.0f;
    int iteration = 0;
    int totalitem = 0;
    int totalline = 0;
    while (1)
    {
	vector_collect_colids(mat->csr_col_id, itempointers, colids);
	for (int euid = 0; euid < 16; euid++)
	{
	    int numlines = 0;
	    int numitems = 0;
	    vector_find_cachelines(colids, numlines, numitems, euid * 16);
	    double ratio = 0.0f;
	    if (numlines > 0)
	    {
		ratio = (double)numitems * 4.0 / ((double)numlines * 64.0);
		//if (ratio > 1.0f)
		//ratio = 1.0f;
		if (ratio > maxusage)
		    maxusage = ratio;
		if (ratio < minusage)
		    minusage = ratio;
		sum += ratio;
		iteration++;
		totalitem += numitems;
		totalline += numlines;
	    }
	}
	if (!vector_nextchunk(mat->csr_row_ptr, rowpointers, itempointers, mat->matinfo.height))
	    break;
    }
    printf("Vector Cache Behavior Avg %f  Max %f  Min %f total item %d total cacheline %d\n", sum / (double)iteration, maxusage, minusage, totalitem, totalline);
}


void spmv_csr_vector(char* oclfilename, coo_matrix<int, float>* mat, int dim2Size, int ntimes, cl_device_type deviceType)
{
    printMatInfo(mat);
    csr_matrix<int, float> csrmat;
    coo2csr<int, float>(mat, &csrmat);
    float* vec = (float*)malloc(sizeof(float)*mat->matinfo.width);
    float* res = (float*)malloc(sizeof(float)*mat->matinfo.height);
    initVectorOne<int, float>(vec, mat->matinfo.width);	
    initVectorZero<int, float>(res, mat->matinfo.height);
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    spmv_only(mat, vec, coores);


    {
	double opttime1 = 10000.0f;
	int optmethod1 = 0;

	//vector_cache_behavior(&csrmat);

	spmv_csr_vector_ocl(&csrmat, vec, res, 0, dim2Size, opttime1, optmethod1, oclfilename, deviceType, coores, ntimes);

	double opttime2 = 10000.0f;
	int optmethod2 = 0;

	csr_matrix<int, float> padcsr;
	pad_csr(&csrmat, &padcsr, WARPSIZE / 2);
	printf("\nNNZ Before %d After %d\n", csrmat.matinfo.nnz, padcsr.matinfo.nnz);
	spmv_csr_vector_ocl(&padcsr, vec, res, 16, dim2Size, opttime2, optmethod2, oclfilename, deviceType, coores, ntimes);
	free_csr_matrix(padcsr);

	int nnz = mat->matinfo.nnz;
	double gflops = (double)nnz*2/opttime1/(double)1e9;
	printf("\n------------------------------------------------------------------------\n");
	printf("CSR VEC without padding best time %f ms best method %d gflops %f", opttime1*1000.0, optmethod1, gflops);
	printf("\n------------------------------------------------------------------------\n");
	gflops = (double)nnz*2/opttime2/(double)1e9;
	printf("CSR VEC with padding best time %f ms best method %d gflops %f", opttime2*1000.0, optmethod2, gflops);
	printf("\n------------------------------------------------------------------------\n");
    }



    free(vec);
    free(res);
    free_csr_matrix(csrmat);
    free(coores);
}

