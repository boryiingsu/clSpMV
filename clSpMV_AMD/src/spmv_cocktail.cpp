#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "CL/cl.h"

#include "oclcommon.h"
#include "util.h"
#include "spmv_serial.h"
#include "constant.h"
#include "matrix_storage.h"
#include "fileio.h"
#include "analyze.h"
#include "spmv_cocktail.h"

void init_cocktail_gpu(cocktail_gpu& mat)
{
    mat.ifusebdia  = false;
    mat.ifusedia   = false;
    mat.ifusesbell = false;
    mat.ifusebell  = false;
    mat.ifusebcsr  = false;
    mat.ifusesell  = false;
    mat.ifuseell   = false;
    mat.ifusecsr   = false;
    mat.ifusecoo   = false;
    
    mat.ifteximage	= false;
    mat.iftexr		= false;
    mat.iftexrgba	= false;
    mat.ifsbelltex	= false;
    mat.ifbelltex	= false;
    mat.ifbcsrtex	= false;
    mat.ifselltex	= false;
    mat.ifelltex	= false;
    mat.ifcsrtex	= false;
    mat.ifcootex	= false;
}

void free_cocktail_gpu(cocktail_gpu& mat)
{
    if (mat.vec)
	clReleaseMemObject(mat.vec);
    if (mat.res)
	clReleaseMemObject(mat.res);
    if (mat.iftexr)
	clReleaseMemObject(mat.vec_tex_r);
    if (mat.iftexrgba)
	clReleaseMemObject(mat.vec_tex_rgba);
    if (mat.ifteximage)
	clReleaseMemObject(mat.vec_image);
    if (mat.ifusebdia || mat.ifusedia)
    {
	if (mat.vec_extended)
	    clReleaseMemObject(mat.vec_extended);
    }
    if (mat.ifusebdia)
    {
	if (mat.bdia_band_ptr)
	    clReleaseMemObject(mat.bdia_band_ptr);
	if (mat.bdia_offsets)
	    clReleaseMemObject(mat.bdia_offsets);
	if (mat.bdia_data)
	    clReleaseMemObject(mat.bdia_data);
    }
    if (mat.ifusedia)
    {
	if (mat.dia_offsets)
	    clReleaseMemObject(mat.dia_offsets);
	if (mat.dia_data)
	    clReleaseMemObject(mat.dia_data);
    }
    if (mat.ifusesbell)
    {
	if (mat.sbell_slice_ptr)
	    clReleaseMemObject(mat.sbell_slice_ptr);
	if (mat.sbell_col_id)
	    clReleaseMemObject(mat.sbell_col_id);
	if (mat.sbell_data)
	    clReleaseMemObject(mat.sbell_data);
    }
    if (mat.ifusebell)
    {
	if (mat.bell_col_id)
	    clReleaseMemObject(mat.bell_col_id);
	if (mat.bell_data)
	    clReleaseMemObject(mat.bell_data);
    }
    if (mat.ifusebcsr)
    {
	if (mat.bcsr_row_ptr)
	    clReleaseMemObject(mat.bcsr_row_ptr);
	if (mat.bcsr_col_id)
	    clReleaseMemObject(mat.bcsr_col_id);
	if (mat.bcsr_data)
	    clReleaseMemObject(mat.bcsr_data);
    }
    if (mat.ifusesell)
    {
	if (mat.sell_slice_ptr)
	    clReleaseMemObject(mat.sell_slice_ptr);
	if (mat.sell_col_id)
	    clReleaseMemObject(mat.sell_col_id);
	if (mat.sell_data)
	    clReleaseMemObject(mat.sell_data);
    }
    if (mat.ifuseell)
    {
	if (mat.ell_col_id)
	    clReleaseMemObject(mat.ell_col_id);
	if (mat.ell_data)
	    clReleaseMemObject(mat.ell_data);
    }
    if (mat.ifusecsr)
    {
	if (mat.csr_row_ptr)
	    clReleaseMemObject(mat.csr_row_ptr);
	if (mat.csr_col_id)
	    clReleaseMemObject(mat.csr_col_id);
	if (mat.csr_data)
	    clReleaseMemObject(mat.csr_data);
    }
    if (mat.ifusecoo)
    {
	if (mat.coo_row_id)
	    clReleaseMemObject(mat.coo_row_id);
	if (mat.coo_col_id)
	    clReleaseMemObject(mat.coo_col_id);
	if (mat.coo_data)
	    clReleaseMemObject(mat.coo_data);
	if (mat.coo_tmp_row)
	    clReleaseMemObject(mat.coo_tmp_row);
	if (mat.coo_tmp_data)
	    clReleaseMemObject(mat.coo_tmp_data);
    }
}

void init_cocktail_kernels(cocktail_kernels& cocktail)
{
    cocktail.devices = NULL;
    cocktail.context = NULL;
    cocktail.cmdQueue = NULL;
    cocktail.program = NULL;

    cocktail.bdia_kernel = NULL;
    cocktail.dia_kernel = NULL;
    cocktail.sbell_kernel = NULL;
    cocktail.bell_kernel = NULL;
    cocktail.bcsr_kernel = NULL;
    cocktail.sell_kernel = NULL;
    cocktail.ell_kernel = NULL;
    cocktail.csr_kernel = NULL;
    cocktail.coo_kernel_s1 = NULL;
    cocktail.coo_kernel_s2 = NULL;
}

void free_cocktail_kernels(cocktail_kernels& cocktail)
{
    if (cocktail.bdia_kernel)
	clReleaseKernel(cocktail.bdia_kernel);
    if (cocktail.dia_kernel)
	clReleaseKernel(cocktail.dia_kernel);
    if (cocktail.sbell_kernel)
	clReleaseKernel(cocktail.sbell_kernel);
    if (cocktail.bell_kernel)
	clReleaseKernel(cocktail.bell_kernel);
    if (cocktail.bcsr_kernel)
	clReleaseKernel(cocktail.bcsr_kernel);
    if (cocktail.sell_kernel)
	clReleaseKernel(cocktail.sell_kernel);
    if (cocktail.ell_kernel)
	clReleaseKernel(cocktail.ell_kernel);
    if (cocktail.csr_kernel)
	clReleaseKernel(cocktail.csr_kernel);
    if (cocktail.coo_kernel_s1)
	clReleaseKernel(cocktail.coo_kernel_s1);
    if (cocktail.coo_kernel_s2)
	clReleaseKernel(cocktail.coo_kernel_s2);
}

void cpy_vec_tex(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* vec, int vec_size)
{
    cl_int errorCode = CL_SUCCESS;
    const cl_image_format floatFormat =
    {
	CL_R,
	CL_FLOAT,
    };
    const cl_image_format float4Format =
    {
	CL_RGBA,
	CL_FLOAT,
    };
    
    int width = VEC2DWIDTH;
    int height = (vec_size + VEC2DWIDTH - 1)/VEC2DWIDTH;
    if (height % 4 != 0)
	height += (4 - (height % 4));
    float* image2dVec = (float*)malloc(sizeof(float)*width*height);
    for (int i = 0; i < width * height; i++)
	image2dVec[i] = 0.0f;
    for (int i = 0; i < vec_size; i++)
    {
	image2dVec[i] = vec[i];
    }
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height, 1};
    size_t vector4Size[] = {width, height/4, 1};
    if (gpumat.iftexr || gpumat.iftexrgba)
    {
	gpumat.ifteximage = true;
	ALLOCATE_GPU_READ(gpumat.vec_image, image2dVec, sizeof(float)*width*height);
    }
    if (gpumat.iftexr)
    {
	gpumat.vec_tex_r = clCreateImage2D(context, CL_MEM_READ_ONLY, &floatFormat, width, height, 0, NULL, &errorCode); CHECKERROR;
	errorCode = clEnqueueWriteImage(cmdQueue, gpumat.vec_tex_r, CL_TRUE, origin, vectorSize, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    }
    if (gpumat.iftexrgba)
    {
	gpumat.vec_tex_rgba = clCreateImage2D(context, CL_MEM_READ_ONLY, &float4Format, width, height/4, 0, NULL, &errorCode); CHECKERROR;
	errorCode = clEnqueueWriteImage(cmdQueue, gpumat.vec_tex_rgba, CL_TRUE, origin, vector4Size, 0, 0, image2dVec, 0, NULL, NULL); CHECKERROR;
    }

    clFinish(cmdQueue);
    free(image2dVec);
}

void cpy_shared(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* vec, int vec_size, float* res, int res_size)
{
    cl_int errorCode = CL_SUCCESS;
    //Copy the result vector to gpu
    int padreslen = findPaddedSize(res_size, 8 * SELL_GROUP_SIZE);
    gpumat.res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*padreslen, NULL, &errorCode); CHECKERROR;
    errorCode = clEnqueueWriteBuffer(cmdQueue, gpumat.res, CL_TRUE, 0, sizeof(float)*res_size, res, 0, NULL, NULL); CHECKERROR;

    //Copy the multiplied vector to gpu
    int padveclen = findPaddedSize(vec_size, 8);
    float* newvec = (float*)malloc(sizeof(float)*padveclen);
    memcpy(newvec, vec, sizeof(float)*vec_size);
    for (int i = vec_size; i < padveclen; i++)
	newvec[i] = 0.0f;
    ALLOCATE_GPU_READ(gpumat.vec, newvec, sizeof(float)*padveclen);

    clFinish(cmdQueue);
    free(newvec);

    cpy_vec_tex(gpumat, context, cmdQueue, vec, vec_size);
}


void cpy_vec_extended(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* vec, int vec_size)
{
    cl_int errorCode = CL_SUCCESS;
    int minoffset = cpumat.mat_width;
    int maxoffset = -cpumat.mat_width;
    if (cpumat.ifusebdia)
    {
	int bandnum = cpumat.bdia.bdia_band_num;
	assert(bandnum <= MAX_BAND_NUM);
	for (int i = 0; i < bandnum; i++)
	{
	    int curOffset = cpumat.bdia.bdia_offsets[i];
	    if (curOffset < minoffset)
		minoffset = curOffset;
	    if (curOffset > maxoffset)
		maxoffset = curOffset;
	}
	maxoffset += MAX_BAND_WIDTH;
    }
    if (cpumat.ifusedia)
    {
	int dianum = cpumat.dia.dia_num;
	assert(dianum <= MAX_DIA_NUM);
	for (int i = 0; i < dianum; i++)
	{
	    int curOffset = cpumat.dia.dia_offsets[i];
	    if (curOffset < minoffset)
		minoffset = curOffset;
	    if (curOffset > maxoffset)
		maxoffset = curOffset;
	}
    }
    int padveclength = cpumat.mat_width;
    int leftoffset = (minoffset >= 0) ? 0 : (-minoffset);
    assert(leftoffset >= 0);
    padveclength += leftoffset;
    if (maxoffset > 0)
	padveclength += maxoffset;
    padveclength += 100;
    float* padvec = (float*)malloc(sizeof(float)*padveclength);
    for (int i = 0; i < padveclength; i++)
	padvec[i] = 0.0f;
    assert(cpumat.mat_width == vec_size);
    memcpy(padvec + leftoffset, vec, sizeof(float)*vec_size);
    ALLOCATE_GPU_READ(gpumat.vec_extended, padvec, sizeof(float)*padveclength);
    clFinish(cmdQueue);
    gpumat.vec_offset = leftoffset;
    free(padvec);
}


void cpy_bdia_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int aligned_length = cpumat.bdia.bdia_length_aligned;
    int bandnum = cpumat.bdia.bdia_band_num;
    assert(bandnum <= MAX_BAND_NUM);
    int* padBandPtr = (int*)malloc(sizeof(int)*(MAX_BAND_NUM + 1));
    int* padOffset = (int*)malloc(sizeof(int)*(MAX_BAND_NUM + 1));
    for (int i = 0; i < MAX_BAND_NUM + 1; i++)
    {
	padBandPtr[i] = padOffset[i] = 0;
    }
    for (int i = 0; i <= bandnum; i++)
    {
	padBandPtr[i] = cpumat.bdia.bdia_bptr[i];
    }
    for (int i = 0; i < bandnum; i++)
    {
	padOffset[i] = cpumat.bdia.bdia_offsets[i];
    }
    int dianum = cpumat.bdia.bdia_bptr[bandnum];
    ALLOCATE_GPU_READ(gpumat.bdia_band_ptr, padBandPtr, sizeof(int)*(MAX_BAND_NUM + 1));
    ALLOCATE_GPU_READ(gpumat.bdia_offsets, padOffset, sizeof(int)*(MAX_BAND_NUM + 1));
    ALLOCATE_GPU_READ(gpumat.bdia_data, cpumat.bdia.bdia_data, sizeof(float)*aligned_length*dianum);
    clFinish(cmdQueue);
    gpumat.bdia_length = aligned_length;
    gpumat.bdia_length4 = aligned_length / 4;
    gpumat.bdia_bandnum = bandnum;
    free(padBandPtr);
    free(padOffset);
}

void set_bdia_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (cpumat.bdia_meth_num == 0)
    {
	kernels.bdia_kernel = clCreateKernel(program, "gpu_bdia_nlvec", &errorCode); CHECKERROR;
    }
    else if (cpumat.bdia_meth_num == 1)
    {
	kernels.bdia_kernel = clCreateKernel(program, "gpu_bdia", &errorCode); CHECKERROR;
    }
    else if (cpumat.bdia_meth_num == 2)
    {
	kernels.bdia_kernel = clCreateKernel(program, "gpu_bdia_g4", &errorCode); CHECKERROR;
    }
    else if (cpumat.bdia_meth_num == 3)
    {
	kernels.bdia_kernel = clCreateKernel(program, "gpu_bdia_v4_nlvec", &errorCode); CHECKERROR;
    }
    else if (cpumat.bdia_meth_num == 4)
    {
	kernels.bdia_kernel = clCreateKernel(program, "gpu_bdia_v4", &errorCode); CHECKERROR;
    }
    else
    { 
	printf("\n!!!Unknown method in BDIA format!!! Abort\n");
	assert(false);
    }

    errorCode = clSetKernelArg(kernels.bdia_kernel, 0, sizeof(cl_mem), &gpumat.bdia_band_ptr); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bdia_kernel, 1, sizeof(cl_mem), &gpumat.bdia_offsets); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bdia_kernel, 2, sizeof(cl_mem), &gpumat.bdia_data); CHECKERROR;
    if (cpumat.bdia_meth_num < 3)
    {
	errorCode = clSetKernelArg(kernels.bdia_kernel, 3, sizeof(int),    &gpumat.bdia_length); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.bdia_kernel, 3, sizeof(int),    &gpumat.bdia_length4); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.bdia_kernel, 4, sizeof(int),    &gpumat.bdia_bandnum); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bdia_kernel, 5, sizeof(cl_mem), &gpumat.vec_extended); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bdia_kernel, 6, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bdia_kernel, 7, sizeof(int),    &gpumat.vec_offset); CHECKERROR;

    int rownum = cpumat.bdia.matinfo.height;
    int rownum4 = rownum / 4;
    if (rownum % 4 != 0)
	rownum4++;
    kernels.bdia_block[0] = WORK_GROUP_SIZE;
    kernels.bdia_block[1] = 1;
    if (cpumat.bdia_meth_num < 2)
    {
	kernels.bdia_global[0] = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.bdia_global[1] = 1;
    }
    else
    {
	kernels.bdia_global[0] = ((rownum4 + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.bdia_global[1] = 1;
    }
}

void cpy_dia_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int aligned_length = cpumat.dia.dia_length_aligned;
    int dianum = cpumat.dia.dia_num;
    assert(dianum <= MAX_DIA_NUM);
    int* padOffset = (int*)malloc(sizeof(int)*(MAX_DIA_NUM));
    for (int i = 0; i < MAX_DIA_NUM; i++)
    {
	padOffset[i] = 0;
    }
    for (int i = 0; i < dianum; i++)
    {
	padOffset[i] = cpumat.dia.dia_offsets[i];
    }
    ALLOCATE_GPU_READ(gpumat.dia_offsets, padOffset, sizeof(int)*(MAX_DIA_NUM));
    ALLOCATE_GPU_READ(gpumat.dia_data, cpumat.dia.dia_data, sizeof(float)*aligned_length*dianum);
    gpumat.dia_length = aligned_length;
    gpumat.dia_length4 = aligned_length / 4;
    gpumat.dia_dianum = dianum;
    clFinish(cmdQueue);
    free(padOffset);
}

void set_dia_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (cpumat.dia_meth_num == 0)
    {
	kernels.dia_kernel = clCreateKernel(program, "gpu_dia", &errorCode); CHECKERROR;
    }
    else if (cpumat.dia_meth_num == 1)
    {
	kernels.dia_kernel = clCreateKernel(program, "gpu_dia_v4", &errorCode); CHECKERROR;
    }
    else if (cpumat.dia_meth_num == 2)
    {
	kernels.dia_kernel = clCreateKernel(program, "gpu_dia_tx", &errorCode); CHECKERROR;
    }
    else if (cpumat.dia_meth_num == 3)
    {
	kernels.dia_kernel = clCreateKernel(program, "gpu_dia_v4_tx", &errorCode); CHECKERROR;
    }
    else
    { 
	printf("\n!!!Unknown method in DIA format!!! Abort\n");
	assert(false);
    }
    errorCode = clSetKernelArg(kernels.dia_kernel, 0, sizeof(cl_mem), &gpumat.dia_offsets); CHECKERROR;
    errorCode = clSetKernelArg(kernels.dia_kernel, 1, sizeof(cl_mem), &gpumat.dia_data); CHECKERROR;
    if (cpumat.dia_meth_num == 0 || cpumat.dia_meth_num == 2)
    {
	errorCode = clSetKernelArg(kernels.dia_kernel, 2, sizeof(int),    &gpumat.dia_length); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.dia_kernel, 2, sizeof(int),    &gpumat.dia_length4); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.dia_kernel, 3, sizeof(int),    &gpumat.dia_dianum); CHECKERROR;
    if (cpumat.dia_meth_num < 2)
    {
	errorCode = clSetKernelArg(kernels.dia_kernel, 4, sizeof(cl_mem), &gpumat.vec_extended); CHECKERROR;
    }
    else if (cpumat.dia_meth_num == 2)
    {
	errorCode = clSetKernelArg(kernels.dia_kernel, 4, sizeof(cl_mem), &gpumat.vec_tex_r); CHECKERROR;
    }
    else if (cpumat.dia_meth_num == 3)
    {
	errorCode = clSetKernelArg(kernels.dia_kernel, 4, sizeof(cl_mem), &gpumat.vec_tex_rgba); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.dia_kernel, 5, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.dia_kernel, 6, sizeof(int),    &gpumat.vec_offset); CHECKERROR;

    int rownum = cpumat.dia.matinfo.height;
    int rownum4 = rownum / 4;
    if (rownum % 4 != 0)
	rownum4++;
    kernels.dia_block[0] = WORK_GROUP_SIZE;
    kernels.dia_block[1] = 1;
    if (cpumat.dia_meth_num == 0 || cpumat.dia_meth_num == 2)
    {
	kernels.dia_global[0] = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.dia_global[1] = 1;
    }
    else
    {
	kernels.dia_global[0] = ((rownum4 + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.dia_global[1] = 1;
    }
}

void cpy_sbell_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int slicenum = cpumat.sbell.sbell_slice_num;
    int bwidth = cpumat.sbell.sbell_bwidth;
    int bheight = cpumat.sbell.sbell_bheight;
    int totalsize = cpumat.sbell.sbell_slice_ptr[slicenum];
    ALLOCATE_GPU_READ(gpumat.sbell_slice_ptr, cpumat.sbell.sbell_slice_ptr, sizeof(int)*(slicenum + 1));
    ALLOCATE_GPU_READ(gpumat.sbell_col_id, cpumat.sbell.sbell_col_id, sizeof(int)*totalsize);
    ALLOCATE_GPU_READ(gpumat.sbell_data, cpumat.sbell.sbell_data, sizeof(float)*totalsize*bwidth*bheight);
    gpumat.sbell_slicenum = slicenum;
    clFinish(cmdQueue);
}

void set_sbell_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    int bheight = cpumat.sbell.sbell_bheight;
    int bwidth = cpumat.sbell.sbell_bwidth;
    char kernelname[100] = "gpu_sbell00";
    char kernelname_tx[100] = "gpu_sbell00_tx";
    kernelname[9] += bheight;
    kernelname[10] += bwidth;
    kernelname_tx[9] += bheight;
    kernelname_tx[10] += bwidth;
    if (gpumat.ifsbelltex)
    {
	kernels.sbell_kernel = clCreateKernel(program, kernelname_tx, &errorCode); CHECKERROR;
    }
    else
    {
	kernels.sbell_kernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.sbell_kernel, 0, sizeof(cl_mem), &gpumat.sbell_slice_ptr); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sbell_kernel, 1, sizeof(cl_mem), &gpumat.sbell_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sbell_kernel, 2, sizeof(cl_mem), &gpumat.sbell_data); CHECKERROR;
    if (gpumat.ifsbelltex)
    {
	errorCode = clSetKernelArg(kernels.sbell_kernel, 3, sizeof(cl_mem), &gpumat.vec_tex_rgba); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.sbell_kernel, 3, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.sbell_kernel, 4, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sbell_kernel, 5, sizeof(int),    &gpumat.sbell_slicenum); CHECKERROR;

    int blockrownum = cpumat.sbell.sbell_row_num;
    kernels.sbell_block[0] = SELL_GROUP_SIZE;
    kernels.sbell_block[1] = 1;
    kernels.sbell_global[0] = ((blockrownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
    kernels.sbell_global[1] = 1;
}

void cpy_bell_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int col_align = cpumat.bell.b4ell_height_aligned;
    int data_align = cpumat.bell.b4ell_float4_aligned;
    int blockrownum = cpumat.bell.b4ell_row_num;
    int b4ellnum = cpumat.bell.b4ell_block_num;
    int bwidth = cpumat.bell.b4ell_bwidth;
    int bheight = cpumat.bell.b4ell_bheight;
    int width4num = bwidth / 4;
    ALLOCATE_GPU_READ(gpumat.bell_col_id, cpumat.bell.b4ell_col_id, sizeof(int)*col_align*b4ellnum);
    ALLOCATE_GPU_READ(gpumat.bell_data, cpumat.bell.b4ell_data, sizeof(float)*data_align*bheight*width4num*b4ellnum);
    gpumat.bell_data_align = data_align / 4;
    gpumat.bell_col_align = col_align;
    gpumat.bell_ellnum = b4ellnum;
    gpumat.bell_brownum = blockrownum;
    clFinish(cmdQueue);
}

void set_bell_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    int bwidth = cpumat.bell.b4ell_bwidth;
    int bheight = cpumat.bell.b4ell_bheight;
    if (cpumat.bell_meth_num == 0)
    {
	char kernelname[100] = "gpu_bell00";
	char kernelname_tx[100] = "gpu_bell00_tx";
	kernelname[8] += bheight;
	kernelname[9] += bwidth;
	kernelname_tx[8] += bheight;
	kernelname_tx[9] += bwidth;
	if (gpumat.ifbelltex)
	{
	    kernels.bell_kernel = clCreateKernel(program, kernelname_tx, &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.bell_kernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	}
    }
    else if (cpumat.bell_meth_num == 1)
    {
	char kernelname[100] = "gpu_bell00_mad";
	char kernelname_tx[100] = "gpu_bell00_mad_tx";
	kernelname[8] += bheight;
	kernelname[9] += bwidth;
	kernelname_tx[8] += bheight;
	kernelname_tx[9] += bwidth;
	if (gpumat.ifbelltex)
	{
	    kernels.bell_kernel = clCreateKernel(program, kernelname_tx, &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.bell_kernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
	}
    }
    else
    { 
	printf("\n!!!Unknown method in BELL format!!! Abort\n");
	assert(false);
    }
    errorCode = clSetKernelArg(kernels.bell_kernel, 0, sizeof(cl_mem), &gpumat.bell_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bell_kernel, 1, sizeof(cl_mem), &gpumat.bell_data); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bell_kernel, 2, sizeof(int),    &gpumat.bell_data_align); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bell_kernel, 3, sizeof(int),    &gpumat.bell_col_align); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bell_kernel, 4, sizeof(int),    &gpumat.bell_ellnum); CHECKERROR;
    if (gpumat.ifbelltex)
    {
	errorCode = clSetKernelArg(kernels.bell_kernel, 5, sizeof(cl_mem), &gpumat.vec_tex_rgba); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.bell_kernel, 5, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.bell_kernel, 6, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bell_kernel, 7, sizeof(int),    &gpumat.bell_brownum); CHECKERROR;

    int blockrownum = gpumat.bell_brownum;
    kernels.bell_block[0] = BELL_GROUP_SIZE;
    kernels.bell_block[1] = 1;
    kernels.bell_global[0] = ((blockrownum + BELL_GROUP_SIZE - 1)/BELL_GROUP_SIZE)*BELL_GROUP_SIZE;
    kernels.bell_global[1] = 1;
}

void cpy_bcsr_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int data_align = cpumat.bcsr.b4csr_aligned_size;
    int blockrownum = cpumat.bcsr.b4csr_row_num;
    int blocknum = cpumat.bcsr.b4csr_block_num;
    int bwidth = cpumat.bcsr.b4csr_bwidth;
    int bheight = cpumat.bcsr.b4csr_bheight;
    int width4num = bwidth / 4;
    ALLOCATE_GPU_READ(gpumat.bcsr_row_ptr, cpumat.bcsr.b4csr_row_ptr, sizeof(int)*(blockrownum + 1));
    ALLOCATE_GPU_READ(gpumat.bcsr_col_id, cpumat.bcsr.b4csr_col_id, sizeof(int)*blocknum);
    ALLOCATE_GPU_READ(gpumat.bcsr_data, cpumat.bcsr.b4csr_data, sizeof(float)*data_align*width4num*bheight);
    gpumat.bcsr_data_align = data_align / 4;
    clFinish(cmdQueue);
}

void set_bcsr_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    int bheight = cpumat.bcsr.b4csr_bheight;
    int bwidth = cpumat.bcsr.b4csr_bwidth;
    char kernelname[100] = "gpu_bcsr_red_00";
    char kernelname_tx[100] = "gpu_bcsr_red_00_tx";
    kernelname[13] += bheight;
    kernelname[14] += bwidth;
    kernelname_tx[13] += bheight;
    kernelname_tx[14] += bwidth;
    if (gpumat.ifbcsrtex)
    {
	kernels.bcsr_kernel = clCreateKernel(program, kernelname_tx, &errorCode); CHECKERROR;
    }
    else
    {
	kernels.bcsr_kernel = clCreateKernel(program, kernelname, &errorCode); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.bcsr_kernel, 0, sizeof(cl_mem), &gpumat.bcsr_row_ptr); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bcsr_kernel, 1, sizeof(cl_mem), &gpumat.bcsr_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bcsr_kernel, 2, sizeof(cl_mem), &gpumat.bcsr_data); CHECKERROR;
    if (gpumat.ifbcsrtex)
    {
	errorCode = clSetKernelArg(kernels.bcsr_kernel, 3, sizeof(cl_mem), &gpumat.vec_tex_rgba); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.bcsr_kernel, 3, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.bcsr_kernel, 4, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.bcsr_kernel, 5, sizeof(int),    &gpumat.bcsr_data_align); CHECKERROR;

    int blockrownum = cpumat.bcsr.b4csr_row_num;
    kernels.bcsr_block[0] = BCSR_VEC_GROUP_SIZE;
    kernels.bcsr_block[1] = 1;
    kernels.bcsr_global[0] = BCSR_VEC_GROUP_SIZE * blockrownum;
    kernels.bcsr_global[1] = 1;
}

void cpy_sell_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int slicenum = cpumat.sell.sell_slice_num;
    int datasize = cpumat.sell.sell_slice_ptr[slicenum];
    ALLOCATE_GPU_READ(gpumat.sell_slice_ptr, cpumat.sell.sell_slice_ptr, sizeof(int)*(slicenum + 1));
    ALLOCATE_GPU_READ(gpumat.sell_col_id, cpumat.sell.sell_col_id, sizeof(int)*datasize);
    ALLOCATE_GPU_READ(gpumat.sell_data, cpumat.sell.sell_data, sizeof(float)*datasize);
    gpumat.sell_slicenum = slicenum;
    clFinish(cmdQueue);
}

void set_sell_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (cpumat.sell_meth_num == 0)
    {
	if (gpumat.ifselltex)
	{
	    kernels.sell_kernel = clCreateKernel(program, "gpu_sell_warp_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.sell_kernel = clCreateKernel(program, "gpu_sell_warp", &errorCode); CHECKERROR;
	}
    }
    else if (cpumat.sell_meth_num == 1)
    {
	if (gpumat.ifselltex)
	{
	    kernels.sell_kernel = clCreateKernel(program, "gpu_sell_group_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.sell_kernel = clCreateKernel(program, "gpu_sell_group", &errorCode); CHECKERROR;
	}
    }
    else
    { 
	printf("\n!!!Unknown method in SELL format!!! Abort\n");
	assert(false);
    }
    errorCode = clSetKernelArg(kernels.sell_kernel, 0, sizeof(cl_mem), &gpumat.sell_slice_ptr); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sell_kernel, 1, sizeof(cl_mem), &gpumat.sell_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sell_kernel, 2, sizeof(cl_mem), &gpumat.sell_data); CHECKERROR;
    if (gpumat.ifselltex)
    {
	errorCode = clSetKernelArg(kernels.sell_kernel, 3, sizeof(cl_mem), &gpumat.vec_tex_r); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.sell_kernel, 3, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.sell_kernel, 4, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.sell_kernel, 5, sizeof(int),    &gpumat.sell_slicenum); CHECKERROR;

    int rownum = cpumat.sell.matinfo.height;
    int slicenum = gpumat.sell_slicenum;
    kernels.sell_block[0] = SELL_GROUP_SIZE;
    kernels.sell_block[1] = 1;
    if (cpumat.sell_meth_num == 0)
    {
	kernels.sell_global[0] = ((rownum + SELL_GROUP_SIZE - 1)/SELL_GROUP_SIZE)*SELL_GROUP_SIZE;
	kernels.sell_global[1] = 1;
    }
    else if (cpumat.sell_meth_num == 1)
    {
	kernels.sell_global[0] = SELL_GROUP_SIZE * slicenum;
	kernels.sell_global[1] = 1;
    }
}


void cpy_ell_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int aligned_length = cpumat.ell.ell_height_aligned;
    int rownum = cpumat.ell.matinfo.height;
    int rownum4 = rownum / 4;
    if (rownum % 4 != 0)
	rownum4++;
    int ellnum = cpumat.ell.ell_num;
    ALLOCATE_GPU_READ(gpumat.ell_col_id, cpumat.ell.ell_col_id, sizeof(int)*aligned_length*ellnum);
    ALLOCATE_GPU_READ(gpumat.ell_data, cpumat.ell.ell_data, sizeof(float)*aligned_length*ellnum);
    gpumat.ell_length = aligned_length;
    gpumat.ell_length4 = aligned_length / 4;
    gpumat.ell_ellnum = ellnum;
    gpumat.ell_rownum = rownum;
    gpumat.ell_rownum4 = rownum4;
    clFinish(cmdQueue);
}

void set_ell_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (cpumat.ell_meth_num == 0)
    {
	if (gpumat.ifelltex)
	{
	    kernels.ell_kernel = clCreateKernel(program, "gpu_ell_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.ell_kernel = clCreateKernel(program, "gpu_ell", &errorCode); CHECKERROR;
	}
    }
    else if (cpumat.ell_meth_num == 1)
    {
	if (gpumat.ifelltex)
	{
	    kernels.ell_kernel = clCreateKernel(program, "gpu_ell_v4_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.ell_kernel = clCreateKernel(program, "gpu_ell_v4", &errorCode); CHECKERROR;
	}
    }
    else
    { 
	printf("\n!!!Unknown method in ELL format!!! Abort\n");
	assert(false);
    }
    errorCode = clSetKernelArg(kernels.ell_kernel, 0, sizeof(cl_mem), &gpumat.ell_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.ell_kernel, 1, sizeof(cl_mem), &gpumat.ell_data); CHECKERROR;
    if (cpumat.ell_meth_num == 0)
    {
	errorCode = clSetKernelArg(kernels.ell_kernel, 2, sizeof(int),    &gpumat.ell_length); CHECKERROR;
	errorCode = clSetKernelArg(kernels.ell_kernel, 6, sizeof(int),    &gpumat.ell_rownum); CHECKERROR;
    }
    else if (cpumat.ell_meth_num == 1)
    {
	errorCode = clSetKernelArg(kernels.ell_kernel, 2, sizeof(int),    &gpumat.ell_length4); CHECKERROR;
	errorCode = clSetKernelArg(kernels.ell_kernel, 6, sizeof(int),    &gpumat.ell_rownum4); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.ell_kernel, 3, sizeof(int),    &gpumat.ell_ellnum); CHECKERROR;
    if (gpumat.ifelltex)
    {
	errorCode = clSetKernelArg(kernels.ell_kernel, 4, sizeof(cl_mem), &gpumat.vec_tex_r); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.ell_kernel, 4, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.ell_kernel, 5, sizeof(cl_mem), &gpumat.res); CHECKERROR;

    int rownum = gpumat.ell_rownum;
    int rownum4 = gpumat.ell_rownum4;
    kernels.ell_block[0] = WORK_GROUP_SIZE;
    kernels.ell_block[1] = 1;
    if (cpumat.ell_meth_num == 0)
    {
	kernels.ell_global[0] = ((rownum + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.ell_global[1] = 1;
    }
    else if (cpumat.ell_meth_num == 1)
    {
	kernels.ell_global[0] = ((rownum4 + WORK_GROUP_SIZE - 1)/WORK_GROUP_SIZE)*WORK_GROUP_SIZE;
	kernels.ell_global[1] = 1;
    }
}


void cpy_csr_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int nnz = cpumat.csr.matinfo.nnz;
    int rownum = cpumat.csr.matinfo.height;
    int padrowsize = findPaddedSize(rownum, CSR_VEC_GROUP_SIZE/WARPSIZE);
    int* rowptrpad = (int*)malloc(sizeof(int)*(padrowsize+1));
    memset(rowptrpad, 0, sizeof(int)*(padrowsize+1));
    for (int i = 0; i <= rownum; i++)
	rowptrpad[i] = cpumat.csr.csr_row_ptr[i];
    ALLOCATE_GPU_READ(gpumat.csr_row_ptr, rowptrpad, sizeof(int)*(padrowsize+1));
    ALLOCATE_GPU_READ(gpumat.csr_col_id, cpumat.csr.csr_col_id, sizeof(int)*nnz);
    ALLOCATE_GPU_READ(gpumat.csr_data, cpumat.csr.csr_data, sizeof(float)*nnz);
    gpumat.csr_rownum = rownum;
    clFinish(cmdQueue);
    free(rowptrpad);
}

void set_csr_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (cpumat.csr_meth_num == 0)
    {
	if (gpumat.ifcsrtex)
	{
	    kernels.csr_kernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.csr_kernel = clCreateKernel(program, "gpu_csr_ve_slm_pm_fs", &errorCode); CHECKERROR;
	}
    }
    else if (cpumat.csr_meth_num == 1)
    {
	if (gpumat.ifcsrtex)
	{
	    kernels.csr_kernel = clCreateKernel(program, "gpu_csr_ve_reduction_fs_tx", &errorCode); CHECKERROR;
	}
	else
	{
	    kernels.csr_kernel = clCreateKernel(program, "gpu_csr_ve_reduction_fs", &errorCode); CHECKERROR;
	}
    }
    else
    { 
	printf("\n!!!Unknown method in CSR format!!! Abort\n");
	assert(false);
    }
    errorCode = clSetKernelArg(kernels.csr_kernel, 0, sizeof(cl_mem), &gpumat.csr_row_ptr); CHECKERROR;
    errorCode = clSetKernelArg(kernels.csr_kernel, 1, sizeof(cl_mem), &gpumat.csr_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.csr_kernel, 2, sizeof(cl_mem), &gpumat.csr_data); CHECKERROR;
    if (gpumat.ifcsrtex)
    {
	errorCode = clSetKernelArg(kernels.csr_kernel, 3, sizeof(cl_mem), &gpumat.vec_tex_r); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.csr_kernel, 3, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.csr_kernel, 4, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.csr_kernel, 5, sizeof(int), &gpumat.csr_rownum); CHECKERROR;

    kernels.csr_block[0] = CSR_VEC_GROUP_SIZE;
    kernels.csr_block[1] = 1;
    kernels.csr_global[0] = CSR_VEC_GROUP_SIZE * cpumat.csr_group_num;
    kernels.csr_global[1] = 1;
}


void cpy_coo_mat(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue)
{
    cl_int errorCode = CL_SUCCESS;
    int maxgroupnum = cpumat.coo_group_num;
    int nnz = cpumat.coo.matinfo.nnz;
    int num_units = nnz / COO_GROUP_SIZE;
    if (nnz % COO_GROUP_SIZE != 0)
	num_units++;
    int group_num = (num_units < maxgroupnum) ? num_units : maxgroupnum;
    int work_size = group_num * COO_GROUP_SIZE;
    int num_iters = nnz / work_size;
    if (nnz % work_size != 0)
	num_iters++;
    int process_size = num_iters * COO_GROUP_SIZE;
    int active_warp = num_units / num_iters;
    if (num_units % num_iters != 0)
	active_warp++;
    int paddedNNZ = findPaddedSize(nnz, COO_ALIGNMENT);
    int* paddedRow = (int*)malloc(sizeof(int)*paddedNNZ);
    int* paddedCol = (int*)malloc(sizeof(int)*paddedNNZ);
    float* paddedData = (float*)malloc(sizeof(float)*paddedNNZ);
    memcpy(paddedRow, cpumat.coo.coo_row_id, sizeof(int)*nnz); 
    memcpy(paddedCol, cpumat.coo.coo_col_id, sizeof(int)*nnz); 
    memcpy(paddedData, cpumat.coo.coo_data, sizeof(float)*nnz); 
    for (int i = nnz; i < paddedNNZ; i++)
    {
	paddedRow[i] = cpumat.coo.coo_row_id[nnz - 1];
	paddedCol[i] = cpumat.coo.coo_col_id[nnz - 1];
	paddedData[i] = 0.0f;
    }

    ALLOCATE_GPU_READ(gpumat.coo_row_id, paddedRow, sizeof(int)*paddedNNZ);
    ALLOCATE_GPU_READ(gpumat.coo_col_id, paddedCol, sizeof(int)*paddedNNZ);
    ALLOCATE_GPU_READ(gpumat.coo_data, paddedData, sizeof(float)*paddedNNZ);
    gpumat.coo_tmp_row = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*maxgroupnum, NULL, &errorCode); CHECKERROR;
    gpumat.coo_tmp_data = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*maxgroupnum, NULL, &errorCode); CHECKERROR;
    gpumat.coo_process_size = process_size;
    gpumat.coo_paddednnz = paddedNNZ;
    gpumat.coo_activewarp = active_warp;
    gpumat.coo_group_num = group_num;
    clFinish(cmdQueue);
    free(paddedRow);
    free(paddedCol);
    free(paddedData);
}

void set_coo_kernel(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program)
{
    cl_int errorCode = CL_SUCCESS;
    if (gpumat.ifcootex)
    {
	kernels.coo_kernel_s1 = clCreateKernel(program, "gpu_coo_s1_tx", &errorCode); CHECKERROR;
    }
    else
    {
	kernels.coo_kernel_s1 = clCreateKernel(program, "gpu_coo_s1", &errorCode); CHECKERROR;
    }
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 0, sizeof(cl_mem), &gpumat.coo_row_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 1, sizeof(cl_mem), &gpumat.coo_col_id); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 2, sizeof(cl_mem), &gpumat.coo_data); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 3, sizeof(int),    &gpumat.coo_process_size); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 4, sizeof(int),    &gpumat.coo_paddednnz); CHECKERROR;
    if (gpumat.ifcootex)
    {
	errorCode = clSetKernelArg(kernels.coo_kernel_s1, 5, sizeof(cl_mem), &gpumat.vec_tex_r); CHECKERROR;
    }
    else
    {
	errorCode = clSetKernelArg(kernels.coo_kernel_s1, 5, sizeof(cl_mem), &gpumat.vec); CHECKERROR;
    }

    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 6, sizeof(cl_mem), &gpumat.res); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 7, sizeof(cl_mem), &gpumat.coo_tmp_row); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s1, 8, sizeof(cl_mem), &gpumat.coo_tmp_data); CHECKERROR;

    kernels.coo_block_s1[0] = COO_GROUP_SIZE;
    kernels.coo_block_s1[1] = 1;
    kernels.coo_global_s1[0] = gpumat.coo_group_num * COO_GROUP_SIZE;
    kernels.coo_global_s1[1] = 1;

    kernels.coo_kernel_s2 = clCreateKernel(program, "gpu_coo_s2", &errorCode); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s2, 0, sizeof(cl_mem), &gpumat.coo_tmp_row); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s2, 1, sizeof(cl_mem), &gpumat.coo_tmp_data); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s2, 2, sizeof(int), &gpumat.coo_activewarp); CHECKERROR;
    errorCode = clSetKernelArg(kernels.coo_kernel_s2, 3, sizeof(cl_mem), &gpumat.res); CHECKERROR;

    kernels.coo_block_s2[0] = COO_GROUP_SIZE * 2;
    kernels.coo_block_s2[1] = 1;
    kernels.coo_global_s2[0] = COO_GROUP_SIZE * 2;
    kernels.coo_global_s2[1] = 1;

}


void set_cocktail_kernels(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program, float* vec, int vec_size, float* res, int res_size)
{
    //Set the texture flag by checking dia kernel
    if (cpumat.ifusedia)
    {
	if (cpumat.dia_meth_num == 2)
	    gpumat.iftexr = true;
	if (cpumat.dia_meth_num == 3)
	    gpumat.iftexrgba = true;
    }

    //Copy the shared part to gpu
    cpy_shared(gpumat, context, cmdQueue, vec, vec_size, res, res_size);

    //Copy the extended vector to gpu
    if (cpumat.ifusebdia || cpumat.ifusedia)
	cpy_vec_extended(cpumat, gpumat, context, cmdQueue, vec, vec_size);
    
    if (cpumat.ifusebdia)
    {
	gpumat.ifusebdia = true;
	cpy_bdia_mat(cpumat, gpumat, context, cmdQueue);
	set_bdia_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusedia)
    {
	gpumat.ifusedia = true;
	cpy_dia_mat(cpumat, gpumat, context, cmdQueue);
	set_dia_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusesbell)
    {
	gpumat.ifusesbell = true;
	cpy_sbell_mat(cpumat, gpumat, context, cmdQueue);
	set_sbell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusebell)
    {
	gpumat.ifusebell = true;
	cpy_bell_mat(cpumat, gpumat, context, cmdQueue);
	set_bell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusebcsr)
    {
	gpumat.ifusebcsr = true;
	cpy_bcsr_mat(cpumat, gpumat, context, cmdQueue);
	set_bcsr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusesell)
    {
	gpumat.ifusesell = true;
	cpy_sell_mat(cpumat, gpumat, context, cmdQueue);
	set_sell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifuseell)
    {
	gpumat.ifuseell = true;
	cpy_ell_mat(cpumat, gpumat, context, cmdQueue);
	set_ell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusecsr)
    {
	gpumat.ifusecsr = true;
	cpy_csr_mat(cpumat, gpumat, context, cmdQueue);
	set_csr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusecoo)
    {
	gpumat.ifusecoo = true;
	cpy_coo_mat(cpumat, gpumat, context, cmdQueue);
	set_coo_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
	
}


void do_spmv(cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, int ntimes)
{
    cl_int errorCode = CL_SUCCESS;
    for (int i = 0; i < ntimes; i++)
    {
	if (kernels.bdia_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.bdia_kernel, 2, NULL, kernels.bdia_global, kernels.bdia_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.dia_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.dia_kernel, 2, NULL, kernels.dia_global, kernels.dia_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.sbell_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.sbell_kernel, 2, NULL, kernels.sbell_global, kernels.sbell_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.bell_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.bell_kernel, 2, NULL, kernels.bell_global, kernels.bell_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.bcsr_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.bcsr_kernel, 2, NULL, kernels.bcsr_global, kernels.bcsr_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.sell_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.sell_kernel, 2, NULL, kernels.sell_global, kernels.sell_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.ell_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.ell_kernel, 2, NULL, kernels.ell_global, kernels.ell_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.csr_kernel != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.csr_kernel, 2, NULL, kernels.csr_global, kernels.csr_block, 0, NULL, NULL); CHECKERROR;
	}
	if (kernels.coo_kernel_s1 != NULL)
	{
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.coo_kernel_s1, 2, NULL, kernels.coo_global_s1, kernels.coo_block_s1, 0, NULL, NULL); CHECKERROR;
	    errorCode = clEnqueueNDRangeKernel(cmdQueue, kernels.coo_kernel_s2, 2, NULL, kernels.coo_global_s2, kernels.coo_block_s2, 0, NULL, NULL); CHECKERROR;
	}
    }
    clFinish(cmdQueue);
}

void check_and_time(cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, float* result, float* coores, int res_size, int nnz)
{
    int ntimes = 1000;
    cl_int errorCode = CL_SUCCESS;
    //Check correctness
    errorCode = clEnqueueWriteBuffer(cmdQueue, gpumat.res, CL_TRUE, 0, sizeof(float)*res_size, result, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
    do_spmv(kernels, context, cmdQueue, 1);
    float* tmpresult = (float*)malloc(sizeof(float)*res_size);
    errorCode = clEnqueueReadBuffer(cmdQueue, gpumat.res, CL_TRUE, 0, sizeof(float)*res_size, tmpresult, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
    two_vec_compare(coores, tmpresult, res_size);
    free(tmpresult);

    double teststart = timestamp();
    do_spmv(kernels, context, cmdQueue, ntimes);
    double testend = timestamp();
    double time_in_sec = (testend - teststart);
    double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
    printf("\nCocktail cpu time %lf ms GFLOPS %lf\n\n",   time_in_sec / (double) ntimes * 1000, gflops);
}

double time_one_kernel(cl_kernel& kernel, size_t* blocksize, size_t* globalsize, cl_context& context, cl_command_queue& cmdQueue, int ntimes)
{
    cl_int errorCode = CL_SUCCESS;
    double teststart = timestamp();
    for (int i = 0; i < ntimes; i++)
    {
	errorCode = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalsize, blocksize, 0, NULL, NULL); CHECKERROR;
    }
    clFinish(cmdQueue);
    double testend = timestamp();
    double time_in_sec = (testend - teststart);
    //double gflops = (double)nnz*2/(time_in_sec/(double)ntimes)/(double)1e9;
    return time_in_sec;
}

void set_cocktail_with_tex(cocktail<int, int, float>& cpumat, cocktail_gpu& gpumat, cocktail_kernels& kernels, cl_context& context, cl_command_queue& cmdQueue, cl_program& program, float* vec, int vec_size, float* res, int res_size)
{
    int eval_times = 20;
    //Set texture flag
    gpumat.iftexr = true;
    gpumat.iftexrgba = true;

    //Copy the shared part to gpu
    cpy_shared(gpumat, context, cmdQueue, vec, vec_size, res, res_size);

    //Copy the extended vector to gpu
    if (cpumat.ifusebdia || cpumat.ifusedia)
	cpy_vec_extended(cpumat, gpumat, context, cmdQueue, vec, vec_size);
    
    if (cpumat.ifusebdia)
    {
	gpumat.ifusebdia = true;
	cpy_bdia_mat(cpumat, gpumat, context, cmdQueue);
	set_bdia_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusedia)
    {
	gpumat.ifusedia = true;
	cpy_dia_mat(cpumat, gpumat, context, cmdQueue);
	set_dia_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusesbell)
    {
	gpumat.ifusesbell = true;
	cpy_sbell_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifsbelltex = false;
	set_sbell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.sbell_kernel, kernels.sbell_block, kernels.sbell_global, context, cmdQueue, eval_times);
	if (kernels.sbell_kernel)
	    clReleaseKernel(kernels.sbell_kernel);
	gpumat.ifsbelltex = true;
	set_sbell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.sbell_kernel, kernels.sbell_block, kernels.sbell_global, context, cmdQueue, eval_times);
	if (kernels.sbell_kernel)
	    clReleaseKernel(kernels.sbell_kernel);
	if (withtextime < notextime)
	    gpumat.ifsbelltex = true;
	else
	    gpumat.ifsbelltex = false;
	printf("sbell no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_sbell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusebell)
    {
	gpumat.ifusebell = true;
	cpy_bell_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifbelltex = false;
	set_bell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.bell_kernel, kernels.bell_block, kernels.bell_global, context, cmdQueue, eval_times);
	if (kernels.bell_kernel)
	    clReleaseKernel(kernels.bell_kernel);
	gpumat.ifbelltex = true;
	set_bell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.bell_kernel, kernels.bell_block, kernels.bell_global, context, cmdQueue, eval_times);
	if (kernels.bell_kernel)
	    clReleaseKernel(kernels.bell_kernel);
	if (withtextime < notextime)
	    gpumat.ifbelltex = true;
	else
	    gpumat.ifbelltex = false;
	printf("bell no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_bell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusebcsr)
    {
	gpumat.ifusebcsr = true;
	cpy_bcsr_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifbcsrtex = false;
	set_bcsr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.bcsr_kernel, kernels.bcsr_block, kernels.bcsr_global, context, cmdQueue, eval_times);
	if (kernels.bcsr_kernel)
	    clReleaseKernel(kernels.bcsr_kernel);
	gpumat.ifbcsrtex = true;
	set_bcsr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.bcsr_kernel, kernels.bcsr_block, kernels.bcsr_global, context, cmdQueue, eval_times);
	if (kernels.bcsr_kernel)
	    clReleaseKernel(kernels.bcsr_kernel);
	if (withtextime < notextime)
	    gpumat.ifbcsrtex = true;
	else
	    gpumat.ifbcsrtex = false;
	printf("bcsr no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_bcsr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusesell)
    {
	gpumat.ifusesell = true;
	cpy_sell_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifselltex = false;
	set_sell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.sell_kernel, kernels.sell_block, kernels.sell_global, context, cmdQueue, eval_times);
	if (kernels.sell_kernel)
	    clReleaseKernel(kernels.sell_kernel);
	gpumat.ifselltex = true;
	set_sell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.sell_kernel, kernels.sell_block, kernels.sell_global, context, cmdQueue, eval_times);
	if (kernels.sell_kernel)
	    clReleaseKernel(kernels.sell_kernel);
	if (withtextime < notextime)
	    gpumat.ifselltex = true;
	else
	    gpumat.ifselltex = false;
	printf("sell no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_sell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifuseell)
    {
	gpumat.ifuseell = true;
	cpy_ell_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifelltex = false;
	set_ell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.ell_kernel, kernels.ell_block, kernels.ell_global, context, cmdQueue, eval_times);
	if (kernels.ell_kernel)
	    clReleaseKernel(kernels.ell_kernel);
	gpumat.ifelltex = true;
	set_ell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.ell_kernel, kernels.ell_block, kernels.ell_global, context, cmdQueue, eval_times);
	if (kernels.ell_kernel)
	    clReleaseKernel(kernels.ell_kernel);
	if (withtextime < notextime)
	    gpumat.ifelltex = true;
	else
	    gpumat.ifelltex = false;
	printf("ell no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_ell_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusecsr)
    {
	gpumat.ifusecsr = true;
	cpy_csr_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifcsrtex = false;
	set_csr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.csr_kernel, kernels.csr_block, kernels.csr_global, context, cmdQueue, eval_times);
	if (kernels.csr_kernel)
	    clReleaseKernel(kernels.csr_kernel);
	gpumat.ifcsrtex = true;
	set_csr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.csr_kernel, kernels.csr_block, kernels.csr_global, context, cmdQueue, eval_times);
	if (kernels.csr_kernel)
	    clReleaseKernel(kernels.csr_kernel);
	if (withtextime < notextime)
	    gpumat.ifcsrtex = true;
	else
	    gpumat.ifcsrtex = false;
	printf("csr no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_csr_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
    if (cpumat.ifusecoo)
    {
	gpumat.ifusecoo = true;
	cpy_coo_mat(cpumat, gpumat, context, cmdQueue);
	gpumat.ifcootex = false;
	set_coo_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double notextime = time_one_kernel(kernels.coo_kernel_s1, kernels.coo_block_s1, kernels.coo_global_s1, context, cmdQueue, eval_times);
	if (kernels.coo_kernel_s1)
	    clReleaseKernel(kernels.coo_kernel_s1);
	if (kernels.coo_kernel_s2)
	    clReleaseKernel(kernels.coo_kernel_s2);
	gpumat.ifcootex = true;
	set_coo_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
	double withtextime = time_one_kernel(kernels.coo_kernel_s1, kernels.coo_block_s1, kernels.coo_global_s1, context, cmdQueue, eval_times);
	if (kernels.coo_kernel_s1)
	    clReleaseKernel(kernels.coo_kernel_s1);
	if (kernels.coo_kernel_s2)
	    clReleaseKernel(kernels.coo_kernel_s2);
	if (withtextime < notextime)
	    gpumat.ifcootex = true;
	else
	    gpumat.ifcootex = false;
	printf("coo no tex time %f ms tex time %f ms use %s \n", notextime*1000.0, withtextime*1000.0, (withtextime < notextime) ? ("Tex") : ("NoTex"));
	set_coo_kernel(cpumat, gpumat, kernels, context, cmdQueue, program);
    }
	
}

void output_cocktail_info(cocktail<int, int, float>& cpumat)
{
    printf("\n*****************************\n");
    printf("Cocktail matrix info\n");
    printf("*****************************\n");
    printf("Original Matrix Width %d Height %d nnz %d\n\n", cpumat.mat_width, cpumat.mat_height, cpumat.mat_nnz);
    if (cpumat.ifusebdia)
    {
	printf("Use bdia\n");
	printf("Implementation no. %d\n", cpumat.bdia_meth_num);
	int bandnum = cpumat.bdia.bdia_band_num;
	printf("NNZ %d\n", cpumat.bdia.matinfo.nnz);
	printf("Band num %d total diagonals %d\n", bandnum, cpumat.bdia.bdia_bptr[bandnum]);
	printf("------------------------------\n");
    }
    if (cpumat.ifusedia)
    {
	printf("Use dia\n");
	printf("Implementation no. %d\n", cpumat.dia_meth_num);
	printf("NNZ %d\n", cpumat.dia.matinfo.nnz);
	printf("Num of diagonals %d\n", cpumat.dia.dia_num);
	printf("------------------------------\n");
    }
    if (cpumat.ifusesbell)
    {
	printf("Use sbell\n");
	printf("Implementation no. %d\n", cpumat.sbell_meth_num);
	printf("NNZ %d\n", cpumat.sbell.matinfo.nnz);
	printf("block height %d block width %d slice height %d\n", cpumat.sbell.sbell_bheight, cpumat.sbell.sbell_bwidth, cpumat.sbell.sbell_slice_height);
	printf("------------------------------\n");
    }
    if (cpumat.ifusebell)
    {
	printf("Use bell\n");
	printf("Implementation no. %d\n", cpumat.bell_meth_num);
	printf("NNZ %d\n", cpumat.bell.matinfo.nnz);
	printf("block height %d block width %d block num %d\n", cpumat.bell.b4ell_bheight, cpumat.bell.b4ell_bwidth, cpumat.bell.b4ell_block_num);
	printf("------------------------------\n");
    }
    if (cpumat.ifusebcsr)
    {
	printf("Use bcsr\n");
	printf("Implementation no. %d\n", cpumat.bcsr_meth_num);
	printf("NNZ %d\n", cpumat.bcsr.matinfo.nnz);
	printf("block height %d block width %d total blocks %d\n", cpumat.bcsr.b4csr_bheight, cpumat.bcsr.b4csr_bwidth, cpumat.bcsr.b4csr_block_num);
	printf("------------------------------\n");
    }
    if (cpumat.ifusesell)
    {
	printf("Use sell\n");
	printf("Implementation no. %d\n", cpumat.sell_meth_num);
	printf("NNZ %d\n", cpumat.sell.matinfo.nnz);
	printf("slice height %d\n", cpumat.sell.sell_slice_height);
	printf("------------------------------\n");
    }
    if (cpumat.ifuseell)
    {
	printf("Use ell\n");
	printf("Implementation no. %d\n", cpumat.ell_meth_num);
	printf("NNZ %d\n", cpumat.ell.matinfo.nnz);
	printf("ELL num %d\n", cpumat.ell.ell_num);
	printf("------------------------------\n");
    }
    if (cpumat.ifusecsr)
    {
	printf("Use csr\n");
	printf("Implementation no. %d\n", cpumat.csr_meth_num);
	printf("NNZ %d\n", cpumat.csr.matinfo.nnz);
	printf("------------------------------\n");
    }
    if (cpumat.ifusecoo)
    {
	printf("Use coo\n");
	printf("Implementation no. %d\n", cpumat.coo_meth_num);
	printf("NNZ %d\n", cpumat.coo.matinfo.nnz);
	printf("------------------------------\n");
    }

}

void evaluate(cocktail<int, int, float>& cpumat, float* vec, float* res, float* coores)
{
    //printf("clmem size %d int size %d float size %d\n", sizeof(cl_mem), sizeof(int), sizeof(float));
    output_cocktail_info(cpumat);
    cocktail_gpu gpumat;
    cocktail_kernels kernels;
    init_cocktail_gpu(gpumat);
    init_cocktail_kernels(kernels);

    char* clspmvpath = getenv("CLSPMVPATH");
    char oclfilename[1000];
    sprintf(oclfilename, "%s%s", clspmvpath, "/kernels/kernel_all.cl");
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    assert(initialization(deviceType, kernels.devices, &kernels.context, &kernels.cmdQueue, &kernels.program, oclfilename) == 1);


    //set_cocktail_kernels(cpumat, gpumat, kernels, kernels.context, kernerls.cmdQueue, kernels.program, vec, cpumat.mat_width, res, cpumat.mat_height);
    set_cocktail_with_tex(cpumat, gpumat, kernels, kernels.context, kernels.cmdQueue, kernels.program, vec, cpumat.mat_width, res, cpumat.mat_height);
    check_and_time(gpumat, kernels, kernels.context, kernels.cmdQueue, res, coores, cpumat.mat_height, cpumat.mat_nnz); 

    free_cocktail_kernels(kernels);
    free_cocktail_gpu(gpumat);
    freeObjects(kernels.devices, &kernels.context, &kernels.cmdQueue, &kernels.program);
}


//Assumption: either set_cocktail_kernels or set_cocktail_with_tex is called, so all the OpenCL memory objects are created, do not need to create objects again, just do the copy
void cpy_vector_from_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* vec, int vec_size)
{
    cl_int errorCode = CL_SUCCESS;
    const cl_image_format floatFormat =
    {
	CL_R,
	CL_FLOAT,
    };
    const cl_image_format float4Format =
    {
	CL_RGBA,
	CL_FLOAT,
    };
    
    int width = VEC2DWIDTH;
    int height = (vec_size + VEC2DWIDTH - 1)/VEC2DWIDTH;
    if (height % 4 != 0)
	height += (4 - (height % 4));

    errorCode = clEnqueueWriteBuffer(cmdQueue, gpumat.vec, CL_TRUE, 0, sizeof(float)*vec_size, vec, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);

    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height, 1};
    size_t vector4Size[] = {width, height/4, 1};
    if (gpumat.iftexr || gpumat.iftexrgba)
    {
	errorCode = clEnqueueCopyBuffer(cmdQueue, gpumat.vec, gpumat.vec_image, 0, 0, sizeof(float)*vec_size, 0, NULL, NULL);
	clFinish(cmdQueue);
    }
    if (gpumat.iftexr)
    {
	errorCode = clEnqueueCopyBufferToImage(cmdQueue, gpumat.vec_image, gpumat.vec_tex_r, 0, origin, vectorSize, 0, NULL, NULL); CHECKERROR;
    }
    if (gpumat.iftexrgba)
    {
	errorCode = clEnqueueCopyBufferToImage(cmdQueue, gpumat.vec_image, gpumat.vec_tex_rgba, 0, origin, vector4Size, 0, NULL, NULL); CHECKERROR;
    }

    if (gpumat.ifusebdia || gpumat.ifusedia)
    {
	int leftoffset = gpumat.vec_offset;
	errorCode = clEnqueueCopyBuffer(cmdQueue, gpumat.vec, gpumat.vec_extended, 0, leftoffset, sizeof(float)*vec_size, 0, NULL, NULL);
	clFinish(cmdQueue);
    }

    clFinish(cmdQueue);
}


//Assumption: either set_cocktail_kernels or set_cocktail_with_tex is called, so all the OpenCL memory objects are created, do not need to create objects again, just do the copy
void cpy_vector_from_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem vec, int vec_size)
{
    cl_int errorCode = CL_SUCCESS;
    const cl_image_format floatFormat =
    {
	CL_R,
	CL_FLOAT,
    };
    const cl_image_format float4Format =
    {
	CL_RGBA,
	CL_FLOAT,
    };
    
    errorCode = clEnqueueCopyBuffer(cmdQueue, vec, gpumat.vec, 0, 0, sizeof(float)*vec_size, 0, NULL, NULL);
    clFinish(cmdQueue);

    int width = VEC2DWIDTH;
    int height = (vec_size + VEC2DWIDTH - 1)/VEC2DWIDTH;
    if (height % 4 != 0)
	height += (4 - (height % 4));
    size_t origin[] = {0, 0, 0};
    size_t vectorSize[] = {width, height, 1};
    size_t vector4Size[] = {width, height/4, 1};
    if (gpumat.iftexr || gpumat.iftexrgba)
    {
	errorCode = clEnqueueCopyBuffer(cmdQueue, gpumat.vec, gpumat.vec_image, 0, 0, sizeof(float)*vec_size, 0, NULL, NULL);
	clFinish(cmdQueue);
    }
    if (gpumat.iftexr)
    {
	errorCode = clEnqueueCopyBufferToImage(cmdQueue, gpumat.vec_image, gpumat.vec_tex_r, 0, origin, vectorSize, 0, NULL, NULL); CHECKERROR;
    }
    if (gpumat.iftexrgba)
    {
	errorCode = clEnqueueCopyBufferToImage(cmdQueue, gpumat.vec_image, gpumat.vec_tex_rgba, 0, origin, vector4Size, 0, NULL, NULL); CHECKERROR;
    }

    if (gpumat.ifusebdia || gpumat.ifusedia)
    {
	int leftoffset = gpumat.vec_offset;
	errorCode = clEnqueueCopyBuffer(cmdQueue, gpumat.vec, gpumat.vec_extended, 0, leftoffset, sizeof(float)*vec_size, 0, NULL, NULL);
	clFinish(cmdQueue);
    }

    clFinish(cmdQueue);
}

//Assumption: either set_cocktail_kernels or set_cocktail_with_tex is called, so all the OpenCL memory objects are created, do not need to create objects again, just do the copy
void cpy_result_from_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* res, int res_size)
{
    cl_int errorCode = CL_SUCCESS;
    errorCode = clEnqueueWriteBuffer(cmdQueue, gpumat.res, CL_TRUE, 0, sizeof(float)*res_size, res, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
}


//Assumption: either set_cocktail_kernels or set_cocktail_with_tex is called, so all the OpenCL memory objects are created, do not need to create objects again, just do the copy
void cpy_result_from_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem res, int res_size)
{
    cl_int errorCode = CL_SUCCESS;
    errorCode = clEnqueueCopyBuffer(cmdQueue, res, gpumat.res, 0, 0, sizeof(float)*res_size, 0, NULL, NULL);
    clFinish(cmdQueue);
}

void cpy_result_to_cpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, float* res, int res_size)
{
    cl_int errorCode = CL_SUCCESS;
    errorCode = clEnqueueReadBuffer(cmdQueue, gpumat.res, CL_TRUE, 0, sizeof(float)*res_size, res, 0, NULL, NULL); CHECKERROR;
    clFinish(cmdQueue);
}

void cpy_result_to_gpu(cocktail_gpu& gpumat, cl_context& context, cl_command_queue& cmdQueue, cl_mem res, int res_size)
{
    cl_int errorCode = CL_SUCCESS;
    errorCode = clEnqueueCopyBuffer(cmdQueue, gpumat.res, res, 0, 0, sizeof(float)*res_size, 0, NULL, NULL);
    clFinish(cmdQueue);
}


void init_mat_kernels(cocktail<int, int, float>& cpumat, float* vec, float* res, cocktail_gpu& gpumat, cocktail_kernels& kernels, bool use_tex)
{
    init_cocktail_gpu(gpumat);
    init_cocktail_kernels(kernels);
    
    char* clspmvpath = getenv("CLSPMVPATH");
    char oclfilename[1000];
    sprintf(oclfilename, "%s%s", clspmvpath, "/kernels/kernel_all.cl");
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;

    assert(initialization(deviceType, kernels.devices, &kernels.context, &kernels.cmdQueue, &kernels.program, oclfilename) == 1);

    if (use_tex)
    {
	set_cocktail_with_tex(cpumat, gpumat, kernels, kernels.context, kernels.cmdQueue, kernels.program, vec, cpumat.mat_width, res, cpumat.mat_height);
    }
    else
    {
	set_cocktail_kernels(cpumat, gpumat, kernels, kernels.context, kernels.cmdQueue, kernels.program, vec, cpumat.mat_width, res, cpumat.mat_height);
    }
    
}

void free_mat_kernels(cocktail_gpu& gpumat, cocktail_kernels& kernels)
{
    free_cocktail_kernels(kernels);
    free_cocktail_gpu(gpumat);
    freeObjects(kernels.devices, &kernels.context, &kernels.cmdQueue, &kernels.program);
}


