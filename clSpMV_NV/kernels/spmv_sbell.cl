#include "constant.h"

// The slice sice is equal to the warp size

__kernel void gpu_sbell14(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0] + laneid;
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    for (int i = start; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[i];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}


__kernel void gpu_sbell14_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0] + laneid;
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    for (int i = start; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[i];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}


__kernel void gpu_sbell24(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    row *= 2;
    accumulant.x = result[row];
    accumulant2.x = result[row + 1];
    int matoffset = start * 2 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}


__kernel void gpu_sbell24_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    row *= 2;
    accumulant.x = result[row];
    accumulant2.x = result[row + 1];
    int matoffset = start * 2 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}


__kernel void gpu_sbell44(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = result[row];
    int matoffset = start * 4 + laneid;
    
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant;
}


__kernel void gpu_sbell44_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = result[row];
    int matoffset = start * 4 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant;
}


__kernel void gpu_sbell84(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    row *= 2;
    float4 accumulant = result[row];
    float4 accumulant2 = result[row + 1];
    int matoffset = start * 8 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant;
    result[row + 1] = accumulant2;
}


__kernel void gpu_sbell84_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    row *= 2;
    float4 accumulant = result[row];
    float4 accumulant2 = result[row + 1];
    int matoffset = start * 8 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant;
    result[row + 1] = accumulant2;
}


__kernel void gpu_sbell18(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int matoffset = start * 2 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	
	vecelem = vec[vecid + 1];
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}


__kernel void gpu_sbell18_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulant.x = result[row];
    int matoffset = start * 2 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);

	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
}

__kernel void gpu_sbell28(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    row *= 2;
    accumulant.x = result[row];
    accumulant2.x = result[row + 1];
    int matoffset = start * 4 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);

	vecelem = vec[vecid + 1];
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}


__kernel void gpu_sbell28_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 accumulant2 = {0.0f, 0.0f, 0.0f, 0.0f};
    row *= 2;
    accumulant.x = result[row];
    accumulant2.x = result[row + 1];
    int matoffset = start * 4 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);

	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant = mad(matrixelem, vecelem, accumulant);

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	accumulant2 = mad(matrixelem, vecelem, accumulant2);
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant.x + accumulant.y + accumulant.z + accumulant.w;
    result[row + 1] = accumulant2.x + accumulant2.y + accumulant2.z + accumulant2.w;
}

__kernel void gpu_sbell48(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = result[row];
    int matoffset = start * 8 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	vecelem = vec[vecid + 1];
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant;
}


__kernel void gpu_sbell48_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    float4 accumulant = result[row];
    int matoffset = start * 8 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;

	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant;
}

__kernel void gpu_sbell88(__global int* slice_ptr, __global int* col_id, __global float4* data, __global float4* vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    row *= 2;
    float4 accumulant = result[row];
    float4 accumulant2 = result[row + 1];
    int matoffset = start * 16 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float4 matrixelem = data[matoffset];
	float4 vecelem = vec[vecid];
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;

	vecelem = vec[vecid + 1];
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    
    result[row] = accumulant;
    result[row + 1] = accumulant2;
}


__kernel void gpu_sbell88_tx(__global int* slice_ptr, __global int* col_id, __global float4* data, __read_only image2d_t vec, __global float4* result, int slice_num)
{
    int row = get_global_id(0);
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int sliceid = row / WARPSIZE;
    if (sliceid >= slice_num)
	return;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int warpid = localid / WARPSIZE;
    int laneid = localid % WARPSIZE;
    if (laneid < 2)
    {
	lslice_ptr[warpid][laneid] = slice_ptr[sliceid + laneid];
    }

    int start = lslice_ptr[warpid][0];
    int end = lslice_ptr[warpid][1];
    row *= 2;
    float4 accumulant = result[row];
    float4 accumulant2 = result[row + 1];
    int matoffset = start * 16 + laneid;
    for (int i = start + laneid; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 matrixelem = data[matoffset];
	float4 vecelem = read_imagef(vec, smp, coord);
	float4 tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	vecid++;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	vecelem = read_imagef(vec, smp, coord);
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.x = accumulant.x + tmp.x + tmp.y + tmp.z + tmp.w;

	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.y = accumulant.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.z = accumulant.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant.w = accumulant.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.x = accumulant2.x + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.y = accumulant2.y + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.z = accumulant2.z + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
	matrixelem = data[matoffset];
	tmp = matrixelem * vecelem;
	accumulant2.w = accumulant2.w + tmp.x + tmp.y + tmp.z + tmp.w;
	
	matoffset += WARPSIZE;
    }
    result[row] = accumulant;
    result[row + 1] = accumulant2;
}

