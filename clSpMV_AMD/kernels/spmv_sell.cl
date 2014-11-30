#include "constant.h"


__kernel void gpu_sell_warp(__global int* slice_ptr, __global int* col_id, __global float* data, __global float* vec, __global float* result, int slice_num)
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
    float accumulant = result[row];
    
    for (int i = start; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	float matrixelem = data[i];
	float vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
    }
    
    result[row] = accumulant;
}

__kernel void gpu_sell_group(__global int* slice_ptr, __global int* col_id, __global float* data, __global float* vec, __global float* result, int slice_num)
{
    __local int lslice_ptr[2];
    int row = get_global_id(0);
    int size = get_local_size(0);
    int sliceid = row / size;
    int localid = get_local_id(0);
    if (localid < 2)
    {
	lslice_ptr[localid] = slice_ptr[sliceid + localid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int start = lslice_ptr[0] + localid;
    int end = lslice_ptr[1];
    float accumulant = result[row];
    for (int i = start; i < end; i += size)
    {
	int vecid = col_id[i];
	float matrixelem = data[i];
	float vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
    }
    
    result[row] = accumulant;
}

__kernel void gpu_sell_warp_tx(__global int* slice_ptr, __global int* col_id, __global float* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    __local int lslice_ptr[SELL_GROUP_SIZE / WARPSIZE][2];
    int row = get_global_id(0);
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
    float accumulant = result[row];
    for (int i = start; i < end; i += WARPSIZE)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float matrixelem = data[i];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem.x, accumulant);
    }
    
    result[row] = accumulant;
}

__kernel void gpu_sell_group_tx(__global int* slice_ptr, __global int* col_id, __global float* data, __read_only image2d_t vec, __global float* result, int slice_num)
{
    __local int lslice_ptr[2];
    int row = get_global_id(0);
    int size = get_local_size(0);
    int sliceid = row / size;
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    if (localid < 2)
    {
	lslice_ptr[localid] = slice_ptr[sliceid + localid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int start = lslice_ptr[0] + localid;
    int end = lslice_ptr[1];
    float accumulant = result[row];
    for (int i = start; i < end; i += size)
    {
	int vecid = col_id[i];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float matrixelem = data[i];
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem.x, accumulant);
    }
    
    result[row] = accumulant;
}

