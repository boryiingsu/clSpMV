#include "constant.h"


__kernel void gpu_dia(__global int* offset, __global float* data, int aligned_length, int dia_num, __global float* vec, __global float* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_offset[MAX_DIA_NUM];
    int local_id = get_local_id(0);
    if (local_id < dia_num)
    {
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < dia_num; i++)
    {
	int diaoffset = l_offset[i];
	int vecid = row + vecoffset + diaoffset;
	float matrixelem = data[matoffset];
	float vecelem = vec[vecid];
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;

}

__kernel void gpu_dia_v4(__global int* offset, __global float4* data, int aligned_length, int dia_num, __global float* vec, __global float4* result, int vecoffset)
{
    int row = get_global_id(0);
    __local int l_offset[MAX_DIA_NUM];
    int local_id = get_local_id(0);
    if (local_id < dia_num)
    {
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < dia_num; i++)
    {
	int diaoffset = l_offset[i];
	int vecid = row * 4 + vecoffset + diaoffset;
	float4 matrixelem = data[matoffset];
	float4 vecelem = vload4(0, vec + vecid);
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;

}

__kernel void gpu_dia_tx(__global int* offset, __global float* data, int aligned_length, int dia_num, __read_only image2d_t vec, __global float* result, int vecoffset)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int l_offset[MAX_DIA_NUM];
    int local_id = get_local_id(0);
    if (local_id < dia_num)
    {
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < dia_num; i++)
    {
	int diaoffset = l_offset[i];
	int vecid = row + diaoffset;
	float matrixelem = data[matoffset];
	int2 coord;
	coord.x = vecid & WIDTH_MASK;
	coord.y = vecid >> LOG_WIDTH;
	float4 vecelem = read_imagef(vec, smp, coord);
	accumulant = mad(matrixelem, vecelem.x, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;
}

__kernel void gpu_dia_v4_tx(__global int* offset, __global float4* data, int aligned_length, int dia_num, __read_only image2d_t vec, __global float4* result, int vecoffset)
{
    int row = get_global_id(0);
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    __local int l_offset[MAX_DIA_NUM];
    int local_id = get_local_id(0);
    if (local_id < dia_num)
    {
	l_offset[local_id] = offset[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 accumulant = result[row];
    int matoffset = row;
    for (int i = 0; i < dia_num; i++)
    {
	int diaoffset = l_offset[i];
	int vecid = row * 4 + diaoffset;
	float4 matrixelem = data[matoffset];
	int alignment = vecid % 4;
	if (alignment < 0)
	    alignment += 4;
	vecid = vecid >> 2;
	float4 vecelem;
	if (alignment == 0)
	{
	    int2 coord;
	    coord.x = vecid & WIDTH_MASK;
	    coord.y = vecid >> LOG_WIDTH;
	    vecelem = read_imagef(vec, smp, coord);
	}
	else if (alignment == 1)
	{
	    int2 coord;
	    coord.x = vecid & WIDTH_MASK;
	    coord.y = vecid >> LOG_WIDTH;
	    vecid++;
	    int2 coord1;
	    coord1.x = vecid & WIDTH_MASK;
	    coord1.y = vecid >> LOG_WIDTH;
	    float4 tmp1 = read_imagef(vec, smp, coord);
	    float4 tmp2 = read_imagef(vec, smp, coord1);
	    vecelem.xyz = tmp1.yzw;
	    vecelem.w = tmp2.x;
	}
	else if (alignment == 2)
	{
	    int2 coord;
	    coord.x = vecid & WIDTH_MASK;
	    coord.y = vecid >> LOG_WIDTH;
	    vecid++;
	    int2 coord1;
	    coord1.x = vecid & WIDTH_MASK;
	    coord1.y = vecid >> LOG_WIDTH;
	    float4 tmp1 = read_imagef(vec, smp, coord);
	    float4 tmp2 = read_imagef(vec, smp, coord1);
	    vecelem.xy = tmp1.zw;
	    vecelem.zw = tmp2.xy;
	}
	else if (alignment == 3)
	{
	    int2 coord;
	    coord.x = vecid & WIDTH_MASK;
	    coord.y = vecid >> LOG_WIDTH;
	    vecid++;
	    int2 coord1;
	    coord1.x = vecid & WIDTH_MASK;
	    coord1.y = vecid >> LOG_WIDTH;
	    float4 tmp1 = read_imagef(vec, smp, coord);
	    float4 tmp2 = read_imagef(vec, smp, coord1);
	    vecelem.x = tmp1.w;
	    vecelem.yzw = tmp2.xyz;
	}
	accumulant = mad(matrixelem, vecelem, accumulant);
	matoffset += aligned_length;
    }
    result[row] = accumulant;

}

