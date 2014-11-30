#include "constant.h"


//Pad matrix
__kernel void gpu_csr_sc_pm(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num)
{
    
    int row = get_global_id(0);
    
    //if (row < row_num)
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	for (int i = start; i < end; i++)
	{
	    int col = colid[i];
	    sum = mad(data[i], vec[col], sum);
	}
	result[row] = sum;
    }
}

//Pad matrix unroll 4
__kernel void gpu_csr_sc_pm_u4(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num)
{
    
    int row = get_global_id(0);
    
    //if (row < row_num)
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	int end4 = end - (end - start) % 4;
	int i = start;
	for (; i < end4; )
	{
	    int col = colid[i];
	    sum = mad(data[i], vec[col], sum);
	    i++;
	    col = colid[i];
	    sum = mad(data[i], vec[col], sum);
            i++;
	    col = colid[i];
	    sum = mad(data[i], vec[col], sum);
            i++;
	    col = colid[i];
	    sum = mad(data[i], vec[col], sum);
            i++;
	}
	for (; i < end; i++)
	{
	    int col = colid[i];
	    sum = mad(data[i], vec[col], sum);
	}
	result[row] = sum;
    }
}


//Pad matrix float4 noif 
__kernel void gpu_csr_sc_pm_float4_noif(__global int* rowptr,  __global int* colid, __global float* data, __global float* vec, __global float* result, int row_num)
{
    
    int row = get_global_id(0);
    
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	for (int i = start; i < end; i+=4)
	{
            float4 data4 = vload4(0, data+i);
            int4 col4 = vload4(0, colid + i);
	    sum = mad(data4.x, vec[col4.x], sum);
	    sum = mad(data4.y, vec[col4.y], sum);
	    sum = mad(data4.z, vec[col4.z], sum);
	    sum = mad(data4.w, vec[col4.w], sum);
	}
	result[row] = sum;
    }
}


__kernel void gpu_csr_sc_pm_tx(__global int* rowptr,  __global int* colid, __global float* data, __read_only image2d_t vec, __global float* result, int row_num)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int row = get_global_id(0);
    
    //if (row < row_num)
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	for (int i = start; i < end; i++)
	{
	    int col = colid[i];
		int2 coord;
		coord.x = col & WIDTH_MASK;
		coord.y = col >> LOG_WIDTH;
		float4 ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
	}
	result[row] = sum;
    }
}

//Pad matrix unroll 4
__kernel void gpu_csr_sc_pm_u4_tx(__global int* rowptr,  __global int* colid, __global float* data, __read_only image2d_t vec, __global float* result, int row_num)
{
    
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int row = get_global_id(0);
    
    //if (row < row_num)
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	int end4 = end - (end - start) % 4;
	int i = start;
	for (; i < end4; )
	{
	    int col = colid[i];
	    int2 coord;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    float4 ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
            i++;
	    col = colid[i];
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
            i++;
	    col = colid[i];
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
            i++;
	    col = colid[i];
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
            i++;
	}
        for (;i < end; i++)
	{            
	    int col = colid[i];
	    int2 coord;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    float4 ans = read_imagef(vec, smp, coord);
	    sum = mad(data[i], ans.x, sum);
	}
	result[row] = sum;
    }
}

__kernel void gpu_csr_sc_pm_float4_noif_tx(__global int* rowptr,  __global int* colid, __global float* data, __read_only image2d_t vec, __global float* result, int row_num)
{
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int row = get_global_id(0);
    
    //if (row < row_num)
    {
	float sum = result[row];
	int start = rowptr[row];
	int end = rowptr[row+1];
	for (int i = start; i < end; i+=4)
	{
        float4 data4 = vload4(0, data + i);
        int4 col4 = vload4(0, colid + i);
		int2 coord;
		coord.x = col4.x & WIDTH_MASK;
		coord.y = col4.x >> LOG_WIDTH;
		float4 ans = read_imagef(vec, smp, coord);
	    sum = mad(data4.x, ans.x, sum);
		coord.x = col4.y & WIDTH_MASK;
		coord.y = col4.y >> LOG_WIDTH;
		ans = read_imagef(vec, smp, coord);
	    sum = mad(data4.y, ans.x, sum);
		coord.x = col4.z & WIDTH_MASK;
		coord.y = col4.z >> LOG_WIDTH;
		ans = read_imagef(vec, smp, coord);
	    sum = mad(data4.z, ans.x, sum);
		coord.x = col4.w & WIDTH_MASK;
		coord.y = col4.w >> LOG_WIDTH;
		ans = read_imagef(vec, smp, coord);
	    sum = mad(data4.w, ans.x, sum);
	}
	result[row] = sum;
    }
}


