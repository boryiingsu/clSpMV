#include "constant.h"


__kernel void gpu_coo_s1(__global int* row_id, __global int* col_id, __global float* data, int process_size, int nnz, __global float* vec, __global float* result, __global int* tmp_rowid, __global float* tmp_data)
{
    __local int l_rowid[COO_GROUP_SIZE];
    __local float l_data[COO_GROUP_SIZE];
    int localid = get_local_id(0);
    int global_warpid = get_group_id(0);
    int start = global_warpid * process_size;
    int gsize = get_local_size(0);
    int end = start + process_size;
    if (end > nnz)
	end = nnz;
    int iter_num = (end - start) / gsize;

    if (localid == gsize - 1)
    {
	l_rowid[localid] = row_id[start];
	l_data[localid] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int lastlocalid = localid + gsize - 1;
    int i = start + localid;
    for (int iter = 0; iter < iter_num; iter++)
    {
	int nzrow = 0;
	float val = 0.0f;
	if (i < nnz)
	{
	    nzrow = row_id[i];
	    val = data[i] * vec[col_id[i]];
	}
	if (localid == 0)
	{
	    if (nzrow == l_rowid[lastlocalid])
	    {
		val += l_data[lastlocalid];
	    }
	    else
	    {
		result[l_rowid[lastlocalid]] += l_data[lastlocalid];
	    }
	}
	l_rowid[localid] = nzrow;
	l_data[localid] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 1 && nzrow == l_rowid[localid - 1])
	    l_data[localid] = val = val + l_data[localid - 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 2 && nzrow == l_rowid[localid - 2])
	   l_data[localid] = val = val + l_data[localid - 2];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 4 && nzrow == l_rowid[localid - 4])
	    l_data[localid] = val = val + l_data[localid - 4];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 8 && nzrow == l_rowid[localid - 8])
	    l_data[localid] = val = val + l_data[localid - 8];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 16 && nzrow == l_rowid[localid - 16])
	    l_data[localid] = val = val + l_data[localid - 16];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 32 && nzrow == l_rowid[localid - 32])
	    l_data[localid] = val = val + l_data[localid - 32];
	barrier(CLK_LOCAL_MEM_FENCE);
	   
	if (localid < (gsize - 1) && nzrow != l_rowid[localid + 1])
	    result[nzrow] += val;	
	barrier(CLK_LOCAL_MEM_FENCE);
	    
	i += gsize;
	
    }

    if (localid == gsize - 1)
    {
	tmp_rowid[global_warpid] = l_rowid[localid];
	tmp_data[global_warpid] = l_data[localid];
    }

}


__kernel void gpu_coo_s2(__global int* tmp_rowid, __global float* tmp_data, int warp_num, __global float* result)
{
    __local int l_rowid[COO_GROUP_SIZE * 2 + 1];
    __local float l_data[COO_GROUP_SIZE * 2 + 1];

    int localid = get_local_id(0);
    if (localid == 0)
    {
	l_rowid[COO_GROUP_SIZE * 2] = -1;
	l_data[COO_GROUP_SIZE * 2] = 0.0f;
    }
    int blocknum = warp_num / (COO_GROUP_SIZE * 2);

    barrier(CLK_LOCAL_MEM_FENCE);

    
    int tmpid = localid;
    for (int i = 0; i < blocknum; i++)
    {
	
	l_rowid[localid] = tmp_rowid[tmpid];
	l_data[localid] = tmp_data[tmpid];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	float left = 0.0f;
	if (localid >= 1 && l_rowid[localid] == l_rowid[localid - 1])
	    left = l_data[localid - 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 2 && l_rowid[localid] == l_rowid[localid - 2])
	    left = l_data[localid - 2];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 4 && l_rowid[localid] == l_rowid[localid - 4])
	    left = l_data[localid - 4];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 8 && l_rowid[localid] == l_rowid[localid - 8])
	    left = l_data[localid - 8];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 16 && l_rowid[localid] == l_rowid[localid - 16])
	    left = l_data[localid - 16];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 32 && l_rowid[localid] == l_rowid[localid - 32])
	    left = l_data[localid - 32];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 64 && l_rowid[localid] == l_rowid[localid - 64])
	    left = l_data[localid - 64];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (l_rowid[localid] != l_rowid[localid + 1])
	{
	    int row = l_rowid[localid];
	    result[row] += l_data[localid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	tmpid += COO_GROUP_SIZE * 2;
    }

    if (warp_num % (COO_GROUP_SIZE*2) != 0)
    {
	if (tmpid < warp_num)
	{
	    l_rowid[localid] = tmp_rowid[tmpid];
	    l_data[localid] = tmp_data[tmpid];
	}
	else
	{
	    l_rowid[localid] = -1;
	    l_data[localid] = 0.0f;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	float left = 0.0f;
	if (localid >= 1 && l_rowid[localid] == l_rowid[localid - 1])
	    left = l_data[localid - 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 2 && l_rowid[localid] == l_rowid[localid - 2])
	    left = l_data[localid - 2];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 4 && l_rowid[localid] == l_rowid[localid - 4])
	    left = l_data[localid - 4];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 8 && l_rowid[localid] == l_rowid[localid - 8])
	    left = l_data[localid - 8];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 16 && l_rowid[localid] == l_rowid[localid - 16])
	    left = l_data[localid - 16];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 32 && l_rowid[localid] == l_rowid[localid - 32])
	    left = l_data[localid - 32];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 64 && l_rowid[localid] == l_rowid[localid - 64])
	    left = l_data[localid - 64];
	barrier(CLK_LOCAL_MEM_FENCE);
	l_data[localid] += left; left = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tmpid < warp_num)
	{
	    if (l_rowid[localid] != l_rowid[localid + 1])
	    {
		int row = l_rowid[localid];
		result[row] += l_data[localid];
	    }	    
	}
    }
    

}

__kernel void gpu_coo_s1_tx(__global int* row_id, __global int* col_id, __global float* data, int process_size, int nnz, __read_only image2d_t vec, __global float* result, __global int* tmp_rowid, __global float* tmp_data)
{
    __local int l_rowid[COO_GROUP_SIZE];
    __local float l_data[COO_GROUP_SIZE];
    const sampler_t smp =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;
    int localid = get_local_id(0);
    int global_warpid = get_group_id(0);
    int start = global_warpid * process_size;
    int gsize = get_local_size(0);
    int end = start + process_size;
    if (end > nnz)
	end = nnz;
    int iter_num = (end - start) / gsize;

    if (localid == gsize - 1)
    {
	l_rowid[localid] = row_id[start];
	l_data[localid] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int lastlocalid = localid + gsize - 1;
    int i = start + localid;
    for (int iter = 0; iter < iter_num; iter++)
    {
	int nzrow = 0;
	float val = 0.0f;
	if (i < nnz)
	{
	    nzrow = row_id[i];
	    int col = col_id[i];
	    int2 coord;
	    coord.x = col & WIDTH_MASK;
	    coord.y = col >> LOG_WIDTH;
	    float4 vecelem = read_imagef(vec, smp, coord);
	    val = data[i] * vecelem.x;
	}
	if (localid == 0)
	{
	    if (nzrow == l_rowid[lastlocalid])
	    {
		val += l_data[lastlocalid];
	    }
	    else
	    {
		result[l_rowid[lastlocalid]] += l_data[lastlocalid];
	    }
	}
	l_rowid[localid] = nzrow;
	l_data[localid] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 1 && nzrow == l_rowid[localid - 1])
	    l_data[localid] = val = val + l_data[localid - 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 2 && nzrow == l_rowid[localid - 2])
	   l_data[localid] = val = val + l_data[localid - 2];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 4 && nzrow == l_rowid[localid - 4])
	    l_data[localid] = val = val + l_data[localid - 4];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 8 && nzrow == l_rowid[localid - 8])
	    l_data[localid] = val = val + l_data[localid - 8];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 16 && nzrow == l_rowid[localid - 16])
	    l_data[localid] = val = val + l_data[localid - 16];
	barrier(CLK_LOCAL_MEM_FENCE);
	if (localid >= 32 && nzrow == l_rowid[localid - 32])
	    l_data[localid] = val = val + l_data[localid - 32];
	barrier(CLK_LOCAL_MEM_FENCE);
	   
	if (localid < (gsize - 1) && nzrow != l_rowid[localid + 1])
	    result[nzrow] += val;	
	barrier(CLK_LOCAL_MEM_FENCE);
	    
	i += gsize;
	
    }

    if (localid == gsize - 1)
    {
	tmp_rowid[global_warpid] = l_rowid[localid];
	tmp_data[global_warpid] = l_data[localid];
    }

}

