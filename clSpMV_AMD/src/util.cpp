#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "spmv_serial.h"
#include "matrix_storage.h"
#include "util.h"


double timestamp ()
{
    struct timeval tv;
    gettimeofday (&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

int findPaddedSize(int realSize, int alignment)
{
    if (realSize % alignment == 0)
	return realSize;
    return realSize + alignment - realSize % alignment;
}

unsigned int getRandMax() {return RAND_MAX;}

unsigned int getRand(unsigned int upper)
{
    return rand() % upper;
}

void pad_csr(csr_matrix<int, float>* source, csr_matrix<int, float>* dest, int alignment)
{
	using namespace std;	
	dest->matinfo.height = source->matinfo.height;
	dest->matinfo.width = source->matinfo.width;
	dest->csr_row_ptr = (int*)malloc(sizeof(int)*(source->matinfo.height+1));
	vector<int> padcol;
	vector<float> paddata;
	padcol.reserve(source->matinfo.nnz*2);
	paddata.reserve(source->matinfo.nnz*2);
	
	dest->csr_row_ptr[0] = 0;
	
	for (int row = 0; row < source->matinfo.height; row++)
	{
		int start = source->csr_row_ptr[row];
		int end = source->csr_row_ptr[row+1];
		int size = end - start;
		int paddedsize = findPaddedSize(size, alignment);
		dest->csr_row_ptr[row+1] = dest->csr_row_ptr[row] + paddedsize;
		int i = 0;
		for (; i < size; i++)
		{
			padcol.push_back(source->csr_col_id[start + i]);
			paddata.push_back(source->csr_data[start + i]);
		}
		int lastcol = padcol[padcol.size() - 1];
		for (; i < paddedsize; i++)
		{
			padcol.push_back(lastcol);
			paddata.push_back(0.0f);
		}
	}
	dest->csr_col_id = (int*)malloc(sizeof(int)*padcol.size());
	dest->csr_data = (float*)malloc(sizeof(float)*paddata.size());
	dest->matinfo.nnz = padcol.size();
	for (unsigned int i = 0; i < padcol.size(); i++)
	{
		dest->csr_col_id[i] = padcol[i];
		dest->csr_data[i] = paddata[i];
	}
}

double distance(float* vec1, float* vec2, int size)
{
	double sum = 0.0f;
	for (int i = 0; i < size; i++)
	{
		double tmp = vec1[i] - vec2[i];
		sum += tmp * tmp;
	}
	return sqrt(sum);
}

void spmv_only(coo_matrix<int, float>* mat, float* vec, float* coores)
{
    int ressize = mat->matinfo.height;
    for (int i = 0; i < ressize; i++)
	coores[i] = (float)0;
    coo_spmv<int, float>(mat, vec, coores, mat->matinfo.width);
}

void two_vec_compare(float* coovec, float* newvec, int size)
{
    double dist = distance(coovec, newvec, size);

    double maxdiff = 0.0f;
    int maxdiffid = 0;
    double maxratiodiff = 0.0f;
    int count = 0;
    for (int i = 0; i < size; i++)
    {
	float tmpa = coovec[i];
	if (tmpa < 0)
	    tmpa *= (-1);
	float tmpb = newvec[i];
	if (tmpb < 0)
	    tmpb *= (-1);
	double diff = tmpa - tmpb;
	if (diff < 0)
	    diff *= (-1);
	float maxab = (tmpa > tmpb)?tmpa:tmpb;
	double ratio = 0.0f;
	if (maxab > 0)
	    ratio = diff / maxab;
	if (diff > maxdiff)
	{
	    maxdiff = diff;
	    maxdiffid = i;
	}
	if (ratio > maxratiodiff)
	    maxratiodiff = ratio;
	if (coovec[i] != newvec[i] && count < 10)
	{
	    printf("Error i %d coo res %f res %f \n", i, coovec[i], newvec[i]);
	    count++;
	}
    }
    printf("Max diff id %d coo res %f res %f \n", maxdiffid, coovec[maxdiffid], newvec[maxdiffid]);
    printf("\nCorrectness Check: Distance %e max diff %e max diff ratio %e vec size %d\n", dist, maxdiff, maxratiodiff, size);
}


void correctness_check(coo_matrix<int, float>* mat, float* vec, float* res)
{
    float* coores = (float*)malloc(sizeof(float)*mat->matinfo.height);
    for (int i = 0; i < mat->matinfo.height; i++)
	coores[i] = 0.0f;
    coo_spmv<int, float>(mat, vec, coores, mat->matinfo.width);
    double dist = distance(coores, res, mat->matinfo.height);

    double maxdiff = 0.0f;
    double maxratiodiff = 0.0f;
    int count = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	float tmpa = coores[i];
	if (tmpa < 0)
	    tmpa *= (-1);
	float tmpb = res[i];
	if (tmpb < 0)
	    tmpb *= (-1);
	double diff = tmpa - tmpb;
	if (diff < 0)
	    diff *= (-1);
	float maxab = (tmpa > tmpb)?tmpa:tmpb;
	double ratio = 0.0f;
	if (maxab > 0)
	    ratio = diff / maxab;
	if (diff > maxdiff)
	    maxdiff = diff;
	if (ratio > maxratiodiff)
	    maxratiodiff = ratio;
	if (coores[i] != res[i] && count < 10)
	{
	    printf("Error i %d coo res %f res %f \n", i, coores[i], res[i]);
	    count++;
	}
    }
    printf("\nCorrectness Check: Distance %e max diff %e max diff ratio %e vec size %d\n", dist, maxdiff, maxratiodiff, mat->matinfo.height);
    free(coores);
}

void printMatInfo(coo_matrix<int, float>* mat)
{
    printf("\nMatInfo: Width %d Height %d NNZ %d\n", mat->matinfo.width, mat->matinfo.height, mat->matinfo.nnz);
    int minoffset = mat->matinfo.width;
    int maxoffset = -minoffset;
    int nnz = mat->matinfo.nnz;
    int lessn16 = 0;
    int inn16 = 0;
    int less16 = 0;
    int large16 = 0;
    for (int i = 0; i < nnz; i++)
    {
	int rowid = mat->coo_row_id[i];
	int colid = mat->coo_col_id[i];
	int diff = rowid - colid;
	if (diff < minoffset)
	    minoffset = diff;
	if (diff > maxoffset)
	    maxoffset = diff;
	if (diff < -15)
	    lessn16++;
	else if (diff < 0)
	    inn16++;
	else if (diff < 16)
	    less16++;
	else
	    large16++;
    }
    printf("Max Offset %d Min Offset %d\n", maxoffset, minoffset);
    printf("Histogram: <-15: %d -15~-1 %d < 0-15 %d > 16 %d\n", lessn16, inn16, less16, large16);

    if (!if_sorted_coo(mat))
    {
	assert(sort_coo(mat) == true);
    }

    int* cacheperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    int* elemperrow = (int*)malloc(sizeof(int)*mat->matinfo.height);
    memset(cacheperrow, 0, sizeof(int)*mat->matinfo.height);
    memset(elemperrow, 0, sizeof(int)*mat->matinfo.height);
    int index = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (i < mat->coo_row_id[index])
	    continue;
	int firstline = mat->coo_col_id[index]/16;
	cacheperrow[i] = 1;
	elemperrow[i] = 1;
	index++;
	while (mat->coo_row_id[index] == i)
	{
	    int nextline = mat->coo_col_id[index]/16;
	    if (nextline != firstline)
	    {
		firstline = nextline;
		cacheperrow[i]++;
	    }
	    elemperrow[i]++;
	    index++;
	}
    }
    int maxcacheline = 0;
    int mincacheline = 100000000;
    int sum = 0;
    for (int i = 0; i < mat->matinfo.height; i++)
    {
	if (cacheperrow[i] < mincacheline)
	    mincacheline = cacheperrow[i];
	if (cacheperrow[i] > maxcacheline)
	    maxcacheline = cacheperrow[i];
	sum += cacheperrow[i];
    }
    printf("Cacheline usage per row: max %d min %d avg %f\n", maxcacheline, mincacheline, (double)sum/(double)mat->matinfo.height);
}



/**
 * Only accept 1x4, 2x4, 4x4, 8x4, 1x8, 2x8, 4x8, 8x8 blocks
 * Example storage
 * |0 1 2 3| g h i j| 
 * |4 5 6 7| k l m n|
 *
 * |8 9 a b| o p q r|
 * |c d e f| s t u v|
 *
 * 2 x 4 block
 * Col_id: [0, 0, 1, 1]
 * data: [0 1 2 3 8 9 a b padding, 4 5 6 7 c d e f padding, 
 *        g h i j o p q r padding, k l m n s t u v padding]
 *
 * 2 x 8 block will be the same data layout
 * Col_id: [0, 0]
 * data: [0 1 2 3 8 9 a b padding, 4 5 6 7 c d e f padding, 
 *        g h i j o p q r padding, k l m n s t u v padding]
 *        
 */


void rearrange_bell_4col(bell_matrix<int, float>* mat, int alignment)
{
    int bwidth = mat->bell_bwidth;
    int bheight = mat->bell_bheight;
    int ellnum = mat->bell_block_num;
    int rownum = mat->bell_row_num;
    int originheight = mat->bell_height_aligned;
    assert(bwidth % 4 == 0);
    int width4num = bwidth / 4;
    int newheight = findPaddedSize(4 * rownum, alignment);
    float* newdata = (float*)malloc(sizeof(float)*newheight*bheight*width4num*ellnum);
    
    int blockcolsize = originheight * bwidth * bheight;
    int blockhsize = originheight * bwidth;

    int newblockcolsize = newheight * width4num * bheight;
    int newblockhsize = newheight * width4num;
    int newblockw4size = newheight * bheight;
    for (int i = 0; i < ellnum; i++)
    {
	for (int h = 0; h < bheight; h++)
	{
	    for (int w = 0; w < bwidth; w++)
	    {
		for (int j = 0; j < rownum; j++)
		{
		    float data = mat->bell_data[i * blockcolsize + h * blockhsize + w * originheight + j];
		    int w4num = w / 4;
		    int wremain = w % 4;
		    newdata[i * newblockcolsize + h * newheight + w4num * newblockw4size + j * 4 + wremain] = data;
		}
	    }
	}
    }

    mat->bell_float4_aligned = newheight;
    free(mat->bell_data);
    mat->bell_data = newdata;

    //Update col id (use float4 id)
    for (int i = 0; i < originheight * ellnum; i++)
	mat->bell_col_id[i] /= 4;
}






