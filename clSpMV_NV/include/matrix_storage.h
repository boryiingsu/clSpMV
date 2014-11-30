#ifndef MATSTORAGE__H__
#define MATSTORAGE__H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <algorithm>
#include <vector>
#include <map>

#include "constant.h"

template <class dimType>
struct matrixInfo
{
    /** Matrix width*/
    dimType width;
     /** Matrix height*/
    dimType height;
    /** Number of non zeros*/
    dimType nnz;

};

//The original definition of diagonal format.
//The maximum number of diagonals of a square matrix: 2 * width - 1
//Diagonal index ranging from -width + 1 to width - 1
template <class dimType, class offsetType, class dataType>
struct dia_matrix
{
    matrixInfo<dimType> matinfo;

     /** Number of diagonals*/
    dimType dia_num;
     /** Diagonal length*/
    dimType dia_length;
     /** Aligned length (with padding)*/
    dimType dia_length_aligned;

    /** Diagonal offsets, from the smallest offset to the largest offset, size dia_num */
    offsetType* dia_offsets;
    /** Padded data, size dia_num * dia_length_aligned 
     * (for each dia_num, store a dia_length_aligned consecutively. )*/
    dataType* dia_data;
};

//My own diagonal representation. If diagonal i and j satisfy i % width == j % width, they will be stored on the same diagonal. 
//The maximum number of diagonals of a square matrix: width
//Diagonal index ranging from 0 to width - 1
template <class dimType, class offsetType, class dataType>
struct dia_ext_matrix
{
    matrixInfo<dimType> matinfo;

     /** Number of diagonals*/
    dimType dia_num;
     /** Diagonal length*/
    dimType dia_length;
     /** Aligned length (with padding)*/
    dimType dia_length_aligned;

    /** Diagonal offsets, from the smallest offset to the largest offset, size dia_num */
    offsetType* dia_offsets;
    /** Padded data, size dia_num * dia_length_aligned 
     * (for each dia_num, store a dia_length_aligned consecutively. )*/
    dataType* dia_data;
};

//Banded diatongal matrix
template <class dimType, class offsetType, class dataType>
struct bdia_matrix
{
    matrixInfo<dimType> matinfo;
     
    /** Number of bands */
    dimType bdia_band_num;
     /** Diagonal length*/
    dimType bdia_length;
     /** Aligned length (with padding)*/
    dimType bdia_length_aligned;

    /** Diagonal offsets, from the smallest offset to the largest offset, size bdia_band_num */
    offsetType* bdia_offsets;
    /** Offsets of each band, size bdia_band_num + 1
     * for a band i, let start = bptr[i], end = bprt[i+1], the band stores in 
     * data[start * length_aligned] to data[end * length_aligned - 1]*/
    dimType* bdia_bptr;
    /** Padded data, size dia_num * dia_length_aligned 
     * (for each dia_num, store a dia_length_aligned consecutively. )*/
    dataType* bdia_data;

};

template <class dimType, class dataType>
struct ell_matrix
{
    matrixInfo<dimType> matinfo;

     /** Number of elements per row*/
    dimType ell_num;
     /** Aligned height (padded with zero)*/
    dimType ell_height_aligned;

    /** Padded column index, size ell_num * ell_height_aligned
     * (for each ell_num, store an ell_height_aligned consecutively.)*/
    dimType* ell_col_id;
    /** Padded data, size ell_num * ell_height_aligned.*/
    dataType* ell_data;
};


/** Each slice has height sell_slice_height. 
 *  The last slice is padded to make sure the height of the slice is the same.
 *  The col id of slices i locates from sell_slice_ptr[i] to sell_slice_ptr[i+1]
 *  The data of slices i locates from sell_slice_ptr[i] to sell_slice_ptr[i+1]
 *
 *  The offset of each col can be computed by 
 *  (sell_slice_ptr[i+1] - sell_slice_ptr[i]) / sell_slice_num;
 */
template <class dimType, class dataType>
struct sell_matrix
{
    matrixInfo<dimType> matinfo;

    /** The height of each slice. */
    unsigned int sell_slice_height;
    /**The number of slices in the format*/
    dimType sell_slice_num;
     /** The starting and ending index of each slice*/
    dimType* sell_slice_ptr;
    /** Padded column index, size = sell_slice_ptr[sell_slice_num]*/
    dimType* sell_col_id;
    /** Padded data, size = sell_slice_ptr[sell_slice_num]*/
    dataType* sell_data;
};



template <class dimType, class dataType>
struct coo_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row index, size nnz*/
    dimType* coo_row_id;
    /** Column index, size nnz*/
    dimType* coo_col_id;
    /** Data, size nnz */
    dataType* coo_data;
};


template <class dimType, class dataType>
struct csr_matrix
{
    matrixInfo<dimType> matinfo;

    /** Row pointer, size height + 1*/
    dimType* csr_row_ptr;
    /** Column index, size nnz*/
    dimType* csr_col_id;
    /** Data, size nnz */
    dataType* csr_data;
};


/** Examplar Storage
 *  | 0 1 | 4 5 |
 *  | 2 3 | 6 7 |
 *
 *  Row_ptr: [0, 2]
 *  Col_id: [0, 2]
 *  data: [0, 4, padding, 1, 5, padding, 2, 6, padding, 3, 7, padding]
 */

template <class dimType, class dataType>
struct bcsr_matrix
{
    matrixInfo<dimType> matinfo;

    /** block width*/
    unsigned int bcsr_bwidth;
    /** block height*/
    unsigned int bcsr_bheight;
    /** Number of blocked rows, eqaul to the size of the bcsr_row_ptr array - 1 */
    dimType bcsr_row_num;
    /** Align the data array to the aligned_size */
    dimType bcsr_aligned_size;
    /** number of blocks */
    dimType bcsr_block_num;
    /** Row pointer, size ceil(height / bcsr_bheight) + 1*/
    dimType* bcsr_row_ptr;
    /** Column index, size bcsr_block_num */
    dimType* bcsr_col_id;
    /** Data, size bcsr_aligned_size * bcsr_bwidth * bcsr_bheight */
    dataType* bcsr_data;
};

/** Examplar Storage
 *  | 0 1 | 4 5 |
 *  | 2 3 | 6 7 |
 *
 *  | 8 9 | a b |
 *  | c d | e f |
 * 
 *  2 x 2 block
 *  Col_id: [0, 0, 2, 2]
 *  data: [0, 8, padding, 1, 9, padding, 2, c, padding, 3, d, padding, 
 *         4, a, padding, 5, b, padding, 6, e, padding, 7, f, padding ]
 */


template <class dimType, class dataType>
struct bell_matrix
{
    matrixInfo<dimType> matinfo;

    /** block width*/
    unsigned int bell_bwidth;
    /** block height*/
    unsigned int bell_bheight;
    /** Number of blocked rows, eqaul to ceil(matinfo.height/bell_bheight) */
    dimType bell_row_num;
    /** number of blocks per block row*/
    dimType bell_block_num;
    /** aligned block height */
    dimType bell_height_aligned;
    /** aligned float4 block height */
    dimType bell_float4_aligned;
    /** Column index, size: bell_block_num * bell_height_aligned */
    dimType* bell_col_id;
    /** Data, size: bell_block_num * bell_bwidth * bell_bheight * bell_height_aligned */
    dataType* bell_data;
};

/**
 * Only accept 1x4, 2x4, 4x4, 8x4, 1x8, 2x8, 4x8, 8x8 blocks
 * The column id is stored using the float4 indexing
 * Example storage
 * |0 1 2 3| g h i j| 
 * |4 5 6 7| k l m n|
 *
 * |8 9 a b| o p q r|
 * |c d e f| s t u v|
 *
 * 2 x 4 block
 * Row_ptr: [0, 2, 4]
 * Col_id: [0, 1, 0, 1]
 * data: [0 1 2 3 g h i j 8 9 a b o p q r padding, 
 *        4 5 6 7 k l m n c d e f s t u v padding]
 *      
 *
 * 2 x 8 block 
 * Row_ptr: [0, 1, 2]
 * Col_id: [0, 0]
 * data: [0 1 2 3 8 9 a b padding,
 *        4 5 6 7 c d e f padding,
 *        g h i j o p q r padding,
 *        k l m n s t u v padding]
 *
 *        
 */

template <class dimType, class dataType>
struct b4csr_matrix
{
    matrixInfo<dimType> matinfo;

    /** block width*/
    unsigned int b4csr_bwidth;
    /** block height*/
    unsigned int b4csr_bheight;
    /** Number of blocked rows, eqaul to the size of the b4csr_row_ptr array - 1 */
    dimType b4csr_row_num;
    /** Align the data array to the aligned_size */
    dimType b4csr_aligned_size;
    /** number of blocks */
    dimType b4csr_block_num;
    /** Row pointer, size ceil(height / b4csr_bheight) + 1*/
    dimType* b4csr_row_ptr;
    /** Column index, size b4csr_block_num */
    dimType* b4csr_col_id;
    /** Data, size b4csr_aligned_size * (b4csr_bwidth/4) * b4csr_bheight */
    dataType* b4csr_data;
};

/**
 * Only accept 1x4, 2x4, 4x4, 8x4, 1x8, 2x8, 4x8, 8x8 blocks
 * The column id is stored using the float4 indexing
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

template <class dimType, class dataType>
struct b4ell_matrix
{
    matrixInfo<dimType> matinfo;

    /** block width*/
    unsigned int b4ell_bwidth;
    /** block height*/
    unsigned int b4ell_bheight;
    /** Number of blocked rows, eqaul to ceil(matinfo.height/bell_bheight) */
    dimType b4ell_row_num;
    /** number of blocks per block row*/
    dimType b4ell_block_num;
    /** aligned block height */
    dimType b4ell_height_aligned;
    /** aligned float4 block height */
    dimType b4ell_float4_aligned;
    /** Column index, size: bell_block_num * bell_height_aligned */
    dimType* b4ell_col_id;
    /** Data, size: bell_block_num * bell_bwidth * bell_bheight * bell_height_aligned */
    dataType* b4ell_data;
};

/**
 * This format is similar to b4ell. The only difference is that each slice has different width. 
 *  The last slice is padded to make sure the height of the slice is the same.
 *  The col id of slices i locates from sbell_slice_ptr[i] to sbell_slice_ptr[i+1]
 *  The data of slices i locates from sbell_slice_ptr[i]*bwidth*bheight to sbell_slice_ptr[i+1]*bwidth*bheight
 *
 *  The offset of each col can be computed by 
 *  (sbell_slice_ptr[i+1] - sbell_slice_ptr[i]) / sbell_slice_height;
 * 
 */
template <class dimType, class dataType>
struct sbell_matrix
{
    matrixInfo<dimType> matinfo;

    /** block width*/
    unsigned int sbell_bwidth;
    /** block height*/
    unsigned int sbell_bheight;
    /** The height of each slice*/
    unsigned int sbell_slice_height;
    /** The number of slices in the format */
    dimType sbell_slice_num;
    /** Number of blocked rows, eqaul to ceil(matinfo.height/bell_bheight) */
    dimType sbell_row_num;
    /** The starting and ending index of each blocked row, size sbell_slice_num + 1*/
    dimType* sbell_slice_ptr;
    /** Column index, size: sbell_slice_ptr[sbell_slice_num] */
    dimType* sbell_col_id;
    /** Data, size: sbell_slice_ptr[sbell_slice_num] * bwidth * bheight */
    dataType* sbell_data;
};

/** Initialization routines*/


template <class dimType>
void init_mat_info(matrixInfo<dimType>& info)
{
    info.width = (dimType)0;
    info.height = (dimType)0;
    info.nnz = (dimType)0;
}


template <class dimType, class offsetType, class dataType>
void init_dia_matrix(dia_matrix<dimType, offsetType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.dia_num = (dimType)0;
    mat.dia_length = (dimType)0;
    mat.dia_length_aligned = (dimType)0;
    mat.dia_offsets = NULL;
    mat.dia_data = NULL;
}


template <class dimType, class offsetType, class dataType>
void init_dia_ext_matrix(dia_ext_matrix<dimType, offsetType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.dia_num = (dimType)0;
    mat.dia_length = (dimType)0;
    mat.dia_length_aligned = (dimType)0;
    mat.dia_offsets = NULL;
    mat.dia_data = NULL;
}

template <class dimType, class offsetType, class dataType>
void init_bdia_matrix(bdia_matrix<dimType, offsetType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.bdia_band_num = (dimType)0;
    mat.bdia_length = (dimType)0;
    mat.bdia_length_aligned = (dimType)0;
    mat.bdia_offsets = NULL;
    mat.bdia_bptr = NULL;
    mat.bdia_data = NULL;
}

template <class dimType, class dataType>
void init_ell_matrix(ell_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.ell_num = (dimType)0;
    mat.ell_height_aligned = (dimType)0;
    mat.ell_col_id = NULL;
    mat.ell_data = NULL;
}


template <class dimType, class dataType>
void init_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.coo_row_id = NULL;
    mat.coo_col_id = NULL;
    mat.coo_data = NULL;
}


template <class dimType, class dataType>
void init_csr_matrix(csr_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.csr_row_ptr = NULL;
    mat.csr_col_id = NULL;
    mat.csr_data = NULL;
}


template <class dimType, class dataType>
void init_sell_matrix(sell_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.sell_slice_height = 0;
    mat.sell_slice_num = (dimType)0;
    mat.sell_slice_ptr = NULL;
    mat.sell_col_id = NULL;
    mat.sell_data = NULL;
}

template <class dimType, class dataType>
void init_bcsr_matrix(bcsr_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.bcsr_bwidth = 0;
    mat.bcsr_bheight = 0;
    mat.bcsr_row_num = (dimType)0;
    mat.bcsr_block_num = (dimType)0;
    mat.bcsr_aligned_size = (dimType)0;
    mat.bcsr_row_ptr = NULL;
    mat.bcsr_col_id = NULL;
    mat.bcsr_data = NULL;
}


template <class dimType, class dataType>
void init_bell_matrix(bell_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.bell_bwidth = 0;
    mat.bell_bheight = 0;
    mat.bell_row_num = (dimType)0;
    mat.bell_block_num = (dimType)0;
    mat.bell_height_aligned = (dimType)0;
    mat.bell_col_id = NULL;
    mat.bell_data = NULL;
}

template <class dimType, class dataType>
void init_b4csr_matrix(b4csr_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.b4csr_bwidth = 0;
    mat.b4csr_bheight = 0;
    mat.b4csr_row_num = (dimType)0;
    mat.b4csr_block_num = (dimType)0;
    mat.b4csr_aligned_size = (dimType)0;
    mat.b4csr_row_ptr = NULL;
    mat.b4csr_col_id = NULL;
    mat.b4csr_data = NULL;
}


template <class dimType, class dataType>
void init_b4ell_matrix(b4ell_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.b4ell_bwidth = 0;
    mat.b4ell_bheight = 0;
    mat.b4ell_row_num = (dimType)0;
    mat.b4ell_block_num = (dimType)0;
    mat.b4ell_height_aligned = (dimType)0;
    mat.b4ell_col_id = NULL;
    mat.b4ell_data = NULL;
}

template <class dimType, class dataType>
void init_sbell_matrix(sbell_matrix<dimType, dataType>& mat)
{
    init_mat_info(mat.matinfo);
    mat.sbell_bwidth = 0;
    mat.sbell_bheight = 0;
    mat.sbell_slice_height = 0;
    mat.sbell_slice_num = (dimType)0;
    mat.sbell_row_num = (dimType)0;
    mat.sbell_slice_ptr = NULL;
    mat.sbell_col_id = NULL;
    mat.sbell_data = NULL;
}

/** Release the matrix memories. Information remains the same, only the arrays are freed.*/


template <class dimType, class offsetType, class dataType>
void free_dia_matrix(dia_matrix<dimType, offsetType, dataType>& mat)
{
    if (mat.dia_offsets != NULL)
	free(mat.dia_offsets);
    if (mat.dia_data != NULL)
	free(mat.dia_data);
}

template <class dimType, class offsetType, class dataType>
void free_dia_ext_matrix(dia_ext_matrix<dimType, offsetType, dataType>& mat)
{
    if (mat.dia_offsets != NULL)
	free(mat.dia_offsets);
    if (mat.dia_data != NULL)
	free(mat.dia_data);
}

template <class dimType, class offsetType, class dataType>
void free_bdia_matrix(bdia_matrix<dimType, offsetType, dataType>& mat)
{
    if (mat.bdia_offsets != NULL)
	free(mat.bdia_offsets);
    if (mat.bdia_bptr != NULL)
	free(mat.bdia_bptr);
    if (mat.bdia_data != NULL)
	free(mat.bdia_data);
}

template <class dimType, class dataType>
void free_ell_matrix(ell_matrix<dimType, dataType>& mat)
{
    if (mat.ell_col_id != NULL)
	free(mat.ell_col_id);
    if (mat.ell_data != NULL)
	free(mat.ell_data);
}


template <class dimType, class dataType>
void free_coo_matrix(coo_matrix<dimType, dataType>& mat)
{
    if (mat.coo_row_id != NULL)
	free(mat.coo_row_id);
    if (mat.coo_col_id != NULL)
	free(mat.coo_col_id);
    if (mat.coo_data != NULL)
	free(mat.coo_data);
}


template <class dimType, class dataType>
void free_csr_matrix(csr_matrix<dimType, dataType>& mat)
{
    if (mat.csr_row_ptr != NULL)
	free(mat.csr_row_ptr);
    if (mat.csr_col_id != NULL)
	free(mat.csr_col_id);
    if (mat.csr_data != NULL)
	free(mat.csr_data);
}


template <class dimType, class dataType>
void free_sell_matrix(sell_matrix<dimType, dataType>& mat)
{
    if (mat.sell_slice_ptr != NULL)
	free(mat.sell_slice_ptr);
    if (mat.sell_col_id != NULL)
	free(mat.sell_col_id);
    if (mat.sell_data != NULL)
	free(mat.sell_data);
}

template <class dimType, class dataType>
void free_bcsr_matrix(bcsr_matrix<dimType, dataType>& mat)
{
    if (mat.bcsr_row_ptr != NULL)
	free(mat.bcsr_row_ptr);
    if (mat.bcsr_col_id != NULL)
	free(mat.bcsr_col_id);
    if (mat.bcsr_data != NULL)
	free(mat.bcsr_data);
}

template <class dimType, class dataType>
void free_bell_matrix(bell_matrix<dimType, dataType>& mat)
{
    if (mat.bell_col_id != NULL)
	free(mat.bell_col_id);
    if (mat.bell_data != NULL)
	free(mat.bell_data);
}

template <class dimType, class dataType>
void free_b4csr_matrix(b4csr_matrix<dimType, dataType>& mat)
{
    if (mat.b4csr_row_ptr != NULL)
	free(mat.b4csr_row_ptr);
    if (mat.b4csr_col_id != NULL)
	free(mat.b4csr_col_id);
    if (mat.b4csr_data != NULL)
	free(mat.b4csr_data);
}

template <class dimType, class dataType>
void free_b4ell_matrix(b4ell_matrix<dimType, dataType>& mat)
{
    if (mat.b4ell_col_id != NULL)
	free(mat.b4ell_col_id);
    if (mat.b4ell_data != NULL)
	free(mat.b4ell_data);
}

template <class dimType, class dataType>
void free_sbell_matrix(sbell_matrix<dimType, dataType>& mat)
{
    if (mat.sbell_slice_ptr != NULL)
	free(mat.sbell_slice_ptr);
    if (mat.sbell_col_id != NULL)
	free(mat.sbell_col_id);
    if (mat.sbell_data != NULL)
	free(mat.sbell_data);
}



template <class dimType, class dataType>
void write_coo(char* filename, coo_matrix<dimType, dataType>* coo)
{
    FILE* outfile;
    outfile = fopen(filename, "w");
    dimType nnz = coo->matinfo.nnz;
    fwrite(&(coo->matinfo.width), sizeof(dimType), 1, outfile);
    fwrite(&(coo->matinfo.height), sizeof(dimType), 1, outfile);
    fwrite(&(coo->matinfo.nnz), sizeof(dimType), 1, outfile);
    
    fwrite(coo->coo_row_id, sizeof(dimType), nnz, outfile);
    fwrite(coo->coo_col_id, sizeof(dimType), nnz, outfile);
    fwrite(coo->coo_data, sizeof(dataType), nnz, outfile);

    fclose(outfile);
}

template <class dimType, class dataType>
void read_coo(char* filename, coo_matrix<dimType, dataType>* coo)
{
    FILE* infile;
    infile = fopen(filename, "r");

    dimType* width;
    dimType* height;
    dimType* nnz;
    
    fread(width, sizeof(dimType), 1, infile);
    fread(height, sizeof(dimType), 1, infile);
    fread(nnz, sizeof(dimType), 1, infile);
    
    coo->matinfo.width = (*width);
    coo->matinfo.height = (*height);
    coo->matinfo.nnz = (*nnz);

    fread(coo->coo_row_id, sizeof(dimType), (*nnz), infile);
    fread(coo->coo_col_id, sizeof(dimType), (*nnz), infile);
    fread(coo->coo_data, sizeof(dataType), (*nnz), infile);

    fclose(infile);
}

template <class dimType, class dataType>
bool is_equal_coo(coo_matrix<dimType, dataType>* mata, coo_matrix<dimType, dataType>* matb)
{
    if (mata->matinfo.width != matb->matinfo.width)
	return false;
    if (mata->matinfo.height != matb->matinfo.height)
	return false;
    if (mata->matinfo.nnz != matb->matinfo.nnz)
	return false;

    dimType nnz = mata->matinfo.nnz;
    for (dimType i = (dimType) 0; i < nnz; i++)
    {
	if (mata->coo_row_id[i] != matb->coo_row_id[i])
	{
	    printf("Diff: index %d rowa %d rowb %d", i, mata->coo_row_id[i], matb->coo_row_id[i]);
	    return false;
	}
	if (mata->coo_col_id[i] != matb->coo_col_id[i])
	{
	    printf("Diff: index %d cola %d colb %d", i, mata->coo_col_id[i], matb->coo_col_id[i]);
	    return false;
	}
	if (mata->coo_data[i] != matb->coo_data[i])
	{
	    printf("Diff: index %d dataa %f datab %f", i, mata->coo_data[i], matb->coo_data[i]);
	    return false;
	}
    }

    return true;
}

template <class dimType, class dataType>
bool if_sorted_coo(coo_matrix<dimType, dataType>* mat)
{
    dimType nnz = mat->matinfo.nnz;
    for (int i = 0; i < nnz - 1; i++)
    {
        if ((mat->coo_row_id[i] > mat->coo_row_id[i+1]) || (mat->coo_row_id[i] == mat->coo_row_id[i+1] && mat->coo_col_id[i] > mat->coo_col_id[i+1]))
            return false;
    }
    return true;
}


template <class dimType, class dataType>
bool sort_coo(coo_matrix<dimType, dataType>* mat)
{

    int i = 0;
    dimType  beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    dimType pivrow, pivcol;
    dataType pivdata;

    beg[0]=0; 
    end[0]=mat->matinfo.nnz;
    while (i>=0) 
    {
	L=beg[i];
	if (end[i] - 1 > end[i])
	    R = end[i];
	else
	    R = end[i] - 1;
	if (L<R) 
	{
	    dimType middle = (L+R)/2;
	    pivrow=mat->coo_row_id[middle]; 
	    pivcol=mat->coo_col_id[middle];
	    pivdata=mat->coo_data[middle];
	    mat->coo_row_id[middle] = mat->coo_row_id[L];
	    mat->coo_col_id[middle] = mat->coo_col_id[L];
	    mat->coo_data[middle] = mat->coo_data[L];
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    if (i==MAX_LEVELS-1) 
		return false;
	    while (L<R) 
	    {
		while (((mat->coo_row_id[R] > pivrow) || 
			    (mat->coo_row_id[R] == pivrow && mat->coo_col_id[R] > pivcol)) 
			&& L<R) 
		    R--; 
		if (L<R) 
		{
		    mat->coo_row_id[L] = mat->coo_row_id[R];
		    mat->coo_col_id[L] = mat->coo_col_id[R];
		    mat->coo_data[L] = mat->coo_data[R];
		    L++;
		}
		while (((mat->coo_row_id[L] < pivrow) || 
			    (mat->coo_row_id[L] == pivrow && mat->coo_col_id[L] < pivcol)) 
			&& L<R) 
		    L++; 
		if (L<R) 
		{
		    mat->coo_row_id[R] = mat->coo_row_id[L];
		    mat->coo_col_id[R] = mat->coo_col_id[L];
		    mat->coo_data[R] = mat->coo_data[L];
		    R--;
		}
	    }
	    mat->coo_row_id[L] = pivrow;
	    mat->coo_col_id[L] = pivcol;
	    mat->coo_data[L] = pivdata;
	    beg[i+1]=L+1; 
	    end[i+1]=end[i]; 
	    end[i++]=L; 
	}
	else 
	{
	    i--; 
	}
    }

    return true;
}

template <class dimType>
dimType aligned_length(dimType length, dimType alignment)
{
    if (length % alignment == 0)
        return length;
    return length + alignment - length % alignment;
}

template <class dimType, class offsetType, class dataType>
void dia2coo(dia_matrix<dimType, offsetType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);

    dimType cooid = dimType(0);
    dimType dianum = source->dia_num;
    dimType dialength = source->dia_length_aligned;
    for (dimType row = 0; row < dialength; row++)
    {
        for (dimType i = 0; i < dianum; i++)
        {
            offsetType col = row + source->dia_offsets[i];
            if (col >= dimType(0) && col < source->dia_length)
            {
                dataType tmpData = source->dia_data[i * dialength + row];
                if ( tmpData != dataType(0))
                {
                    dest->coo_row_id[cooid] = row;
                    dest->coo_col_id[cooid] = col;
                    dest->coo_data[cooid] = tmpData;
                    cooid = cooid + 1;
                }
            }
        }
    }
    if (!if_sorted_coo<dimType, dataType>(dest))
    {
        bool res = sort_coo<dimType, dataType>(dest);
        assert(res == true);
    }
}

template<class offsetType, class dataType>
struct diagonal
{
    offsetType offset;
    dataType* data;
};

template <class dimType, class offsetType, class dataType>
bool sort_diagonal(diagonal<offsetType, dataType>* dias, dimType size)
{

    int i = 0;
    dimType  beg[MAX_LEVELS], end[MAX_LEVELS], L, R ;
    offsetType pivoffset;
    dataType* pivdataptr;

    beg[0]=0; 
    end[0]=size;

    while (i>=0) 
    {
	L=beg[i];
	if (end[i] - 1 > end[i])
	    R = end[i];
	else
	    R = end[i] - 1;
	if (L<R) 
	{
	    pivoffset = dias[L].offset;
	    pivdataptr = dias[L].data;

	    if (i==MAX_LEVELS-1) 
		return false;

	    while (L<R) 
	    {
		while (dias[R].offset > pivoffset && L<R) 
		    R--; 
		if (L<R) 
		{
		    dias[L].offset = dias[R].offset;
		    dias[L].data = dias[R].data;
		    L++;
		}
		while (dias[L].offset < pivoffset && L<R) 
		    L++; 
		if (L<R) 
		{
		    dias[R].offset = dias[L].offset;
		    dias[R].data = dias[L].data;
		    R--;
		}
	    }

	    dias[L].offset = pivoffset;
	    dias[L].data = pivdataptr;
	    beg[i+1]=L+1; 
	    end[i+1]=end[i]; 
	    end[i++]=L; 

	}
	else 
	{
	    i--; 
	}
    }


    return true;
}

template <class dimType, class offsetType, class dataType>
bool coo2dia(coo_matrix<dimType, dataType>* source, dia_matrix<dimType, offsetType, dataType>* dest, dimType diaAlignment)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;
    
    if (dest->matinfo.width < dest->matinfo.height)
	    dest->dia_length = dest->matinfo.width;
    else
	    dest->dia_length = dest->matinfo.height;
    dimType newlength = aligned_length(dest->dia_length, diaAlignment);

    dimType diaSize = (dimType)DIAINCSIZE;
    diagonal<offsetType, dataType>* dias = (diagonal<offsetType, dataType>*)malloc(sizeof(diagonal<offsetType, dataType>) * diaSize);
    dimType diaCount = (dimType)0;
    for (dimType i = (dimType)0; i < dest->matinfo.nnz; i++)
    {
	dimType row = source->coo_row_id[i];
	dimType col = source->coo_col_id[i];
	dataType data = source->coo_data[i];
	offsetType offset = (offsetType)col - (offsetType)row;
	bool ifDiaExist = false;
	dimType diaid = (dimType)0;
	for (dimType j = (dimType)0; j < diaCount; j++)
	{
	    if (dias[j].offset == offset)
	    {
		ifDiaExist = true;
		dias[j].data[row] = data;
		break;
	    }
	}
    
	if (!ifDiaExist)
	{
        if (diaCount * newlength >= source->matinfo.nnz * 2)
            {
                dest->dia_num = diaCount;
                dest->dia_length_aligned = newlength;
                printf("\ndiaCount %d length %d nnz %d", diaCount, newlength, source->matinfo.nnz);
                printf("\nThe number of padded numbers is too large, should not represetned by the dia format\n");
                return false;
            }
	    if (diaCount + 1>= diaSize)
	    {
            
            
		diaSize += DIAINCSIZE;
		diagonal<offsetType, dataType>* newdias = (diagonal<offsetType, dataType>*)realloc(dias, sizeof(diagonal<offsetType, dataType>) * diaSize);
		dias = newdias;
	    }
	    dias[diaCount].offset = offset;
	    dias[diaCount].data = (dataType*)malloc(sizeof(dataType)*newlength);
	    memset(dias[diaCount].data, 0, sizeof(dataType)*newlength);
	    dias[diaCount].data[row] = data;
	    diaCount++;
	}
	
    }
    dest->dia_num = diaCount;
    bool res = sort_diagonal<dimType, offsetType, dataType>(dias, diaCount);
    assert(res == true);
    //assert(sort_diagonal<dimType, offsetType, dataType>(dias, diaCount) == true);
    dest->dia_length_aligned = newlength;
    dest->dia_offsets = (offsetType*)malloc(sizeof(offsetType)*diaCount);
    dest->dia_data = (dataType*)malloc(sizeof(dataType)*diaCount*newlength);
    for (dimType i = (dimType)0; i < diaCount; i++)
    {
	dest->dia_offsets[i] = dias[i].offset;
	memcpy(dest->dia_data + i * newlength, dias[i].data, sizeof(dataType)*newlength);
    }
    for (dimType i = (dimType)0; i < diaCount; i++)
    {
        free(dias[i].data);
    }
    free(dias);
    return true;
}

template <class dimType, class offsetType, class dataType>
void diaext2coo(dia_ext_matrix<dimType, offsetType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);

    dimType cooid = dimType(0);
    dimType dianum = source->dia_num;
    dimType dialength = source->dia_length_aligned;
    dimType width = source->matinfo.width;
    for (dimType row = 0; row < dialength; row++)
    {
        for (dimType i = 0; i < dianum; i++)
        {
            offsetType col = row + source->dia_offsets[i];
            col = col % width;
            dataType tmpData = source->dia_data[i * dialength + row];
            if ( tmpData != dataType(0))
            {
                dest->coo_row_id[cooid] = row;
                dest->coo_col_id[cooid] = col;
                dest->coo_data[cooid] = tmpData;
                cooid = cooid + 1;
            }
        }
    }
    if (!if_sorted_coo<dimType, dataType>(dest))
    {
	    bool res = sort_coo<dimType, dataType>(dest);
        assert(res == true);
    }
}

template <class dimType, class offsetType, class dataType>
bool coo2diaext(coo_matrix<dimType, dataType>* source, dia_ext_matrix<dimType, offsetType, dataType>* dest, dimType diaAlignment)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;
    
    if (dest->matinfo.width < dest->matinfo.height)
	    dest->dia_length = dest->matinfo.width;
    else
	    dest->dia_length = dest->matinfo.height;
    dimType newlength = aligned_length(dest->dia_length, diaAlignment);

    dimType diaSize = (dimType)DIAINCSIZE;
    diagonal<offsetType, dataType>* dias = (diagonal<offsetType, dataType>*)malloc(sizeof(diagonal<offsetType, dataType>) * diaSize);
    dimType diaCount = (dimType)0;
    for (dimType i = (dimType)0; i < dest->matinfo.nnz; i++)
    {
	dimType row = source->coo_row_id[i];
	dimType col = source->coo_col_id[i];
	dataType data = source->coo_data[i];
	offsetType offset = (offsetType)col - (offsetType)row;
    if (offset < (offsetType)0)
        offset += source->matinfo.width;
	bool ifDiaExist = false;
	dimType diaid = (dimType)0;
	for (dimType j = (dimType)0; j < diaCount; j++)
	{
	    if (dias[j].offset == offset)
	    {
		ifDiaExist = true;
		dias[j].data[row] = data;
		break;
	    }
	}
    
	if (!ifDiaExist)
	{
        if (diaCount * newlength >= source->matinfo.nnz * 2)
        {
            dest->dia_num = diaCount;
            dest->dia_length_aligned = newlength;
            printf("\ndiaCount %d length %d nnz %d", diaCount, newlength, source->matinfo.nnz);
            printf("\nThe number of padded numbers is too large, should not represetned by the dia format\n");
            return false;
        }
	    if (diaCount + 1>= diaSize)
	    {
            
		diaSize += DIAINCSIZE;
		diagonal<offsetType, dataType>* newdias = (diagonal<offsetType, dataType>*)realloc(dias, sizeof(diagonal<offsetType, dataType>) * diaSize);
		dias = newdias;
	    }
	    dias[diaCount].offset = offset;
	    dias[diaCount].data = (dataType*)malloc(sizeof(dataType)*newlength);
	    memset(dias[diaCount].data, 0, sizeof(dataType)*newlength);
	    dias[diaCount].data[row] = data;
	    diaCount++;
	}
	
    }
    dest->dia_num = diaCount;
    bool res = sort_diagonal<dimType, offsetType, dataType>(dias, diaCount);
    assert(res == true);
    //assert(sort_diagonal<dimType, offsetType, dataType>(dias, diaCount) == true);
    dest->dia_length_aligned = newlength;
    dest->dia_offsets = (offsetType*)malloc(sizeof(offsetType)*diaCount);
    dest->dia_data = (dataType*)malloc(sizeof(dataType)*diaCount*newlength);
    for (dimType i = (dimType)0; i < diaCount; i++)
    {
	dest->dia_offsets[i] = dias[i].offset;
	memcpy(dest->dia_data + i * newlength, dias[i].data, sizeof(dataType)*newlength);
    }
    for (dimType i = (dimType)0; i < diaCount; i++)
    {
        free(dias[i].data);
    }
    free(dias);
    return true;
}

template <class dimType, class offsetType, class dataType>
void bdia2coo(bdia_matrix<dimType, offsetType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);

    dimType cooid = dimType(0);
    dimType bandnum = source->bdia_band_num;
    dimType length = source->bdia_length_aligned;
    for (dimType band = (dimType)0; band < bandnum; band++)
    {
	dimType start = source->bdia_bptr[band];
	dimType end = source->bdia_bptr[band + 1];
	dimType offset = source->bdia_offsets[band];
        for (dimType i = start; i < end; i++)
        {
	    for (dimType row = 0; row < length; row++)
	    {
		offsetType col = row + offset;
		if (col >= dimType(0) && col < source->bdia_length)
		{
		    dataType tmpData = source->bdia_data[i * length + row];
		    if ( tmpData != dataType(0))
		    {
			dest->coo_row_id[cooid] = row;
			dest->coo_col_id[cooid] = col;
			dest->coo_data[cooid] = tmpData;
			cooid = cooid + 1;
		    }
		}
	    }
	    offset++;
        }
    }
    if (!if_sorted_coo<dimType, dataType>(dest))
    {
        bool res = sort_coo<dimType, dataType>(dest);
        assert(res == true);
    }
}


template <class dimType, class offsetType, class dataType>
bool coo2bdia(coo_matrix<dimType, dataType>* source, bdia_matrix<dimType, offsetType, dataType>* dest, dimType diaAlignment)
{
    using namespace std;
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dia_matrix<dimType, offsetType, dataType> diamat;
    bool diaflag = coo2dia<dimType, offsetType, dataType>(source, &diamat, diaAlignment);
    if (!diaflag)
	return diaflag;
    vector<offsetType> bandOffset;
    vector<unsigned int> bandCount;
    bandOffset.reserve(diamat.dia_num);
    bandCount.reserve(diamat.dia_num);
    dimType diaid = (dimType)0;
    dimType lastOffset = diamat.dia_offsets[0];
    bandOffset.push_back(lastOffset);
    bandCount.push_back(1);
    diaid++;
    while (diaid < diamat.dia_num)
    {
	dimType curOffset = diamat.dia_offsets[diaid];
	if (curOffset - lastOffset == 1)
	{
	    bandCount[bandCount.size() - 1]++;
	}
	else
	{
	    bandOffset.push_back(curOffset);
	    bandCount.push_back(1);
	}
	lastOffset = curOffset;
	diaid++;
    }
    assert(bandOffset.size() == bandCount.size());
    
    dest->bdia_band_num = bandOffset.size();
    dest->bdia_length = diamat.dia_length;
    dest->bdia_length_aligned = diamat.dia_length_aligned;
    dest->bdia_offsets = (offsetType*)malloc(sizeof(offsetType)*bandOffset.size());
    dest->bdia_bptr = (dimType*)malloc(sizeof(dimType)*(bandOffset.size() + 1));
    dest->bdia_data = (dataType*)malloc(sizeof(dataType)*diamat.dia_length_aligned*diamat.dia_num);
    for (dimType i = (dimType)0; i < bandOffset.size(); i++)
	dest->bdia_offsets[i] = bandOffset[i];
    dest->bdia_bptr[0] = 0;
    for (dimType i = (dimType)0; i < bandOffset.size(); i++)
	dest->bdia_bptr[i+1] = dest->bdia_bptr[i] + bandCount[i];
    memcpy(dest->bdia_data, diamat.dia_data, sizeof(dataType)*diamat.dia_length_aligned*diamat.dia_num);
    free_dia_matrix(diamat);
    return true;
}

template <class dimType, class dataType>
void csr2coo(csr_matrix<dimType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    if (!if_sorted_coo(source))
    {
	assert(sort_coo(source) == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);

    memcpy(dest->coo_data, source->csr_data, sizeof(dataType)*nnz);
    memcpy(dest->coo_col_id, source->csr_col_id, sizeof(dimType)*nnz);
    for (dimType i = (dimType)0; i < source->matinfo.height; i++)
    {
	dimType start = source->csr_row_ptr[i];
	dimType end = source->csr_row_ptr[i+1];
	for (dimType j = start; j < end; j++)
	{
	    dest->coo_row_id[j] = i;
	}
    }
}

template <class dimType, class dataType>
void coo2csr(coo_matrix<dimType, dataType>* source, csr_matrix<dimType, dataType>* dest)
{
    if (!if_sorted_coo(source))
    {
	assert(sort_coo(source) == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->csr_row_ptr = (dimType*)malloc(sizeof(dimType)*(source->matinfo.height + 1));
    dest->csr_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->csr_data = (dataType*)malloc(sizeof(dataType)*nnz);

    memcpy(dest->csr_data, source->coo_data, sizeof(dataType)*nnz);
    memcpy(dest->csr_col_id, source->coo_col_id, sizeof(dimType)*nnz);

    dest->csr_row_ptr[0] = 0;
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	dest->csr_row_ptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    dest->csr_row_ptr[curRow] = dest->csr_row_ptr[curRow - 1];
	    curRow++;
	}
    }
}

template <class dimType, class dataType>
void ell2coo(ell_matrix<dimType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);

    dimType cooid = (dimType)0;
    for (dimType j = (dimType)0; j < source->matinfo.height; j++)
    {
	for (dimType i = (dimType)0; i < source->ell_num; i++)
	{
	    if (source->ell_data[i * source->ell_height_aligned + j] != (dataType)0)
	    {
		dest->coo_row_id[cooid] = j;
		dest->coo_col_id[cooid] = source->ell_col_id[i * source->ell_height_aligned + j];
		dest->coo_data[cooid] = source->ell_data[i * source->ell_height_aligned + j];
		cooid++;
	    }
	}
    }
    assert(cooid == nnz);
    
}

// if ellnum == 0, use the max number of nonzero per row as the ellnum value
template <class dimType, class dataType>
void coo2ell(coo_matrix<dimType, dataType>* source, ell_matrix<dimType, dataType>* dest, dimType alignment, dimType ellnum)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;
    
    csr_matrix<dimType, dataType> csrmat;
    coo2csr<dimType, dataType>(source, &csrmat);
   
    if (ellnum == (dimType)0)
    {
	for (dimType i = (dimType)0; i < csrmat.matinfo.height; i++)
	{
	    dimType size = csrmat.csr_row_ptr[i+1] - csrmat.csr_row_ptr[i];
	    if (size > ellnum)
		ellnum = size;
	}
    }
    dest->ell_num = ellnum;
    dimType newlength = aligned_length(source->matinfo.height, alignment);
    dest->ell_height_aligned = newlength;

    dest->ell_col_id = (dimType*)malloc(sizeof(dimType)*newlength*ellnum);
    dest->ell_data = (dataType*)malloc(sizeof(dataType)*newlength*ellnum);
    for (dimType i = (dimType)0; i < newlength * ellnum; i++)
    {
	dest->ell_col_id[i] = (dimType)0;
	dest->ell_data[i] = (dataType)0;
    }

    for (dimType i = (dimType)0; i < source->matinfo.height; i++)
    {
	dimType start = csrmat.csr_row_ptr[i];
	dimType end = csrmat.csr_row_ptr[i+1];
	assert(end - start <= ellnum);
	dimType lastcolid = (dimType)0;
	for (dimType j = start; j < end; j++)
	{
	    dimType colid = csrmat.csr_col_id[j];
	    dataType data = csrmat.csr_data[j];
	    dest->ell_col_id[i + (j - start) * newlength] = colid;
	    dest->ell_data[i + (j - start) * newlength] = data;
	    lastcolid = colid;
	}
	for (dimType j = end; j < start + ellnum; j++)
	{
	    dest->ell_col_id[i + (j - start) * newlength] = lastcolid;
	    dest->ell_data[i + (j - start) * newlength] = (dataType)0;
	}
    }

    free_csr_matrix(csrmat);

}


template <class dimType, class dataType>
void coo2sell(coo_matrix<dimType, dataType>* source, sell_matrix<dimType, dataType>* dest, unsigned int sliceheight)
{
    using namespace std;
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->sell_slice_height = sliceheight;
    dimType height = source->matinfo.height;
    dimType slicenum = height / sliceheight;
    if (height % sliceheight != 0)
	slicenum++;
    dest->sell_slice_num = slicenum;
    vector<dimType>csr_row_ptr(height + 1);
    csr_row_ptr[0] = 0;
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	csr_row_ptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    csr_row_ptr[curRow] = csr_row_ptr[curRow - 1];
	    curRow++;
	}
    }

    vector<dimType*> colid(slicenum, NULL);
    vector<dataType*> data(slicenum, NULL);
    vector<dimType> slicesize(slicenum, 0);
    dest->sell_slice_ptr = (dimType*)malloc(sizeof(dimType)*(slicenum + 1));
    dest->sell_slice_ptr[0] = 0;
    
    for (dimType i = (dimType)0; i < slicenum; i++)
    {
	dimType maxwidth = 0;
	for (dimType j = i * sliceheight; j < (i+1)*sliceheight && j < height; j++)
	{
	    dimType size = csr_row_ptr[j + 1] - csr_row_ptr[j];
	    if (size > maxwidth)
		maxwidth = size;
	}
	dimType newsize = dest->sell_slice_ptr[i] + maxwidth * sliceheight;
	dest->sell_slice_ptr[i + 1] = newsize;
	colid[i] = (dimType*)malloc(sizeof(dimType)*maxwidth*sliceheight);
	data[i] = (dataType*)malloc(sizeof(dataType)*maxwidth*sliceheight);
	colid[i][0] = (dimType)0;
	slicesize[i] = maxwidth*sliceheight;
	
	dimType offset = i * sliceheight;
	for (dimType j = offset; j < offset + sliceheight && j < height; j++)
	{
	    dimType start = csr_row_ptr[j];
	    dimType end = csr_row_ptr[j + 1];
	    for (dimType k = start; k < end; k++)
	    {
		colid[i][(k - start) * sliceheight + j - offset] = source->coo_col_id[k];
		data[i][(k - start) * sliceheight + j - offset] = source->coo_data[k];
	    }
	    
	    for (dimType k = end; k < start + maxwidth; k++)
	    {
		dimType cpyid = (k-start)*sliceheight+j - offset;
		if (cpyid >= sliceheight)
		    colid[i][cpyid] = colid[i][cpyid - sliceheight];
		else
		    colid[i][cpyid] = colid[i][0];
		data[i][cpyid] = (dataType)0;
	    }
	    
	}
	
	if (i == (slicenum - 1) && (slicenum * sliceheight) > height)
	{
	    for (dimType j = height; j < (slicenum*sliceheight); j++)
	    {
		for (dimType k = (dimType)0; k < maxwidth; k++)
		{
		    dimType cpyid = k*sliceheight+j - offset;
		    if (cpyid >= sliceheight)
			colid[i][cpyid] = colid[i][cpyid - sliceheight];
		    else
			colid[i][cpyid] = colid[i][0];
		    data[i][cpyid] = (dataType)0;
		}
	    }
	}
    }

    dimType totalsize = dest->sell_slice_ptr[slicenum];
    dest->sell_col_id = (dimType*)malloc(sizeof(dimType)*totalsize);
    dest->sell_data = (dataType*)malloc(sizeof(dataType)*totalsize);
    dimType index = (dimType)0;
    for (dimType i = 0; i < slicenum; i++)
    {
	for (dimType j = 0; j < slicesize[i]; j ++)
	{
	    dest->sell_col_id[index] = colid[i][j];
	    dest->sell_data[index] = data[i][j];
	    index++;
	}
    }
    for (dimType i = 0; i < slicenum; i++)
    {
	free(colid[i]);
	free(data[i]);
    }
}

template<class dimType, class dataType>
struct oneElem
{
    dimType rowid;
    dimType colid;
    dataType data;
};


template<class dimType, class dataType>
bool compareByColumn(const oneElem<dimType, dataType>& a, const oneElem<dimType, dataType>& b)
{
    if (a.colid != b.colid)
	return a.colid < b.colid;
    return a.rowid < b.rowid;
}

template<class dimType, class dataType>
struct compareCol
{
    bool operator()(const oneElem<dimType, dataType>& a, const oneElem<dimType, dataType>& b)
    {
	if (a.colid != b.colid)
	    return a.colid < b.colid;
	return a.rowid < b.rowid;
    }
};



template <class dimType, class dataType>
void bcsr2coo(const bcsr_matrix<dimType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    using namespace std;
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;
   
    unsigned int bwidth = source->bcsr_bwidth;
    unsigned int bheight = source->bcsr_bheight;
    unsigned int blocksize = bwidth * bheight;
    dimType nnz = source->matinfo.nnz;
    dimType brownum = source->bcsr_row_num;

    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);
    
    dimType counter = (dimType)0;
    for (dimType i = (dimType)0; i < brownum; i++)
    {
	dimType start = source->bcsr_row_ptr[i];
	dimType end = source->bcsr_row_ptr[i+1];
	dimType outerRowid = i * bheight;
	for (unsigned int k = 0; k < blocksize; k++)
	{
	    unsigned int innerColid = k % bwidth;
	    unsigned int innerRowid = k / bwidth;
	    for (dimType j = start; j < end; j++)
	    {
		dimType outerColid = source->bcsr_col_id[j];
		if (source->bcsr_data[j + k * source->bcsr_aligned_size] != (dataType)0)
		{
		    dest->coo_row_id[counter] = outerRowid + innerRowid;
		    dest->coo_col_id[counter] = outerColid + innerColid;
		    dest->coo_data[counter] = source->bcsr_data[j + k * source->bcsr_aligned_size];
		    counter++;
		}
	    }
	}
    }
    assert(counter == nnz);

    assert(sort_coo(dest) == true);
}



// On Nvidia Divice, the global memory should be padded to a multiple of 128 bytes, or 32 floats. 
template <class dimType, class dataType>
void coo2bcsr(coo_matrix<dimType, dataType>* source, bcsr_matrix<dimType, dataType>* dest, unsigned int bwidth, unsigned int bheight, dimType alignment)
{
    using namespace std;
    if (!if_sorted_coo<dimType, dataType>(source))
    {
	bool res = sort_coo<dimType, dataType>(source);
	assert(res == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dest->bcsr_bwidth = bwidth;
    dest->bcsr_bheight = bheight;

    // Compute the row pointer array as in the csr case
    vector<dimType> rowptr(source->matinfo.height + 1, 0);
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    dimType nnz = source->matinfo.nnz;
    rowptr[0] = 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	rowptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    rowptr[curRow] = rowptr[curRow - 1];
	    curRow++;
	}
    }

    vector<dimType> blockrowptr;
    vector<dimType> blockcolid;
    vector<dataType> blockdata;
    dimType browsize = source->matinfo.height / bheight;
    if (source->matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(nnz);
    blockdata.reserve(nnz*2);
    unsigned int blocksize = bwidth * bheight;
    dimType curdataid = dimType(0);
    for (row = dimType(0); row < source->matinfo.height; row += bheight)
    {
	dimType start = rowptr[row];
	dimType end;
	if (row + bheight <= source->matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[source->matinfo.height];
	dimType size = end - start;
	dimType blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<dimType, dataType> > elements(size);
	for (dimType i = start; i < end; i++)
	{
	    elements[i - start].rowid = source->coo_row_id[i];
	    elements[i - start].colid = source->coo_col_id[i];
	    elements[i - start].data = source->coo_data[i];
	}
	compareCol<dimType, dataType> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	dimType blocknum = dimType(0);
	dimType elemid = dimType(0);
	while (elemid < size)
	{
	    dimType rowid = elements[elemid].rowid;
	    dimType colid = elements[elemid].colid;
	    dataType data = elements[elemid].data;
	    blocknum++;
	    dimType bcolid = colid - (colid % bwidth);
	    dimType browid = rowid - (rowid % bheight);
	    dimType curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    for (unsigned int i = 0; i < blocksize; i++)
		blockdata.push_back((dataType)0);
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    while (elemid < size)
	    {
		rowid = elements[elemid].rowid;
		colid = elements[elemid].colid;
		data = elements[elemid].data;
		bcolid = colid - (colid % bwidth);
		if (bcolid != curbcolid)
		{
		    elemid--;
		    break;
		}
		browid = rowid - (rowid % bheight);
		innercolid = colid - bcolid;
		innerrowid = rowid - browid;
		innerid = innerrowid * bwidth + innercolid;
		blockdata[curdataid + innerid] = data;
		elemid++;
	    }
	    elemid++;
	    curdataid += blocksize;
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());

    dest->bcsr_row_num = blockrowptr.size() - 1;
    dest->bcsr_row_ptr = (dimType*)malloc(sizeof(dimType)*blockrowptr.size());
    dest->bcsr_col_id = (dimType*) malloc(sizeof(dimType)*blockcolid.size());
    dimType newlength = aligned_length<dimType>((dimType)blockcolid.size(), alignment);
    dest->bcsr_data = (dataType*)malloc(sizeof(dataType)*newlength*blocksize);
    memset(dest->bcsr_data, 0, sizeof(dataType)*newlength*blocksize);
    dest->bcsr_aligned_size = newlength;
    dest->bcsr_block_num = blockcolid.size();
    for (dimType i = dimType(0); i < blockrowptr.size(); i++)
	dest->bcsr_row_ptr[i] = blockrowptr[i];
    for (dimType i = dimType(0); i < blockcolid.size(); i++)
	dest->bcsr_col_id[i] = blockcolid[i];
    for (dimType i = dimType(0); i < blockcolid.size(); i++)
    {
	for (unsigned j = 0; j < blocksize; j++)
	{
	    dest->bcsr_data[j * newlength + i] = blockdata[i * blocksize + j];
	}
    }


}

// On Nvidia Divice, the global memory should be padded to a multiple of 128 bytes, or 32 floats. 
template <class dimType, class dataType>
bool coo2b4csr(coo_matrix<dimType, dataType>* source, b4csr_matrix<dimType, dataType>* dest, unsigned int bwidth, unsigned int bheight, dimType alignment)
{
    using namespace std;
    assert(bwidth % 4 == 0);
    if (!if_sorted_coo<dimType, dataType>(source))
    {
	bool res = sort_coo<dimType, dataType>(source);
	assert(res == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dest->b4csr_bwidth = bwidth;
    dest->b4csr_bheight = bheight;

    // Compute the row pointer array as in the csr case
    vector<dimType> rowptr(source->matinfo.height + 1, 0);
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    dimType nnz = source->matinfo.nnz;
    rowptr[0] = 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	rowptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    rowptr[curRow] = rowptr[curRow - 1];
	    curRow++;
	}
    }

    vector<dimType> blockrowptr;
    vector<dimType> blockcolid;
    vector<dataType> blockdata;
    dimType browsize = source->matinfo.height / bheight;
    if (source->matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(nnz);
    blockdata.reserve(nnz*2);
    unsigned int blocksize = bwidth * bheight;
    dimType curdataid = dimType(0);
    for (row = dimType(0); row < source->matinfo.height; row += bheight)
    {
	dimType start = rowptr[row];
	dimType end;
	if (row + bheight <= source->matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[source->matinfo.height];
	dimType size = end - start;
	dimType blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<dimType, dataType> > elements(size);
	for (dimType i = start; i < end; i++)
	{
	    elements[i - start].rowid = source->coo_row_id[i];
	    elements[i - start].colid = source->coo_col_id[i];
	    elements[i - start].data = source->coo_data[i];
	}
	compareCol<dimType, dataType> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	dimType blocknum = dimType(0);
	dimType elemid = dimType(0);
	while (elemid < size)
	{
	    dimType rowid = elements[elemid].rowid;
	    dimType colid = elements[elemid].colid;
	    dataType data = elements[elemid].data;
	    blocknum++;
	    dimType bcolid = colid - (colid % bwidth);
	    dimType browid = rowid - (rowid % bheight);
	    dimType curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    for (unsigned int i = 0; i < blocksize; i++)
		blockdata.push_back((dataType)0);
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    while (elemid < size)
	    {
		rowid = elements[elemid].rowid;
		colid = elements[elemid].colid;
		data = elements[elemid].data;
		bcolid = colid - (colid % bwidth);
		if (bcolid != curbcolid)
		{
		    elemid--;
		    break;
		}
		browid = rowid - (rowid % bheight);
		innercolid = colid - bcolid;
		innerrowid = rowid - browid;
		innerid = innerrowid * bwidth + innercolid;
		blockdata[curdataid + innerid] = data;
		elemid++;
	    }
	    elemid++;
	    curdataid += blocksize;
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());

    if (blockcolid.size() > (MAX_MEM_OBJ/(bwidth*bheight*sizeof(dataType))))
    {
	printf("BCSR too large blocknum %d bwidth %d bheight %d\n", blockcolid.size(), bwidth, bheight);
	return false;
    }

    dest->b4csr_row_num = blockrowptr.size() - 1;
    dest->b4csr_row_ptr = (dimType*)malloc(sizeof(dimType)*blockrowptr.size());
    dest->b4csr_col_id = (dimType*) malloc(sizeof(dimType)*blockcolid.size());
    dimType newlength = aligned_length<dimType>((dimType)blockcolid.size() * 4, alignment);
    unsigned int bwidth4num = bwidth / 4;
    dest->b4csr_data = (dataType*)malloc(sizeof(dataType)*newlength*bheight*bwidth4num);
    memset(dest->b4csr_data, 0, sizeof(dataType)*newlength*bheight*bwidth4num);
    dest->b4csr_aligned_size = newlength;
    dest->b4csr_block_num = blockcolid.size();
    for (dimType i = dimType(0); i < blockrowptr.size(); i++)
	dest->b4csr_row_ptr[i] = blockrowptr[i];
    for (dimType i = dimType(0); i < blockcolid.size(); i++)
	dest->b4csr_col_id[i] = blockcolid[i] / 4;
    dimType onerowlength = newlength * bheight;
    for (dimType i = dimType(0); i < blockcolid.size(); i++)
    {
	for (unsigned int h = 0; h < bheight; h++)
	{
	    for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
	    {
		for (unsigned int w = 0; w < 4; w++)
		{		
		    dest->b4csr_data[w4 * onerowlength + h * newlength + i * 4 + w] = blockdata[i * blocksize + h * bwidth + w4 * 4 + w];
		}
	    }
	}
    }
    return true;
}


template <class dimType, class dataType>
void bell2coo(bell_matrix<dimType, dataType>* source, coo_matrix<dimType, dataType>* dest)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dimType nnz = source->matinfo.nnz;
    dest->coo_row_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_col_id = (dimType*)malloc(sizeof(dimType)*nnz);
    dest->coo_data = (dataType*)malloc(sizeof(dataType)*nnz);
    
    dimType blockcolsize = source->bell_height_aligned * source->bell_bwidth * source->bell_bheight;
    dimType cooid = (dimType)0;
    for (dimType i = (dimType)0; i < source->bell_block_num; i++)
    {
	for (dimType j = (dimType)0; j < source->bell_row_num; j++)
	{
	    for (unsigned int k = 0; k < source->bell_bwidth * source->bell_bheight; k++)
	    {
		dataType data = source->bell_data[j + k * source->bell_height_aligned + i * blockcolsize];
		if (data != (dataType)0)
		{
		    unsigned int innerColid = k % source->bell_bwidth;
		    unsigned int innerRowid = k / source->bell_bwidth;
		    dest->coo_row_id[cooid] = j * source->bell_bheight + innerRowid;
		    dest->coo_col_id[cooid] = source->bell_col_id[i * source->bell_height_aligned + j] + innerColid;
		    dest->coo_data[cooid] = data;
		    cooid++;
		}
	    }
	}
    }
    assert(cooid == nnz);
    if (!if_sorted_coo(dest))
    {
	assert(sort_coo(dest) == true);
    }
}

template <class dimType, class dataType>
void coo2bell(coo_matrix<dimType, dataType>* source, bell_matrix<dimType, dataType>* dest, unsigned int bwidth, unsigned int bheight, dimType alignment, dimType ellnum)
{
    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;
    dest->bell_bwidth = bwidth;
    dest->bell_bheight = bheight;
    
    bcsr_matrix<dimType, dataType> bcsrmat;
    coo2bcsr<dimType, dataType>(source, &bcsrmat, bwidth, bheight, alignment);
   
    if (ellnum == (dimType)0)
    {
	for (dimType i = (dimType)0; i < bcsrmat.bcsr_row_num; i++)
	{
	    dimType size = bcsrmat.bcsr_row_ptr[i+1] - bcsrmat.bcsr_row_ptr[i];
	    if (size > ellnum)
		ellnum = size;
	}
    }
    dest->bell_block_num = ellnum;
    dest->bell_row_num = bcsrmat.bcsr_row_num;
    dimType newlength = aligned_length(bcsrmat.bcsr_row_num, alignment);
    dest->bell_height_aligned = newlength;

    dest->bell_col_id = (dimType*)malloc(sizeof(dimType)*newlength*ellnum);
    dest->bell_data = (dataType*)malloc(sizeof(dataType)*newlength*ellnum*bwidth*bheight);
    for (dimType i = (dimType)0; i < newlength * ellnum; i++)
    {
	dest->bell_col_id[i] = (dimType)0;
	for (unsigned int j = 0; j < bwidth * bheight; j++)
	{
	    dest->bell_data[i + j * newlength * ellnum] = (dataType)0;
	}
    }

    dimType blockcolsize = newlength * bwidth * bheight;
    for (dimType i = (dimType)0; i < bcsrmat.bcsr_row_num; i++)
    {
	dimType start = bcsrmat.bcsr_row_ptr[i];
	dimType end = bcsrmat.bcsr_row_ptr[i+1];
	assert(end - start <= ellnum);
	dimType lastcolid = (dimType)0;
	for (dimType j = start; j < end; j++)
	{
	    dimType colid = bcsrmat.bcsr_col_id[j];
	    dest->bell_col_id[i + (j - start) * newlength] = colid;
	    for (unsigned int k = 0; k < bwidth * bheight; k++)
	    {
		dest->bell_data[i + k * newlength + (j - start) * blockcolsize] = bcsrmat.bcsr_data[j + k * bcsrmat.bcsr_aligned_size];
	    }
	    lastcolid = colid;
	}
	for (dimType j = end; j < start + ellnum; j++)
	{
	    dest->bell_col_id[i + (j - start) * newlength] = lastcolid;
	    for (unsigned int k = 0; k < bwidth * bheight; k++)
	    {
		dest->bell_data[i + k * newlength + (j - start) * blockcolsize] = (dataType)0;
	    }
	}
    }

    free_bcsr_matrix(bcsrmat);

}

// On Nvidia Divice, the global memory should be padded to a multiple of 128 bytes, or 32 floats. 
template <class dimType, class dataType>
bool coo2b4ell(coo_matrix<dimType, dataType>* source, b4ell_matrix<dimType, dataType>* dest, unsigned int bwidth, unsigned int bheight, dimType alignment, dimType ellnum)
{
    using namespace std;
    assert(bwidth % 4 == 0);
    if (!if_sorted_coo<dimType, dataType>(source))
    {
	bool res = sort_coo<dimType, dataType>(source);
	assert(res == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dest->b4ell_bwidth = bwidth;
    dest->b4ell_bheight = bheight;

    // Compute the row pointer array as in the csr case
    vector<dimType> rowptr(source->matinfo.height + 1, 0);
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    dimType nnz = source->matinfo.nnz;
    rowptr[0] = 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	rowptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    rowptr[curRow] = rowptr[curRow - 1];
	    curRow++;
	}
    }

    vector<dimType> blockrowptr;
    vector<dimType> blockcolid;
    vector<dataType> blockdata;
    dimType browsize = source->matinfo.height / bheight;
    if (source->matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(nnz);
    blockdata.reserve(nnz*2);
    unsigned int blocksize = bwidth * bheight;
    dimType curdataid = dimType(0);
    for (row = dimType(0); row < source->matinfo.height; row += bheight)
    {
	dimType start = rowptr[row];
	dimType end;
	if (row + bheight <= source->matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[source->matinfo.height];
	dimType size = end - start;
	dimType blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<dimType, dataType> > elements(size);
	for (dimType i = start; i < end; i++)
	{
	    elements[i - start].rowid = source->coo_row_id[i];
	    elements[i - start].colid = source->coo_col_id[i];
	    elements[i - start].data = source->coo_data[i];
	}
	compareCol<dimType, dataType> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	dimType blocknum = dimType(0);
	dimType elemid = dimType(0);
	while (elemid < size)
	{
	    dimType rowid = elements[elemid].rowid;
	    dimType colid = elements[elemid].colid;
	    dataType data = elements[elemid].data;
	    blocknum++;
	    dimType bcolid = colid - (colid % bwidth);
	    dimType browid = rowid - (rowid % bheight);
	    dimType curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    for (unsigned int i = 0; i < blocksize; i++)
		blockdata.push_back((dataType)0);
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    while (elemid < size)
	    {
		rowid = elements[elemid].rowid;
		colid = elements[elemid].colid;
		data = elements[elemid].data;
		bcolid = colid - (colid % bwidth);
		if (bcolid != curbcolid)
		{
		    elemid--;
		    break;
		}
		browid = rowid - (rowid % bheight);
		innercolid = colid - bcolid;
		innerrowid = rowid - browid;
		innerid = innerrowid * bwidth + innercolid;
		blockdata[curdataid + innerid] = data;
		elemid++;
	    }
	    elemid++;
	    curdataid += blocksize;
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());

    if (ellnum == (dimType)0)
    {
	for (dimType i = (dimType)0; i < blockrowptr.size() - 1; i++)
	{
	    dimType size = blockrowptr[i + 1] - blockrowptr[i];
	    if (size > ellnum)
		ellnum = size;
	}
    }
    if ((blockrowptr.size()-1)*ellnum > (MAX_MEM_OBJ/(bwidth*bheight*sizeof(dataType))))
    {
	printf("BELL too large ellnum %d bwidth %d bheight %d\n", ellnum, bwidth, bheight);
	return false;
    }
    

    dest->b4ell_row_num = blockrowptr.size() - 1;
    dest->b4ell_block_num = ellnum; 
    dimType newlength = aligned_length<dimType>(dest->b4ell_row_num, alignment);
    dimType newf4length = aligned_length<dimType>(4 * dest->b4ell_row_num, alignment);
    dest->b4ell_height_aligned = newlength;
    dest->b4ell_float4_aligned = newf4length;

    dest->b4ell_col_id = (dimType*) malloc(sizeof(dimType)*newlength*ellnum);
    unsigned int bwidth4num = bwidth / 4;
    dest->b4ell_data = (dataType*)malloc(sizeof(dataType)*newf4length*bheight*bwidth4num*ellnum);

    dimType newblockcolsize = newf4length * bwidth4num * bheight;
    dimType newblockw4size = newf4length * bheight;
    for (dimType r = (dimType)0; r < dest->b4ell_row_num; r++)
    {
	dimType start = blockrowptr[r];
	dimType end = blockrowptr[r + 1];
	assert(end - start <= ellnum);
	dimType lastcolid = (dimType)0;
	for (dimType j = start; j < end; j++)
	{
	    dimType colid = blockcolid[j] / 4;
	    dest->b4ell_col_id[r + (j - start) * newlength] = colid;
	    lastcolid = colid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			dest->b4ell_data[(j - start) * newblockcolsize + h * newf4length + w4 * newblockw4size + r * 4 + w] = blockdata[j * blocksize + h * bwidth + w4 * 4 + w];
		    }
		}
	    }
	}
	for (dimType j = end; j < start + ellnum; j++)
	{   
	    dest->b4ell_col_id[r + (j - start) * newlength] = lastcolid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			dest->b4ell_data[(j - start) * newblockcolsize + h * newf4length + w4 * newblockw4size + r * 4 + w] = (dataType)0;
		    }
		}
	    }
	}
    }
    return true;

}

template <class dimType, class dataType>
bool coo2sbell(coo_matrix<dimType, dataType>* source, sbell_matrix<dimType, dataType>* dest, unsigned int bwidth, unsigned int bheight, unsigned int sliceheight)
{
    using namespace std;
    assert(bwidth % 4 == 0);
    if (!if_sorted_coo<dimType, dataType>(source))
    {
	bool res = sort_coo<dimType, dataType>(source);
	assert(res == true);
    }

    dest->matinfo.width = source->matinfo.width;
    dest->matinfo.height = source->matinfo.height;
    dest->matinfo.nnz = source->matinfo.nnz;

    dest->sbell_bwidth = bwidth;
    dest->sbell_bheight = bheight;

    // Compute the row pointer array as in the csr case
    vector<dimType> rowptr(source->matinfo.height + 1, 0);
    dimType row = (dimType) 0;
    dimType curRow = (dimType) 0;
    dimType nnz = source->matinfo.nnz;
    rowptr[0] = 0;
    while (row < nnz)
    {
	while (source->coo_row_id[row] == curRow && row < nnz)
	    row++;
	curRow++;
	rowptr[curRow] = row;
    }
    if (curRow < source->matinfo.height)
    {
	curRow++;
	while (curRow <= source->matinfo.height)
	{
	    rowptr[curRow] = rowptr[curRow - 1];
	    curRow++;
	}
    }

    vector<dimType> blockrowptr;
    vector<dimType> blockcolid;
    vector<dataType> blockdata;
    dimType browsize = source->matinfo.height / bheight;
    if (source->matinfo.height % bheight != 0)
	browsize++;

    dimType slicenum = browsize / sliceheight;
    if (browsize % sliceheight != 0)
	slicenum++;
    dest->sbell_slice_num = slicenum;
    dest->sbell_row_num = browsize;
    dest->sbell_slice_height = sliceheight;

    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(nnz);
    blockdata.reserve(nnz*2);
    unsigned int blocksize = bwidth * bheight;
    dimType curdataid = dimType(0);
    for (row = dimType(0); row < source->matinfo.height; row += bheight)
    {
	dimType start = rowptr[row];
	dimType end;
	if (row + bheight <= source->matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[source->matinfo.height];
	dimType size = end - start;
	dimType blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<dimType, dataType> > elements(size);
	for (dimType i = start; i < end; i++)
	{
	    elements[i - start].rowid = source->coo_row_id[i];
	    elements[i - start].colid = source->coo_col_id[i];
	    elements[i - start].data = source->coo_data[i];
	}
	compareCol<dimType, dataType> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	dimType blocknum = dimType(0);
	dimType elemid = dimType(0);
	while (elemid < size)
	{
	    dimType rowid = elements[elemid].rowid;
	    dimType colid = elements[elemid].colid;
	    dataType data = elements[elemid].data;
	    blocknum++;
	    dimType bcolid = colid - (colid % bwidth);
	    dimType browid = rowid - (rowid % bheight);
	    dimType curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    for (unsigned int i = 0; i < blocksize; i++)
		blockdata.push_back((dataType)0);
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    while (elemid < size)
	    {
		rowid = elements[elemid].rowid;
		colid = elements[elemid].colid;
		data = elements[elemid].data;
		bcolid = colid - (colid % bwidth);
		if (bcolid != curbcolid)
		{
		    elemid--;
		    break;
		}
		browid = rowid - (rowid % bheight);
		innercolid = colid - bcolid;
		innerrowid = rowid - browid;
		innerid = innerrowid * bwidth + innercolid;
		blockdata[curdataid + innerid] = data;
		elemid++;
	    }
	    elemid++;
	    curdataid += blocksize;
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());

    dest->sbell_slice_ptr = (dimType*)malloc(sizeof(dimType)*(slicenum+1));
    dest->sbell_slice_ptr[0] = 0;
    vector<dimType> slicewidth(slicenum, 0);
    for (dimType i = (dimType)0; i < slicenum; i++)
    {
	dimType maxwidth = 0;
	for (dimType j = i * sliceheight; j < (i+1)*sliceheight && j < browsize; j++)
	{
	    dimType size = blockrowptr[j + 1] - blockrowptr[j];
	    if (size > maxwidth)
		maxwidth = size;
	}
	slicewidth[i] = maxwidth;
	dest->sbell_slice_ptr[i + 1] = dest->sbell_slice_ptr[i] + maxwidth * sliceheight;
    }

    dimType totalsize = dest->sbell_slice_ptr[slicenum];
    if (totalsize > (MAX_MEM_OBJ/(bwidth*bheight*sizeof(dataType))))
    {
	if (dest->sbell_slice_ptr)
	    free(dest->sbell_slice_ptr);
	printf("SBELL too large totalsize %d bwidth %d bheight %d\n", totalsize, bwidth, bheight);
	return false;
    }
    dest->sbell_col_id = (dimType*)malloc(sizeof(dimType)*totalsize);
    dest->sbell_data = (dataType*)malloc(sizeof(dataType)*totalsize*bwidth*bheight);
    memset(dest->sbell_col_id, 0, sizeof(dimType)*totalsize);
    memset(dest->sbell_data, 0, sizeof(dataType)*totalsize*bwidth*bheight);

    unsigned int bwidth4num = bwidth / 4;

    dimType newblockcolsize = sliceheight * bwidth4num * bheight * 4;
    dimType newblockw4size = sliceheight * bheight * 4;
    dimType heightfour = sliceheight * 4;
    for (dimType r = (dimType)0; r < browsize; r++)
    {
	dimType start = blockrowptr[r];
	dimType end = blockrowptr[r + 1];
	dimType sliceid = r / sliceheight;
	dimType rowid = r % sliceheight;
	dimType colstart = dest->sbell_slice_ptr[sliceid];
	dimType datastart = colstart * bwidth * bheight;
	dimType lastcolid = (dimType)0;
	for (dimType j = start; j < end; j++)
	{
	    dimType colid = blockcolid[j] / 4;
	    dest->sbell_col_id[colstart + rowid + (j - start) * sliceheight] = colid;
	    lastcolid = colid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			dest->sbell_data[datastart + (j - start) * newblockcolsize + h * heightfour + w4 * newblockw4size + rowid * 4 + w] = blockdata[j * blocksize + h * bwidth + w4 * 4 + w];
		    }
		}
	    }
	}
	for (dimType j = end; j < start + slicewidth[sliceid]; j++)
	{   
	    dest->sbell_col_id[colstart + rowid + (j - start) * sliceheight] = lastcolid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			dest->sbell_data[datastart + (j - start) * newblockcolsize + h * heightfour + w4 * newblockw4size + rowid * 4 + w] = (dataType)0;
		    }
		}
	    }
	}
    }
    return true;

}

#endif

