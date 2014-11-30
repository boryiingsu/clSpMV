#ifndef __SPMV_SERIAL__H__
#define __SPMV_SERIAL__H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "matrix_storage.h"

template <class dimType, class offsetType, class dataType>
void dia_spmv(dia_matrix<dimType, offsetType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType dianum = mat->dia_num;
    dimType length = mat->dia_length_aligned;

    for (dimType row = (dimType)0; row < length; row++)
    {
	dataType accumulant = (dataType)0;
	for (dimType j = (dimType)0; j < dianum; j++)
	{
	    offsetType col = row + mat->dia_offsets[j];
	    dataType matelement = mat->dia_data[row + j * length];
	    dataType vecelement = (dataType) 0;
	    if (col >= (dimType)0 && col < vec_size)
		vecelement = vec[col];
	    accumulant += vecelement * matelement;
	}
	result[row] += accumulant;
    }
  
}

template <class dimType, class offsetType, class dataType>
void dia_ext_spmv(dia_ext_matrix<dimType, offsetType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType dianum = mat->dia_num;
    dimType length = mat->dia_length_aligned;

    for (dimType row = (dimType)0; row < length; row++)
    {
	dataType accumulant = (dataType)0;
	for (dimType j = (dimType)0; j < dianum; j++)
	{
	    offsetType col = row + mat->dia_offsets[j];
	    dataType matelement = mat->dia_data[row + j * length];
	    dataType vecelement = vec[col];
	    accumulant += vecelement * matelement;
	}
	result[row] += accumulant;
    }
  
}

template <class dimType, class offsetType, class dataType>
void bdia_spmv(bdia_matrix<dimType, offsetType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType bandnum = mat->bdia_band_num;
    dimType length = mat->bdia_length_aligned;

    for (dimType row = (dimType)0; row < length; row++)
    {
	dataType accumulant = (dataType)0;
	for (dimType band = (dimType)0; band < bandnum; band++)
	{
	    dimType start = mat->bdia_bptr[band];
	    dimType end = mat->bdia_bptr[band + 1];
	    dimType offset = mat->bdia_offsets[band];
	    for (dimType i = start; i < end; i++)
	    {
		offsetType col = row + offset;
		dataType matelement = mat->bdia_data[row + i * length];
		dataType vecelement = (dataType) 0;
		if (col >= (dimType)0 && col < vec_size)
		    vecelement = vec[col];
		accumulant += vecelement * matelement;
		offset++;
	    }
	}
	result[row] += accumulant;
    }
  
}


template <class dimType, class dataType>
void ell_spmv(ell_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType ellnum = mat->ell_num;
    dimType paddedheight = mat->ell_height_aligned;
    dimType height = mat->matinfo.height;
    
    for (dimType i = (dimType)0; i < ellnum; i++)
    {
	for (dimType j = (dimType)0; j < height; j++)
	{
	    dimType colid = mat->ell_col_id[i * paddedheight + j];
	    dataType data = mat->ell_data[i * paddedheight + j];
	    result[j] += data * vec[colid];
	}
    }
}

template <class dimType, class dataType>
void sell_spmv(sell_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    unsigned int sliceheight = mat->sell_slice_height;
    dimType slicenum = mat->sell_slice_num;
    dimType height = mat->matinfo.height;
    
    for (dimType i = (dimType)0; i < slicenum; i++)
    {
	dimType start = mat->sell_slice_ptr[i];
	dimType end = mat->sell_slice_ptr[i + 1];
	dimType ellnum = (end - start) / sliceheight;
	for (dimType j = (dimType)0; j < ellnum; j++)
	{
	    dimType offset = i * sliceheight;
	    for (dimType row = i * sliceheight; row < (i+1)*sliceheight && row < height; row++)
	    {
		dimType colid = mat->sell_col_id[start + row - offset + j * sliceheight];
		dataType data = mat->sell_data[start + row - offset + j * sliceheight];
		result[row] += data * vec[colid];
	    }	    
	}
    }
}

template <class dimType, class dataType>
void sbell_spmv(sbell_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    unsigned int bwidth = mat->sbell_bwidth;
    unsigned int bheight = mat->sbell_bheight;
    unsigned int sliceheight = mat->sbell_slice_height;
    dimType slicenum = mat->sbell_slice_num;
    unsigned int blocksize = bwidth * bheight;
    for (dimType i = (dimType)0; i < slicenum; i++)
    {
	dimType start = mat->sbell_slice_ptr[i];
	dimType end = mat->sbell_slice_ptr[i + 1];
	dimType ellnum = (end - start) / sliceheight;
	for (unsigned int s = 0; s < sliceheight; s++)
	{
	    unsigned int width4num = bwidth / 4;
	    dimType coloffset = start + s;
	    dimType dataoffset = start * blocksize + s * 4;
	    for (dimType j = (dimType)0; j < ellnum; j++)
	    {
		dimType col = mat->sbell_col_id[coloffset] * 4;
		for (unsigned int w4 = 0; w4 < width4num; w4++)
		{
		    for (unsigned int h = 0; h < bheight; h++)
		    {
			dimType row = (i * sliceheight + s) * bheight + h;
			if (row < mat->matinfo.height)
			{
			    for (unsigned int w = 0; w < 4; w++)
			    {
				dataType data = mat->sbell_data[dataoffset + w];
				if (col + w >= mat->matinfo.width)
				    break;
				result[row] += data * vec[col + w];
			    }
			}
			dataoffset += sliceheight * 4;
		    }
		    col += 4;
		}
		coloffset += sliceheight;
	    }
	}
    }
}

template <class dimType, class dataType>
void coo_spmv(coo_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType nnz = mat->matinfo.nnz;
    for (dimType i = (dimType)0; i < nnz; i++)
    {
	dimType row = mat->coo_row_id[i];
	dimType col = mat->coo_col_id[i];
	dataType data = mat->coo_data[i];
	result[row] += data * vec[col];
    }
}

template <class dimType, class dataType>
void csr_spmv(csr_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType height = mat->matinfo.height;
    for (dimType row = (dimType)0; row < height; row++)
    {
	dimType start = mat->csr_row_ptr[row];
	dimType end = mat->csr_row_ptr[row+1];
	dataType accumulant = (dataType) 0;
	for (dimType j = start; j < end; j++)
	{
	    dimType col = mat->csr_col_id[j];
	    dataType data = mat->csr_data[j];
	    accumulant += data * vec[col];
	}
	result[row] += accumulant;
    }
}

template<class dimType, class dataType>
void bcsr_spmv(bcsr_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;
    dimType rownum = mat->bcsr_row_num;
    unsigned int bwidth = mat->bcsr_bwidth;
    unsigned int bheight = mat->bcsr_bheight;
    unsigned int blocksize = bwidth * bheight;
    for (dimType i = (dimType)0; i < rownum; i++)
    {
	dimType start = mat->bcsr_row_ptr[i];
	dimType end = mat->bcsr_row_ptr[i+1];
	dimType outerRowid = i * bheight;
	for (unsigned int k = 0; k < blocksize; k++)
	{
	    unsigned int innerColid = k % bwidth;
	    unsigned int innerRowid = k / bwidth;
	    for (dimType j = start; j < end; j++)
	    {
		dimType outerColid = mat->bcsr_col_id[j];
		dimType colid = outerColid + innerColid;
		dimType rowid = outerRowid + innerRowid;
		if (colid < vec_size && rowid < mat->matinfo.height)
		{
		    result[rowid] += mat->bcsr_data[j + k * mat->bcsr_aligned_size] * vec[colid];
		}
	    }
	}
    }
}

template<class dimType, class dataType>
void bell_spmv(bell_matrix<dimType, dataType>* mat, dataType* vec, dataType* result, dimType vec_size)
{
    //for (dimType i = (dimType)0; i < mat->matinfo.height; i++)
	//result[i] = (dataType)0;

    unsigned int bwidth = mat->bell_bwidth;
    unsigned int bheight = mat->bell_bheight;
    dimType rownum = mat->bell_row_num;
    dimType blocknum = mat->bell_block_num;
    dimType paddedrownum = mat->bell_height_aligned;

    dimType blockcolsize = paddedrownum * bwidth * bheight;
    for (dimType i = (dimType)0; i < blocknum; i++)
    {
	for (dimType j = (dimType)0; j < rownum; j++)
	{
	    for (unsigned int k = 0; k < bwidth * bheight; k++)
	    {
		dataType data = mat->bell_data[j + k * paddedrownum + i * blockcolsize];
		unsigned int innerColid = k % bwidth;
		unsigned int innerRowid = k / bwidth;
		dimType colid = mat->bell_col_id[i * paddedrownum + j] + innerColid;
		result[j * bheight + innerRowid] += data * vec[colid];
	    }
	}
    }
    
}


template<class dimType, class dataType>
bool compareVectors(dataType* vec1, dataType* vec2, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
        dataType diff = vec1[i] - vec2[i];
        
        //diff = diff;
        if (diff < 0)
            diff *= (dataType)(-1);
	    if (vec1[i] != vec2[i])
        {
            printf("Diff: id %d vec1 %f vec2 %f diff %f\n", i, vec1[i], vec2[i], diff);
	        return false;
        }
    }
    return true;
}


template<class dimType, class dataType>
void initVectorZero(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)0;
    }
}

template<class dimType, class dataType>
void initVectorOne(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)1;
    }
}



template<class dimType, class dataType>
void initVectorArbitrary(dataType* vec, dimType vec_size)
{
    for (dimType i = 0; i < vec_size; i++)
    {
	vec[i] = (dataType)(i % 1024);
    }
}


template <class dimType, class offsetType, class dataType>
void doDIA(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    int alignment = 512;
    dia_matrix<dimType, offsetType, dataType> dia_mat;
    bool res = coo2dia<dimType, offsetType, dataType>(mat, &dia_mat, alignment);
    
    if (res)
    {
    dimType padded_size = dia_mat.dia_length_aligned;
    dataType* padded_vec = (dataType*)malloc(sizeof(dataType)*padded_size);
    memcpy(padded_vec, vec, sizeof(dataType)*vec_size);
    

    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*padded_size);
    initVectorZero<dimType, dataType>(result_coo, padded_size);
    coo_spmv<dimType, dataType>(mat, padded_vec, result_coo, padded_size);

    dataType* result_dia = (dataType*)malloc(sizeof(dataType)*padded_size);
    initVectorZero<dimType, dataType>(result_dia, padded_size);
    
    dia_spmv<dimType, offsetType, dataType>(&dia_mat, padded_vec, result_dia, padded_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_dia, padded_size))
	printf("Spmv of coo and dia are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and dia are different!\n");
    free(result_coo);
    free(result_dia);
    free(padded_vec);
    free_dia_matrix(dia_mat);
    }
    printf("Diagonal number %d length %d total element %d nnz %d\n", dia_mat.dia_num, dia_mat.dia_length_aligned, dia_mat.dia_num * dia_mat.dia_length_aligned, dia_mat.matinfo.nnz);
    
    
    
    
    
}

template <class dimType, class offsetType, class dataType>
void doDIAext(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    int alignment = 512;
    dia_ext_matrix<dimType, offsetType, dataType> dia_mat;
    bool res = coo2diaext<dimType, offsetType, dataType>(mat, &dia_mat, alignment);
    
    if (res)
    {
    dimType padded_size = dia_mat.dia_length_aligned;
    dataType* padded_vec = (dataType*)malloc(sizeof(dataType)*vec_size*2);
    memcpy(padded_vec, vec, sizeof(dataType)*vec_size);
    memcpy(padded_vec + vec_size, vec, sizeof(dataType)*vec_size);

    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*padded_size);
    initVectorZero<dimType, dataType>(result_coo, padded_size);
    coo_spmv<dimType, dataType>(mat, padded_vec, result_coo, padded_size);

    dataType* result_dia = (dataType*)malloc(sizeof(dataType)*padded_size);
    initVectorZero<dimType, dataType>(result_dia, padded_size);
    if (res)
        dia_ext_spmv<dimType, offsetType, dataType>(&dia_mat, padded_vec, result_dia, padded_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_dia, padded_size))
	printf("Spmv of coo and dia ext are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and dia ext are different!\n");
    free(result_coo);
    free(result_dia);
    free(padded_vec);
    free_dia_ext_matrix(dia_mat);
    }
    printf("Diagonal number %d length %d total element %d nnz %d\n", dia_mat.dia_num, dia_mat.dia_length_aligned, dia_mat.dia_num * dia_mat.dia_length_aligned, dia_mat.matinfo.nnz);
    
    
    
    
}

template <class dimType, class offsetType, class dataType>
void doBDIA(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    int alignment = 512;
    bdia_matrix<dimType, offsetType, dataType> bdia_mat;
    bool res = coo2bdia<dimType, offsetType, dataType>(mat, &bdia_mat, alignment);
    
    if (res)
    {
	dimType padded_size = bdia_mat.bdia_length_aligned;
	dataType* padded_vec = (dataType*)malloc(sizeof(dataType)*padded_size);
	memcpy(padded_vec, vec, sizeof(dataType)*vec_size);



	dataType* result_coo = (dataType*)malloc(sizeof(dataType)*padded_size);
	initVectorZero<dimType, dataType>(result_coo, padded_size);
	coo_spmv<dimType, dataType>(mat, padded_vec, result_coo, padded_size);

	dataType* result_bdia = (dataType*)malloc(sizeof(dataType)*padded_size);
	initVectorZero<dimType, dataType>(result_bdia, padded_size);

	bdia_spmv<dimType, offsetType, dataType>(&bdia_mat, padded_vec, result_bdia, padded_size);

	if (compareVectors<dimType, dataType>(result_coo, result_bdia, padded_size))
	    printf("Spmv of coo and bdia are equivalent!\n");
	else
	    printf("Wrong!!!!! Spmv of coo and bdia are different!\n");
	free(result_coo);
	free(result_bdia);
	free(padded_vec);
	free_bdia_matrix(bdia_mat);
    }
    
}

template <class dimType, class dataType>
void doELL(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    ell_matrix<dimType, dataType> ell_mat;
    coo2ell<dimType, dataType>(mat, &ell_mat, 32, 0);
    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_coo, vec_size);
    coo_spmv<dimType, dataType>(mat, vec, result_coo, vec_size);

    dataType* result_ell = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_ell, vec_size);
    ell_spmv<dimType, dataType>(&ell_mat, vec, result_ell, vec_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_ell, vec_size))
	printf("Spmv of coo and ell are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and ell are different!\n");

    free_ell_matrix(ell_mat);
    free(result_coo);
    free(result_ell);
}

template <class dimType, class dataType>
void doBELL(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    bell_matrix<dimType, dataType> bell_mat;
    coo2bell<dimType, dataType>(mat, &bell_mat, 4, 4, 32, 0);
    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_coo, vec_size);
    coo_spmv<dimType, dataType>(mat, vec, result_coo, vec_size);

    dataType* result_bell = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_bell, vec_size);
    bell_spmv<dimType, dataType>(&bell_mat, vec, result_bell, vec_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_bell, vec_size))
	printf("Spmv of coo and bell are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and bell are different!\n");

    free_bell_matrix(bell_mat);
    free(result_coo);
    free(result_bell);
}


template <class dimType, class dataType>
void doCSR(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    csr_matrix<dimType, dataType> csr_mat;
    coo2csr<dimType, dataType>(mat, &csr_mat);
    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_coo, vec_size);
    coo_spmv<dimType, dataType>(mat, vec, result_coo, vec_size);

    dataType* result_csr = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_csr, vec_size);
    csr_spmv<dimType, dataType>(&csr_mat, vec, result_csr, vec_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_csr, vec_size))
	printf("Spmv of coo and csr are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and csr are different!\n");

    free_csr_matrix(csr_mat);
    free(result_coo);
    free(result_csr);
}

template <class dimType, class dataType>
void doBCSR(coo_matrix<dimType, dataType>* mat, dataType* vec, dimType vec_size)
{
    bcsr_matrix<dimType, dataType> bcsr_mat;
    coo2bcsr<dimType, dataType>(mat, &bcsr_mat, 4, 4, 32);
    
    dataType* result_coo = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_coo, vec_size);
    coo_spmv<dimType, dataType>(mat, vec, result_coo, vec_size);

    dataType* result_bcsr = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorZero<dimType, dataType>(result_bcsr, vec_size);
    bcsr_spmv<dimType, dataType>(&bcsr_mat, vec, result_bcsr, vec_size);
    
    if (compareVectors<dimType, dataType>(result_coo, result_bcsr, vec_size))
	printf("Spmv of coo and bcsr are equivalent!\n");
    else
	printf("Wrong!!!!! Spmv of coo and bcsr are different!\n");

    free_bcsr_matrix(bcsr_mat);
    free(result_coo);
    free(result_bcsr);
}

template <class dimType, class offsetType, class dataType>
void doExp(coo_matrix<dimType, dataType>* mat)
{
    dimType width = mat->matinfo.width;
    dimType vec_size = width;
    
    dataType* vec = (dataType*)malloc(sizeof(dataType)*vec_size);
    initVectorOne<dimType, dataType>(vec, vec_size);

    
    //doCSR<dimType, dataType>(mat, vec, vec_size);
    //doDIAext<dimType, offsetType, dataType>(mat, vec, vec_size);
    //doDIA<dimType, offsetType, dataType>(mat, vec, vec_size);
    doELL<dimType, dataType>(mat, vec, vec_size);
    //doBDIA<dimType, offsetType, dataType>(mat, vec, vec_size);
    //doBCSR<dimType, dataType>(mat, vec, vec_size);
    doBELL<dimType, dataType>(mat, vec, vec_size);
    
    free(vec);
}


#endif
