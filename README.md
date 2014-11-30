clSpMV
======

This is the code base of my publication in 2012:
Bor-Yiing Su, Kurt Keutzer, "clSpMV: A Cross-Platform OpenCL SpMV Framework on GPUs," in International Conference on Supercomputing (ICS 2012), Italy, June 2012.

Package Description
===================

The clSpMV framework optimizes the Sparse Matrix Vector multiplication (SpMV) kernel on OpenCL enabled GPU platforms. Given a sparse matrix, the ultimate goal is to find the best representation of any sparse matrix, and choose the best kernel implementation on the platform of choice automatically.

Matrix Format
=============

The clSpMV framework uses the Cocktail Format to store a sparse matrix. The Cocktail Format is a combination of different sparse matrix formats. The idea is to represent specialized submatrices using specialized formats. Under the case that the sprase matrix should be represented by a single format, the clSpMV framework will also tell you which single format to choose. So far the package supports 9 different formats in the Cocktail Format:

Diagonal based formats: Diagonal, Banded Diagonal

Flat based formats: Sliced ELL, ELL, CSR, COO

Block based formats: Sliced Blocked ELL, Blocked ELL, Blocked CSR

Executables
===========

The package has two major executables:

spmv_cocktail: Given a sparse matrix, analyze it, represent the matrix using the Cocktail Format, and perform the SpMV kernel. 
Usage: spmv_cocktail matrix.mtx

spmv_all: Represent the matrix using any 1 of the 9 supported format, and evaluate the execution time of different implementations of that format.
Usage: spmv_all input_matrix.mtx method execution_times
The method is the format you want to use:
Method 0: measure the memory bandwidth and kernel launch overhead only
Method 1: use the csr matrix format, using the scalar implementations
Method 2: use the csr matrix format, using the vector implementations
Method 3: use the bdia matrix format
Method 4: use the dia matrix format
Method 5: use the ell matrix format
Method 6: use the coo matrix format
Method 7: use the bell matrix format
Method 8: use the bcsr matrix format
Method 9: use the sell matrix format
Method 10: use the sbell matrix format

Citation
========
Bor-Yiing Su, Kurt Keutzer, "clSpMV: A Cross-Platform OpenCL SpMV Framework on GPUs," in International Conference on Supercomputing (ICS 2012), Italy, June 2012.
@inproceedings{ics2012-su, 
author = {Bor-Yiing Su and Kurt Keutzer}, 
title = {clSpMV: A Cross-Platform OpenCL SpMV Framework on GPUs},
booktitle = {Proceedings of the international conference on Supercomputing},
series = {ICS '12},
year = {2012},
location = {Venice, Italy}, }




