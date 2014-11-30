#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <set>
#include <map>

#include "matrix_storage.h"
#include "analyze.h"
#include "fileio.h"
#include "util.h"


double estimate_flat_time(vector<int>& rownnz, int nnz, benchmark_all& bench);

void ReadOverhead(vector<overhead_entry>& overhead, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/overhead.ben");
    printf("filename %s\n", filename);
    fflush(stdout);
    FILE* infile = fopen(filename, "r");
    overhead.reserve(100);
    overhead_entry tmp;
    while (fscanf(infile, "%d %d %lf", &tmp.groupnum, &tmp.threadnum, &tmp.microsec) != EOF)
	overhead.push_back(tmp);
    fclose(infile);
    printf("filename %s\n", filename);
    fflush(stdout);
}

void ReadBDIA(vector<bdia_entry>& bdia, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/bdia.ben");
    FILE* infile = fopen(filename, "r");
    bdia.reserve(300);
    bdia_entry tmp;
    while (fscanf(infile, "%d %d", &tmp.dimension, &tmp.width) != EOF)
    {
	for (unsigned int i = 0; i < BDIA_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.bdia_gflops + i);
	bdia.push_back(tmp);
    }
    fclose(infile);
}

void ReadDIA(vector<dia_entry>& dia, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/dia.ben");
    FILE* infile = fopen(filename, "r");
    dia.reserve(300);
    dia_entry tmp;
    while (fscanf(infile, "%d %d", &tmp.dimension, &tmp.width) != EOF)
    {
	for (unsigned int i = 0; i < DIA_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.dia_gflops + i);
	dia.push_back(tmp);
    }
    fclose(infile);
}

void ReadSBELL(vector<sbell_entry>& sbell, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/sbell.ben");
    FILE* infile = fopen(filename, "r");
    sbell.reserve(1200);
    sbell_entry tmp;
    while (fscanf(infile, "%d %d %d %d", &tmp.bheight, &tmp.bwidth, &tmp.dimension, &tmp.blocknum) != EOF)
    {
	for (unsigned int i = 0; i < SBELL_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.sbell_gflops + i);
	sbell.push_back(tmp);
    }
    fclose(infile);
}

void ReadBELL(vector<bell_entry>& bell, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/bell.ben");
    FILE* infile = fopen(filename, "r");
    bell.reserve(1200);
    bell_entry tmp;
    while (fscanf(infile, "%d %d %d %d", &tmp.bheight, &tmp.bwidth, &tmp.dimension, &tmp.blocknum) != EOF)
    {
	for (unsigned int i = 0; i < BELL_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.bell_gflops + i);
	bell.push_back(tmp);
    }
    fclose(infile);
}

void ReadBCSR(vector<bcsr_entry>& bcsr, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/bcsr.ben");
    FILE* infile = fopen(filename, "r");
    bcsr.reserve(600);
    bcsr_entry tmp;
    while (fscanf(infile, "%d %d %d %d", &tmp.bheight, &tmp.bwidth, &tmp.dimension, &tmp.blocknum) != EOF)
    {
	for (unsigned int i = 0; i < BCSR_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.bcsr_gflops + i);
	bcsr.push_back(tmp);
    }
    fclose(infile);
}

void ReadSELL(vector<sell_entry>& sell, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/sell.ben");
    FILE* infile = fopen(filename, "r");
    sell.reserve(100);
    sell_entry tmp;
    while (fscanf(infile, "%d %d", &tmp.dimension, &tmp.width) != EOF)
    {
	for (unsigned int i = 0; i < SELL_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.sell_gflops + i);
	sell.push_back(tmp);
    }
    fclose(infile);
}

void ReadELL(vector<ell_entry>& ell, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/ell.ben");
    FILE* infile = fopen(filename, "r");
    ell.reserve(100);
    ell_entry tmp;
    while (fscanf(infile, "%d %d", &tmp.dimension, &tmp.width) != EOF)
    {
	for (unsigned int i = 0; i < ELL_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.ell_gflops + i);
	ell.push_back(tmp);
    }
    fclose(infile);
}

void ReadCSR(vector<csr_entry>& csr, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/csr.ben");
    FILE* infile = fopen(filename, "r");
    csr.reserve(600);
    csr_entry tmp;
    while (fscanf(infile, "%d %d %d", &tmp.dimension, &tmp.width, &tmp.csr_groupnum) != EOF)
    {
	for (unsigned int i = 0; i < CSR_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.csr_gflops + i);
	csr.push_back(tmp);
    }
    fclose(infile);
}

void ReadCOO(vector<coo_entry>& coo, char* clspmvpath)
{
    char filename[1000];
    sprintf(filename, "%s%s", clspmvpath, "/benchmark/coo.ben");
    FILE* infile = fopen(filename, "r");
    coo.reserve(600);
    coo_entry tmp;
    while (fscanf(infile, "%d %d %d", &tmp.dimension, &tmp.width, &tmp.coo_groupnum) != EOF)
    {
	for (unsigned int i = 0; i < COO_IMP_NUM; i++)
	    fscanf(infile, "%lf", tmp.coo_gflops + i);
	coo.push_back(tmp);
    }
    fclose(infile);
}


void ReadBench(benchmark_all& bench)
{
    char* clspmvpath = getenv("CLSPMVPATH");
    printf("Read Overhead\n");
    ReadOverhead(bench.overhead, clspmvpath);
    printf("%d Entries read\n", bench.overhead.size());
    printf("Read bdia\n");
    ReadBDIA(bench.bdia, clspmvpath);
    printf("%d Entries read\n", bench.bdia.size());
    printf("Read dia\n");
    ReadDIA(bench.dia, clspmvpath);
    printf("%d Entries read\n", bench.dia.size());
    printf("Read sbell\n");
    ReadSBELL(bench.sbell, clspmvpath);
    printf("%d Entries read\n", bench.sbell.size());
    printf("Read bell\n");
    ReadBELL(bench.bell, clspmvpath);
    printf("%d Entries read\n", bench.bell.size());
    printf("Read bcsr\n");
    ReadBCSR(bench.bcsr, clspmvpath);
    printf("%d Entries read\n", bench.bcsr.size());
    printf("Read sell\n");
    ReadSELL(bench.sell, clspmvpath);
    printf("%d Entries read\n", bench.sell.size());
    printf("Read ell\n");
    ReadELL(bench.ell, clspmvpath);
    printf("%d Entries read\n", bench.ell.size());
    printf("Read csr\n");
    ReadCSR(bench.csr, clspmvpath);
    printf("%d Entries read\n", bench.csr.size());
    printf("Read coo\n");
    ReadCOO(bench.coo, clspmvpath);
    printf("%d Entries read\n", bench.coo.size());
}

void OutBench(benchmark_all& bench, int num)
{
    for (int i = 0; i < num; i++)
    {
	overhead_entry tmp = bench.overhead[i];
	printf("Overhead gn %d tn %d time %lf\n", tmp.groupnum, tmp.threadnum, tmp.microsec);
    }
    for (int i = 0; i < num; i++)
    {
	bdia_entry tmp = bench.bdia[i];
	printf("bdia di %d wi %d ", tmp.dimension, tmp.width);
	for (int j = 0; j < BDIA_IMP_NUM; j++)
	    printf("%lf ", tmp.bdia_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	dia_entry tmp = bench.dia[i];
	printf("dia di %d wi %d ", tmp.dimension, tmp.width);
	for (int j = 0; j < DIA_IMP_NUM; j++)
	    printf("%lf ", tmp.dia_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	sbell_entry tmp = bench.sbell[i];
	printf("sbell bh %d bw %d di %d bm %d ", tmp.bheight, tmp.bwidth, tmp.dimension, tmp.blocknum);
	for (int j = 0; j < SBELL_IMP_NUM; j++)
	    printf("%lf ", tmp.sbell_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	bell_entry tmp = bench.bell[i];
	printf("bell bh %d bw %d di %d bm %d ", tmp.bheight, tmp.bwidth, tmp.dimension, tmp.blocknum);
	for (int j = 0; j < BELL_IMP_NUM; j++)
	    printf("%lf ", tmp.bell_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	bcsr_entry tmp = bench.bcsr[i];
	printf("bcsr bh %d bw %d di %d bm %d ", tmp.bheight, tmp.bwidth, tmp.dimension, tmp.blocknum);
	for (int j = 0; j < BCSR_IMP_NUM; j++)
	    printf("%lf ", tmp.bcsr_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	sell_entry tmp = bench.sell[i];
	printf("sell di %d wi %d ", tmp.dimension, tmp.width);
	for (int j = 0; j < SELL_IMP_NUM; j++)
	    printf("%lf ", tmp.sell_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	ell_entry tmp = bench.ell[i];
	printf("ell di %d wi %d ", tmp.dimension, tmp.width);
	for (int j = 0; j < ELL_IMP_NUM; j++)
	    printf("%lf ", tmp.ell_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	csr_entry tmp = bench.csr[i];
	printf("csr di %d wi %d gn %d ", tmp.dimension, tmp.width, tmp.csr_groupnum);
	for (int j = 0; j < CSR_IMP_NUM; j++)
	    printf("%lf ", tmp.csr_gflops[j]);
	printf("\n");
    }
    for (int i = 0; i < num; i++)
    {
	coo_entry tmp = bench.coo[i];
	printf("coo di %d wi %d gn %d ", tmp.dimension, tmp.width, tmp.coo_groupnum);
	for (int j = 0; j < COO_IMP_NUM; j++)
	    printf("%lf ", tmp.coo_gflops[j]);
	printf("\n");
    }
}


void find_fourid(vector<basic_entry>& basic, four_index& index, int dim, int width)
{
    vector<int> dims;
    vector<int> offset;
    dims.push_back(basic[0].dimension);
    offset.push_back(0);
    for (int i = 0; i < basic.size(); i++)
    {
	if (basic[i].dimension != dims[dims.size() - 1])
	{
	    dims.push_back(basic[i].dimension);
	    offset.push_back(i);
	}
    }
    offset.push_back(basic.size());
    int dim_s = 0;
    int dim_l = 0;
    if (dim < dims[0])
	dim_s = dim_l = 0;
    else if (dim > dims[dims.size() - 1])
	dim_s = dim_l = dims.size() - 1;
    else
    {
	for (int i = 0; i < dims.size(); i++)
	{
	    if (dim == dims[i])
	    {
		dim_s = dim_l = i;
		break;
	    }
	    if (i < dims.size() - 1)
	    {
		if (dims[i] < dim && dims[i + 1] > dim)
		{
		    dim_s = i;
		    dim_l = i+1;
		    break;
		}
	    }
	}
    }
    if (width < basic[offset[dim_s]].width)
    {
	index.dim_s_wid_s = index.dim_s_wid_l = offset[dim_s];
    }
    else if (width > basic[offset[dim_s+1] - 1].width)
    {
	index.dim_s_wid_s = index.dim_s_wid_l = offset[dim_s+1] - 1;
    }
    else
    {
	for (int i = offset[dim_s]; i < offset[dim_s + 1]; i++)
	{
	    if (width == basic[i].width)
	    {
		index.dim_s_wid_s = index.dim_s_wid_l = i;
		break;
	    }
	    if (i < offset[dim_s + 1] - 1)
	    {
		if (basic[i].width < width && basic[i+1].width > width)
		{
		    index.dim_s_wid_s = i;
		    index.dim_s_wid_l = i+1;
		}
	    }
	}
    }
    if (width < basic[offset[dim_l]].width)
    {
	index.dim_l_wid_s = index.dim_l_wid_l = offset[dim_l];
    }
    else if (width > basic[offset[dim_l+1] - 1].width)
    {
	index.dim_l_wid_s = index.dim_l_wid_l = offset[dim_l+1] - 1;
    }
    else
    {
	for (int i = offset[dim_l]; i < offset[dim_l + 1]; i++)
	{
	    if (width == basic[i].width)
	    {
		index.dim_l_wid_s = index.dim_l_wid_l = i;
		break;
	    }
	    if (i < offset[dim_l + 1] - 1)
	    {
		if (basic[i].width < width && basic[i+1].width > width)
		{
		    index.dim_l_wid_s = i;
		    index.dim_l_wid_l = i+1;
		}
	    }
	}
    }
    /*
    printf("(%d, %d) -- (%d, %d) ; (%d, %d) -- (%d, %d) Target (%d, %d)\n", 
	    basic[index.dim_s_wid_s].dimension, basic[index.dim_s_wid_s].width,
	    basic[index.dim_s_wid_l].dimension, basic[index.dim_s_wid_l].width,
	    basic[index.dim_l_wid_s].dimension, basic[index.dim_l_wid_s].width,
	    basic[index.dim_l_wid_l].dimension, basic[index.dim_l_wid_l].width,
	    dim, width
	    );
    */

}

double linear_inter(int small, double svalue, int large, double lvalue, int index)
{
    if (small == large)
    {
	assert(lvalue == svalue);
	return lvalue;
    }
    assert(small <= index);
    assert(index <= large);
    return svalue + ((double)(index - small))*(lvalue - svalue)/((double)(large-small));
}

double find_overhead(vector<overhead_entry>& benchover, int groupnum, int threadnum)
{
    vector<basic_entry> basic;
    basic.resize(benchover.size());
    for (int i = 0; i < benchover.size(); i++)
    {
	basic[i].dimension = benchover[i].groupnum;
	basic[i].width = benchover[i].threadnum;
    }
    four_index index;
    find_fourid(basic, index, groupnum, threadnum);
    double svalue = linear_inter(benchover[index.dim_s_wid_s].threadnum, benchover[index.dim_s_wid_s].microsec, benchover[index.dim_s_wid_l].threadnum, benchover[index.dim_s_wid_l].microsec, threadnum);
    double lvalue = linear_inter(benchover[index.dim_l_wid_s].threadnum, benchover[index.dim_l_wid_s].microsec, benchover[index.dim_l_wid_l].threadnum, benchover[index.dim_l_wid_l].microsec, threadnum);
    return linear_inter(benchover[index.dim_s_wid_s].groupnum, svalue, benchover[index.dim_l_wid_l].groupnum, lvalue, groupnum);

}


void maxflop_bdia(vector<bdia_entry>& benchbdia, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<basic_entry> basic;
    basic.resize(benchbdia.size());
    for (int i = 0; i < benchbdia.size(); i++)
    {
	basic[i].dimension = benchbdia[i].dimension;
	basic[i].width = benchbdia[i].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.bdia_meth_num = 0;
    info.bdia_max_flop = 0.0f;
    for (int i = 0; i < BDIA_IMP_NUM; i++)
    {
	double svalue = linear_inter(benchbdia[index.dim_s_wid_s].width, benchbdia[index.dim_s_wid_s].bdia_gflops[i], benchbdia[index.dim_s_wid_l].width, benchbdia[index.dim_s_wid_l].bdia_gflops[i], nnzperrow);
	double lvalue = linear_inter(benchbdia[index.dim_l_wid_s].width, benchbdia[index.dim_l_wid_s].bdia_gflops[i], benchbdia[index.dim_l_wid_l].width, benchbdia[index.dim_l_wid_l].bdia_gflops[i], nnzperrow);
	double finalvalue = linear_inter(benchbdia[index.dim_s_wid_s].dimension, svalue, benchbdia[index.dim_l_wid_l].dimension, lvalue, rownum);
	if (finalvalue > info.bdia_max_flop)
	{
	    info.bdia_meth_num = i;
	    info.bdia_max_flop = finalvalue;
	}
	//printf("bdia Method %d dim %d width %d value %f\n", i, rownum, nnzperrow, finalvalue);
    }
    printf("bdia Max method %d value %f\n", info.bdia_meth_num, info.bdia_max_flop);
}

void maxflop_dia(vector<dia_entry>& benchdia, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<basic_entry> basic;
    basic.resize(benchdia.size());
    for (int i = 0; i < benchdia.size(); i++)
    {
	basic[i].dimension = benchdia[i].dimension;
	basic[i].width = benchdia[i].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.dia_meth_num = 0;
    info.dia_max_flop = 0.0f;
    for (int i = 0; i < DIA_IMP_NUM; i++)
    {
	double svalue = linear_inter(benchdia[index.dim_s_wid_s].width, benchdia[index.dim_s_wid_s].dia_gflops[i], benchdia[index.dim_s_wid_l].width, benchdia[index.dim_s_wid_l].dia_gflops[i], nnzperrow);
	double lvalue = linear_inter(benchdia[index.dim_l_wid_s].width, benchdia[index.dim_l_wid_s].dia_gflops[i], benchdia[index.dim_l_wid_l].width, benchdia[index.dim_l_wid_l].dia_gflops[i], nnzperrow);
	double finalvalue = linear_inter(benchdia[index.dim_s_wid_s].dimension, svalue, benchdia[index.dim_l_wid_l].dimension, lvalue, rownum);
	if (finalvalue > info.dia_max_flop)
	{
	    info.dia_meth_num = i;
	    info.dia_max_flop = finalvalue;
	}
	//printf("dia Method %d dim %d width %d value %f\n", i, rownum, nnzperrow, finalvalue);
    }
    printf("dia Max method %d value %f\n", info.dia_meth_num, info.dia_max_flop);
}

void maxflop_sell(vector<sell_entry>& benchsell, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<basic_entry> basic;
    basic.resize(benchsell.size());
    for (int i = 0; i < benchsell.size(); i++)
    {
	basic[i].dimension = benchsell[i].dimension;
	basic[i].width = benchsell[i].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.sell_meth_num = 0;
    info.sell_max_flop = 0.0f;
    for (int i = 0; i < SELL_IMP_NUM; i++)
    {
	double svalue = linear_inter(benchsell[index.dim_s_wid_s].width, benchsell[index.dim_s_wid_s].sell_gflops[i], benchsell[index.dim_s_wid_l].width, benchsell[index.dim_s_wid_l].sell_gflops[i], nnzperrow);
	double lvalue = linear_inter(benchsell[index.dim_l_wid_s].width, benchsell[index.dim_l_wid_s].sell_gflops[i], benchsell[index.dim_l_wid_l].width, benchsell[index.dim_l_wid_l].sell_gflops[i], nnzperrow);
	double finalvalue = linear_inter(benchsell[index.dim_s_wid_s].dimension, svalue, benchsell[index.dim_l_wid_l].dimension, lvalue, rownum);
	if (finalvalue > info.sell_max_flop)
	{
	    info.sell_meth_num = i;
	    info.sell_max_flop = finalvalue;
	}
	//printf("sell Method %d dim %d width %d value %f\n", i, rownum, nnzperrow, finalvalue);
    }
    printf("sell Max method %d value %f\n", info.sell_meth_num, info.sell_max_flop);
}

void maxflop_ell(vector<ell_entry>& benchell, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<basic_entry> basic;
    basic.resize(benchell.size());
    for (int i = 0; i < benchell.size(); i++)
    {
	basic[i].dimension = benchell[i].dimension;
	basic[i].width = benchell[i].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.ell_meth_num = 0;
    info.ell_max_flop = 0.0f;
    for (int i = 0; i < ELL_IMP_NUM; i++)
    {
	double svalue = linear_inter(benchell[index.dim_s_wid_s].width, benchell[index.dim_s_wid_s].ell_gflops[i], benchell[index.dim_s_wid_l].width, benchell[index.dim_s_wid_l].ell_gflops[i], nnzperrow);
	double lvalue = linear_inter(benchell[index.dim_l_wid_s].width, benchell[index.dim_l_wid_s].ell_gflops[i], benchell[index.dim_l_wid_l].width, benchell[index.dim_l_wid_l].ell_gflops[i], nnzperrow);
	double finalvalue = linear_inter(benchell[index.dim_s_wid_s].dimension, svalue, benchell[index.dim_l_wid_l].dimension, lvalue, rownum);
	if (finalvalue > info.ell_max_flop)
	{
	    info.ell_meth_num = i;
	    info.ell_max_flop = finalvalue;
	}
	//printf("ell Method %d dim %d width %d value %f\n", i, rownum, nnzperrow, finalvalue);
    }
    printf("ell Max method %d value %f\n", info.ell_meth_num, info.ell_max_flop);
}

void maxflop_sbell(vector<sbell_entry>& benchsbell, maxflop_info& info, int rownum, int nnzperrow)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    vector<int> offset(9, 0);
    offset[0] = 0;
    int b = 1;
    for (int i = 0; i < benchsbell.size(); i++)
    {
	if (benchsbell[i].bwidth == bw[b] && benchsbell[i].bheight == bh[b])
	{
	    offset[b] = i;
	    b++;
	    if (b == 9)
		break;
	}
    }
    offset[8] = benchsbell.size();
    for (b = 0; b < 8; b++)
    {
	vector<basic_entry> basic;
	basic.resize(offset[b+1] - offset[b]);
	int basicid = 0;
	int blocknum = (int)(((double)nnzperrow) / ((double)bw[b]) + 0.5);
	for (int i = offset[b]; i < offset[b+1]; i++)
	{
	    basic[basicid].dimension = benchsbell[i].dimension;
	    basic[basicid].width = benchsbell[i].blocknum;
	    basicid++;
	}
	four_index index;
	find_fourid(basic, index, rownum, blocknum);
	info.sbell_meth_num[b] = 0;
	info.sbell_max_flop[b] = 0.0f;
	
	index.dim_s_wid_s += offset[b];
	index.dim_s_wid_l += offset[b];
	index.dim_l_wid_s += offset[b];
	index.dim_l_wid_l += offset[b];
	for (int i = 0; i < SBELL_IMP_NUM; i++)
	{
	    double svalue = linear_inter(benchsbell[index.dim_s_wid_s].blocknum, benchsbell[index.dim_s_wid_s].sbell_gflops[i], benchsbell[index.dim_s_wid_l].blocknum, benchsbell[index.dim_s_wid_l].sbell_gflops[i], blocknum);
	    double lvalue = linear_inter(benchsbell[index.dim_l_wid_s].blocknum, benchsbell[index.dim_l_wid_s].sbell_gflops[i], benchsbell[index.dim_l_wid_l].blocknum, benchsbell[index.dim_l_wid_l].sbell_gflops[i], blocknum);
	    double finalvalue = linear_inter(benchsbell[index.dim_s_wid_s].dimension, svalue, benchsbell[index.dim_l_wid_l].dimension, lvalue, rownum);
	    if (finalvalue > info.sbell_max_flop[b])
	    {
		info.sbell_meth_num[b] = i;
		info.sbell_max_flop[b] = finalvalue;
	    }
	    //printf("sbell Method %d bh %d bw %d dim %d block %d value %f\n", i, bh[b], bw[b], rownum, blocknum, finalvalue);
	}
	printf("sbell Max method %d value %f\n", info.sbell_meth_num[b], info.sbell_max_flop[b]);
    }
}

void maxflop_bell(vector<bell_entry>& benchbell, maxflop_info& info, int rownum, int nnzperrow)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    vector<int> offset(9, 0);
    offset[0] = 0;
    int b = 1;
    for (int i = 0; i < benchbell.size(); i++)
    {
	if (benchbell[i].bwidth == bw[b] && benchbell[i].bheight == bh[b])
	{
	    offset[b] = i;
	    b++;
	    if (b == 9)
		break;
	}
    }
    offset[8] = benchbell.size();
    for (b = 0; b < 8; b++)
    {
	vector<basic_entry> basic;
	basic.resize(offset[b+1] - offset[b]);
	int basicid = 0;
	int blocknum = (int)(((double)nnzperrow) / ((double)bw[b]) + 0.5);
	for (int i = offset[b]; i < offset[b+1]; i++)
	{
	    basic[basicid].dimension = benchbell[i].dimension;
	    basic[basicid].width = benchbell[i].blocknum;
	    basicid++;
	}
	four_index index;
	find_fourid(basic, index, rownum, blocknum);
	info.bell_meth_num[b] = 0;
	info.bell_max_flop[b] = 0.0f;
	
	index.dim_s_wid_s += offset[b];
	index.dim_s_wid_l += offset[b];
	index.dim_l_wid_s += offset[b];
	index.dim_l_wid_l += offset[b];
	for (int i = 0; i < BELL_IMP_NUM; i++)
	{
	    double svalue = linear_inter(benchbell[index.dim_s_wid_s].blocknum, benchbell[index.dim_s_wid_s].bell_gflops[i], benchbell[index.dim_s_wid_l].blocknum, benchbell[index.dim_s_wid_l].bell_gflops[i], blocknum);
	    double lvalue = linear_inter(benchbell[index.dim_l_wid_s].blocknum, benchbell[index.dim_l_wid_s].bell_gflops[i], benchbell[index.dim_l_wid_l].blocknum, benchbell[index.dim_l_wid_l].bell_gflops[i], blocknum);
	    double finalvalue = linear_inter(benchbell[index.dim_s_wid_s].dimension, svalue, benchbell[index.dim_l_wid_l].dimension, lvalue, rownum);
	    if (finalvalue > info.bell_max_flop[b])
	    {
		info.bell_meth_num[b] = i;
		info.bell_max_flop[b] = finalvalue;
	    }
	    //printf("bell Method %d bh %d bw %d dim %d block %d value %f\n", i, bh[b], bw[b], rownum, blocknum, finalvalue);
	}
	printf("bell Max method %d value %f\n", info.bell_meth_num[b], info.bell_max_flop[b]);
    }
}

void maxflop_bcsr(vector<bcsr_entry>& benchbcsr, maxflop_info& info, int rownum, int nnzperrow)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    vector<int> offset(9, 0);
    offset[0] = 0;
    int b = 1;
    for (int i = 0; i < benchbcsr.size(); i++)
    {
	if (benchbcsr[i].bwidth == bw[b] && benchbcsr[i].bheight == bh[b])
	{
	    offset[b] = i;
	    b++;
	    if (b == 9)
		break;
	}
    }
    offset[8] = benchbcsr.size();
    for (b = 0; b < 8; b++)
    {
	vector<basic_entry> basic;
	basic.resize(offset[b+1] - offset[b]);
	int basicid = 0;
	int blocknum = (int)(((double)nnzperrow) / ((double)bw[b]) + 0.5);
	for (int i = offset[b]; i < offset[b+1]; i++)
	{
	    basic[basicid].dimension = benchbcsr[i].dimension;
	    basic[basicid].width = benchbcsr[i].blocknum;
	    basicid++;
	}
	four_index index;
	find_fourid(basic, index, rownum, blocknum);
	info.bcsr_meth_num[b] = 0;
	info.bcsr_max_flop[b] = 0.0f;
	
	index.dim_s_wid_s += offset[b];
	index.dim_s_wid_l += offset[b];
	index.dim_l_wid_s += offset[b];
	index.dim_l_wid_l += offset[b];
	for (int i = 0; i < BCSR_IMP_NUM; i++)
	{
	    double svalue = linear_inter(benchbcsr[index.dim_s_wid_s].blocknum, benchbcsr[index.dim_s_wid_s].bcsr_gflops[i], benchbcsr[index.dim_s_wid_l].blocknum, benchbcsr[index.dim_s_wid_l].bcsr_gflops[i], blocknum);
	    double lvalue = linear_inter(benchbcsr[index.dim_l_wid_s].blocknum, benchbcsr[index.dim_l_wid_s].bcsr_gflops[i], benchbcsr[index.dim_l_wid_l].blocknum, benchbcsr[index.dim_l_wid_l].bcsr_gflops[i], blocknum);
	    double finalvalue = linear_inter(benchbcsr[index.dim_s_wid_s].dimension, svalue, benchbcsr[index.dim_l_wid_l].dimension, lvalue, rownum);
	    if (finalvalue > info.bcsr_max_flop[b])
	    {
		info.bcsr_meth_num[b] = i;
		info.bcsr_max_flop[b] = finalvalue;
	    }
	    //printf("bcsr Method %d bh %d bw %d dim %d block %d value %f\n", i, bh[b], bw[b], rownum, blocknum, finalvalue);
	}
	printf("bcsr Max method %d value %f\n", info.bcsr_meth_num[b], info.bcsr_max_flop[b]);
    }
}

void maxflop_csr(vector<csr_entry>& benchcsr, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<int> offset;
    basic_entry prev_entry;
    prev_entry.dimension = benchcsr[0].dimension;
    prev_entry.width = benchcsr[0].width;
    offset.push_back(0);
    for (int i = 0; i < benchcsr.size(); i++)
    {
	if (prev_entry.dimension != benchcsr[i].dimension || prev_entry.width != benchcsr[i].width)
	{
	    offset.push_back(i);
	    prev_entry.dimension = benchcsr[i].dimension;
	    prev_entry.width = benchcsr[i].width;
	}
    }
    offset.push_back(benchcsr.size());
    vector<basic_entry> basic;
    basic.resize(offset.size() - 1);
    for (int i = 0; i < offset.size() - 1; i++)
    {
	basic[i].dimension = benchcsr[offset[i]].dimension;
	basic[i].width = benchcsr[offset[i]].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.csr_meth_num = 0;
    info.csr_max_flop = 0.0f;
    int group_choice = offset[1] - offset[0];
    for (int g = 0; g < group_choice; g++)
    {
	for (int i = 0; i < CSR_IMP_NUM; i++)
	{
	    double svalue = linear_inter(benchcsr[offset[index.dim_s_wid_s] + g].width, benchcsr[offset[index.dim_s_wid_s] + g].csr_gflops[i], benchcsr[offset[index.dim_s_wid_l] + g].width, benchcsr[offset[index.dim_s_wid_l] + g].csr_gflops[i], nnzperrow);
	    double lvalue = linear_inter(benchcsr[offset[index.dim_l_wid_s] + g].width, benchcsr[offset[index.dim_l_wid_s] + g].csr_gflops[i], benchcsr[offset[index.dim_l_wid_l] + g].width, benchcsr[offset[index.dim_l_wid_l] + g].csr_gflops[i], nnzperrow);
	    double finalvalue = linear_inter(benchcsr[offset[index.dim_s_wid_s] + g].dimension, svalue, benchcsr[offset[index.dim_l_wid_l] + g].dimension, lvalue, rownum);
	    if (finalvalue > info.csr_max_flop)
	    {
		info.csr_meth_num = i;
		info.csr_groupnum = benchcsr[offset[index.dim_s_wid_s] + g].csr_groupnum;
		info.csr_max_flop = finalvalue;
	    }
	    //printf("csr Method %d dim %d width %d gnum %d value %f\n", i, rownum, nnzperrow, benchcsr[offset[index.dim_s_wid_s] + g].csr_groupnum, finalvalue);
	}
    }
    printf("csr Max method %d gnum %d value %f\n", info.csr_meth_num, info.csr_groupnum, info.csr_max_flop);
}

void maxflop_coo(vector<coo_entry>& benchcoo, maxflop_info& info, int rownum, int nnzperrow)
{
    vector<int> offset;
    basic_entry prev_entry;
    prev_entry.dimension = benchcoo[0].dimension;
    prev_entry.width = benchcoo[0].width;
    offset.push_back(0);
    for (int i = 0; i < benchcoo.size(); i++)
    {
	if (prev_entry.dimension != benchcoo[i].dimension || prev_entry.width != benchcoo[i].width)
	{
	    offset.push_back(i);
	    prev_entry.dimension = benchcoo[i].dimension;
	    prev_entry.width = benchcoo[i].width;
	}
    }
    offset.push_back(benchcoo.size());
    vector<basic_entry> basic;
    basic.resize(offset.size() - 1);
    for (int i = 0; i < offset.size() - 1; i++)
    {
	basic[i].dimension = benchcoo[offset[i]].dimension;
	basic[i].width = benchcoo[offset[i]].width;
    }
    four_index index;
    find_fourid(basic, index, rownum, nnzperrow);
    info.coo_meth_num = 0;
    info.coo_max_flop = 0.0f;
    int group_choice = offset[1] - offset[0];
    for (int g = 0; g < group_choice; g++)
    {
	for (int i = 0; i < COO_IMP_NUM; i++)
	{
	    double svalue = linear_inter(benchcoo[offset[index.dim_s_wid_s] + g].width, benchcoo[offset[index.dim_s_wid_s] + g].coo_gflops[i], benchcoo[offset[index.dim_s_wid_l] + g].width, benchcoo[offset[index.dim_s_wid_l] + g].coo_gflops[i], nnzperrow);
	    double lvalue = linear_inter(benchcoo[offset[index.dim_l_wid_s] + g].width, benchcoo[offset[index.dim_l_wid_s] + g].coo_gflops[i], benchcoo[offset[index.dim_l_wid_l] + g].width, benchcoo[offset[index.dim_l_wid_l] + g].coo_gflops[i], nnzperrow);
	    double finalvalue = linear_inter(benchcoo[offset[index.dim_s_wid_s] + g].dimension, svalue, benchcoo[offset[index.dim_l_wid_l] + g].dimension, lvalue, rownum);
	    if (finalvalue > info.coo_max_flop)
	    {
		info.coo_meth_num = i;
		info.coo_groupnum = benchcoo[offset[index.dim_s_wid_s] + g].coo_groupnum;
		info.coo_max_flop = finalvalue;
	    }
	    //printf("coo Method %d dim %d width %d gnum %d value %f\n", i, rownum, nnzperrow, benchcoo[offset[index.dim_s_wid_s] + g].coo_groupnum, finalvalue);
	}
    }
    printf("coo Max method %d gnum %d value %f\n", info.coo_meth_num, info.coo_groupnum, info.coo_max_flop);
}

void count_dia_nnz(coo_matrix<int, float>& coomat, vector<int>& diacount, int threshold)
{
    diacount.resize(2 * threshold + 1, 0); 
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	int col = coomat.coo_col_id[i];
	int offset = col - row + threshold;
	if (offset >= 0 && offset < diacount.size())
	    diacount[offset]++;
    }
}

struct bdia_vs_dia
{
    int min_width;
    int range_s;
    int range_l;
    double bdia_min_width_gflop;
    double dia_min_width_gflop;
    double bdia_single_gflop;
    double dia_single_gflop;
};

void find_min_bdia_width(vector<bdia_entry>& bdia, vector<dia_entry>& dia, int dimension, bdia_vs_dia& bvsd)
{
    vector<int> dims;
    vector<int> offset;
    dims.push_back(bdia[0].dimension);
    offset.push_back(0);
    for (int i = 0; i < bdia.size(); i++)
    {
	if (bdia[i].dimension != dims[dims.size() - 1])
	{
	    dims.push_back(bdia[i].dimension);
	    offset.push_back(i);
	}
    }
    offset.push_back(bdia.size());
    int dim_s = 0;
    int dim_l = 0;
    if (dimension < dims[0])
	dim_s = dim_l = 0;
    else if (dimension > dims[dims.size() - 1])
	dim_s = dim_l = dims.size() - 1;
    else
    {
	for (int i = 0; i < dims.size(); i++)
	{
	    if (dimension == dims[i])
	    {
		dim_s = dim_l = i;
		break;
	    }
	    if (i < dims.size() - 1)
	    {
		if (dims[i] < dimension && dims[i + 1] > dimension)
		{
		    dim_s = i;
		    dim_l = i+1;
		    break;
		}
	    }
	}
    }
    int near_dim = 0;
    if (dim_s == dim_l)
	near_dim = dim_s;
    else
    {
	int dist_s = dimension - bdia[offset[dim_s]].dimension;
	int dist_l = bdia[offset[dim_l]].dimension - dimension;
	assert(dist_s > 0 && dist_l > 0);
	near_dim = (dist_s < dist_l) ? dim_s : dim_l;
    }
    bvsd.range_s = offset[near_dim];
    bvsd.range_l = offset[near_dim + 1];
    for (int i = offset[near_dim]; i < offset[near_dim + 1]; i++)
    {
	int width = bdia[i].width;
	double max_bdia_flop = 0.0f;
	double max_dia_flop = 0.0f;
	for (int j = 0 ; j < BDIA_IMP_NUM; j++)
	{
	    if (bdia[i].bdia_gflops[j] > max_bdia_flop)
		max_bdia_flop = bdia[i].bdia_gflops[j];
	}
	for (int j = 0 ; j < DIA_IMP_NUM; j++)
	{
	    if (dia[i].dia_gflops[j] > max_dia_flop)
		max_dia_flop = dia[i].dia_gflops[j];
	}
	if (width == 1)
	{
	    bvsd.bdia_single_gflop = max_bdia_flop;
	    bvsd.dia_single_gflop = max_dia_flop;
	}
	printf("Width %d bdia %f dia %f\n", width, max_bdia_flop, max_dia_flop);
	if (max_bdia_flop > max_dia_flop)
	{
	    bvsd.min_width = width;
	    bvsd.bdia_min_width_gflop = max_bdia_flop;
	    bvsd.dia_min_width_gflop = max_dia_flop;
	    return;
	}
    }
    bvsd.min_width = 1000000;
}

void bdia_gflop(vector<bdia_entry>& bdia, int range_s, int range_l, int width, double& max_bdia_gflop, int& max_bdia_meth)
{
    int index = 0;
    if (width < bdia[range_s].width)
	index = range_s;
    else if (width > bdia[range_l - 1].width)
	index = range_l - 1;
    else
    {
	for (int i = range_s; i < range_l; i++)
	{
	    if (width == bdia[i].width)
	    {
		index = i;
		break;
	    }
	    if (i < range_l - 1)
	    {
		if (bdia[i].width < width && bdia[i + 1].width > width)
		{
		    index = i;
		    break;
		}
	    }
	}	
    }
    max_bdia_gflop = 0.0f;
    max_bdia_meth = 0;
    for (int i = 0; i < BDIA_IMP_NUM; i++)
    {
	if (bdia[index].bdia_gflops[i] > max_bdia_gflop)
	{
	    max_bdia_gflop = bdia[index].bdia_gflops[i];
	    max_bdia_meth = i;
	}
    }
}

void dia_gflop(vector<dia_entry>& dia, int range_s, int range_l, int width, double& max_dia_gflop, int& max_dia_meth)
{
    int index = 0;
    if (width < dia[range_s].width)
	index = range_s;
    else if (width > dia[range_l - 1].width)
	index = range_l - 1;
    else
    {
	for (int i = range_s; i < range_l; i++)
	{
	    if (width == dia[i].width)
	    {
		index = i;
		break;
	    }
	    if (i < range_l - 1)
	    {
		if (dia[i].width < width && dia[i + 1].width > width)
		{
		    index = i;
		    break;
		}
	    }
	}	
    }
    max_dia_gflop = 0.0f;
    max_dia_meth = 0;
    for (int i = 0; i < DIA_IMP_NUM; i++)
    {
	if (dia[index].dia_gflops[i] > max_dia_gflop)
	{
	    max_dia_gflop = dia[index].dia_gflops[i];
	    max_dia_meth = i;
	}
    }
}

void extract_bdia_only(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, vector<pair<int, int> >& select, int threshold, int bdia_meth)
{
    mat.ifusebdia = true;
    mat.bdia_meth_num = bdia_meth;
    
    int length_aligned = aligned_length(coomat.matinfo.height, GPU_ALIGNMENT);
    mat.bdia.bdia_length = coomat.matinfo.height;
    mat.bdia.bdia_length_aligned = length_aligned;

    vector<int> diaoff(select.size(), 0);
    int bdiannz = 0;
    for (int i = 0; i < select.size(); i++)
    {
	diaoff[i] = select[i].first - threshold;
	bdiannz += select[i].second;
    }
    mat.bdia.matinfo.width = coomat.matinfo.width;
    mat.bdia.matinfo.height = coomat.matinfo.height;
    mat.bdia.matinfo.nnz = bdiannz;
    int nnzremain = coomat.matinfo.nnz - bdiannz;

    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    cooremain.matinfo.nnz = nnzremain;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_data = (float*)malloc(sizeof(float)*nnzremain);

    mat.bdia.bdia_data = (float*)malloc(sizeof(float)*diaoff.size()*length_aligned);
    memset(mat.bdia.bdia_data, 0, sizeof(float)*diaoff.size()*length_aligned);

    int remainid = 0;
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	int col = coomat.coo_col_id[i];
	float data = coomat.coo_data[i];

	int offset = col - row;
	int j = 0;
	for (j = 0; j < diaoff.size(); j++)
	{
	    if (diaoff[j] == offset)
	    {
		mat.bdia.bdia_data[j * length_aligned + row] = data;
		break;
	    }
	}
	if (j == diaoff.size())
	{
	    cooremain.coo_row_id[remainid] = row; 
	    cooremain.coo_col_id[remainid] = col; 
	    cooremain.coo_data[remainid] = data; 
	    remainid++;
	}
    }
    assert(remainid == nnzremain);
    
    vector<int> bandOffset;
    vector<unsigned int> bandCount;
    bandOffset.reserve(diaoff.size());
    bandCount.reserve(diaoff.size());
    int diaid = 0;
    int lastOffset = diaoff[0];
    bandOffset.push_back(lastOffset);
    bandCount.push_back(1);
    diaid++;
    while (diaid < diaoff.size())
    {
	int curOffset = diaoff[diaid];
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

    mat.bdia.bdia_band_num = bandOffset.size();
    mat.bdia.bdia_offsets = (int*)malloc(sizeof(int)*bandOffset.size());
    mat.bdia.bdia_bptr = (int*)malloc(sizeof(int)*(bandOffset.size() + 1));
    for (int i = 0; i < bandOffset.size(); i++)
	mat.bdia.bdia_offsets[i] = bandOffset[i];
    mat.bdia.bdia_bptr[0] = 0;
    for (int i = 0; i < bandOffset.size(); i++)
	mat.bdia.bdia_bptr[i+1] = mat.bdia.bdia_bptr[i] + bandCount[i];

}

void extract_dia_only(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, vector<pair<int, int> >& select, int threshold, int dia_meth)
{
    mat.ifusedia = true;
    mat.dia_meth_num = dia_meth;
    
    int length_aligned = aligned_length(coomat.matinfo.height, GPU_ALIGNMENT);
    mat.dia.dia_length = coomat.matinfo.height;
    mat.dia.dia_length_aligned = length_aligned;

    vector<int> diaoff(select.size(), 0);
    int diannz = 0;
    for (int i = 0; i < select.size(); i++)
    {
	diaoff[i] = select[i].first - threshold;
	diannz += select[i].second;
    }
    mat.dia.matinfo.width = coomat.matinfo.width;
    mat.dia.matinfo.height = coomat.matinfo.height;
    mat.dia.matinfo.nnz = diannz;
    mat.dia.dia_num = diaoff.size();
    mat.dia.dia_offsets = (int*)malloc(sizeof(int)*diaoff.size());
    for (int i = 0; i < diaoff.size(); i++)
	mat.dia.dia_offsets[i] = diaoff[i];
    int nnzremain = coomat.matinfo.nnz - diannz;

    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    cooremain.matinfo.nnz = nnzremain;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_data = (float*)malloc(sizeof(float)*nnzremain);

    mat.dia.dia_data = (float*)malloc(sizeof(float)*diaoff.size()*length_aligned);
    memset(mat.dia.dia_data, 0, sizeof(float)*diaoff.size()*length_aligned);

    int remainid = 0;
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	int col = coomat.coo_col_id[i];
	float data = coomat.coo_data[i];

	int offset = col - row;
	int j = 0;
	for (j = 0; j < diaoff.size(); j++)
	{
	    if (diaoff[j] == offset)
	    {
		mat.dia.dia_data[j * length_aligned + row] = data;
		break;
	    }
	}
	if (j == diaoff.size())
	{
	    cooremain.coo_row_id[remainid] = row; 
	    cooremain.coo_col_id[remainid] = col; 
	    cooremain.coo_data[remainid] = data; 
	    remainid++;
	}
    }
    assert(remainid == nnzremain);
}

void extract_bdia_and_dia(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, vector<pair<int, int> >& select, int threshold, int least_band, int bdia_meth, int dia_meth)
{
    mat.ifusebdia = true;
    mat.bdia_meth_num = bdia_meth;
    mat.ifusedia = true;
    mat.dia_meth_num = dia_meth;

    int length_aligned = aligned_length(coomat.matinfo.height, GPU_ALIGNMENT);
    vector<int> dia_all(select.size(), 0);
    int bothnnz = 0;
    for (int i = 0; i < select.size(); i++)
    {
	dia_all[i] = select[i].first - threshold;
	bothnnz += select[i].second;
    }
    int nnzremain = coomat.matinfo.nnz - bothnnz;

    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    cooremain.matinfo.nnz = nnzremain;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*nnzremain);
    cooremain.coo_data = (float*)malloc(sizeof(float)*nnzremain); 
    vector<int> bandOffset;
    vector<unsigned int> bandCount;
    vector<int> bandnnz;
    bandOffset.reserve(dia_all.size());
    bandCount.reserve(dia_all.size());
    int diaid = 0;
    int lastOffset = dia_all[0];
    bandOffset.push_back(lastOffset);
    bandCount.push_back(1);
    bandnnz.push_back(select[0].second);
    diaid++;
    while (diaid < dia_all.size())
    {
	int curOffset = dia_all[diaid];
	if (curOffset - lastOffset == 1)
	{
	    bandCount[bandCount.size() - 1]++;
	    bandnnz[bandnnz.size() - 1] += select[diaid].second;
	}
	else
	{
	    bandOffset.push_back(curOffset);
	    bandCount.push_back(1);
	    bandnnz.push_back(select[diaid].second);
	}
	lastOffset = curOffset;
	diaid++;
    }
    assert(bandOffset.size() == bandCount.size());
    assert(bandnnz.size() == bandCount.size());

    // Separate the nnz of bdia and the nnz of dia
    vector<int> diaoff;
    diaoff.reserve(select.size());
    int diannz = 0;
    
    vector<int> bdiaoff;
    bdiaoff.reserve(select.size());
    int bdiannz = 0;
    int bandnum = 0;
    
    for (int i = 0; i < bandCount.size(); i++)
    {
	if (bandCount[i] >= least_band)
	{
	    bandnum++;
	    bdiannz += bandnnz[i];
	    for (int j = 0; j < bandCount[i]; j++)
	    {
		bdiaoff.push_back(bandOffset[i] + j);
	    }
	}
	else
	{
	    diannz += bandnnz[i];
	    for (int j = 0; j < bandCount[i]; j++)
	    {
		diaoff.push_back(bandOffset[i] + j);
	    }
	}
    }
    assert(diannz + bdiannz == bothnnz);
    assert(diaoff.size() + bdiaoff.size() == select.size());
    mat.dia.matinfo.width = coomat.matinfo.width;
    mat.dia.matinfo.height = coomat.matinfo.height;
    mat.dia.matinfo.nnz = diannz;
    mat.dia.dia_length = coomat.matinfo.height;
    mat.dia.dia_length_aligned = length_aligned;
    mat.dia.dia_num = diaoff.size();
    mat.dia.dia_offsets = (int*)malloc(sizeof(int)*diaoff.size());
    for (int i = 0; i < diaoff.size(); i++)
	mat.dia.dia_offsets[i] = diaoff[i];
    mat.dia.dia_data = (float*)malloc(sizeof(float)*diaoff.size()*length_aligned);
    memset(mat.dia.dia_data, 0, sizeof(float)*diaoff.size()*length_aligned);

    mat.bdia.matinfo.width = coomat.matinfo.width;
    mat.bdia.matinfo.height = coomat.matinfo.height;
    mat.bdia.matinfo.nnz = bdiannz;
    mat.bdia.bdia_length = coomat.matinfo.height;
    mat.bdia.bdia_length_aligned = length_aligned;
    mat.bdia.bdia_band_num = bandnum;
    mat.bdia.bdia_offsets = (int*)malloc(sizeof(int)*bandnum);
    mat.bdia.bdia_bptr = (int*)malloc(sizeof(int)*(bandnum+1));
    mat.bdia.bdia_bptr[0] = 0;
    int bdiaid = 0;
    for (int i = 0; i < bandCount.size(); i++)
    {
	if (bandCount[i] >= least_band)
	{
	    mat.bdia.bdia_offsets[bdiaid] = bandOffset[i];
	    mat.bdia.bdia_bptr[bdiaid + 1] = mat.bdia.bdia_bptr[bdiaid] + bandCount[i];
	    bdiaid++;
	}
    }
    assert(bdiaid == bdiaoff.size());
    mat.bdia.bdia_data = (float*)malloc(sizeof(float)*bdiaoff.size()*length_aligned);
    memset(mat.bdia.bdia_data, 0, sizeof(float)*bdiaoff.size()*length_aligned);

    int remainid = 0;
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	int col = coomat.coo_col_id[i];
	float data = coomat.coo_data[i];

	int offset = col - row;
	bool ifdia = false;
	bool ifbdia = false;
	for (int j = 0; j < diaoff.size(); j++)
	{
	    if (diaoff[j] == offset)
	    {
		mat.dia.dia_data[j * length_aligned + row] = data;
		ifdia = true;
		break;
	    }
	}
	if (ifdia == false)
	{
	    for (int j = 0; j < bdiaoff.size(); j++)
	    {
		if (bdiaoff[j] == offset)
		{
		    mat.bdia.bdia_data[j * length_aligned + row] = data;
		    ifbdia = true;
		    break;
		}
	    }
	}
	if (ifdia == false && ifbdia == false)
	{
	    cooremain.coo_row_id[remainid] = row; 
	    cooremain.coo_col_id[remainid] = col; 
	    cooremain.coo_data[remainid] = data; 
	    remainid++;
	}
    }
    assert(remainid == nnzremain);
    
}

void update_info_dia(coo_matrix<int, float>& coomat, maxflop_info& info, benchmark_all& bench)
{
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_bdia(bench.bdia, info, coomat.matinfo.height, nnzperrow);
    maxflop_dia(bench.dia, info, coomat.matinfo.height, nnzperrow);
    double max_dia_base = info.bdia_max_flop;
    if (info.dia_max_flop > max_dia_base)
	max_dia_base = info.dia_max_flop;
    info.max_dia_base_flop = max_dia_base;
}

void update_info_block(coo_matrix<int, float>& coomat, maxflop_info& info, benchmark_all& bench)
{
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_sbell(bench.sbell, info, coomat.matinfo.height, nnzperrow);
    maxflop_bell(bench.bell, info, coomat.matinfo.height, nnzperrow);
    maxflop_bcsr(bench.bcsr, info, coomat.matinfo.height, nnzperrow);
    double max_block_base = 0.0f;
    for (int i = 0; i < 8; i++)
    {
	if (info.sbell_max_flop[i] > max_block_base)
	    max_block_base = info.sbell_max_flop[i];
	if (info.bell_max_flop[i] > max_block_base)
	    max_block_base = info.bell_max_flop[i];
	if (info.bcsr_max_flop[i] > max_block_base)
	    max_block_base = info.bcsr_max_flop[i];
    }
    info.max_block_base_flop = max_block_base;
}

void update_info_flat(coo_matrix<int, float>& coomat, maxflop_info& info, benchmark_all& bench)
{
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_sell(bench.sell, info, coomat.matinfo.height, nnzperrow);
    maxflop_ell(bench.ell, info, coomat.matinfo.height, nnzperrow);
    maxflop_csr(bench.csr, info, coomat.matinfo.height, nnzperrow);
    maxflop_coo(bench.coo, info, coomat.matinfo.height, nnzperrow);
    double max_flat_base = info.sell_max_flop;
    if (info.ell_max_flop > max_flat_base)
	max_flat_base = info.ell_max_flop;
    if (info.csr_max_flop > max_flat_base)
	max_flat_base = info.csr_max_flop;
    if (info.coo_max_flop > max_flat_base)
	max_flat_base = info.coo_max_flop;
    info.max_flat_base_flop = max_flat_base;
}

void update_info(coo_matrix<int, float>& coomat, maxflop_info& info, benchmark_all& bench)
{
   update_info_dia(coomat, info, bench); 
   update_info_block(coomat, info, bench); 
   update_info_flat(coomat, info, bench); 
}

bool extract_dia(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, coo_matrix<int, float>& cooremain, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Extract Diagonals");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No bdia or dia extracted\n");
	return false;
    }
    //Possibly that the blocks are extracted, so it is necessary to update the flop information
    update_info_dia(coomat, info, bench);
    update_info_flat(coomat, info, bench);
    double margin = 1.05;
    double ratio = info.max_flat_base_flop / info.max_dia_base_flop; 
    int least_nnz_dia = (int)(ratio * (double)coomat.matinfo.height * margin);
    if (least_nnz_dia > coomat.matinfo.height)
    {
	printf("No bdia or dia extracted\n");
	return false;
    }
    int threshold = coomat.matinfo.height - least_nnz_dia;

    //Count nnz per diagonal
    vector<int> diacount;
    count_dia_nnz(coomat, diacount, threshold);
    vector<pair<int, int> >valid_dia;
    for (int i = 0; i < diacount.size(); i++)
    {
	if (diacount[i] > least_nnz_dia)
	{
	    valid_dia.push_back(pair<int, int>(diacount[i], i));
	}
    }
    sort(valid_dia.begin(), valid_dia.end(), greater<pair<int, int> >());
    double overhead = find_overhead(bench.overhead, coomat.matinfo.height / WORK_GROUP_SIZE, WORK_GROUP_SIZE);
    overhead /= 1000.0;

    int currentnnz = 0;
    int dia_num = 0;
    bool ifdia = false;
    double bestdiatime = 1000000.0f;
    //for (int i = 0; i < MAX_DIA_NUM && i < valid_dia.size(); i++)
    for (int i = 0; i <= MAX_DIA_NUM && i <= valid_dia.size(); i++)
    {
	
	if (i > 0)
	    currentnnz += valid_dia[i-1].first;
	int nnzremain = coomat.matinfo.nnz - currentnnz;
	double totaltime = ((double)i)*((double)coomat.matinfo.height)/1000000.0/info.max_dia_base_flop + 
	    ((double)nnzremain)/1000000.0/info.max_flat_base_flop;
	if (i > 0)
	    totaltime += overhead;
	if (nnzremain > 0)
	    totaltime += overhead;
	if (totaltime < bestdiatime)
	{
	    bestdiatime = totaltime;
	    dia_num = i;
	}
	
	/*
	currentnnz += valid_dia[i].first;
	double flattime = ((double)currentnnz)/1000000.0/info.max_flat_base_flop;
	double diatime  = ((double)i)*((double)coomat.matinfo.height)/1000000.0/info.max_dia_base_flop + overhead;
	if (diatime < flattime)
	{
	    ifdia = true;
	    dia_num = i + 1;
	}
	else
	{
	    if (ifdia == true)
		break;
	}
	*/
	
    }
    /*
    if (currentnnz == coomat.matinfo.nnz && ifdia == false)
    {
	ifdia = true;
	dia_num = min(MAX_DIA_NUM, (int)valid_dia.size());
    }
    */
    if (dia_num > 0)
	ifdia = true;
    
    printf("max flat %f max dia %f overhead %f cur nnz %d total nnz %d\n", info.max_flat_base_flop, info.max_dia_base_flop, overhead, currentnnz, coomat.matinfo.nnz);
    printf("Extract %d diagonals\n", dia_num);

    if (ifdia == false)
	return false;
    //Decide whether using dia or bdia or bdia + dia
    vector<pair<int, int> > select;
    for (int i = 0; i < dia_num; i++)
	select.push_back(pair<int, int>(valid_dia[i].second, valid_dia[i].first));
    sort(select.begin(), select.end());
    //Find bands
    bdia_vs_dia bvsd;
    find_min_bdia_width(bench.bdia, bench.dia, coomat.matinfo.height, bvsd);
    int least_band = bvsd.min_width;
    int band_num = 0;
    int current_width = 1;
    for (int i = 1; i < select.size(); i++)
    {
	if (select[i].first == select[i-1].first + 1)
	    current_width++;
	else
	{
	    if (current_width >= least_band)
		band_num += current_width;
	    current_width = 1;
	}
	if (i == select.size() - 1)
	{
	    if (current_width >= least_band)
		band_num += current_width;
	}
    }
    printf("Find %d bands min band width %d \n", band_num, least_band);
    int remain_dia = dia_num - band_num;
    double bdia_partial_gflop = 0.0001f;
    int bdia_partial_meth = 0;
    if (band_num > 0)
	bdia_gflop(bench.bdia, bvsd.range_s, bvsd.range_l, band_num, bdia_partial_gflop, bdia_partial_meth);
    double dia_partial_gflop = 0.0001f;
    int dia_partial_meth = 0;
    if (remain_dia > 0)
	dia_gflop(bench.dia, bvsd.range_s, bvsd.range_l, remain_dia, dia_partial_gflop, dia_partial_meth);
    double dia_all_gflop = 0.0001f;
    int dia_all_meth = 0;
    dia_gflop(bench.dia, bvsd.range_s, bvsd.range_l, dia_num, dia_all_gflop, dia_all_meth);

    double bdia_only = ((double)band_num*(double)coomat.matinfo.height)/1000000.0/bdia_partial_gflop + 
	((double)remain_dia*(double)coomat.matinfo.height)/1000000.0/(dia_partial_gflop * bvsd.bdia_single_gflop / bvsd.dia_single_gflop) + overhead;
    double dia_only = ((double)dia_num*(double)coomat.matinfo.height)/1000000.0/dia_all_gflop + overhead;
    double bdia_and_dia = ((double)band_num*(double)coomat.matinfo.height)/1000000.0/bdia_partial_gflop + 
	((double)remain_dia*(double)coomat.matinfo.height)/1000000.0/dia_partial_gflop + 2 * overhead;

    //printf("bdia_partial %f dia_partial %f dia_all %f\n", bdia_partial_gflop, dia_partial_gflop, dia_all_gflop);
    if (bdia_only <= dia_only && bdia_only <= bdia_and_dia)
    {
	printf("Use bdia only\n");
	extract_bdia_only(coomat, mat, cooremain, select, threshold, bdia_partial_meth);
    }
    else if (dia_only <= bdia_only && dia_only <= bdia_and_dia)
    {
	printf("Use dia only\n");
	extract_dia_only(coomat, mat, cooremain, select, threshold, dia_all_meth);
    }
    else
    {
	printf("Use bdia and dia\n");
	extract_bdia_and_dia(coomat, mat, cooremain, select, threshold, least_band, bdia_partial_meth, dia_partial_meth);
    }

    return true;
}

struct block_info
{
    int partial_block_num;
    int partial_nnz;
    int full_block_num;
    int full_nnz;
};

void compute_rowptr(coo_matrix<int, float>& coomat, vector<int>& rowptr)
{
    int row = 0;
    int curRow = 0;
    rowptr.resize(coomat.matinfo.height + 1, 0);
    rowptr[0] = 0;
    while (row < coomat.matinfo.nnz)
    {
	while (coomat.coo_row_id[row] == curRow && row < coomat.matinfo.nnz)
	    row++;
	curRow++;
	rowptr[curRow] = row;
    }
    if (curRow < coomat.matinfo.height)
    {
	curRow++;
	while (curRow <= coomat.matinfo.height)
	{
	    rowptr[curRow] = rowptr[curRow - 1];
	    curRow++;
	}
    }
}

void find_thre_max_flop(maxflop_info& info, vector<int>& threshold, vector<double>& best_block_flop)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    best_block_flop.resize(8, 0.0f);
    for (int i = 0; i < 8; i++)
    {
	if (info.sbell_max_flop[i] > best_block_flop[i])
	    best_block_flop[i] = info.sbell_max_flop[i];
	if (info.bell_max_flop[i] > best_block_flop[i])
	    best_block_flop[i] = info.bell_max_flop[i];
	if (info.bcsr_max_flop[i] > best_block_flop[i])
	    best_block_flop[i] = info.bcsr_max_flop[i];
    }
    threshold.resize(8, 0);
    //Each block should have more than threshold[i] zeros to be considered as a dense block
    for (int i = 0; i < 8; i++)
    {
	threshold[i] = (int)((double)bh[i]*(double)bw[i]*info.max_flat_base_flop/best_block_flop[i]);
	if (threshold[i] > bh[i] * bw[i] - 1)
	    threshold[i] = bh[i] * bw[i] - 1;
    }
}

void count_blocks(coo_matrix<int, float>& coomat, maxflop_info& info, benchmark_all& bench, vector<int>& threshold, vector<vector<block_info> >& block_count, vector<int>& rowptr)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};

    block_count.resize(8);
    for (int i = 0; i < 8; i++)
    {
	int size = coomat.matinfo.height/bh[i];
	if (coomat.matinfo.height % bh[i] != 0)
	    size++;
	block_count[i].resize(size);
	for (int j = 0; j < size; j++)
	{
	    block_count[i][j].partial_block_num = 0;
	    block_count[i][j].partial_nnz = 0;
	    block_count[i][j].full_block_num = 0;
	    block_count[i][j].full_nnz = 0;
	}
    }
    int row8num = coomat.matinfo.height / 8;
    if (coomat.matinfo.height % 8 != 0)
	row8num++;
    //Count 8 rows in a loop
    //The block is indexed by (block_row, block_col), the value is the nnz in the block
    vector<map<pair<int, int>, int> > blocks;
    blocks.resize(8);
    for (int r = 0; r < row8num; r++)
    {
	map<pair<int, int>, int>::iterator itr;
	for (int i = 0; i < 8; i++)
	    blocks[i].clear();
	int start = rowptr[r*8];
	int end = 0;
	if (r*8+8 >= rowptr.size())
	    end = rowptr[rowptr.size() - 1];
	else
	    end = rowptr[r*8+8];
	for (int i = start; i < end; i++)
	{
	    int row = coomat.coo_row_id[i];
	    int col = coomat.coo_col_id[i];
	    for (int j = 0; j < 8; j++)
	    {
		int newrow = row / bh[j];
		int newcol = col / bw[j];
		itr = blocks[j].find(pair<int, int>(newrow, newcol));
		if (itr == blocks[j].end())
		{
		    //New block, create it.
		    blocks[j][pair<int, int>(newrow, newcol)] = 1;
		}
		else
		{
		    //find the block, increase the count
		    (*itr).second++;
		}
	    }
	}
	//Do the counting
	for (int i = 0; i < 8; i++)
	{
	    for (itr = blocks[i].begin(); itr != blocks[i].end(); itr++)
	    {
		int newrow = itr->first.first;
		int curCount = itr->second;
		block_count[i][newrow].full_block_num++;
		block_count[i][newrow].full_nnz += curCount;
		if (curCount > threshold[i])
		{
		    block_count[i][newrow].partial_block_num++;
		    block_count[i][newrow].partial_nnz += curCount;
		}
	    }
	}
    }
}

void extract_sbell_part(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int bwidth, int bheight, int alignment, int threshold, int sbell_meth, vector<int>& sbell_num, int slice_height)
{
    printf("\n---------------------------------------------\n");
    printf("Extract sbell ");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No sbell extracted\n");
	return;
    }
    
    mat.ifusesbell = true;
    mat.sbell_meth_num = sbell_meth;
    mat.sbell.matinfo.width = coomat.matinfo.width;
    mat.sbell.matinfo.height = coomat.matinfo.height;
    mat.sbell.sbell_bwidth = bwidth;
    mat.sbell.sbell_bheight = bheight;
    
    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    int remainnnz = coomat.matinfo.nnz;
    cooremain.matinfo.nnz = remainnnz;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_data = (float*)malloc(sizeof(float)*remainnnz);

    vector<int> blockrowptr;
    vector<int> blockcolid;
    vector<float> blockdata;
    int browsize = coomat.matinfo.height / bheight;
    if (coomat.matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    int totalsize = 0;
    int slicenum = browsize / slice_height;
    if (browsize % slice_height != 0)
	slicenum++;
    printf("browsize %d slicenum %d sbell_num.size %d\n", browsize, slicenum, sbell_num.size());
    assert(slicenum == sbell_num.size());
    for (int i = 0; i < sbell_num.size(); i++)
	totalsize += slice_height * sbell_num[i];
    blockcolid.reserve(totalsize);
    blockdata.reserve(totalsize*bwidth*bheight);
    unsigned int blocksize = bwidth * bheight;
    int curdataid = 0;
    int remainid = 0;
    for (int row = 0; row < coomat.matinfo.height; row += bheight)
    {
	int sliceid = row / (slice_height * bheight);
	int start = rowptr[row];
	int end;
	if (row + bheight <= coomat.matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[coomat.matinfo.height];
	int size = end - start;
	int blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<int, float> > elements(size);
	for (int i = start; i < end; i++)
	{
	    elements[i - start].rowid = coomat.coo_row_id[i];
	    elements[i - start].colid = coomat.coo_col_id[i];
	    elements[i - start].data = coomat.coo_data[i];
	}
	compareCol<int, float> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	int blocknum = 0;
	int elemid = 0;
	while (elemid < size)
	{
	    int rowid = elements[elemid].rowid;
	    int colid = elements[elemid].colid;
	    float data = elements[elemid].data;
	    blocknum++;
	    int bcolid = colid - (colid % bwidth);
	    int browid = rowid - (rowid % bheight);
	    int curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    if (blockdata.size() <= curdataid)
	    {
		for (unsigned int i = 0; i < blocksize; i++)
		    blockdata.push_back(0.0f);
	    }
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    vector<int> tmprowid;
	    vector<int> tmpcolid;
	    vector<float> tmpdata;
	    tmprowid.clear();
	    tmpcolid.clear();
	    tmpdata.clear();
	    tmprowid.reserve(bwidth*bheight);
	    tmpcolid.reserve(bwidth*bheight);
	    tmpdata.reserve(bwidth*bheight);
	    tmprowid.push_back(rowid);
	    tmpcolid.push_back(colid);
	    tmpdata.push_back(data);
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
		tmprowid.push_back(rowid);
		tmpcolid.push_back(colid);
		tmpdata.push_back(data);
	    }
	    elemid++;
	    if (tmprowid.size() > threshold && blocknum <= sbell_num[sliceid])
	    {
		curdataid += blocksize;
	    }
	    else
	    {
		for (int k = 0; k < tmprowid.size(); k++)
		{
		    cooremain.coo_row_id[remainid] = tmprowid[k];	
		    cooremain.coo_col_id[remainid] = tmpcolid[k];	
		    cooremain.coo_data[remainid] = tmpdata[k];	
		    remainid++;
		}
		for (int k = 0; k < blocksize; k++)
		{
		    blockdata.pop_back();
		}
		blockcolid.pop_back();
		blocknum--;
	    }
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    //assert(remainid == remainnnz);
    cooremain.matinfo.nnz = remainid;
    mat.sbell.matinfo.nnz = coomat.matinfo.nnz - remainid;
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());
   
    mat.sbell.sbell_slice_height = slice_height;
    mat.sbell.sbell_slice_num = slicenum;
    mat.sbell.sbell_row_num = browsize;
    mat.sbell.sbell_slice_ptr = (int*)malloc(sizeof(int)*(slicenum+1));
    mat.sbell.sbell_slice_ptr[0] = 0;
    for (int i = 0; i < slicenum; i++)
    {
	mat.sbell.sbell_slice_ptr[i + 1] = mat.sbell.sbell_slice_ptr[i] + sbell_num[i] * slice_height;
    }
    assert(totalsize == mat.sbell.sbell_slice_ptr[slicenum]);
    mat.sbell.sbell_col_id = (int*)malloc(sizeof(int)*totalsize);
    mat.sbell.sbell_data = (float*)malloc(sizeof(float)*totalsize*bwidth*bheight);
    memset(mat.sbell.sbell_data, 0, sizeof(float)*totalsize*bwidth*bheight);

    unsigned int bwidth4num = bwidth / 4;

    int newblockcolsize = slice_height * bwidth4num * bheight * 4;
    int newblockw4size = slice_height * bheight * 4;
    int heightfour = slice_height * 4;
    for (int r = 0; r < browsize; r++)
    {
	int start = blockrowptr[r];
	int end = blockrowptr[r + 1];
	int sliceid = r / slice_height;
	int rowid = r % slice_height;
	int colstart = mat.sbell.sbell_slice_ptr[sliceid];
	int datastart = colstart * bwidth * bheight;
	int lastcolid = 0;

	assert(end <= start + sbell_num[sliceid]);
	for (int j = start; j < end; j++)
	{
	    int colid = blockcolid[j] / 4;
	    mat.sbell.sbell_col_id[colstart + rowid + (j - start) * slice_height] = colid;
	    lastcolid = colid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			mat.sbell.sbell_data[datastart + (j - start) * newblockcolsize + h * heightfour + w4 * newblockw4size + rowid * 4 + w] = blockdata[j * blocksize + h * bwidth + w4 * 4 + w];
		    }
		}
	    }
	}
	for (int j = end; j < start + sbell_num[sliceid]; j++)
	{   
	    mat.sbell.sbell_col_id[colstart + rowid + (j - start) * slice_height] = lastcolid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			mat.sbell.sbell_data[datastart + (j - start) * newblockcolsize + h * heightfour + w4 * newblockw4size + rowid * 4 + w] = 0.0f;
		    }
		}
	    }
	}
    }

    if (!if_sorted_coo<int, float>(&cooremain))
    {
	bool res = sort_coo<int, float>(&cooremain);
	assert(res == true);
    }
}

void extract_bell_part(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int bwidth, int bheight, int alignment, int threshold, int bell_meth, int bell_num)
{
    printf("\n---------------------------------------------\n");
    printf("Extract bell ");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No bell extracted\n");
	return;
    }
    if (bell_num == 0)
    {
	printf("No bell extracted\n");
	cooremain.matinfo.width = coomat.matinfo.width;
	cooremain.matinfo.height = coomat.matinfo.height;
	int nnz = coomat.matinfo.nnz;
	cooremain.matinfo.nnz = nnz;
	cooremain.coo_row_id = (int*)malloc(sizeof(int)*nnz);
	cooremain.coo_col_id = (int*)malloc(sizeof(int)*nnz);
	cooremain.coo_data = (float*)malloc(sizeof(float)*nnz);
	memcpy(cooremain.coo_row_id, coomat.coo_row_id, sizeof(int)*nnz);
	memcpy(cooremain.coo_col_id, coomat.coo_col_id, sizeof(int)*nnz);
	memcpy(cooremain.coo_data, coomat.coo_data, sizeof(float)*nnz);
	return;
    }
    
    mat.ifusebell = true;
    mat.bell_meth_num = bell_meth;
    mat.bell.matinfo.width = coomat.matinfo.width;
    mat.bell.matinfo.height = coomat.matinfo.height;
    mat.bell.b4ell_bwidth = bwidth;
    mat.bell.b4ell_bheight = bheight;
    
    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    int remainnnz = coomat.matinfo.nnz;
    //cooremain.matinfo.nnz = remainnnz;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_data = (float*)malloc(sizeof(float)*remainnnz);

    vector<int> blockrowptr;
    vector<int> blockcolid;
    vector<float> blockdata;
    int browsize = coomat.matinfo.height / bheight;
    if (coomat.matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(browsize * bell_num);
    blockdata.reserve(browsize * bell_num*bwidth*bheight);
    unsigned int blocksize = bwidth * bheight;
    int curdataid = 0;
    int remainid = 0;
    for (int row = 0; row < coomat.matinfo.height; row += bheight)
    {
	int start = rowptr[row];
	int end;
	if (row + bheight <= coomat.matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[coomat.matinfo.height];
	int size = end - start;
	int blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<int, float> > elements(size);
	for (int i = start; i < end; i++)
	{
	    elements[i - start].rowid = coomat.coo_row_id[i];
	    elements[i - start].colid = coomat.coo_col_id[i];
	    elements[i - start].data = coomat.coo_data[i];
	}
	compareCol<int, float> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	int blocknum = 0;
	int elemid = 0;
	while (elemid < size)
	{
	    int rowid = elements[elemid].rowid;
	    int colid = elements[elemid].colid;
	    float data = elements[elemid].data;
	    blocknum++;
	    int bcolid = colid - (colid % bwidth);
	    int browid = rowid - (rowid % bheight);
	    int curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    if (blockdata.size() <= curdataid)
	    {
		for (unsigned int i = 0; i < blocksize; i++)
		    blockdata.push_back(0.0f);
	    }
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    vector<int> tmprowid;
	    vector<int> tmpcolid;
	    vector<float> tmpdata;
	    tmprowid.clear();
	    tmpcolid.clear();
	    tmpdata.clear();
	    tmprowid.reserve(bwidth*bheight);
	    tmpcolid.reserve(bwidth*bheight);
	    tmpdata.reserve(bwidth*bheight);
	    tmprowid.push_back(rowid);
	    tmpcolid.push_back(colid);
	    tmpdata.push_back(data);
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
		tmprowid.push_back(rowid);
		tmpcolid.push_back(colid);
		tmpdata.push_back(data);
	    }
	    elemid++;
	    if (tmprowid.size() > threshold && blocknum <= bell_num)
	    {
		curdataid += blocksize;
	    }
	    else
	    {
		for (int k = 0; k < tmprowid.size(); k++)
		{
		    cooremain.coo_row_id[remainid] = tmprowid[k];	
		    cooremain.coo_col_id[remainid] = tmpcolid[k];	
		    cooremain.coo_data[remainid] = tmpdata[k];	
		    remainid++;
		}
		for (int k = 0; k < blocksize; k++)
		{
		    blockdata.pop_back();
		}
		blockcolid.pop_back();
		blocknum--;
	    }
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
    }
    //assert(remainid == remainnnz);
    cooremain.matinfo.nnz = remainid;
    mat.bell.matinfo.nnz = coomat.matinfo.nnz - remainid;
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());

    mat.bell.b4ell_row_num = blockrowptr.size() - 1;
    mat.bell.b4ell_block_num = bell_num; 
    int newlength = aligned_length<int>(mat.bell.b4ell_row_num, alignment);
    int newf4length = aligned_length<int>(4 * mat.bell.b4ell_row_num, alignment);
    mat.bell.b4ell_height_aligned = newlength;
    mat.bell.b4ell_float4_aligned = newf4length;

    mat.bell.b4ell_col_id = (int*) malloc(sizeof(int)*newlength*bell_num);
    unsigned int bwidth4num = bwidth / 4;
    mat.bell.b4ell_data = (float*)malloc(sizeof(float)*newf4length*bheight*bwidth4num*bell_num);
    memset(mat.bell.b4ell_data, 0, sizeof(float)*newf4length*bheight*bwidth4num*bell_num);

    int newblockcolsize = newf4length * bwidth4num * bheight;
    int newblockw4size = newf4length * bheight;
    for (int r = 0; r < mat.bell.b4ell_row_num; r++)
    {
	int start = blockrowptr[r];
	int end = blockrowptr[r + 1];
	int lastcolid = 0;
	assert(end <= start + bell_num);
	for (int j = start; j < end ; j++)
	{
	    int colid = blockcolid[j] / 4;
	    mat.bell.b4ell_col_id[r + (j - start) * newlength] = colid;
	    lastcolid = colid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			mat.bell.b4ell_data[(j - start) * newblockcolsize + h * newf4length + w4 * newblockw4size + r * 4 + w] = blockdata[j * blocksize + h * bwidth + w4 * 4 + w];
		    }
		}
	    }
	}
	for (int j = end; j < start + bell_num; j++)
	{   
	    mat.bell.b4ell_col_id[r + (j - start) * newlength] = lastcolid;
	    for (unsigned int h = 0; h < bheight; h++)
	    {
		for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
		{
		    for (unsigned int w = 0; w < 4; w++)
		    {
			mat.bell.b4ell_data[(j - start) * newblockcolsize + h * newf4length + w4 * newblockw4size + r * 4 + w] = 0.0f;
		    }
		}
	    }
	}
    }


    if (!if_sorted_coo<int, float>(&cooremain))
    {
	bool res = sort_coo<int, float>(&cooremain);
	assert(res == true);
    }
}


void extract_bcsr_part(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int bwidth, int bheight, int alignment, int bcsr_nnz, int bcsr_block, int threshold, int bcsr_meth)
{
    printf("\n---------------------------------------------\n");
    printf("Extract bcsr ");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No bcsr extracted\n");
	return;
    }
    
    mat.ifusebcsr = true;
    mat.bcsr_meth_num = bcsr_meth;
    mat.bcsr.matinfo.width = coomat.matinfo.width;
    mat.bcsr.matinfo.height = coomat.matinfo.height;
    mat.bcsr.matinfo.nnz = bcsr_nnz;
    mat.bcsr.b4csr_bwidth = bwidth;
    mat.bcsr.b4csr_bheight = bheight;
    
    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    int remainnnz = coomat.matinfo.nnz - bcsr_nnz;
    cooremain.matinfo.nnz = remainnnz;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_data = (float*)malloc(sizeof(float)*remainnnz);

    vector<int> blockrowptr;
    vector<int> blockcolid;
    vector<float> blockdata;
    int browsize = coomat.matinfo.height / bheight;
    if (coomat.matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(bcsr_block);
    blockdata.reserve(bcsr_block*bwidth*bheight);
    unsigned int blocksize = bwidth * bheight;
    int curdataid = 0;
    int remainid = 0;
    for (int row = 0; row < coomat.matinfo.height; row += bheight)
    {
	int start = rowptr[row];
	int end;
	if (row + bheight <= coomat.matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[coomat.matinfo.height];
	int size = end - start;
	int blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<int, float> > elements(size);
	for (int i = start; i < end; i++)
	{
	    elements[i - start].rowid = coomat.coo_row_id[i];
	    elements[i - start].colid = coomat.coo_col_id[i];
	    elements[i - start].data = coomat.coo_data[i];
	}
	compareCol<int, float> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	int blocknum = 0;
	int elemid = 0;
	while (elemid < size)
	{
	    int rowid = elements[elemid].rowid;
	    int colid = elements[elemid].colid;
	    float data = elements[elemid].data;
	    blocknum++;
	    int bcolid = colid - (colid % bwidth);
	    int browid = rowid - (rowid % bheight);
	    int curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    if (blockdata.size() <= curdataid)
	    {
		for (unsigned int i = 0; i < blocksize; i++)
		    blockdata.push_back(0.0f);
	    }
	    blockdata[curdataid + innerid] = data;
	    blockcolid.push_back(bcolid);
	    elemid++;
	    vector<int> tmprowid;
	    vector<int> tmpcolid;
	    vector<float> tmpdata;
	    tmprowid.clear();
	    tmpcolid.clear();
	    tmpdata.clear();
	    tmprowid.reserve(bwidth*bheight);
	    tmpcolid.reserve(bwidth*bheight);
	    tmpdata.reserve(bwidth*bheight);
	    tmprowid.push_back(rowid);
	    tmpcolid.push_back(colid);
	    tmpdata.push_back(data);
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
		tmprowid.push_back(rowid);
		tmpcolid.push_back(colid);
		tmpdata.push_back(data);
	    }
	    elemid++;
	    if (tmprowid.size() > threshold)
	    {
		curdataid += blocksize;
	    }
	    else
	    {
		for (int k = 0; k < tmprowid.size(); k++)
		{
		    cooremain.coo_row_id[remainid] = tmprowid[k];	
		    cooremain.coo_col_id[remainid] = tmpcolid[k];	
		    cooremain.coo_data[remainid] = tmpdata[k];	
		    remainid++;
		}
		for (int k = 0; k < blocksize; k++)
		{
		    blockdata.pop_back();
		}
		blockcolid.pop_back();
		blocknum--;
	    }
	}
	blockrowptr[blockrowid + 1] = blockrowptr[blockrowid] + blocknum;
	
    }
    assert(remainid == remainnnz);
    assert(blockrowptr[blockrowptr.size() - 1] == blockcolid.size());
    assert(blockrowptr[blockrowptr.size() - 1] * bwidth * bheight == blockdata.size());
    assert(bcsr_block == blockcolid.size());

    mat.bcsr.b4csr_row_num = blockrowptr.size() - 1;
    mat.bcsr.b4csr_row_ptr = (int*)malloc(sizeof(int)*blockrowptr.size());
    mat.bcsr.b4csr_col_id = (int*) malloc(sizeof(int)*blockcolid.size());
    int newlength = aligned_length<int>((int)blockcolid.size() * 4, alignment);
    unsigned int bwidth4num = bwidth / 4;
    mat.bcsr.b4csr_data = (float*)malloc(sizeof(float)*newlength*bheight*bwidth4num);
    memset(mat.bcsr.b4csr_data, 0, sizeof(float)*newlength*bheight*bwidth4num);
    mat.bcsr.b4csr_aligned_size = newlength;
    mat.bcsr.b4csr_block_num = blockcolid.size();
    assert(blockcolid.size() == bcsr_block);
    for (int i = 0; i < blockrowptr.size(); i++)
	mat.bcsr.b4csr_row_ptr[i] = blockrowptr[i];
    for (int i = 0; i < blockcolid.size(); i++)
	mat.bcsr.b4csr_col_id[i] = blockcolid[i] / 4;
    int onerowlength = newlength * bheight;
    for (int i = 0; i < blockcolid.size(); i++)
    {
	for (unsigned int h = 0; h < bheight; h++)
	{
	    for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
	    {
		for (unsigned int w = 0; w < 4; w++)
		{		
		    mat.bcsr.b4csr_data[w4 * onerowlength + h * newlength + i * 4 + w] = blockdata[i * blocksize + h * bwidth + w4 * 4 + w];
		}
	    }
	}
    }
    if (!if_sorted_coo<int, float>(&cooremain))
    {
	bool res = sort_coo<int, float>(&cooremain);
	assert(res == true);
    }
}

void extract_bcsr_all(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int bwidth, int bheight, int alignment, int bcsr_nnz, int bcsr_block, int bcsr_meth)
{
    printf("\n---------------------------------------------\n");
    printf("Extract bcsr all");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No bcsr extracted\n");
	return;
    }
    mat.ifusebcsr = true;
    mat.bcsr_meth_num = bcsr_meth;
    mat.bcsr.matinfo.width = coomat.matinfo.width;
    mat.bcsr.matinfo.height = coomat.matinfo.height;
    mat.bcsr.matinfo.nnz = bcsr_nnz;
    mat.bcsr.b4csr_bwidth = bwidth;
    mat.bcsr.b4csr_bheight = bheight;
    
    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    cooremain.matinfo.nnz = 0;
    assert(bcsr_nnz == coomat.matinfo.nnz);

    vector<int> blockrowptr;
    vector<int> blockcolid;
    vector<float> blockdata;
    int browsize = coomat.matinfo.height / bheight;
    if (coomat.matinfo.height % bheight != 0)
	browsize++;
    blockrowptr.resize(browsize + 1);
    blockrowptr[0] = 0;
    blockcolid.reserve(bcsr_block);
    blockdata.reserve(bcsr_block*bwidth*bheight);
    unsigned int blocksize = bwidth * bheight;
    int curdataid = 0;
    for (int row = 0; row < coomat.matinfo.height; row += bheight)
    {
	int start = rowptr[row];
	int end;
	if (row + bheight <= coomat.matinfo.height)
	    end = rowptr[row + bheight];
	else
	    end = rowptr[coomat.matinfo.height];
	int size = end - start;
	int blockrowid = row / bheight;
	if (size <= 0)
	{
	    blockrowptr[blockrowid + 1] = blockrowptr[blockrowid];
	    continue;
	}
	vector<oneElem<int, float> > elements(size);
	for (int i = start; i < end; i++)
	{
	    elements[i - start].rowid = coomat.coo_row_id[i];
	    elements[i - start].colid = coomat.coo_col_id[i];
	    elements[i - start].data = coomat.coo_data[i];
	}
	compareCol<int, float> compareobj;
	sort(elements.begin(), elements.end(), compareobj); 
	int blocknum = 0;
	int elemid = 0;
	while (elemid < size)
	{
	    int rowid = elements[elemid].rowid;
	    int colid = elements[elemid].colid;
	    float data = elements[elemid].data;
	    blocknum++;
	    int bcolid = colid - (colid % bwidth);
	    int browid = rowid - (rowid % bheight);
	    int curbcolid = bcolid;
	    unsigned int innercolid = colid - bcolid;
	    unsigned int innerrowid = rowid - browid;
	    unsigned int innerid = innerrowid * bwidth + innercolid;
	    for (unsigned int i = 0; i < blocksize; i++)
		blockdata.push_back(0.0f);
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

    mat.bcsr.b4csr_row_num = blockrowptr.size() - 1;
    mat.bcsr.b4csr_row_ptr = (int*)malloc(sizeof(int)*blockrowptr.size());
    mat.bcsr.b4csr_col_id = (int*) malloc(sizeof(int)*blockcolid.size());
    int newlength = aligned_length<int>((int)blockcolid.size() * 4, alignment);
    unsigned int bwidth4num = bwidth / 4;
    mat.bcsr.b4csr_data = (float*)malloc(sizeof(float)*newlength*bheight*bwidth4num);
    memset(mat.bcsr.b4csr_data, 0, sizeof(float)*newlength*bheight*bwidth4num);
    mat.bcsr.b4csr_aligned_size = newlength;
    mat.bcsr.b4csr_block_num = blockcolid.size();
    assert(blockcolid.size() == bcsr_block);
    for (int i = 0; i < blockrowptr.size(); i++)
	mat.bcsr.b4csr_row_ptr[i] = blockrowptr[i];
    for (int i = 0; i < blockcolid.size(); i++)
	mat.bcsr.b4csr_col_id[i] = blockcolid[i] / 4;
    int onerowlength = newlength * bheight;
    for (int i = 0; i < blockcolid.size(); i++)
    {
	for (unsigned int h = 0; h < bheight; h++)
	{
	    for (unsigned int w4 = 0; w4 < bwidth4num; w4++)
	    {
		for (unsigned int w = 0; w < 4; w++)
		{		
		    mat.bcsr.b4csr_data[w4 * onerowlength + h * newlength + i * 4 + w] = blockdata[i * blocksize + h * bwidth + w4 * 4 + w];
		}
	    }
	}
    }
}

//block_count: first element of the pair: block_num, second element: nnz
void extract_full_sbell_bell(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench, coo_matrix<int, float>& cooremain, vector<int>& rowptr, int wh_num, int threshold, vector<pair<int, int> >& block_count, pair<int, int>& summary)
{
    printf("\n---------------------------------------------\n");
    printf("Choose between full sbell and full bell");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d block nnz %d\n", coomat.matinfo.nnz, summary.second);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No sbell or bell extracted\n");
	return;
    }
     
    int nnzperrow = (int)(((float)summary.second)/((float)coomat.matinfo.height) + 0.5);
    maxflop_sbell(bench.sbell, info, coomat.matinfo.height, nnzperrow);
    maxflop_bell(bench.bell, info, coomat.matinfo.height, nnzperrow);
    
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};

    //Find the best ell time
    int maxblockperrow = 0;
    for (int i = 0; i < block_count.size(); i++)
    {
	if (block_count[i].first > maxblockperrow)
	    maxblockperrow = block_count[i].first;
    }
    int brownum = coomat.matinfo.height / bh[wh_num];
    if (coomat.matinfo.height % bh[wh_num] != 0)
	brownum++;
    int bellnnz = 0;
    double bestelltime = 1000000.0;
    int bestellnum = 0;
    int bestellnnz = 0;
    double bell_overhead = find_overhead(bench.overhead, brownum / BELL_GROUP_SIZE, BELL_GROUP_SIZE);
    bell_overhead /= 1000.0;
    int bwidth4num = bw[wh_num] / 4;
	
    int newlength = aligned_length(4 * brownum, GPU_ALIGNMENT);
    //Average nnz per block
    double avgnnz = ((double)summary.second/(double)summary.first);
    bestelltime = 0.0f;
    bestelltime += ((double)maxblockperrow*(double)newlength*(double)bh[wh_num]*(double)bwidth4num)/1000000.0/info.bell_max_flop[wh_num] + bell_overhead;
    bestellnum = maxblockperrow;
    bestellnnz = coomat.matinfo.nnz;
    
    //Find the best sell time
    int slice_height = WARPSIZE;
    brownum = coomat.matinfo.height / bh[wh_num];
    if (coomat.matinfo.height % bh[wh_num] != 0)
	brownum++;
    int slice_num = brownum / slice_height;
    if (brownum % slice_height != 0)
	slice_num++;
    vector<int>slice_ell_full(slice_num, 0);
    int slice_full_nnz = summary.second;
    for (int i = 0; i < slice_num; i++)
    {
	vector<int> slice_width;
	slice_width.reserve(slice_height);
	for (int j = i * slice_height; j < (i+1) * slice_height && j < block_count.size(); j++)
	{
	    slice_width.push_back(block_count[j].first);
	}
	sort(slice_width.begin(), slice_width.end());
	slice_ell_full[i] = slice_width[slice_width.size() - 1];
    }
    printf("slice full nnz %d\n", slice_full_nnz);
    double sbell_overhead = find_overhead(bench.overhead, brownum/SELL_GROUP_SIZE, SELL_GROUP_SIZE);
    sbell_overhead /= 1000.0;
    int sbell_full = 0;
    for (int i = 0; i < slice_ell_full.size(); i++)
	sbell_full += slice_height * slice_ell_full[i];
    double sbell_full_time = ((double)(sbell_full*bw[wh_num]*bh[wh_num]))/1000000.0/info.sbell_max_flop[wh_num] + sbell_overhead;

    printf("sbell_full block %d nnz %d total nnz %d sbell over %f\n", sbell_full, slice_full_nnz, coomat.matinfo.nnz, sbell_overhead);
    printf("sbell full time %f best bell time %f \n", sbell_full_time, bestelltime);
    if (bestelltime < sbell_full_time)
    {
	extract_bell_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, threshold, info.bell_meth_num[wh_num], bestellnum);
	return;
    }
    extract_sbell_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, threshold, info.sbell_meth_num[wh_num], slice_ell_full, slice_height);

}

//block_count: first element of the pair: block_num, second element: nnz
void extract_sbell_bell(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench, coo_matrix<int, float>& cooremain, vector<int>& rowptr, int wh_num, int threshold, vector<pair<int, int> >& block_count, pair<int, int>& summary)
{
    printf("\n---------------------------------------------\n");
    printf("Choose between sbell and bell");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d block nnz %d\n", coomat.matinfo.nnz, summary.second);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No sbell or bell extracted\n");
	return;
    }
     
    int nnzperrow = (int)(((float)summary.second)/((float)coomat.matinfo.height) + 0.5);
    maxflop_sbell(bench.sbell, info, coomat.matinfo.height, nnzperrow);
    maxflop_bell(bench.bell, info, coomat.matinfo.height, nnzperrow);
    
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};

    double remain_flop = info.max_flat_base_flop;
    int remain_group = coomat.matinfo.height / WORK_GROUP_SIZE;
    int remain_thread = WORK_GROUP_SIZE;
    printf("remain flop %f sbell %f bell %f id %d\n", remain_flop, info.sbell_max_flop[wh_num], info.bell_max_flop[wh_num], wh_num);
    /*
    if (info.csr_max_flop > info.coo_max_flop)
    {
	remain_flop = info.csr_max_flop;
	remain_group = info.csr_groupnum;
	remain_thread = CSR_VEC_GROUP_SIZE;
    }
    else
    {
	remain_flop = info.coo_max_flop;
	remain_group = info.coo_groupnum;
	remain_thread = COO_GROUP_SIZE;
    }
    */
    double remain_overhead = find_overhead(bench.overhead, remain_group, remain_thread);
    remain_overhead /= 1000.0;
    //Find the best ell time
    int maxblockperrow = 0;
    for (int i = 0; i < block_count.size(); i++)
    {
	if (block_count[i].first > maxblockperrow)
	    maxblockperrow = block_count[i].first;
    }
    vector<int> histogram(maxblockperrow + 1, 0);
    for (int i = 0; i < block_count.size(); i++)
	histogram[block_count[i].first]++;
    int brownum = coomat.matinfo.height / bh[wh_num];
    if (coomat.matinfo.height % bh[wh_num] != 0)
	brownum++;
    int bellnnz = 0;
    double bestelltime = 1000000.0;
    int bestellnum = 0;
    int bestellnnz = 0;
    double bell_overhead = find_overhead(bench.overhead, brownum / BELL_GROUP_SIZE, BELL_GROUP_SIZE);
    bell_overhead /= 1000.0;
    int bwidth4num = bw[wh_num] / 4;
	
    int newlength = aligned_length(4 * brownum, GPU_ALIGNMENT);
    //Average nnz per block
    double avgnnz = ((double)summary.second/(double)summary.first);
    
    for (int i = 0; i < histogram.size(); i++)
    {
	brownum -= histogram[i];
	double belltime = 0.0f;
	if (i > 0)
	    belltime += ((double)i*(double)newlength*(double)bh[wh_num]*(double)bwidth4num)/1000000.0/info.bell_max_flop[wh_num] + bell_overhead;
	if (bellnnz < coomat.matinfo.nnz)
	    belltime += ((double)(coomat.matinfo.nnz - bellnnz))/1000000.0/remain_flop + remain_overhead;
	if (belltime < bestelltime)
	{
	    bestelltime = belltime;
	    bestellnum = i;
	    bestellnnz = bellnnz;
	}
	bellnnz += (int)((double)brownum * avgnnz);
    }

    //Find the best sell time
    int slice_height = WARPSIZE;
    brownum = coomat.matinfo.height / bh[wh_num];
    if (coomat.matinfo.height % bh[wh_num] != 0)
	brownum++;
    int slice_num = brownum / slice_height;
    if (brownum % slice_height != 0)
	slice_num++;
    vector<int>slice_ell_partial(slice_num, 0);
    vector<int>slice_ell_full(slice_num, 0);
    int slice_partial_nnz = 0;
    int slice_full_nnz = summary.second;
    for (int i = 0; i < slice_num; i++)
    {
	vector<int> slice_width;
	slice_width.reserve(slice_height);
	for (int j = i * slice_height; j < (i+1) * slice_height && j < block_count.size(); j++)
	{
	    slice_width.push_back(block_count[j].first);
	}
	sort(slice_width.begin(), slice_width.end());
	slice_ell_full[i] = slice_width[slice_width.size() - 1];
	int partialnnz = (int)((double)slice_width[0] * (double)slice_width.size() * avgnnz);
	int currentnnz = partialnnz;
	slice_ell_partial[i] = slice_width[0];
	for (int j = 1; j < slice_width.size(); j++)
	{
	    partialnnz += (int)((double)(slice_width[j] - slice_width[j-1]) * (double)(slice_width.size() - j)*avgnnz);
	    double ratio = remain_flop / info.sbell_max_flop[wh_num]; 
	    if (partialnnz > (int)(((double)(slice_width.size() * slice_width[j] * bw[wh_num]*bh[wh_num])) * ratio))
	    {
		slice_ell_partial[i] = slice_width[j];
		currentnnz = partialnnz;
	    }
	    else
	    {
		break;
	    }
	}
	
	slice_partial_nnz += currentnnz;
    }
    printf("slice partial nnz %d full nnz %d\n", slice_partial_nnz, slice_full_nnz);
    assert(slice_partial_nnz <= slice_full_nnz);
    double sbell_overhead = find_overhead(bench.overhead, brownum/SELL_GROUP_SIZE, SELL_GROUP_SIZE);
    sbell_overhead /= 1000.0;
    int sbell_full = 0;
    int sbell_partial = 0;
    for (int i = 0; i < slice_ell_full.size(); i++)
	sbell_full += slice_height * slice_ell_full[i];
    for (int i = 0; i < slice_ell_partial.size(); i++)
	sbell_partial += slice_height * slice_ell_partial[i];
    double sbell_full_time = ((double)(sbell_full*bw[wh_num]*bh[wh_num]))/1000000.0/info.sbell_max_flop[wh_num] + sbell_overhead +
	((double)(coomat.matinfo.nnz - slice_full_nnz))/1000000.0/remain_flop + remain_overhead;
    double sbell_partial_time = ((double)(sbell_partial*bw[wh_num]*bh[wh_num]))/1000000.0/info.sbell_max_flop[wh_num] + sbell_overhead +
	((double)(coomat.matinfo.nnz - slice_partial_nnz))/1000000.0/remain_flop + remain_overhead;

    printf("sbell_full block %d nnz %d sbell_partial block %d nnz %d total nnz %d sbell over %f flat over %f\n", sbell_full, slice_full_nnz, sbell_partial, slice_partial_nnz, coomat.matinfo.nnz, sbell_overhead, remain_overhead);
    printf("sbell full time %f sbell partial time %f best bell time %f flat time %f\n", sbell_full_time, sbell_partial_time, bestelltime, 
	    ((double)(coomat.matinfo.nnz))/1000000.0/remain_flop + remain_overhead);
    if (bestelltime < sbell_full_time && bestelltime < sbell_partial_time)
    {
	extract_bell_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, threshold, info.bell_meth_num[wh_num], bestellnum);
	return;
    }
    if (sbell_partial_time < sbell_full_time)
    {
	extract_sbell_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, threshold, info.sbell_meth_num[wh_num], slice_ell_partial, slice_height);
	return;
    }
    extract_sbell_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, threshold, info.sbell_meth_num[wh_num], slice_ell_full, slice_height);

}

void extract_full_block(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int wh_num, int threshold, vector<block_info>& block_count, block_info& summary)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    printf("Extract full block original nnz %d block num %d block nnz %d\n", coomat.matinfo.nnz, summary.full_block_num, summary.full_nnz);
    if (info.bcsr_max_flop[wh_num] >= info.sbell_max_flop[wh_num] && info.bcsr_max_flop[wh_num] >= info.bell_max_flop[wh_num])
    {
	extract_bcsr_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, summary.full_nnz, summary.full_block_num, 0, info.bcsr_meth_num[wh_num]); 
	//extract_bcsr_all(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, summary[wh_num].full_nnz, summary[wh_num].full_block_num, info.bcsr_meth_num[wh_num]); 
	return;
    }
    vector<pair<int, int> >count(block_count.size());
    for (int i = 0; i < block_count.size(); i++)
    {
	count[i].first = block_count[i].full_block_num;
	count[i].second = block_count[i].full_nnz;
    }
    pair<int, int> sum;
    sum.first = summary.full_block_num;
    sum.second = summary.full_nnz;
    extract_full_sbell_bell(coomat, mat, info, bench, cooremain, rowptr, wh_num, 0, count, sum);

}

void extract_partial_block(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, coo_matrix<int, float>& cooremain, benchmark_all& bench, vector<int>& rowptr, int wh_num, int threshold, vector<block_info>& block_count, block_info& summary)
{
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    printf("Extract partial block original nnz %d block num %d block nnz %d\n", coomat.matinfo.nnz, summary.partial_block_num, summary.partial_nnz);
    if (info.bcsr_max_flop[wh_num] >= info.sbell_max_flop[wh_num] && info.bcsr_max_flop[wh_num] >= info.bell_max_flop[wh_num])
    {
	extract_bcsr_part(coomat, mat, cooremain, bench, rowptr, bw[wh_num], bh[wh_num], GPU_ALIGNMENT, summary.partial_nnz, summary.partial_block_num, threshold, info.bcsr_meth_num[wh_num]); 
	return;
    }
    vector<pair<int, int> >count(block_count.size());
    for (int i = 0; i < block_count.size(); i++)
    {
	count[i].first = block_count[i].partial_block_num;
	count[i].second = block_count[i].partial_nnz;
    }
    pair<int, int> sum;
    sum.first = summary.partial_block_num;
    sum.second = summary.partial_nnz;
    extract_sbell_bell(coomat, mat, info, bench, cooremain, rowptr, wh_num, 0, count, sum);

}

bool extract_block(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, coo_matrix<int, float>& cooremain, benchmark_all& bench, bool estimate_flat_partial, bool estimate_flat_full)
{
    printf("\n---------------------------------------------\n");
    printf("Extract Blocks");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No block extracted\n");
	return false;
    }
    //Possibly that the diagonals are extracted, so it is necessary to update the flop information
    update_info_block(coomat, info, bench);
    update_info_flat(coomat, info, bench);

    vector<int> rowptr;
    compute_rowptr(coomat, rowptr);
    vector<int> threshold;
    vector<double> best_block_flop;
    find_thre_max_flop(info, threshold, best_block_flop);
    vector<vector<block_info> > block_count;
    count_blocks(coomat, info, bench, threshold, block_count, rowptr);

    //Decide the block size
    int bh[8] = {1, 2, 4, 8, 1, 2, 4, 8};
    int bw[8] = {4, 4, 4, 4, 8, 8, 8, 8};
    vector<double> block_over;
    block_over.resize(8, 0.0f);
    for (int i = 0; i < 8; i++)
    {
	block_over[i] = find_overhead(bench.overhead, coomat.matinfo.height/(bh[i]*SELL_GROUP_SIZE), SELL_GROUP_SIZE); 
	block_over[i] /= 1000.0;
    }
    double bestfulltime = 1000000.0f;
    int bestfullid = 0;
    double bestpartialtime = 1000000.0f;
    int bestpartialid = 0;
    vector<block_info> summary;
    summary.resize(8);
    double flatover = find_overhead(bench.overhead, coomat.matinfo.height / WORK_GROUP_SIZE, WORK_GROUP_SIZE);
    flatover /= 1000.0;
    for (int i = 0; i < 8; i++)
    {
	summary[i].partial_block_num = 0;
	summary[i].partial_nnz = 0;
	summary[i].full_block_num = 0;
	summary[i].full_nnz = 0;
    }
    for (int i = 0; i < 8; i++)
    {
	for (int j = 0; j < block_count[i].size(); j++)
	{
	    summary[i].partial_block_num += block_count[i][j].partial_block_num;
	    summary[i].partial_nnz += block_count[i][j].partial_nnz;
	    summary[i].full_block_num += block_count[i][j].full_block_num;
	    summary[i].full_nnz += block_count[i][j].full_nnz;
	}
	assert(summary[i].full_nnz == coomat.matinfo.nnz);
    }
    for (int i = 0; i < 8; i++)
    {
	double fulltime = ((double)summary[i].full_block_num)*((double)bh[i])*((double)bw[i])/1000000.0/best_block_flop[i] + block_over[i];
	if (fulltime < bestfulltime)
	{
	    bestfulltime = fulltime;
	    bestfullid = i;
	}
	vector<int> rownnz(coomat.matinfo.height);
	for (int j = 0; j < rownnz.size(); j++)
	    rownnz[j] = 0;
	int flatnnz = 0;
	for (int j = 0; j < block_count[i].size(); j++)
	{
	    int diff = block_count[i][j].full_nnz - block_count[i][j].partial_nnz;
	    int rowstart = j * bh[i];
	    int rowend = (j+1)*bh[i];
	    if (rowend > rownnz.size())
		rowend = rownnz.size();
	    int tmprownum = rowend - rowstart;
	    assert(tmprownum > 0);
	    int avg = diff / tmprownum;
	    for (int k = rowstart; k < rowend - 1; k++)
		rownnz[k] = avg;
	    rownnz[rowend-1] = diff - avg * (tmprownum - 1);
	    flatnnz += diff;
	}
	double partialtime = ((double)summary[i].partial_block_num)*((double)bh[i])*((double)bw[i])/1000000.0/best_block_flop[i] + block_over[i];
	if (estimate_flat_partial)
	    partialtime += estimate_flat_time(rownnz, flatnnz, bench);
	else
	    partialtime += ((double)(coomat.matinfo.nnz - summary[i].partial_nnz))/1000000.0/info.max_flat_base_flop + flatover;
	if (partialtime < bestpartialtime)
	{
	    bestpartialtime = partialtime;
	    bestpartialid = i;
	}
	printf("Block size (%d, %d) full block num %d nnz %d partial num %d nnz %d full time %f partial time %f block over %f flat over %f\n", bh[i], bw[i], summary[i].full_block_num, summary[i].full_nnz, summary[i].partial_block_num, summary[i].partial_nnz, fulltime, partialtime, block_over[i], flatover);
    }
    double fullflattime = ((double)coomat.matinfo.nnz)/1000000.0/info.max_flat_base_flop + flatover;
    double flatratio = 1.02; //Favor block representation, the flat format should be better than 2% to be considered
    fullflattime *= flatratio;
    if (estimate_flat_full)
    {
	vector<int> rownnz(coomat.matinfo.height);
	for (int i = 0; i < rownnz.size(); i++)
	    rownnz[i] = rowptr[i+1] - rowptr[i];
	fullflattime = estimate_flat_time(rownnz, coomat.matinfo.nnz, bench);
    }
    if (fullflattime <= bestfulltime && fullflattime <= bestpartialtime)
    {
	printf("No block extracted\n");
	return false;
    }
    if (bestfulltime <= bestpartialtime)
    {
	extract_full_block(coomat, mat, info, cooremain, bench, rowptr, bestfullid, threshold[bestfullid], block_count[bestfullid], summary[bestfullid]);
	return true;
    }

    extract_partial_block(coomat, mat, info, cooremain, bench, rowptr, bestpartialid, threshold[bestpartialid], block_count[bestpartialid], summary[bestpartialid]);
    return true;
}

double estimate_coo_time(int nnz, maxflop_info& info, benchmark_all& bench)
{
    if (nnz == 0)
	return 0.0f;
    double cooover = find_overhead(bench.overhead, info.coo_groupnum, COO_GROUP_SIZE);
    cooover /= 1000.0;
    double cootime = ((double)nnz)/1000000.0/info.coo_max_flop + cooover;
    return cootime;
}

void extract_coo(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Extract coo");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No coo extracted\n");
	return;
    }
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_coo(bench.coo, info, coomat.matinfo.height, nnzperrow);
    mat.ifusecoo = true;
    mat.coo_meth_num = info.coo_meth_num;
    mat.coo_group_num = info.coo_groupnum;
    mat.coo.matinfo.height = coomat.matinfo.height;
    mat.coo.matinfo.width = coomat.matinfo.width;
    int nnz = coomat.matinfo.nnz;
    mat.coo.matinfo.nnz = nnz;
    mat.coo.coo_row_id = (int*)malloc(sizeof(int)*nnz);
    mat.coo.coo_col_id = (int*)malloc(sizeof(int)*nnz);
    mat.coo.coo_data = (float*)malloc(sizeof(float)*nnz);
    memcpy(mat.coo.coo_row_id, coomat.coo_row_id, sizeof(int)*nnz);
    memcpy(mat.coo.coo_col_id, coomat.coo_col_id, sizeof(int)*nnz);
    memcpy(mat.coo.coo_data, coomat.coo_data, sizeof(float)*nnz);
    printf("coo group num %d\n", info.coo_groupnum);
}

double estimate_csr_time(vector<int>& rownnz, int nnz, maxflop_info& info, benchmark_all& bench)
{
    if (nnz == 0)
	return 0.0f;
    vector<int> nnzpergroup(info.csr_groupnum, 0);
    if (info.csr_meth_num == 0)
    {
	//reduction per warp
	int height = CSR_VEC_GROUP_SIZE / WARPSIZE;
	int totalwarpnum = info.csr_groupnum * height;
	int totaliter = rownnz.size() / totalwarpnum;
	if (rownnz.size() % totalwarpnum != 0)
	    totaliter++;
	for (int i = 0; i < totaliter; i++)
	{
	    for (int g = 0; g < info.csr_groupnum; g++)
	    {
		for (int j = 0; j < height; j++)
		{
		    int id = i * totalwarpnum + g * height + j;
		    if (id >= rownnz.size())
			break;
		    nnzpergroup[g] += rownnz[id];
		}
	    }
	}
    }
    else if (info.csr_meth_num == 1)
    {
	//reduction per group
	for (int i = 0; i < rownnz.size(); i++)
	{
	    for (int g = 0; g < info.csr_groupnum; g++)
	    {
		int id = i * info.csr_groupnum + g;
		if (id >= rownnz.size())
		    break;
		nnzpergroup[g] += rownnz[id];
	    }
	}
    }

    int maxnnzgroup = 0;
    for (int i = 0; i < info.csr_groupnum; i++)
    {
	if (nnzpergroup[i] > maxnnzgroup)
	    maxnnzgroup = nnzpergroup[i];
    }
    int newnnz = maxnnzgroup * info.csr_groupnum;
    double csrover = find_overhead(bench.overhead, info.csr_groupnum, CSR_VEC_GROUP_SIZE);
    csrover /= 1000.0;
    double csrtime = ((double)newnnz)/1000000.0/info.csr_max_flop + csrover;
    return csrtime;
}

void extract_csr(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Extract csr");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No csr extracted\n");
	return;
    }
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_csr(bench.csr, info, coomat.matinfo.height, nnzperrow);
    mat.ifusecsr = true;
    mat.csr_meth_num = info.csr_meth_num;
    mat.csr_group_num = info.csr_groupnum;
    coo2csr<int, float>(&coomat, &mat.csr);
    printf("csr group num %d\n", info.csr_groupnum);
}

double estimate_csr_coo_time(vector<int>& rownnz, int nnz, maxflop_info& info, benchmark_all& bench)
{
    if (nnz == 0)
	return 0.0f;
    int nnzperrow = (int)(((float)nnz)/((float)rownnz.size()) + 0.5);
    maxflop_csr(bench.csr, info, rownnz.size(), nnzperrow);
    maxflop_coo(bench.coo, info, rownnz.size(), nnzperrow);
    
    if (info.coo_max_flop >= info.csr_max_flop)
    {
	return estimate_coo_time(nnz, info, bench);
    }

    double cootime = estimate_coo_time(nnz, info, bench);
    double csrtime = estimate_csr_time(rownnz, nnz, info, bench);
    if (cootime < csrtime)
	return cootime;
    return csrtime;
}

void extract_csr_coo(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Choose between csr and coo");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No csr or coo extracted\n");
	return;
    }
    int nnzperrow = (int)(((float)coomat.matinfo.nnz)/((float)coomat.matinfo.height) + 0.5);
    maxflop_csr(bench.csr, info, coomat.matinfo.height, nnzperrow);
    maxflop_coo(bench.coo, info, coomat.matinfo.height, nnzperrow);

    if (info.coo_max_flop >= info.csr_max_flop)
    {
	extract_coo(coomat, mat, info, bench);
	return;
    }

    //csr is better if the variances of the nnz per row is small
    //coo is beeter if the variances of the nnz per row is very large

    //Count nnz for each row
    vector<int> nnzcount(coomat.matinfo.height, 0);
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	nnzcount[row]++;
    }

    double cooover = find_overhead(bench.overhead, info.coo_groupnum, COO_GROUP_SIZE);
    cooover /= 1000.0;
    double cootime = ((double)coomat.matinfo.nnz)/1000000.0/info.coo_max_flop + cooover;

    //Check the nnz that each work group needs to process
    vector<int> nnzpergroup(info.csr_groupnum, 0);
    if (info.csr_meth_num == 0)
    {
	//reduction per warp
	int height = CSR_VEC_GROUP_SIZE / WARPSIZE;
	int totalwarpnum = info.csr_groupnum * height;
	int totaliter = coomat.matinfo.height / totalwarpnum;
	if (coomat.matinfo.height % totalwarpnum != 0)
	    totaliter++;
	for (int i = 0; i < totaliter; i++)
	{
	    for (int g = 0; g < info.csr_groupnum; g++)
	    {
		for (int j = 0; j < height; j++)
		{
		    int id = i * totalwarpnum + g * height + j;
		    if (id >= nnzcount.size())
			break;
		    nnzpergroup[g] += nnzcount[id];
		}
	    }
	}
    }
    else if (info.csr_meth_num == 1)
    {
	//reduction per group
	for (int i = 0; i < coomat.matinfo.height; i++)
	{
	    for (int g = 0; g < info.csr_groupnum; g++)
	    {
		int id = i * info.csr_groupnum + g;
		if (id >= nnzcount.size())
		    break;
		nnzpergroup[g] += nnzcount[id];
	    }
	}
    }

    int maxnnzgroup = 0;
    for (int i = 0; i < info.csr_groupnum; i++)
    {
	if (nnzpergroup[i] > maxnnzgroup)
	    maxnnzgroup = nnzpergroup[i];
    }
    int newnnz = maxnnzgroup * info.csr_groupnum;
    double csrover = find_overhead(bench.overhead, info.csr_groupnum, CSR_VEC_GROUP_SIZE);
    csrover /= 1000.0;
    double csrtime = ((double)newnnz)/1000000.0/info.csr_max_flop + csrover;

    printf("max nnz per group %d nnz %d newnnz %d cootime %f ms csrtime %f ms\n", maxnnzgroup, coomat.matinfo.nnz, newnnz, cootime, csrtime);
    if (cootime <= csrtime)
    {
	extract_coo(coomat, mat, info, bench);
    }
    else
    {
	extract_csr(coomat, mat, info, bench);
    }
}


void extract_sell(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench, coo_matrix<int, float>& cooremain, vector<int>& nnzcount, int slice_height, vector<int>& slice_ell, int slice_nnz)
{
    printf("\n---------------------------------------------\n");
    printf("Extract sell\n");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0 || slice_nnz == 0)
    {
	printf("No sell extracted\n");
	return;
    }
    mat.ifusesell = true;
    mat.sell_meth_num = info.sell_meth_num;
    if (info.sell_meth_num == 0)
	assert(slice_height == WARPSIZE);
    if (info.sell_meth_num == 1)
	assert(slice_height == SELL_GROUP_SIZE);
    
    vector<int> rowptr(nnzcount.size() + 1, 0);
    for (int i = 1; i < rowptr.size(); i++)
	rowptr[i] = rowptr[i-1] + nnzcount[i-1];

    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    int remainnnz = coomat.matinfo.nnz - slice_nnz;
    cooremain.matinfo.nnz = remainnnz;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_data = (float*)malloc(sizeof(float)*remainnnz);

    int slicenum = slice_ell.size();
    mat.sell.matinfo.width = coomat.matinfo.width;
    mat.sell.matinfo.height = coomat.matinfo.height;
    mat.sell.matinfo.nnz = slice_nnz;
    mat.sell.sell_slice_height = slice_height;
    mat.sell.sell_slice_num = slicenum;
    mat.sell.sell_slice_ptr = (int*)malloc(sizeof(int)*(slicenum + 1));
    mat.sell.sell_slice_ptr[0] = 0;
    for (int i = 0; i < slice_ell.size(); i++)
    {
	mat.sell.sell_slice_ptr[i + 1] = mat.sell.sell_slice_ptr[i] + slice_ell[i] * slice_height;
    }
    int totalsize = mat.sell.sell_slice_ptr[slicenum];
    mat.sell.sell_col_id = (int*)malloc(sizeof(int)*totalsize);
    mat.sell.sell_data = (float*)malloc(sizeof(float)*totalsize);
    for (int i = 0; i < totalsize; i++)
    {
	mat.sell.sell_col_id[i] = 0;
	mat.sell.sell_data[i] = 0.0f;
    }
    int remainid = 0;
    for (int i = 0; i < slicenum; i++)
    {
	int offset = i * slice_height;
	int slice_offset = mat.sell.sell_slice_ptr[i];
	for (int j = offset; j < offset + slice_height && j < coomat.matinfo.height ; j++)
	{
	    int start = rowptr[j];
	    int end = rowptr[j + 1];
	    for (int k = start; k < end && k < start + slice_ell[i]; k++)
	    {
		int colid = coomat.coo_col_id[k];
		float data = coomat.coo_data[k];
		mat.sell.sell_col_id[slice_offset + (k - start) * slice_height + j - offset] = colid;
		mat.sell.sell_data[slice_offset + (k - start) * slice_height + j - offset] = data;
	    }
	    for (int k = end; k < start + slice_ell[i]; k++)
	    {
		int cpyid = (k-start)*slice_height+j - offset;
		if (cpyid >= slice_height)
		    mat.sell.sell_col_id[slice_offset + cpyid] = mat.sell.sell_col_id[slice_offset + cpyid - slice_height];
		else
		    mat.sell.sell_col_id[slice_offset + cpyid] = mat.sell.sell_col_id[slice_offset];
		mat.sell.sell_data[slice_offset + cpyid] = 0.0f;
	    }
	    for (int k = start + slice_ell[i]; k < end; k++)
	    {
		int colid = coomat.coo_col_id[k];
		float data = coomat.coo_data[k];
		cooremain.coo_row_id[remainid] = j;
		cooremain.coo_col_id[remainid] = colid;
		cooremain.coo_data[remainid] = data;
		remainid++;
	    }
	}
	
	if (i == (slicenum - 1) && (slicenum * slice_height) > coomat.matinfo.height)
	{
	    for (int j = coomat.matinfo.height; j < (slicenum*slice_height); j++)
	    {
		for (int k = 0; k < slice_ell[i]; k++)
		{
		    int cpyid = k*slice_height+j - offset;
		    if (cpyid >= slice_height)
			mat.sell.sell_col_id[slice_offset + cpyid] = mat.sell.sell_col_id[slice_offset + cpyid - slice_height];
		    else
			mat.sell.sell_col_id[slice_offset + cpyid] = mat.sell.sell_col_id[slice_offset];
		    mat.sell.sell_data[slice_offset + cpyid] = 0.0f;
		}
	    }
	}
    }

    assert(remainid == remainnnz);

}

void extract_ell(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench, coo_matrix<int, float>& cooremain, vector<int>& nnzcount, int ellnum, int ellnnz)
{
    printf("\n---------------------------------------------\n");
    printf("Extract ell ell_num %d \n", ellnum);
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0 || ellnum == 0)
    {
	printf("No ell extracted\n");
	return;
    }
    maxflop_ell(bench.ell, info, coomat.matinfo.height, ellnum);
    mat.ifuseell = true;
    mat.ell_meth_num = info.ell_meth_num;
    
    vector<int> rowptr(nnzcount.size() + 1, 0);
    for (int i = 1; i < rowptr.size(); i++)
	rowptr[i] = rowptr[i-1] + nnzcount[i-1];

    cooremain.matinfo.width = coomat.matinfo.width;
    cooremain.matinfo.height = coomat.matinfo.height;
    int remainnnz = coomat.matinfo.nnz - ellnnz;
    cooremain.matinfo.nnz = remainnnz;
    cooremain.coo_row_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_col_id = (int*)malloc(sizeof(int)*remainnnz);
    cooremain.coo_data = (float*)malloc(sizeof(float)*remainnnz);

    mat.ell.matinfo.width = coomat.matinfo.width;
    mat.ell.matinfo.height = coomat.matinfo.height;
    mat.ell.matinfo.nnz = ellnnz;
    mat.ell.ell_num = ellnum;
    int newlength = aligned_length(coomat.matinfo.height, GPU_ALIGNMENT);
    mat.ell.ell_height_aligned = newlength;
    
    mat.ell.ell_col_id = (int*)malloc(sizeof(int)*newlength*ellnum);
    mat.ell.ell_data = (float*)malloc(sizeof(float)*newlength*ellnum);
    for (int i = 0; i < newlength * ellnum; i++)
    {
	mat.ell.ell_col_id[i] = 0;
	mat.ell.ell_data[i] = 0.0f;
    }
    int remainid = 0;
    for (int i = 0; i < coomat.matinfo.height; i++)
    {
	int start = rowptr[i];
	int end = rowptr[i+1];
	int lastcolid = 0;
	for (int j = start; j < end && j < start + ellnum; j++)
	{
	    int colid = coomat.coo_col_id[j];
	    float data = coomat.coo_data[j];
	    mat.ell.ell_col_id[i + (j - start) * newlength] = colid;
	    mat.ell.ell_data[i + (j - start) * newlength] = data;
	    lastcolid = colid;
	}
	for (int j = end; j < start + ellnum; j++)
	{
	    mat.ell.ell_col_id[i + (j - start) * newlength] = lastcolid;
	    mat.ell.ell_data[i + (j - start) * newlength] = 0.0f;
	}
	for (int j = start + ellnum; j < end; j++)
	{
	    int colid = coomat.coo_col_id[j];
	    float data = coomat.coo_data[j];
	    cooremain.coo_row_id[remainid] = i;
	    cooremain.coo_col_id[remainid] = colid;
	    cooremain.coo_data[remainid] = data;
	    remainid++;
	}
    }
    assert(remainid == remainnnz);
}

double estimate_sell_ell_time(vector<int>& rownnz, int nnz, maxflop_info& info, benchmark_all& bench, vector<int>& rownnzremain, int& nnzremain)
{
    if (nnz == 0)
    {
	nnzremain = 0;
	for (int i = 0; i < rownnzremain.size(); i++)
	    rownnzremain[i] = 0;
	return 0.0f;
    }
    
    double remain_flop;
    int remain_group;
    int remain_thread;
    if (info.csr_max_flop > info.coo_max_flop)
    {
	remain_flop = info.csr_max_flop;
	remain_group = info.csr_groupnum;
	remain_thread = CSR_VEC_GROUP_SIZE;
    }
    else
    {
	remain_flop = info.coo_max_flop;
	remain_group = info.coo_groupnum;
	remain_thread = COO_GROUP_SIZE;
    }
    double remain_overhead = find_overhead(bench.overhead, remain_group, remain_thread);
    remain_overhead /= 1000.0;
    //Find the best ell time
    int maxnnzperrow = 0;
    for (int i = 0; i < rownnz.size(); i++)
    {
	if (rownnz[i] > maxnnzperrow)
	    maxnnzperrow = rownnz[i];
    }
    vector<int> histogram(maxnnzperrow + 1, 0);
    for (int i = 0; i < rownnz.size(); i++)
	histogram[rownnz[i]]++;
    int rownum = rownnz.size();
    int ellnnz = 0;
    double bestelltime = 1000000.0;
    int bestellnum = 0;
    int bestellnnz = 0;
    double ell_overhead = 0.0f;
    if (info.ell_meth_num == 0)
    {
	ell_overhead = find_overhead(bench.overhead, rownnz.size() / WORK_GROUP_SIZE, WORK_GROUP_SIZE);
    }
    else if (info.ell_meth_num == 1)
    {
	ell_overhead = find_overhead(bench.overhead, rownnz.size() / (4*WORK_GROUP_SIZE), WORK_GROUP_SIZE);
    }
    ell_overhead /= 1000.0;
	
    int newlength = aligned_length((int)rownnz.size(), GPU_ALIGNMENT);
    for (int i = 0; i < histogram.size(); i++)
    {
	rownum -= histogram[i];
	double elltime = 0.0f;
	if (i > 0)
	    elltime += ((double)i*(double)newlength)/1000000.0/info.ell_max_flop + ell_overhead;
	if (ellnnz < nnz)
	    elltime += ((double)(nnz - ellnnz))/1000000.0/remain_flop + remain_overhead;
	if (elltime < bestelltime)
	{
	    bestelltime = elltime;
	    bestellnum = i;
	    bestellnnz = ellnnz;
	}
	ellnnz += rownum;
    }

    //Check if the above ellnnz count is correct
    int tmpnnz = 0;
    for (int i = 0; i < rownnz.size(); i++)
    {
	if (rownnz[i] <= bestellnum)
	    tmpnnz += rownnz[i];
	else
	    tmpnnz += bestellnum;
    }
    printf("bestnnz %d tmpnnz %d\n", bestellnnz, tmpnnz);
    assert(bestellnnz == tmpnnz);

    //Find the best sell time
    int slice_height = 0;
    if (info.sell_meth_num == 0)
	slice_height = WARPSIZE;
    else if (info.sell_meth_num == 1)
	slice_height = SELL_GROUP_SIZE;
    int slice_num = rownnz.size() / slice_height;
    if (rownnz.size() % slice_height != 0)
	slice_num++;
    vector<int>slice_ell_partial(slice_num, 0);
    vector<int>slice_ell_full(slice_num, 0);
    int slice_partial_nnz = 0;
    int slice_full_nnz = nnz;
    for (int i = 0; i < slice_num; i++)
    {
	vector<int> slice_width;
	slice_width.reserve(slice_height);
	for (int j = i * slice_height; j < (i+1) * slice_height && j < rownnz.size(); j++)
	{
	    slice_width.push_back(rownnz[j]);
	}
	sort(slice_width.begin(), slice_width.end());
	slice_ell_full[i] = slice_width[slice_width.size() - 1];
	int partialnnz = slice_width[0] * slice_width.size();
	int currentnnz = partialnnz;
	slice_ell_partial[i] = slice_width[0];
	for (int j = 1; j < slice_width.size(); j++)
	{
	    partialnnz += (slice_width[j] - slice_width[j-1]) * (slice_width.size() - j);
	    double ratio = remain_flop / info.sell_max_flop; 
	    if (partialnnz > (int)(((double)(slice_width.size() * slice_width[j])) * ratio))
	    {
		slice_ell_partial[i] = slice_width[j];
		currentnnz = partialnnz;
	    }
	    else
	    {
		break;
	    }
	}
	//check if the nnz count is correct.
	tmpnnz = 0;
	for (int j = i * slice_height; j < (i+1) * slice_height && j < rownnz.size(); j++)
	{
	    if (rownnz[j] <= slice_ell_partial[i])
		tmpnnz += rownnz[j];
	    else
		tmpnnz += slice_ell_partial[i];
	}
	if (tmpnnz != currentnnz)
	    printf("total slice %d slice %d tmpnnz %d currentnnz %d width %d\n", slice_num, i, tmpnnz, currentnnz, slice_ell_partial[i]);
	assert(tmpnnz == currentnnz);
	//end check
	
	slice_partial_nnz += currentnnz;
    }
    printf("slice partial nnz %d full nnz %d\n", slice_partial_nnz, slice_full_nnz);
    assert(slice_partial_nnz <= slice_full_nnz);
    double sell_overhead = find_overhead(bench.overhead, slice_num, SELL_GROUP_SIZE);
    sell_overhead /= 1000.0;
    int sell_full = 0;
    int sell_partial = 0;
    for (int i = 0; i < slice_ell_full.size(); i++)
	sell_full += slice_height * slice_ell_full[i];
    for (int i = 0; i < slice_ell_partial.size(); i++)
	sell_partial += slice_height * slice_ell_partial[i];
    double sell_full_time = ((double)sell_full)/1000000.0/info.sell_max_flop + sell_overhead;
    double sell_partial_time = ((double)sell_partial)/1000000.0/info.sell_max_flop + sell_overhead +
	((double)(nnz - slice_partial_nnz))/1000000.0/remain_flop + remain_overhead;

    if (bestelltime < sell_full_time && bestelltime < sell_partial_time)
    {
	nnzremain = 0;
	for (int i = 0; i < rownnz.size(); i++)
	{
	    if (rownnz[i] <= bestellnum)
		rownnzremain[i] = 0;
	    else
	    {
		int diff = rownnz[i] - bestellnum;
		rownnzremain[i] = diff;
		nnzremain += diff;
	    }
	}
	return bestelltime;
    }
    if (sell_partial_time < sell_full_time)
    {
	nnzremain = 0;
	for (int i = 0; i < rownnz.size(); i++)
	{
	    int sliceid = i / slice_height;
	    int ellwidth = slice_ell_partial[sliceid];
	    if (rownnz[i] <= ellwidth)
		rownnzremain[i] = 0;
	    else
	    {
		int diff = rownnz[i] - ellwidth;
		rownnzremain[i] = diff;
		nnzremain += diff;
	    }
	}
	return sell_partial_time;
    }
    nnzremain = 0;
    for (int i = 0; i < rownnzremain.size(); i++)
	rownnzremain[i] = 0;
    return sell_full_time;
}

void extract_sell_ell(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench, coo_matrix<int, float>& cooremain)
{
    printf("\n---------------------------------------------\n");
    printf("Choose between sell and ell");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No sell or ell extracted\n");
	return;
    }

    //Count nnz for each row
    vector<int> nnzcount(coomat.matinfo.height, 0);
    for (int i = 0; i < coomat.matinfo.nnz; i++)
    {
	int row = coomat.coo_row_id[i];
	nnzcount[row]++;
    }
    
    double remain_flop;
    int remain_group;
    int remain_thread;
    if (info.csr_max_flop > info.coo_max_flop)
    {
	remain_flop = info.csr_max_flop;
	remain_group = info.csr_groupnum;
	remain_thread = CSR_VEC_GROUP_SIZE;
    }
    else
    {
	remain_flop = info.coo_max_flop;
	remain_group = info.coo_groupnum;
	remain_thread = COO_GROUP_SIZE;
    }
    double remain_overhead = find_overhead(bench.overhead, remain_group, remain_thread);
    remain_overhead /= 1000.0;
    //Find the best ell time
    int maxnnzperrow = 0;
    for (int i = 0; i < nnzcount.size(); i++)
    {
	if (nnzcount[i] > maxnnzperrow)
	    maxnnzperrow = nnzcount[i];
    }
    vector<int> histogram(maxnnzperrow + 1, 0);
    for (int i = 0; i < nnzcount.size(); i++)
	histogram[nnzcount[i]]++;
    int rownum = coomat.matinfo.height;
    int ellnnz = 0;
    double bestelltime = 1000000.0;
    int bestellnum = 0;
    int bestellnnz = 0;
    double ell_overhead = 0.0f;
    if (info.ell_meth_num == 0)
    {
	ell_overhead = find_overhead(bench.overhead, coomat.matinfo.height / WORK_GROUP_SIZE, WORK_GROUP_SIZE);
    }
    else if (info.ell_meth_num == 1)
    {
	ell_overhead = find_overhead(bench.overhead, coomat.matinfo.height / (4*WORK_GROUP_SIZE), WORK_GROUP_SIZE);
    }
    ell_overhead /= 1000.0;
	
    int newlength = aligned_length(coomat.matinfo.height, GPU_ALIGNMENT);
    for (int i = 0; i < histogram.size(); i++)
    {
	rownum -= histogram[i];
	double elltime = 0.0f;
	if (i > 0)
	    elltime += ((double)i*(double)newlength)/1000000.0/info.ell_max_flop + ell_overhead;
	if (ellnnz < coomat.matinfo.nnz)
	    elltime += ((double)(coomat.matinfo.nnz - ellnnz))/1000000.0/remain_flop + remain_overhead;
	if (elltime < bestelltime)
	{
	    bestelltime = elltime;
	    bestellnum = i;
	    bestellnnz = ellnnz;
	}
	ellnnz += rownum;
    }

    //Check if the above ellnnz count is correct
    int tmpnnz = 0;
    for (int i = 0; i < nnzcount.size(); i++)
    {
	if (nnzcount[i] <= bestellnum)
	    tmpnnz += nnzcount[i];
	else
	    tmpnnz += bestellnum;
    }
    printf("bestnnz %d tmpnnz %d\n", bestellnnz, tmpnnz);
    assert(bestellnnz == tmpnnz);

    //Find the best sell time
    int slice_height = 0;
    if (info.sell_meth_num == 0)
	slice_height = WARPSIZE;
    else if (info.sell_meth_num == 1)
	slice_height = SELL_GROUP_SIZE;
    int slice_num = coomat.matinfo.height / slice_height;
    if (coomat.matinfo.height % slice_height != 0)
	slice_num++;
    vector<int>slice_ell_partial(slice_num, 0);
    vector<int>slice_ell_full(slice_num, 0);
    int slice_partial_nnz = 0;
    int slice_full_nnz = coomat.matinfo.nnz;
    for (int i = 0; i < slice_num; i++)
    {
	vector<int> slice_width;
	slice_width.reserve(slice_height);
	for (int j = i * slice_height; j < (i+1) * slice_height && j < nnzcount.size(); j++)
	{
	    slice_width.push_back(nnzcount[j]);
	}
	sort(slice_width.begin(), slice_width.end());
	slice_ell_full[i] = slice_width[slice_width.size() - 1];
	int partialnnz = slice_width[0] * slice_width.size();
	int currentnnz = partialnnz;
	slice_ell_partial[i] = slice_width[0];
	for (int j = 1; j < slice_width.size(); j++)
	{
	    partialnnz += (slice_width[j] - slice_width[j-1]) * (slice_width.size() - j);
	    double ratio = remain_flop / info.sell_max_flop; 
	    if (partialnnz > (int)(((double)(slice_width.size() * slice_width[j])) * ratio))
	    {
		slice_ell_partial[i] = slice_width[j];
		currentnnz = partialnnz;
	    }
	    else
	    {
		break;
	    }
	}
	//check if the nnz count is correct.
	tmpnnz = 0;
	for (int j = i * slice_height; j < (i+1) * slice_height && j < nnzcount.size(); j++)
	{
	    if (nnzcount[j] <= slice_ell_partial[i])
		tmpnnz += nnzcount[j];
	    else
		tmpnnz += slice_ell_partial[i];
	}
	if (tmpnnz != currentnnz)
	    printf("total slice %d slice %d tmpnnz %d currentnnz %d width %d\n", slice_num, i, tmpnnz, currentnnz, slice_ell_partial[i]);
	assert(tmpnnz == currentnnz);
	//end check
	
	slice_partial_nnz += currentnnz;
    }
    printf("slice partial nnz %d full nnz %d\n", slice_partial_nnz, slice_full_nnz);
    assert(slice_partial_nnz <= slice_full_nnz);
    double sell_overhead = find_overhead(bench.overhead, slice_num, SELL_GROUP_SIZE);
    sell_overhead /= 1000.0;
    int sell_full = 0;
    int sell_partial = 0;
    for (int i = 0; i < slice_ell_full.size(); i++)
	sell_full += slice_height * slice_ell_full[i];
    for (int i = 0; i < slice_ell_partial.size(); i++)
	sell_partial += slice_height * slice_ell_partial[i];
    double sell_full_time = ((double)sell_full)/1000000.0/info.sell_max_flop + sell_overhead;
    double sell_partial_time = ((double)sell_partial)/1000000.0/info.sell_max_flop + sell_overhead +
	((double)(coomat.matinfo.nnz - slice_partial_nnz))/1000000.0/remain_flop + remain_overhead;

    if (bestelltime < sell_full_time && bestelltime < sell_partial_time)
    {
	extract_ell(coomat, mat, info, bench, cooremain, nnzcount, bestellnum, bestellnnz);
	return;
    }
    if (sell_partial_time < sell_full_time)
    {
	extract_sell(coomat, mat, info, bench, cooremain, nnzcount, slice_height, slice_ell_partial, slice_partial_nnz);
	return;
    }

    extract_sell(coomat, mat, info, bench, cooremain, nnzcount, slice_height, slice_ell_full, slice_full_nnz);

}

double estimate_flat_time(vector<int>& rownnz, int nnz, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Estimate Flat Time");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", nnz);
    if (nnz == 0)
    {
	printf("No flat needed\n");
	return 0.0f;
    }
    maxflop_info info;
    int nnzperrow = (int)(((float)nnz)/((float)rownnz.size()) + 0.5);
    maxflop_sell(bench.sell, info, rownnz.size(), nnzperrow);
    maxflop_ell(bench.ell, info, rownnz.size(), nnzperrow);
    maxflop_csr(bench.csr, info, rownnz.size(), nnzperrow);
    maxflop_coo(bench.coo, info, rownnz.size(), nnzperrow);
    double max_flat_base = info.sell_max_flop;
    if (info.ell_max_flop > max_flat_base)
	max_flat_base = info.ell_max_flop;
    if (info.csr_max_flop > max_flat_base)
	max_flat_base = info.csr_max_flop;
    if (info.coo_max_flop > max_flat_base)
	max_flat_base = info.coo_max_flop;
    info.max_flat_base_flop = max_flat_base;


    //Easy decision here
    if (info.max_flat_base_flop == info.coo_max_flop)
    {
	return estimate_coo_time(nnz, info, bench);
    }
    
    if (info.max_flat_base_flop == info.csr_max_flop)
    {
	return estimate_csr_time(rownnz, nnz, info, bench);
    }

    vector<int> rownnzremain(rownnz.size(), 0);
    int nnzremain = 0;
    double elltime = estimate_sell_ell_time(rownnz, nnz, info, bench, rownnzremain, nnzremain);
    double csrcootime = estimate_csr_coo_time(rownnzremain, nnzremain, info, bench);
    return elltime + csrcootime;

}

void extract_flat(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, maxflop_info& info, benchmark_all& bench)
{
    printf("\n---------------------------------------------\n");
    printf("Extract Flat Elements");
    printf("\n---------------------------------------------\n");
    printf("Matrix nnz %d\n", coomat.matinfo.nnz);
    if (coomat.matinfo.nnz == 0)
    {
	printf("No flat extracted\n");
	return;
    }
    //Possibly that the diagonals and blocks are extracted, so it is necessary to update the flop information
    update_info_flat(coomat, info, bench);

    //Easy decision here
    if (info.max_flat_base_flop == info.coo_max_flop)
    {
	extract_coo(coomat, mat, info, bench);
	return;	
    }
    
    if (info.max_flat_base_flop == info.csr_max_flop)
    {
	extract_csr_coo(coomat, mat, info, bench);
	return;	
    }

    coo_matrix<int, float> cooremain;
    init_coo_matrix(cooremain);
    extract_sell_ell(coomat, mat, info, bench, cooremain);
    if (cooremain.coo_row_id == NULL)
	extract_csr_coo(coomat, mat, info, bench);
    else
	extract_csr_coo(cooremain, mat, info, bench);
    free_coo_matrix(cooremain);

}


void analyze_matrix(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, float dia_priority_multiplier, bool estimate_flat_partial, bool estimate_flat_full)
{
    mat.mat_width = coomat.matinfo.width;
    mat.mat_height = coomat.matinfo.height;
    mat.mat_nnz = coomat.matinfo.nnz;
    benchmark_all bench;
    ReadBench(bench);

    maxflop_info info;
    update_info(coomat, info, bench);
   
    double max_dia_base = info.max_dia_base_flop;
    max_dia_base *= dia_priority_multiplier;
    double max_block_base = 1000000.0f;
    max_block_base = info.max_block_base_flop;
    double max_flat_base = info.max_flat_base_flop;
    printf("dia flop %f block flop %f flat flop %f\n", max_dia_base, max_block_base, max_flat_base);
    coo_matrix<int, float> cooremain1;
    coo_matrix<int, float> cooremain2;
    init_coo_matrix(cooremain1);
    init_coo_matrix(cooremain2);
    double starttime = 0.0f;
    double endtime = 0.0f;
    if (max_dia_base > max_flat_base)
    {
	if (max_block_base > max_flat_base)
	{
	    if (max_dia_base  > max_block_base)
	    {
		if (coomat.matinfo.width == coomat.matinfo.height)
		{
		    starttime = timestamp();
		    bool ifdia = extract_dia(coomat, mat, info, cooremain1, bench);
		    endtime = timestamp();
		    printf("Dia category analysis time %lf s\n", endtime - starttime);
		    if (ifdia)
		    {
			starttime = timestamp();
			bool ifblock = extract_block(cooremain1, mat, info, cooremain2, bench, estimate_flat_partial, estimate_flat_full);
			endtime = timestamp();
			printf("Block category analysis time %lf s\n", endtime - starttime);
			starttime = timestamp();
			if (ifblock)
			    extract_flat(cooremain2, mat, info, bench);
			else
			    extract_flat(cooremain1, mat, info, bench);
			endtime = timestamp();
			printf("Flat category analysis time %lf s\n", endtime - starttime);
		    }
		    else
		    {
			starttime = timestamp();
			bool ifblock = extract_block(coomat, mat, info, cooremain2, bench, estimate_flat_partial, estimate_flat_full);
			endtime = timestamp();
			printf("Block category analysis time %lf s\n", endtime - starttime);
			starttime = timestamp();
			if (ifblock)
			    extract_flat(cooremain2, mat, info, bench);
			else
			    extract_flat(coomat, mat, info, bench);
			endtime = timestamp();
			printf("Flat category analysis time %lf s\n", endtime - starttime);
		    }
		}
		else
		{
		    starttime = timestamp();
		    bool ifblock = extract_block(coomat, mat, info, cooremain1, bench, estimate_flat_partial, estimate_flat_full);
		    endtime = timestamp();
		    printf("Block category analysis time %lf s\n", endtime - starttime);
		    starttime = timestamp();
		    if (ifblock)
			extract_flat(cooremain1, mat, info, bench);
		    else
			extract_flat(coomat, mat, info, bench);
		    endtime = timestamp();
		    printf("Flat category analysis time %lf s\n", endtime - starttime);
		}
	    }
	    else
	    {
		if (coomat.matinfo.width == coomat.matinfo.height)
		{
		    starttime = timestamp();
		    bool ifblock = extract_block(coomat, mat, info, cooremain1, bench, estimate_flat_partial, estimate_flat_full);
		    endtime = timestamp();
		    printf("Block category analysis time %lf s\n", endtime - starttime);
		    if (ifblock)
		    {
			starttime = timestamp();
			bool ifdia = extract_dia(cooremain1, mat, info, cooremain2, bench);
			endtime = timestamp();
			printf("Dia category analysis time %lf s\n", endtime - starttime);
			starttime = timestamp();
			if (ifdia)
			    extract_flat(cooremain2, mat, info, bench);
			else
			    extract_flat(cooremain1, mat, info, bench);
			endtime = timestamp();
			printf("Flat category analysis time %lf s\n", endtime - starttime);
		    }
		    else
		    {
			starttime = timestamp();
			bool ifdia = extract_dia(coomat, mat, info, cooremain2, bench);
			endtime = timestamp();
			printf("Dia category analysis time %lf s\n", endtime - starttime);
			starttime = timestamp();
			if (ifdia)
			    extract_flat(cooremain2, mat, info, bench);
			else
			    extract_flat(coomat, mat, info, bench);
			endtime = timestamp();
			printf("Flat category analysis time %lf s\n", endtime - starttime);
		    }
		}
		else
		{
		    starttime = timestamp();
		    bool ifblock = extract_block(coomat, mat, info, cooremain1, bench, estimate_flat_partial, estimate_flat_full);
		    endtime = timestamp();
		    printf("Block category analysis time %lf s\n", endtime - starttime);
		    starttime = timestamp();
		    if (ifblock)
			extract_flat(cooremain1, mat, info, bench);		
		    else
			extract_flat(coomat, mat, info, bench);		
		    endtime = timestamp();
		    printf("Flat category analysis time %lf s\n", endtime - starttime);
		}
	    }
	}
	else
	{
	    if (coomat.matinfo.width == coomat.matinfo.height)
	    {
		starttime = timestamp();
		bool ifdia = extract_dia(coomat, mat, info, cooremain1, bench);
		endtime = timestamp();
		printf("Dia category analysis time %lf s\n", endtime - starttime);
		starttime = timestamp();
		if (ifdia)
		    extract_flat(cooremain1, mat, info, bench);
		else
		    extract_flat(coomat, mat, info, bench);
		endtime = timestamp();
		printf("Flat category analysis time %lf s\n", endtime - starttime);
	    }
	    else
	    {
		starttime = timestamp();
		extract_flat(coomat, mat, info, bench);
		endtime = timestamp();
		printf("Flat category analysis time %lf s\n", endtime - starttime);
	    }
	}
    }
    else
    {
	if (max_block_base > max_flat_base)
	{
	    starttime = timestamp();
	    bool ifblock = extract_block(coomat, mat, info, cooremain1, bench, estimate_flat_partial, estimate_flat_full);
	    endtime = timestamp();
	    printf("Block category analysis time %lf s\n", endtime - starttime);
	    starttime = timestamp();
	    if (ifblock)
		extract_flat(cooremain1, mat, info, bench);
	    else
		extract_flat(coomat, mat, info, bench);
	    endtime = timestamp();
	    printf("Flat category analysis time %lf s\n", endtime - starttime);
	}
	else
	{
	    starttime = timestamp();
	    extract_flat(coomat, mat, info, bench);
	    endtime = timestamp();
	    printf("Flat category analysis time %lf s\n", endtime - starttime);
	}
    }

    free_coo_matrix(cooremain1);
    free_coo_matrix(cooremain2);


}

