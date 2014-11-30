#ifndef __ANALYZE__H__
#define __ANALYZE__H__

#include <vector>

#include "constant.h"
#include "matrix_storage.h"

using namespace std;


struct overhead_entry
{
    unsigned int groupnum;
    unsigned int threadnum;
    double microsec;
};

class basic_entry
{
    public:
    unsigned int dimension;
    unsigned int width;
};

class block_entry
{
    public:
    unsigned int bheight;
    unsigned int bwidth;
    unsigned int dimension;
    unsigned int blocknum;
};

class bdia_entry:public basic_entry
{
    public:
    double bdia_gflops[BDIA_IMP_NUM];
};

class dia_entry:public basic_entry
{
    public:
    double dia_gflops[DIA_IMP_NUM];
};

class sbell_entry:public block_entry
{
    public:
    double sbell_gflops[SBELL_IMP_NUM]; 
};

class bell_entry:public block_entry
{
    public:
    double bell_gflops[BELL_IMP_NUM]; 
};

class bcsr_entry:public block_entry
{
    public:
    double bcsr_gflops[BCSR_IMP_NUM]; 
};

class sell_entry:public basic_entry
{
    public:
    double sell_gflops[SELL_IMP_NUM];
};

class ell_entry:public basic_entry
{
    public:
    double ell_gflops[ELL_IMP_NUM];
};

class csr_entry:public basic_entry
{
    public:
    unsigned int csr_groupnum;
    double csr_gflops[CSR_IMP_NUM];
};

class coo_entry:public basic_entry
{
    public:
    unsigned int coo_groupnum;
    double coo_gflops[COO_IMP_NUM];
};

struct benchmark_all
{
    vector<overhead_entry> overhead;
    vector<bdia_entry> bdia;
    vector<dia_entry> dia;
    vector<sbell_entry> sbell;
    vector<bell_entry> bell;
    vector<bcsr_entry> bcsr;
    vector<sell_entry> sell;
    vector<ell_entry> ell;
    vector<csr_entry> csr;
    vector<coo_entry> coo;
};

template <class dimType, class offsetType, class dataType>
struct cocktail
{
    bool ifusebdia;
    bool ifusedia;
    bool ifusesbell;
    bool ifusebell;
    bool ifusebcsr;
    bool ifusesell;
    bool ifuseell;
    bool ifusecsr;
    bool ifusecoo;

    int bdia_meth_num;
    int dia_meth_num;
    int sbell_meth_num;
    int bell_meth_num;
    int bcsr_meth_num;
    int sell_meth_num;
    int ell_meth_num;
    int csr_meth_num;
    int coo_meth_num;

    int csr_group_num;
    int coo_group_num;

    int mat_width;
    int mat_height;
    int mat_nnz;

    bdia_matrix<dimType, offsetType, dataType> bdia;
    dia_matrix<dimType, offsetType, dataType> dia;
    sbell_matrix<dimType, dataType> sbell;
    b4ell_matrix<dimType, dataType> bell;
    b4csr_matrix<dimType, dataType> bcsr;
    sell_matrix<dimType, dataType> sell;
    ell_matrix<dimType, dataType> ell;
    csr_matrix<dimType, dataType> csr;
    coo_matrix<dimType, dataType> coo;
};

template<class dimType, class offsetType, class dataType>
void init_cocktail(cocktail<dimType, offsetType, dataType>& mat)
{
    mat.ifusebdia  = false;
    mat.ifusedia   = false;
    mat.ifusesbell = false;
    mat.ifusebell  = false;
    mat.ifusebcsr  = false;
    mat.ifusesell  = false;
    mat.ifuseell   = false;
    mat.ifusecsr   = false;
    mat.ifusecoo   = false;

    mat.bdia_meth_num = 0;
    mat.dia_meth_num = 0;
    mat.sbell_meth_num = 0;
    mat.bell_meth_num = 0;
    mat.bcsr_meth_num = 0;
    mat.sell_meth_num = 0;
    mat.ell_meth_num = 0;
    mat.csr_meth_num = 0;
    mat.coo_meth_num = 0;

    mat.mat_width = 0;
    mat.mat_height = 0;
    mat.mat_nnz = 0;

    init_bdia_matrix(mat.bdia);
    init_dia_matrix(mat.dia);
    init_sbell_matrix(mat.sbell);
    init_b4ell_matrix(mat.bell);
    init_b4csr_matrix(mat.bcsr);
    init_sell_matrix(mat.sell);
    init_ell_matrix(mat.ell);
    init_csr_matrix(mat.csr);
    init_coo_matrix(mat.coo);
}

template<class dimType, class offsetType, class dataType>
void free_cocktail(cocktail<dimType, offsetType, dataType>& mat)
{
    mat.ifusebdia  = false;
    mat.ifusedia   = false;
    mat.ifusesbell = false;
    mat.ifusebell  = false;
    mat.ifusebcsr  = false;
    mat.ifusesell  = false;
    mat.ifuseell   = false;
    mat.ifusecsr   = false;
    mat.ifusecoo   = false;

    free_bdia_matrix(mat.bdia);
    free_dia_matrix(mat.dia);
    free_sbell_matrix(mat.sbell);
    free_b4ell_matrix(mat.bell);
    free_b4csr_matrix(mat.bcsr);
    free_sell_matrix(mat.sell);
    free_ell_matrix(mat.ell);
    free_csr_matrix(mat.csr);
    free_coo_matrix(mat.coo);
}

struct maxflop_info
{
    int bdia_meth_num;
    double bdia_max_flop;

    int dia_meth_num;
    double dia_max_flop;

    //1x4, 2x4, 4x4, 8x4, 1x8, 2x8, 4x8, 8x8
    int sbell_meth_num[8];
    double sbell_max_flop[8];

    int bell_meth_num[8];
    double bell_max_flop[8];

    int bcsr_meth_num[8];
    double bcsr_max_flop[8];

    int sell_meth_num;
    double sell_max_flop;
    
    int ell_meth_num;
    double ell_max_flop;

    int csr_meth_num;
    int csr_groupnum;
    double csr_max_flop;

    int coo_meth_num;
    int coo_groupnum;
    double coo_max_flop;

    double max_dia_base_flop;
    double max_block_base_flop;
    double max_flat_base_flop;
};

struct four_index
{
    //0: smaller dim, smaller width
    //1: smaller dim, larger width
    //2: larger dim, smaller width
    //3: larger dim, larger width
    unsigned int dim_s_wid_s;
    unsigned int dim_s_wid_l;
    unsigned int dim_l_wid_s;
    unsigned int dim_l_wid_l;
};

void analyze_matrix(coo_matrix<int, float>& coomat, cocktail<int, int, float>& mat, float dia_priority_multiplier = 1.0, bool estimate_flat_partial = true, bool estimate_flat_full = false);

#endif

