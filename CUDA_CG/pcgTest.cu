// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include<sys/time.h>


//Utilities
#include "include/helper_debug.h"
// helper function CUDA error checking and initialization
#include "include/helper_cuda.h"  
#include "include/helper_functions.h"
#include "include/helper_cg.h"

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}


// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

void pcgTest_Case1();
void pcgTest_Case2();
void pcgTest_Case3();
void pcgTest_Case4();
void pcgTest_Case5();


int main(int argc, char** argv)
{   
    printf("\n\n= = = = pcgTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    pcgTest_Case1();

    // // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // // pcgTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // pcgTest_Case3();

    // // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // // pcgTest_Case4();

    // // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // // pcgTest_Case5();

    printf("\n\n= = = = end of pcgTest = = = =\n\n");
} // end of main



void pcgTest_Case1()
{
    const int N = 5;



    bool debug = true;
    
    // double mtxA_h[] = {        
    //     10.840188, 0.394383, 0.000000, 0.000000, 0.000000,  
    //     0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 
    //     0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 
    //     0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 
    //     0.000000, 0.000000, 0.000000, 0.768230, 10.277775 
    // };

    //Sparse matrix 
    
    int rowOffSets[] = {0, 2, 5, 8, 11, 13};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double vals[] = {10.840188, 0.394383, 
                      0.394383, 10.783099, 0.798440, 
                      0.798440, 10.911648, 0.197551,
                      0.197551, 10.335223, 0.768230,
                      0.768230, 10.277775};

    int nnz = 13;


    CSRMatrix csrMtxA_h = constructCSRMatrix(N, N, nnz, rowOffSets, colIndices, vals);


    double mtxB_h[] = {-0.957936, 0.099025, -0.312390, -0.141889, 0.429427};

    //(1) Allocate memory
    // double* mtxA_d = NULL;
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
    // CHECK(cudaMemcpy(mtxA_d, mtxA_h, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_vector(mtxSolX_d, N);
        printf("\n\n~~mtxB~~\n\n");
        print_vector(mtxB_d, N);
    }

    //Solve AX = B with bfbcg method
    pcg(csrMtxA_h, mtxSolX_d, mtxB_d, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Vector X📝📝📝~~\n\n");
        print_vector(mtxSolX_d, N);
    }

    //Validate with r - b -Ax with 2 Norm

    printf("\n\n~~🔍👀Validate Solution vector X 🔍👀~~");

    
    double twoNorms = validateCG(csrMtxA_h, N, mtxSolX_d, mtxB_d);
    printf("\n\n= = 1st Column Vector 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case1



