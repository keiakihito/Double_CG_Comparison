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


//After refactor header files
#include "include/utils/checks.h"
#include "include/functions/helper.h"
#include "include/functions/cuBLAS_util.h"
#include "include/functions/cuSPARSE_util.h"
#include "include/functions/cuSOLVER_util.h"
#include "include/functions/pcg.h"
#include "include/CSRMatrix.h"




void pcgTest_Case1();
void pcgTest_Case2();
void pcgTest_Case3();
void pcgTest_Case4();
void pcgTest_Case5();


int main(int argc, char** argv)
{   
    printf("\n\n= = = = pcgTest.cu= = = = \n\n");
    
    // printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    // pcgTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // pcgTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // pcgTest_Case3();

    printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    pcgTest_Case4();

    // // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // // pcgTest_Case5();

    printf("\n\n= = = = end of pcgTest = = = =\n\n");
} // end of main



void pcgTest_Case1()
{
    cudaDeviceSynchronize();    
    const int N = 5;

    bool debug = false;
    
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
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
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
    printf("\n\n~~Valicate : r = b - A * x_sol ~~ \n = =  vector r 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case1



void pcgTest_Case2()
{
    cudaDeviceSynchronize();
    const int N = 10;

    bool debug = false;
    

    /*
    SPD Tridiagonal Dense
    10.840188 0.394383 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.394383 10.783099 0.798440 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.798440 10.911648 0.197551 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.197551 10.335223 0.768230 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.768230 10.277775 0.553970 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.553970 10.477397 0.628871 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.628871 10.364784 0.513401 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.513401 10.952229 0.916195 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.916195 10.635712 0.717297 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.717297 10.141603 
    */
    
    
    //Sparse matrix 
    int rowOffSets[] = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 28};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9};
    double vals[] = {10.840188, 0.394383,
                    0.394383, 10.783099, 0.798440,
                    0.798440, 10.911648, 0.197551,
                    0.197551, 10.335223, 0.768230,
                    0.768230, 10.277775, 0.553970,
                    0.553970, 10.477397, 0.628871, 
                    0.628871, 10.364784, 0.513401, 
                    0.513401, 10.952229, 0.916195,
                    0.916195, 10.635712, 0.717297,
                    0.717297, 10.141603
                    };

    int nnz = 28;


    CSRMatrix csrMtxA_h = constructCSRMatrix(N, N, nnz, rowOffSets, colIndices, vals);


    double mtxB_h[] = {-0.917206, 0.742276, 0.899125, -0.558456, 0.726547,0.343939, -0.319879, -0.901800, 0.898783,-0.885493};

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
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
    printf("\n\n~~Valicate : r = b - A * x_sol ~~ \n = =  vector r 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case2

void pcgTest_Case3()
{
    cudaDeviceSynchronize();
    const int N = 16;

    bool debug = false;
    

    /*
    SPD Tridiagonal Dense
    10.840188 0.394383 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.394383 10.783099 0.798440 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.798440 10.911648 0.197551 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.197551 10.335223 0.768230 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.768230 10.277775 0.553970 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.553970 10.477397 0.628871 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.628871 10.364784 0.513401 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.513401 10.952229 0.916195 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.916195 10.635712 0.717297 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.717297 10.141603 0.606969 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.606969 10.016300 0.242887 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.242887 10.137232 0.804177 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.804177 10.156679 0.400944 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.400944 10.129790 0.108809 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.108809 10.998924 0.218257 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.218257 10.512933 
    */
    
    //Sparse matrix 
    int rowOffSets[] = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 46};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 14, 15};
    double vals[] = {10.840188, 0.394383,
                    0.394383, 10.783099, 0.798440,
                    0.798440, 10.911648, 0.197551, 
                    0.197551, 10.335223, 0.768230, 
                    0.768230, 10.277775, 0.553970,
                    0.553970, 10.477397, 0.628871,
                    0.628871, 10.364784, 0.513401, 
                    0.513401, 10.952229, 0.916195, 
                    0.916195, 10.635712, 0.717297, 
                    0.717297, 10.141603, 0.606969,
                    0.606969, 10.016300, 0.242887,
                    0.242887, 10.137232, 0.804177,
                    0.804177, 10.156679, 0.400944, 
                    0.400944, 10.129790, 0.108809, 
                    0.108809, 10.998924, 0.218257, 
                    0.218257, 10.512933
                    };

    int nnz = 46;


    CSRMatrix csrMtxA_h = constructCSRMatrix(N, N, nnz, rowOffSets, colIndices, vals);


    double mtxB_h[] = {-0.828583, -0.206065, -0.357603, 0.477982, -0.268613, -0.063139, 0.296667, -0.326744, 0.445940, -0.912228, 0.211429, 0.346067, 0.664094, 0.893560, 0.828269, -0.867125};

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
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
    printf("\n\n~~Valicate : r = b - A * x_sol ~~ \n = =  vector r 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case3




void pcgTest_Case4()
{
    const int N = 32768;



    bool debug = false;

    CSRMatrix csrMtxA_h = generateSparseSPDMatrixCSR(N);


    double mtxB_h[N];
    initializeRandom(mtxB_h, N, 1);

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  N * sizeof(double)));

    //(2) Copy value from host to device
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
    printf("\n\n~~Valicate : r = b - A * x_sol ~~ \n = =  vector r 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case4
