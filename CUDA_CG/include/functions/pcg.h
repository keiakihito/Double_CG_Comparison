#ifndef PCG_H
#define PCG_H


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

#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSPARSE_util.h"

// #define CHECK(call){ \
//  const cudaError_t cuda_ret = call; \
//  if(cuda_ret != cudaSuccess){ \
//      printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
//      printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
//      exit(-1); \
//  }\
// }


//Input:
//Process: Conjugate Gradient
//Output: vecSolX
void pcg(CSRMatrix &csrMtxA, double *vecSolX_h, double *vecB_h, int numOfA);


void pcg(CSRMatrix &csrMtxA, double *vecSolX_d, double *vecB_d, int numOfA)
{
    bool debug = false;
    const double THRESHOLD = 1e-6;

    double *r_d = NULL; // Residual
    double *s_d = NULL; // For s <- M * r and delta <- r' * s
    CSRMatrix csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
    double *dirc_d = NULL; // Direction
    double *q_d = NULL; // Vector Ad
    double dot = 0.0f; // temporary val for d^{T} *q to get aplha
    
    //Using for cublas functin argument
    double alpha = 1.0; 


    double initial_delta = 0.0;
    double delta_new = 0.0;
    double delta_old = 0.0;
    double relative_residual = 0.0;

    // In CG iteration alpha and beta
    double alph = 0.0f;
    double ngtAlph = 0.0f;
    double bta = 0.0f;

    

    //Crete handler
    cublasHandle_t cublasHandler = NULL;
    cusparseHandle_t cusparseHandler = NULL;
    // cusolverDnHandle_t cusolverHandler = NULL;
    
    CHECK_CUBLAS(cublasCreate(&cublasHandler));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandler));
    // CHECK_CUBLAS(cusolverDnCreate(&cusolverHandler));

    //(1) Allocate space in global memory
    CHECK(cudaMalloc((void**)&r_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&s_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&dirc_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&q_d, sizeof(double) * numOfA));

    //(2) Copy from host to device
    CHECK(cudaMemcpy(r_d, vecB_d, sizeof(double) * numOfA, cudaMemcpyDeviceToDevice));



    //(5) Iteration
    /* 💫💫💫Begin CG💫💫💫 */
    //Setting up the initial state.


    //r = b - Ax
    den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandler, csrMtxA, vecSolX_d, r_d);
    if(debug){
        printf("\n\nr_{0} = \n");
        print_vector(r_d, numOfA);
    }
    

    //Set d <- M * r;
    //M is Identity matrix for the place holder of precondition
    // CHECK(cudaMemcpy(dirc_d, r_d, N * sizeof(double), cudaMemcpyDeviceToDevice));
    multiply_Sprc_Den_vec(cusparseHandler, csrMtxM, r_d, dirc_d);
    if(debug){
        printf("\n\nd <- M * r");
        printf("\n\n~~vector d~~\n");
        print_vector(dirc_d, numOfA);
    }
    

    //delta_{new} <- r^{T} * d
    // Compute the squared norm of the initial residual vector r (stored in r1).
    CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, r_d, 1, dirc_d, 1, &delta_new));
    //Save it for the relative residual calculation.
    initial_delta = delta_new;
    // //✅
    if(debug){
        printf("\n\ndelta_new{0} = \n %f\n ", initial_delta);    
    }
    
    
    int cntr = 1; // counter
    const int MAX_ITR = 5;

    while(cntr <= MAX_ITR){

        printf("\n\n💫💫💫= = = Iteraion %d= = = 💫💫💫\n", cntr);

        
        //q <- Ad
        // CHECK_CUBLAS(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, dirc_d, strd_x, &beta, q_d, strd_y));
        multiply_Sprc_Den_vec(cusparseHandler, csrMtxA, dirc_d, q_d);
        // //✅
        if(debug){
            printf("\nq = \n");
            print_vector(q_d, numOfA);
        }
        

        //dot <- d^{T} * q
        CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, dirc_d, 1, q_d, 1, &dot));
        //✅
        if(debug){
            printf("\n\n~~(d'* q)~~\n %f\n", dot);
        }
        

        //alpha(a) <- delta_{new} / dot // dot <- d^{T} * q 
        alph = delta_new / dot;
        // //✅
        if(debug){
            printf("\nalpha = %f\n", alph);
        }
        

        //x_{i+1} <- x_{i} + alpha * d_{i}
        CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &alph, dirc_d, 1, vecSolX_d, 1));
        // //✅
        if(debug){
            printf("\nx_sol = \n");
            print_vector(vecSolX_d, numOfA);
        }


        if(cntr % 50 == 0){
            //r <- b -Ax Recompute
            CHECK(cudaMemcpy(r_d, vecB_d, sizeof(double) * numOfA, cudaMemcpyHostToDevice));
            den_vec_subtract_multiplly_Sprc_Den_vec(cusparseHandler, csrMtxA, vecSolX_d, r_d);
            if(debug){
                printf("\n\nr_{0} = \n");
                print_vector(r_d, numOfA);
            }
        }else{
            // Set -alpha
            ngtAlph = -alph;

            //r_{i+1} <- r_{i} -alpha*q
            CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &ngtAlph, q_d, 1, r_d, 1));
            //✅
            if(debug){
                printf("\n\nr = \n");
                print_vector(r_d, numOfA);
            }
            
        }

        //s <- M * r
        multiply_Sprc_Den_vec(cusparseHandler, csrMtxM, r_d, s_d);


        // delta_old <- delta_new
        delta_old = delta_new;
        //✅
        // if(debug){
        //     printf("\n\ndelta_old = %f\n", delta_old);
        // }
        

        // delta_new <- r' * s
        CHECK_CUBLAS(cublasDdot(cublasHandler, numOfA, r_d, 1, s_d, 1, &delta_new));
        //✅
        if(debug){
            printf("\n\ndelta_new = %f\n", delta_new);
        }

        relative_residual = sqrt(delta_new)/sqrt(initial_delta);
        printf("\n\n🫥Relative residual = %f🫥\n", relative_residual);
       
        if(sqrt(delta_new) < THRESHOLD){
            printf("\n\n🌀🌀🌀CONVERGED🌀🌀🌀\n\n");
            break; 
        }

        // bta <- delta_new / delta_old
        bta = delta_new / delta_old;
        //✅
        if(debug){
            printf("\nbta = %f\n", bta);
        }
        

        //d <- s + ßd
        CHECK_CUBLAS(cublasDscal(cublasHandler, numOfA, &bta, dirc_d, 1)); //d <- ßd
        CHECK_CUBLAS(cublasDaxpy(cublasHandler, numOfA, &alpha, s_d, 1, dirc_d, 1)); // d <- s + d
        if(debug){
            printf("\nd = \n");
            print_vector(dirc_d, numOfA);
        }

        cntr++;
    } // end of while




    //(6) Free the GPU memory after use
    CHECK_CUBLAS(cublasDestroy(cublasHandler));

    CHECK(cudaFree(r_d));
    CHECK(cudaFree(s_d));
    CHECK(cudaFree(dirc_d));
    CHECK(cudaFree(q_d););
    
}// end of pcg

#endif // PCG_H