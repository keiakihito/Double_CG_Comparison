#ifndef HELPER_CG_H
#define HELPER_CG_H

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
#include "helper_debug.h"
// helper function CUDA error checking and initialization
#include "helper_cuda.h"  
#include "helper_functions.h"

#define CHECK(call){ \
	const cudaError_t cuda_ret = call; \
	if(cuda_ret != cudaSuccess){ \
		printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
		printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
		exit(-1); \
	}\
}


//Input:
//Process: Conjugate Gradient
//Output: vecSolX
void cg(CSRMatrix &csrMtxA, double *vecSolX_h, double *vecB_h, int numOfA);


void cg(CSRMatrix &csrMtxA, double *vecSolX_h, double *vecB_h, int numOfA)
{
    bool debug = true;
    const double THRESHOLD = 1e-6;
    bool isStop = false;

    double *vecSolX_d = NULL;
    double *r_d = NULL; // Residual
    CSRMatrix csrMtxM = generateSparseIdentityMatrixCSR(numOfA); // Precondtion
    double *dirc_d = NULL; // Direction
    // double *Ax_d = NULL; // Vector Ax
    double *q_d = NULL; // Vector Ad
    double dot = 0.0f; // temporary val for d^{T} *q to get aplha
    
    //Using for cublas functin argument
    double alpha = 1.0; 
    double alphamns1 = -1.0;// negative alpha
    double beta = 0.0;

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
    
	checkCudaErrors(cublasCreate(&cublasHandler));
	checkCudaErrors(cusparseCreate(&cusparseHandler));
	// checkCudaErrors(cusolverDnCreate(&cusolverHandler));

    //(1) Allocate space in global memory
    CHECK(cudaMalloc((void**)&r_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&vecSolX_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&dirc_d, sizeof(double) * numOfA));
    // CHECK(cudaMalloc((void**)&Ax_d, sizeof(double) * numOfA));
    CHECK(cudaMalloc((void**)&q_d, sizeof(double) * numOfA));

    //(2) Copy from host to device
    CHECK(cudaMemcpy(vecSolX_d, vecSolX_h, sizeof(double) * numOfA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(r_d, vecB_h, sizeof(double) * numOfA, cudaMemcpyHostToDevice));



    //(5) Iteration
    /* 💫💫💫Begin CG💫💫💫 */
    //Setting up the initial state.


    //vector Ax
    // checkCudaErrors(cublasDgemv(cublasHandler, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, x_d, strd_x, &beta, Ax_d, strd_y));
    // if(debug){
    //     printf("\n\n~~vec Ax~~\n");
    //     print_vector(Ax_d, N);
    // }
    

    //r = b - Ax
    //This function performs the operation y=αx+y.
    // Update the residual vector r by calculating r_{0} = b - Ax_{0}
    // Given vector b is d_r only used 1 time. We will updateing d_r as a new residual.
    // which is critical for determining the direction and magnitude of the initial search step in the CG algorithm.
    // checkCudaErrors(cublasDaxpy(cublasHandler, N, &alphamns1, Ax_d, strd_x, r_d, strd_y));

    //r = b - Ax
    den_vec_subtract_multiplly_Sprc_Den_vec(csrMtxA, vecSolX_d, r_d);
    if(debug){
        printf("\n\nr_{0} = \n");
        print_vector(r_d, numOfA);
    }
    

    //Set d <- M * r;
    //M is Identity matrix for the place holder of precondition
    // CHECK(cudaMemcpy(dirc_d, r_d, N * sizeof(double), cudaMemcpyDeviceToDevice));
    multiply_Sprc_Den_vec(csrMtxM, r_d, dirc_d);
    if(debug){
        printf("\n\nr <- d");
        printf("\n\n~~vector d~~\n");
        print_vector(dirc_d, numOfA);
    }
    

    //delta_{new} <- r^{T} * d
    // Compute the squared norm of the initial residual vector r (stored in r1).
    checkCudaErrors(cublasDdot(cublasHandler, numOfA, r_d, 1, dirc_d, 1, &delta_new));
    //Save it for the relative residual calculation.
    initial_delta = delta_new;
    // //✅
    if(debug){
        printf("\n\ndelta_new{0} = \n %f\n ", delta_new);    
    }
    
    
    int cntr = 1; // counter
    const int MAX_ITR = 5;

    while(cntr <= MAX_ITR){
        if(debug){
            printf("\n\n💫💫💫= = = Iteraion %d= = = 💫💫💫\n", cntr);
        }
        
        //q <- Ad
        // checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, N, N, &alpha, mtxA_d, N, dirc_d, strd_x, &beta, q_d, strd_y));
        multiply_Sprc_Den_vec(csrMtxA, dirc_d, q_d);
        // //✅
        if(debug){
            printf("\nq = \n");
            print_vector(q_d, numOfA);
        }
        

        //dot <- d^{T} * q
        checkCudaErrors(cublasDdot(cublasHandler, numOfA, dirc_d, 1, q_d, 1, &dot));
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
        checkCudaErrors(cublasDaxpy(cublasHandler, numOfA, &alph, dirc_d, 1, vecSolX_d, 1));
        // //✅

        if(debug){
            printf("\nx_sol = \n");
            print_vector(vecSolX_d, numOfA);
        }


        if(cntr % 50 == 0){
            //r <- b -Ax Recompute
            CHECK(cudaMemcpy(r_d, vecB_h, sizeof(double) * numOfA, cudaMemcpyHostToDevice));
            den_vec_subtract_multiplly_Sprc_Den_vec(csrMtxA, vecSolX_d, r_d);
            if(debug){
                printf("\n\nr_{0} = \n");
                print_vector(r_d, numOfA);
            }
        }else{
            // Set -alpha
            ngtAlph = -alph;

            // //r_{i+1} <- r_{i} -alpha*q
            // if(debug && cntr == 3){
            //     printf("\n\n~~Before _{i+1} <- r_{i} -alpha*q~~\n");
            //     printf("\n\nr = \n");
            //     print_vector(r_d, N);
            //     printf("\n\nalpha = %f\n", alph);
            //     printf("\n\nngtAlpha = %f\n", ngtAlph);
            //     printf("\n\nq = \n");
            //     print_vector(q_d, N);
            // }
            //r_{i+1} <- r_{i} -alpha*q
            checkCudaErrors(cublasDaxpy(cublasHandler, numOfA, &ngtAlph, q_d, 1, r_d, 1));
            //✅
            if(debug){
                printf("\n\nr = \n");
                print_vector(r_d, numOfA);
            }
            
        }

        //r_{i+1} <- M * r_{i}
        //Allocate r2_d
        //copy r2_d <- r1_d
        //Perform r_d <- M * r2_d
        //Delete r2_d

        multiply_Sprc_Den_vec(csrMtxM, r_d, dirc_d);


        // delta_old <- delta_new
        delta_old = delta_new;
        //✅
        if(debug){
            printf("\n\ndelta_old = %f\n", delta_old);
        }
        

        // delta_new <- r'_{i+1} * r_{i+1}
        checkCudaErrors(cublasSdot(cublasHandle, N, r_d, strd_x, r_d, strd_y, &delta_new));
        //✅
        if(debug){
            printf("\n\ndelta_new = %f\n", delta_new);
        }

        // bta <- delta_new / delta_old
        bta = delta_new / delta_old;
        //✅
        if(debug){
            printf("\nbta = %f\n", bta);
        }
        

        //ßd <- bta * d_{i}
        checkCudaErrors(cublasSscal(cublasHandle, N, &bta, dirc_d, strd_x));
        //✅
        // printf("\n\n~~ ß AKA bta * dirc_d_{i} = \n");
        // print_vector(dirc_d, N);

        // d_{i+1} <- r_{i+1} + ßd_{i}
        checkCudaErrors(cublasSaxpy(cublasHandle, N, &alpha, r_d, 1, dirc_d, 1));
        //✅
        if(debug){
            printf("\nd = \n");
            print_vector(dirc_d, N);
        }

        relative_residual = sqrt(delta_new)/sqrt(initial_delta);
        printf("\n\nRelative residual = %f\n", relative_residual);
       
        cudaDeviceSynchronize();



        cntr++;
    } // end of while

    // if(cntr < MAX_ITR){
    //     printf("\n\n✅✅✅Converged at iteration %d✅✅✅\n", cntr-1);
    //     printf("\nRelative Error: delta_new = %f\n", delta_new);
    // }else{
    //     printf("\n\n😫😫😫The iteration did not converged😫😫😫\n");
    //     printf("\nRelative Error: delta_new = %f\n", delta_new);
    // }

    if(debug){
        printf("\n\nIteration %d", cntr - 1);
        printf("\nRelative residual = %f\n", relative_residual);
        printf("\n\n~~vector x_sol~~\n");
        print_vector(x_d, N);
    }
    
    
    double* x_h = (double*)malloc(sizeof(double) * N);
    CHECK(cudaMemcpy(x_h, x_d, N * sizeof(double), cudaMemcpyDeviceToHost));

    //Check error as error = b - A * x_sol
    validate(mtxA_h, x_h, rhs, N);



    //(6) Free the GPU memory after use
    cudaFree(mtxA_d);
    cudaFree(x_d);
    cudaFree(r_d);
    cudaFree(dirc_d);
    cudaFree(Ax_d);
    cudaFree(q_d);
    cublasDestroy(cublasHandle);
    free(x_h);
    free(x);
    free(rhs);



}// end of cg


// // Time tracker for each iteration
// double myCPUTimer()
// {
//     struct timeval tp;
//     gettimeofday(&tp, NULL);
//     return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
// }




int main(int argc, char** argv)
{   
    // double startTime, endTime;
  
    //Declare matrix A, solution vector x, given residual r
    //double pointers for device memoery
    // double *mtxA_d = NULL;
    // double *x_d = NULL;  // Solution vector x
    // double *r_d = NULL; // Residual
    // double *dirc_d = NULL; // Direction
    // double *Ax_d = NULL; // Vector Ax
    // double *q_d = NULL; // Vector Ad
    // double dot = 0.0f; // temporary val for d^{T} *q to get aplha
    
    // //Using for cublas functin argument
    // double alpha = 1.0; 
    // double alphamns1 = -1.0;// negative alpha
    // double beta = 0.0;

    // double initial_delta = 0.0;
    // double delta_new = 0.0;
    // double delta_old = 0.0;
    // double relative_residual = 0.0;
    // const double EPS = 1e-5f;
    // const int MAX_ITR = 4;

    // // In CG iteration alpha and beta
    // double alph = 0.0f;
    // double ngtAlph = 0.0f;
    // double bta = 0.0f;
    
    // //Stride for x and y using contiguous vectors
    // //Using for calling cublasSgemv
    // int strd_x = 1;
    // int strd_y = 1;
     
    // //Print 3 by 3 Matrix
    // printf("\n\n~~ 3 x 3 SPD matrix~~\n");
    // for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
    //     for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
    //         printf("%f ", mtxA_h[rw_wkr * N + clm_wkr]);
    //     }
    //     printf("\n");
    // }

    //Generating Random Dense SPD Matrix
    // double* mtxA_h = generateSPD_DenseMatrix(N);

    //Generating Random Tridiagonal SPD Matrix
    // double* mtxA_h = generate_TriDiagMatrix(N);
    // for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
    //     for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
    //         printf("%f ", mtxA_h[rw_wkr*N + clm_wkr]);
    //     }
    //     printf("\n");
    // }



    // //(0) Set initial guess and given vector b, right hand side
    // double *x = (double*)malloc(sizeof(double) * N);
    // double *rhs = (double*)malloc(sizeof(double) * N);
    
    
    // for (int i = 0; i < N; i++){
    //     x[i] = 0.0;
    //     rhs[i] = 1.0;
    // }//end of for

    // //(1) Allocate space in global memory
    // CHECK(cudaMalloc((void**)&mtxA_d, sizeof(double) * (N * N)));
    // CHECK(cudaMalloc((void**)&x_d, sizeof(double) * N));
    // CHECK(cudaMalloc((void**)&r_d, sizeof(double) * N));
    // CHECK(cudaMalloc((void**)&dirc_d, sizeof(double) * N));
    // CHECK(cudaMalloc((void**)&Ax_d, sizeof(double) * N));
    // CHECK(cudaMalloc((void**)&q_d, sizeof(double) * N));


    // //(2) Copy value from host to device 
    // CHECK(cudaMemcpy(mtxA_d, mtxA_h, sizeof(double) * (N * N), cudaMemcpyHostToDevice)); 
    // //✅
    // // Print the matrix from device to host (Check for Debugging)
    // print_mtx(mtxA_d, N, (N * N));


    // // x_{0}
    // CHECK(cudaMemcpy(x_d, x, N * sizeof(double), cudaMemcpyHostToDevice)); 
    // //✅
    // // printf("\n\nx_{0}\n");
    // // print_vector(x_d, N);

    // // rhs, r_d is b vector initial guess is all 1.
    // // The vector b is used only getting r_{0}, r_{0} = b - Ax where Ax = 0 vector
    // // Then keep updating residual r_{i+1} = r_{i} - alpha*Ad_{i}
    // CHECK(cudaMemcpy(r_d, rhs, N * sizeof(double), cudaMemcpyHostToDevice));
    // //✅
    // // printf("\n\nr_{0} AKA given vector b = \n");
    // // print_vector(r_d, N);


    // //(3) Handle to the CUBLAS context
    // //The cublasHandle variable will be used to store the handle to the cuBLAS library context. 
    // cublasHandle_t cublasHandle = 0;
    // cublasCreate(&cublasHandle);



    
    
    return 0;
} // end of main


#endif // HELPER_CG_H