#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

// helper function CUDA error checking and initialization
#include "helper_cuda.h"  
#include "helper_debug.h"
#include "CSRMatrix.h"  

// // Forward Declarations of functions from helper_orth.h
// void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);
// double* transpose_Den_Mtx(cublasHandle_t cublasHandler, double* mtxX_d, int numOfRow, int numOfClm);



#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

//function headers
//Generate random SPD dense matrix
// N is matrix size
// SPD <- mtxA * mtxA'
double* generateSPD_DenseMatrix(int N);

// N is matrix size
double* generate_TriDiagMatrix(int N);


void initializeRandom(double mtxB_h[], int numOfRow, int numOfClm);


//Dense Diagonally doinat SPD matrix 
double* generateWellConditoinedSPDMatrix(int N);


void validateSol(const double *mtxA_h, const double* x_h, double* rhs, int N);



//Breakdown Free Block Conjugate Gradient (BFBCG) function
//Input: double* mtxA_d, double* mtxB_d, double* mtxSolX_d, int numOfA, int numOfColX
//Process: Solve AX = B where A is sparse matrix, X is solutoinn column vectors and B is given column vectors
//Output: double* mtxSolX_d
void bfbcg(CSRMatrix &csrMtxA, double* mtxSolX_d, double* mtxB_d, int numOfA, int numOfColX);





//Sub-funcstions for BFBCG implementation

//Input: cublasHandle_t cublasHandler, double* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
//Output: boolean
bool checkStop(cublasHandle_t cublasHandler, double *mtxR_d, int numOfRow, int numOfClm, double const threshold);

//Input: cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double* residual as answer
//Process: Extrace the first column vector from the Residual mtrix and calculate dot product
//Output: double& rsdl_h
void calculateResidual(cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double& rsdl_h);


// //Inverse funtion with sub functions
// //Input: double* matrix A, double* matrix A inverse, int N, number of Row or Column
// //Process: Inverse matrix
// //Output: lfoat mtxQPT_d
// void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, double* mtxA_d, double* mtxA_inv_d, int N);


// //Input: double* identity matrix, int numer of Row, Column
// //Process: Creating identity matrix with number of N
// //Output: double* mtxI
// __global__ void identity_matrix(double* mtxI_d, int N);

// //Input: double* identity matrix, int N, which is numer of Row or Column
// //Process: Call the kernel to create identity matrix with number of N
// //Output: double* mtxI
// void createIdentityMtx(double* mtxI_d, int N);


//Input: const CSRMatrix &csrMtx, double *dnsMtxB_d, int numClmB, double * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, double *dnsMtxB_d, int numClmsB, double * dnsMtxC_d);




//Input: double *dnsMtxB_d, const CSRMatrix &csrMtx, double *dnsMtxX_d, int numClmB, double * dnxMtxC_d
//Process: perform C = C - AX
//Output: dnsMtxC_d, dense matrix C in device
void den_mtx_subtract_multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d);

//Input:
//Process: perform vector y = y  - Ax
//Output: dnsVecY_d, dense vector C in device
void den_vec_subtract_multiplly_Sprc_Den_vec(const CSRMatrix &csrMtx, double *dnsVecX_d, double *dnsVecY_d);


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix -A and matrix B
//Result: matrix C as a result
void multiply_ngt_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix C <-  matrix A * matrix B + matrixC with overwrite
//Result: matrix C as a result
void multiply_sum_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxC_d, int numOfRow, int numOfClm);

//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);

//Input: cublasHandle_t cublasHandler, double* mtxB_d, double* mtxA_d, double* mtxSolX_d, int numOfRowA, int numOfClmB
//Process: Perform R = 
//Output: double* mtxB_d as a result with overwritten
void subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);


//Input: cublasHandler_t cublasHandler, double* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: double* matrix X transpose
double* transpose_Den_Mtx(cublasHandle_t cublasHandler, double* mtxX_d, int numOfRow, int numOfClm);





//Input: double* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: double condition number
double computeConditionNumber(double* mtxA_d, int numOfRow, int numOfClm);

//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, double* matrix A
//Process: Extract eigenvalues with full SVD
//Output: double* sngVals_d, singular values in device in column vector
double* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d);

//Input
//Process: perform r = b - Ax, dot product r' * r, then square norm
//Output: double twoNorms residual
double validateCG(const CSRMatrix &csrMtx, int numOfA, double *dnsVecX_d, double* dnsVecY_d);


//Input: CSRMatrix &csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d
//Process: perform R = B - AX, then calculate the first column vector 2 norms
//Output: double twoNorms
double validateBFBCG(const CSRMatrix &csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d);



// = = = Function signatures = = = = 
//Generate random SPD dense matrix
// N is matrix size
double* generateSPD_DenseMatrix(int N){
	double* mtx_h = NULL;
	double* mtx_d = NULL;
	double* mtxSPD_h = NULL;
	double* mtxSPD_d = NULL;

	//Using for cublas function
	const double alpha = 1.0f;
	const double beta = 0.0f;

	//Allocate memoery in Host
	mtx_h = (double*)malloc(sizeof(double)* (N*N));
	mtxSPD_h = (double*)malloc(sizeof(double)* (N*N));

	if(! mtx_h || ! mtxSPD_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h in all elements random values between 0 and 1.
	for (int wkr = 0; wkr < N*N;  wkr++){
		if(wkr % N == 0){
			printf("\n");
		}
		mtx_h[wkr] = ((double)rand()/RAND_MAX);
		// printf("\nmtx_h[%d] = %f", wkr, mtx_h[wkr]);

	}



	//(1)Allocate memoery in device
	CHECK(cudaMalloc((void**)&mtx_d, sizeof(double) * (N*N)));
	CHECK(cudaMalloc((void**)&mtxSPD_d, sizeof(double) * (N*N)));

	//(2) Copy value from host to device
	CHECK(cudaMemcpy(mtx_d, mtx_h, sizeof(double)* (N*N), cudaMemcpyHostToDevice));

	//(3) Calculate SPD matrix <- A' * A
	// Create a cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, mtx_d, N, mtx_d, N, &beta, mtxSPD_d, N));

	//(4) Copy value from device to host
	CHECK(cudaMemcpy(mtxSPD_h, mtxSPD_d, sizeof(double) * (N*N), cudaMemcpyDeviceToHost));
	
	//(5) Free memeory
	cudaFree(mtx_d);
	cudaFree(mtxSPD_d);
	cublasDestroy(handle);
	free(mtx_h);


	return mtxSPD_h;
} // enf of generateSPD_DenseMatrix


// N is matrix size
double* generate_TriDiagMatrix(int N)
{
	//Allocate memoery in Host
	double* mtx_h = (double*)calloc(N*N, sizeof(double));


	if(! mtx_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h tridiagonal random values between 0 and 1.
	mtx_h[0] = ((double)rand()/RAND_MAX)+10.0f;
	mtx_h[1] = ((double)rand()/RAND_MAX);
	for (int wkr = 1; wkr < N -1 ;  wkr++){
		mtx_h[(wkr * N) + (wkr-1)] =((double)rand()/RAND_MAX);
		mtx_h[wkr * N + wkr] = ((double)rand()/RAND_MAX)+10.0f;
		mtx_h[(wkr * N) + (wkr+1)] = ((double)rand()/RAND_MAX);
	}
	mtx_h[(N*(N-1))+ (N-2)] = ((double)rand()/RAND_MAX);
	mtx_h[(N*(N-1)) + (N-1)] = ((double)rand()/RAND_MAX) + 10.0f;
	

	//Scale down
	for (int i = 0; i < N * N; i++) {
        mtx_h[i] /= 10;
    }

	return mtx_h;
} // enf of generate_TriDiagMatrix

//Initialize random values between -1 and 1
void initializeRandom(double mtxB_h[], int numOfRow, int numOfClm)
{
	srand(time(NULL));

	for (int wkr = 0; wkr < numOfRow * numOfClm; wkr++){
		//Generate a random double between -1 and 1
		double rndVal = ((double)rand() / RAND_MAX) * 2.0f - 1.0f;
		mtxB_h[wkr] = rndVal;
	}
} // end of initializeRandom

double* generateWellConditoinedSPDMatrix(int N)
{
	double* mtxA_h = (double*)malloc(N * N * sizeof(double));

	srand(time(NULL));

	//Generate a random matrix
	for (int otWkr = 0; otWkr < N; otWkr++){
		for (int inWkr = 0; inWkr <= otWkr; inWkr++){
			double val = (double)rand() / RAND_MAX;
			mtxA_h[otWkr * N + inWkr] = val;
			mtxA_h[inWkr * N + otWkr] = val;
		} // end of inner loop
	} // end of outer loop

	//Ensure the matrix is diagonally dominant
	for (int wkr = 0; wkr < N; wkr++){
		mtxA_h[wkr * N + wkr] += N;
	}

	return mtxA_h;
} // end of generateWellConditoinedSPDMatrix

void validateSol(const double *mtxA_h, const double* x_h, double* rhs, int N)
{
    double rsum, diff, error = 0.0f;

    for (int rw_wkr = 0; rw_wkr < N; rw_wkr++){
        rsum = 0.0f;
        for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
            rsum += mtxA_h[rw_wkr*N + clm_wkr]* x_h[clm_wkr];
            // printf("\nrsum = %f", rsum);
        } // end of inner loop
        diff = fabs(rsum - rhs[rw_wkr]);
        if(diff > error){
            error = diff;
        }
        
    }// end of outer loop
    
    printf("\n\nTest Summary: Error amount = %f\n", error);

}// end of validateSol




//Stop condition
//Input: cublasHandle_t cublasHandler, double* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
bool checkStop(cublasHandle_t cublasHandler, double *mtxR_d, int numOfRow, int numOfClm, double const threshold)
{
	double *r1_d = NULL;
	double dotPrdct = 0.0f;
	bool debug =false;

	//Extract first column
	CHECK(cudaMalloc((void**)&r1_d, numOfRow * sizeof(double)));
	CHECK(cudaMemcpy(r1_d, mtxR_d, numOfRow * sizeof(double), cudaMemcpyDeviceToDevice));

	if(debug){
		printf("\n\nvector r_1: \n");
		print_vector(r1_d, numOfRow);
	}
	
	//Dot product of r_{1}' * r_{1}, cublasSdot
	checkCudaErrors(cublasDdot(cublasHandler, numOfRow, r1_d, 1, r1_d, 1, &dotPrdct));

	//Square root(dotPrdct)
	if(debug){
		printf("\n\ndot product of r_1: %.10f\n", dotPrdct);
		printf("\n\nsqrt(dot product of r_1): %.10f\n", sqrt(dotPrdct));
		printf("\n\nTHRESHOLD : %f\n", threshold);
	}

	CHECK(cudaFree(r1_d));

	return (sqrt(dotPrdct)< threshold);
}


//Input: cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double* residual as answer
//Process: Extrace the first column vector from the Residual mtrix and calculate dot product
//Output: double& rsdl_h
void calculateResidual(cublasHandle_t cublasHandler, double * mtxR_d, int numOfRow, int numOfClm, double& rsdl_h)
{	
	double* r1_d = NULL;
	bool debug =false;

	//Extract first column
	CHECK(cudaMalloc((void**)&r1_d, numOfRow * sizeof(double)));
	CHECK(cudaMemcpy(r1_d, mtxR_d, numOfRow * sizeof(double), cudaMemcpyDeviceToDevice));

	if(debug){
		printf("\n\nvector r_1: \n");
		print_vector(r1_d, numOfRow);
	}
	
	//Dot product of r_{1}' * r_{1}, cublasSdot
	checkCudaErrors(cublasDdot(cublasHandler, numOfRow, r1_d, 1, r1_d, 1, &rsdl_h));

	//Square root(dotPrdct)
	if(debug){
		printf("\n\ndot product of r_1: %.10f\n", rsdl_h);
		printf("\n\nsqrt(dot product of r_1): %.10f\n", sqrt(rsdl_h));
	}

	// Set residual swuare root of dot product
	rsdl_h = sqrt(rsdl_h);

	CHECK(cudaFree(r1_d));
}

// //Inverse
// //Input: double* matrix A, double* matrix A inverse, int N, number of Row or Column
// //Process: Inverse matrix
// //Output: lfoat mtxQPT_d
// void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, double* mtxA_d, double* mtxA_inv_d, int N)
// {
// 	double* mtxA_cpy_d = NULL;

// 	double *work_d = nullptr;

//     //The devInfo pointer holds the status information after the LU decomposition or solve operations.
//     int *devInfo = nullptr;
    
//     /*
//     A pivots_d pointer holds the pivot indices generated by the LU decomposition. 
//     These indices indicate how the rows of the matrix were permuted during the factorization.
//     */
//     int *pivots_d = nullptr;
    
//     //Status information specific to the LAPACK operations performed by cuSolver.
//     // int *lapackInfo = nullptr;

//     // Size of the workng space
//     int lwork = 0;
// 	bool debug = false;


//     if(debug){
//         printf("\n\n~~mtxA_d~~\n\n");
//         print_mtx_clm_d(mtxA_d, N, N);
//     }

// 	//(1) Make copy of mtxA
// 	CHECK(cudaMalloc((void**)&mtxA_cpy_d, N * N * sizeof(double)));
// 	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice));
	
// 	if(debug){
// 		printf("\n\n~~mtxA_cpy_d~~\n\n");
// 		print_mtx_clm_d(mtxA_cpy_d, N, N);
// 	}

// 	//(2) Create Identity matrix
// 	createIdentityMtx(mtxA_inv_d, N);
// 	if(debug){
// 		printf("\n\n~~mtxI~~\n\n");
// 		print_mtx_clm_d(mtxA_inv_d, N, N);
// 	}

// 	//(3)Calculate work space for cusolver
//     checkCudaErrors(cusolverDnDgetrf_bufferSize(cusolverHandler, N, N, mtxA_cpy_d, N, &lwork));
//     checkCudaErrors(cudaMalloc((void**)&work_d, lwork * sizeof(double)));
// 	CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
//     CHECK(cudaMalloc((void**)&pivots_d, N * sizeof(int)));

// 	//(4.1) Perform the LU decomposition, 
//     checkCudaErrors(cusolverDnDgetrf(cusolverHandler, N, N, mtxA_cpy_d, N, work_d, pivots_d, devInfo));
//     cudaDeviceSynchronize();

// 	//Check LU decomposition was successful or not.
// 	//If not, it can be ill-conditioned or singular.
// 	int devInfo_h;
// 	checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
// 	if(devInfo_h != 0){
// 		printf("\n\nLU decomposition failed in the inverse_Den_Mtx, info = %d\n", devInfo_h);
// 		if(devInfo_h == 11){
// 			printf("\n!!!The matrix potentially is ill-conditioned or singular!!!\n\n");
// 			double* mtxA_check_d = NULL;
// 			CHECK(cudaMalloc((void**)&mtxA_check_d, N * N * sizeof(double)));
// 			CHECK(cudaMemcpy(mtxA_check_d, mtxA_d, N * N * sizeof(double), cudaMemcpyDeviceToDevice));
// 			double conditionNum = computeConditionNumber(mtxA_check_d, N, N);
// 			printf("\n\n🔍Condition number = %f🔍\n\n", conditionNum);
// 			CHECK(cudaFree(mtxA_check_d));
// 		}
// 		exit(1);
// 	}


//     /*
//     mtxA_d will be a compact form such that 

//     A = LU = | 4  1 |
//              | 1  3 |
    
//     L = |1    0 |  U = |4    1  |
//         |0.25 1 |      |0   2.75|
    
//     mtxA_d compact form = | 4      1  |
//                           | 0.25  2.75|
//     */
// 	if(debug){
//         printf("\n\nAfter LU factorization\n");
//         printf("\n\n~~mtxA_cpy_d~~\n");
//         print_mtx_clm_d(mtxA_cpy_d, N, N);
//     }


//     //(4.2)Solve for the iverse, UX = Y
//     /*
//     A = LU
//     A * X = LU * X = I
//     L * (UX) = L * Y = I
//     UX = Y
//     */
//     checkCudaErrors(cusolverDnDgetrs(cusolverHandler, CUBLAS_OP_N, N, N, mtxA_cpy_d, N, pivots_d, mtxA_inv_d, N, devInfo));
// 	cudaDeviceSynchronize();

// 	//Check solver after LU decomposition was successful or not.
// 	checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
// 	if(devInfo_h != 0){
// 		printf("Solve after LU failed, info = %d\n", devInfo_h);
// 		exit(1);
// 	}


// 	//(5)Free memoery
// 	CHECK(cudaFree(mtxA_cpy_d));
// 	CHECK(cudaFree(work_d));
//     CHECK(cudaFree(devInfo));
//     CHECK(cudaFree(pivots_d));

// } // end of inverse_Den_Mtx


// //Input: double* identity matrix, int numer of Row, Column
// //Process: Creating dense identity matrix with number of N
// //Output: double* mtxI
// __global__ void identity_matrix(double* mtxI_d, int N)
// {	
// 	//Get global index 
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
// 	//Set boundry condition
// 	if(idx < (N * N)){
// 		int glbRow = idx / N;
// 		int glbClm = idx % N;

// 		// Index points to the diagonal element
// 		if(glbRow == glbClm){
// 			mtxI_d[idx] = 1.0f;
// 		}else{
// 			mtxI_d[idx] = 0.0f;
// 		}// end of if, store diagonal element

// 	} // end of if, boundtry condition

// }// end of identity_matrix


// //Input: double* identity matrix, int N, which is numer of Row or Column
// //Process: Call the kernel to create identity matrix with number of N
// //Output: double* mtxI
// void createIdentityMtx(double* mtxI_d, int N)
// {		
// 	// Use a 1D block and grid configuration
//     int blockSize = 1024; // Number of threads per block
//     int gridSize = ceil((double)N * N / blockSize); // Number of blocks needed

//     identity_matrix<<<gridSize, blockSize>>>(mtxI_d, N);
    
// 	cudaDeviceSynchronize(); // Ensure the kernel execution completes before proceeding
// }


//Sparse matrix multiplicatation
//Input: const CSRMatrix &csrMtx, double *dnsMtxB_d, int numClmB, double * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, double *dnsMtxB_d, int numClmsB, double * dnsMtxC_d)
{
	int numRowsA = csrMtx.numOfRows;
	int numClmsA = csrMtx.numOfClms;
	int nnz = csrMtx.numOfnz;

	double alpha = 1.0f;
	double beta = 0.0f;

	bool debug = false;


	//(1) Allocate device memoery for CSR matrix
	int	*row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx.row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx.vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Crate cuSPARSE handle and descriptors
	cusparseHandle_t cusparseHandler;
	cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnMatDescr_t mtxB, mtxC;

	checkCudaErrors(cusparseCreateCsr(&mtxA, numRowsA, numClmsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnMat(&mtxB, numClmsA, numClmsB, numClmsA, dnsMtxB_d, CUDA_R_64F, CUSPARSE_ORDER_COL));
	checkCudaErrors(cusparseCreateDnMat(&mtxC, numRowsA, numClmsB, numRowsA, dnsMtxC_d, CUDA_R_64F, CUSPARSE_ORDER_COL));

	//(4) Calculate buffer size of Spase by dense matrix mulply operation
    size_t bufferSize = 0;
    void *dBuffer = NULL;
	checkCudaErrors(cusparseSpMM_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5)Perform sparse-dense matrix Multiplication
	checkCudaErrors(cusparseSpMM(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~mtxC after cusparseSpMM~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numRowsA, numClmsB);
	}

	//(6) Free memeory and destroy descriptors
	checkCudaErrors(cusparseDestroySpMat(mtxA));
	checkCudaErrors(cusparseDestroyDnMat(mtxB));
	checkCudaErrors(cusparseDestroyDnMat(mtxC));
	checkCudaErrors(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));

} // end of multiply_Src_Den_mtx



// Sparse matrix multiplication with a dense vector
// Input: const CSRMatrix &csrMtx, double *dnsVecX_d, double *dnsVecY_d
// Process: Matrix Multiplication Sparse matrix and Dense vector
// Output: dnsVecY_d, dense vector Y in device
void multiply_Sprc_Den_vec(const CSRMatrix &csrMtx, double *dnsVecX_d, double *dnsVecY_d)
{
	int numRowsA = csrMtx.numOfRows;
	int numColsA = csrMtx.numOfClms;
	int nnz = csrMtx.numOfnz;

	double alpha = 1.0f;
	double beta = 0.0f;
	
	bool debug = false;

	//(1) Allocate device memory for CSR matrix
	int *row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx.row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx.vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3)Create cuSPARSE handle and descriptors
	cusparseHandle_t cusparseHandler;
	cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnVecDescr_t vecX, vecY;

	checkCudaErrors(cusparseCreateCsr(&mtxA, numRowsA, numColsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecX, numColsA, dnsVecX_d, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecY, numRowsA, dnsVecY_d, CUDA_R_64F));


	//(4) Calculate buffer size of SpMV operation
	size_t bufferSize = 0;
	void *dBuffer = NULL;
	checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5) Perform sparse-dense vector Multiplication
	checkCudaErrors(cusparseSpMV(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~dnsVecY~~\n");
		print_vector(dnsVecY_d, numRowsA);
	}

	//(6) Free memory and destroy descriptors
	checkCudaErrors(cusparseDestroySpMat(mtxA));
	checkCudaErrors(cusparseDestroyDnVec(vecX));
	checkCudaErrors(cusparseDestroyDnVec(vecY));
	checkCudaErrors(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));

} // end of multiply_Sprc_Den_vec


//Input: double *dnsMtxB_d, const CSRMatrix &csrMtx, double *dnsMtxX_d, int numClmB, double * dnxMtxC_d
//Process: perform C = C - AX
//Output: dnsMtxC_d, dense matrix C in device
void den_mtx_subtract_multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d) {
	int numRowsA = csrMtx.numOfRows;
	int numClmsA = csrMtx.numOfClms;
	int nnz = csrMtx.numOfnz;

	double alpha = -1.0f;
	double beta = 1.0f;

	bool debug = false;


	//(1) Allocate device memoery for CSR matrix
	int	*row_offsets_d = NULL;
	int *col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx.row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx.vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Crate cuSPARSE handle and descriptors
	cusparseHandle_t cusparseHandler;
	cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnMatDescr_t mtxB, mtxC;

	checkCudaErrors(cusparseCreateCsr(&mtxA, numRowsA, numClmsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnMat(&mtxB, numClmsA, numClmsB, numClmsA, dnsMtxX_d, CUDA_R_64F, CUSPARSE_ORDER_COL));
	checkCudaErrors(cusparseCreateDnMat(&mtxC, numRowsA, numClmsB, numRowsA, dnsMtxC_d, CUDA_R_64F, CUSPARSE_ORDER_COL));

	//(4) Calculate buffer size of Spase by dense matrix mulply operation
    size_t bufferSize = 0;
    void *dBuffer = NULL;
	checkCudaErrors(cusparseSpMM_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5)Perform sparse-dense matrix Multiplication
	checkCudaErrors(cusparseSpMM(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~mtxC after cusparseSpMM~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numRowsA, numClmsB);
	}

	//(6) Free memeory and destroy descriptors
	checkCudaErrors(cusparseDestroySpMat(mtxA));
	checkCudaErrors(cusparseDestroyDnMat(mtxB));
	checkCudaErrors(cusparseDestroyDnMat(mtxC));
	checkCudaErrors(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));
} // end ofden_mtx_subtract_multiply_Sprc_Den_mtx



//Input:
//Process: perform vector y = y  - Ax
//Output: dnsVecY_d, dense vector C in device
void den_vec_subtract_multiplly_Sprc_Den_vec(const CSRMatrix &csrMtx, double *dnsVecX_d, double *dnsVecY_d){
	int numRowsA = csrMtx.numOfRows;
	int numColsA = csrMtx.numOfClms;
	int nnz = csrMtx.numOfnz;

	double alpha = -1.0f;
	double beta = 1.0f;

	bool debug = false;

	//(1) Allocate device memory for CSR matrix
	int *row_offsets_d = NULL;
	int * col_indices_d = NULL;
	double *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(double)));

	//(2) copy values from host to devise
	CHECK(cudaMemcpy(row_offsets_d, csrMtx.row_offsets, (numRowsA + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx.vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

	//(3) Create cuSPARSE handle and descriptors
	cusparseHandle_t cusparseHandler;
	cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnVecDescr_t vecX, vecY;

	checkCudaErrors(cusparseCreateCsr(&mtxA, numRowsA, numColsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecX, numRowsA, dnsVecX_d, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecY, numColsA, dnsVecY_d, CUDA_R_64F));

	//(4) Calculate buffer size of SpMV operation
	size_t bufferSize = 0;
	void *dBuffer = NULL;
	checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5) Perform sparse matrix-vector multiplication
	checkCudaErrors(cusparseSpMV(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\nVecY = vecY - mtxA * vecX with sparse function");
		printf("\n~~dnsVecY~~\n");
		print_vector(dnsVecY_d, numRowsA);
	}

	//(6) Free memory and 
	checkCudaErrors(cusparseDestroySpMat(mtxA));
	checkCudaErrors(cusparseDestroyDnVec(vecX));
	checkCudaErrors(cusparseDestroyDnVec(vecY));
	checkCudaErrors(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));


} // end of den_mtx_subtract_multiplly_Sprc_Den_vec

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = 1.0f;
	const double beta = 0.0f;

	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));

}


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix -A and matrix B
//Result: matrix C as a result
void multiply_ngt_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = -1.0f;
	const double beta = 0.0f;

	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));
}


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix C <-  matrix A * matrix B + matrixC with overwrite
//Result: matrix C as a result
void multiply_sum_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const double alpha = 1.0f;
	const double beta = 1.0f;

	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));
}



//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxC_d, int numOfRow, int numOfClm)
{	
	const double alpha = 1.0f;
	const double beta = 0.0f;
	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClm, numOfClm, numOfRow, &alpha, mtxA_d, numOfRow, mtxA_d, numOfRow, &beta, mtxC_d, numOfClm));
	
	// It will be no longer need inside orth(*), and later iteration
	CHECK(cudaFree(mtxA_d)); 
} // end of multiply_Den_ClmM_mtxT_mtx



//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, double* matrix A in device, double* matrix B in device , double* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix B
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB)
{
	
	const double alpha = 1.0f;
	const double beta = 0.0f;

	/*
	Note
	3rd parameter, m: Number of rows of C (or AT), which is numOfColA (K).
	4th parameter, n: Number of columns of C (or B), which is numOfColB (N).
	5th parameter, k: Number of columns of AT (or rows of B), which is numOfRowA (M).
	
	Summary,
	Thinking about 3rd and 4th parameter as matrix C would be colmnA * columnB because A is transposed.
	Then, the 5th parameter is inbetween number as rowA or rowB becuase matrix A is transpose
	*/
	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClmA, numOfClmB, numOfRowA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfRowA, &beta, mtxC_d, numOfClmA));

} // end of multiply_Den_ClmM_mtxT_mtx


//Input: cublasHandle_t cublasHandler, double* mtxB_d, double* mtxA_d, double* mtxSolX_d, int numOfRowA, int numOfClmB
//Process: Perform R = 
//Output: double* mtxB_d as a result with overwritten
void subtract_multiply_Den_mtx_ngtMtx_Mtx(cublasHandle_t cublasHandler, double* mtxA_d, double* mtxB_d, double* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB)
{
	const double alpha = -1.0f;
	const double beta = 1.0f;

	checkCudaErrors(cublasDgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfClmB, numOfClmA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfClmA, &beta, mtxC_d, numOfRowA));
}




//TODO check this function is need or not.
//Input: cublasHandler_t cublasHandler, double* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: double* matrix X transpose
double* transpose_Den_Mtx(cublasHandle_t cublasHandler, double* mtxX_d, int numOfRow, int numOfClm)
{	
	double* mtxXT_d = NULL;
	const double alpha = 1.0f;
	const double beta = 0.0f;

	//Allocate a new memory space for mtxXT
	CHECK(cudaMalloc((void**)&mtxXT_d, numOfRow * numOfClm * sizeof(double)));
	
	//Transpose mtxX
	// checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));
    checkCudaErrors(cublasDgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfRow, numOfClm, &alpha, mtxX_d, numOfClm, &beta, mtxX_d, numOfRow, mtxXT_d, numOfRow));

	//Free memory the original matrix X
	CHECK(cudaFree(mtxX_d));

	return mtxXT_d;
}





//Input: double* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: double condition number
double computeConditionNumber(double* mtxA_d, int numOfRow, int numOfClm)
{
	bool debug = false;

	//Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cusolverDnCreate(&cusolverHandler);

	double* sngVals_d = extractSngVals(cusolverHandler, numOfRow, numOfClm, numOfRow, mtxA_d);
	if(debug){
		printf("\n\nsngular values after SVD decomp\n\n");
		print_vector(sngVals_d, numOfClm);
	}

	double* sngVals_h = (double*)malloc(numOfClm * sizeof(double));
	CHECK(cudaMemcpy(sngVals_h, sngVals_d, numOfClm * sizeof(double), cudaMemcpyDeviceToHost));
	double conditionNum = sngVals_h[0] / sngVals_h[numOfClm-1];
	
	cusolverDnDestroy(cusolverHandler);
	CHECK(cudaFree(sngVals_d));
	free(sngVals_h);
	
	return conditionNum;

} // end of computeConditionNumber



//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, double* matrix A
//Process: Extract eigenvalues with full SVD
//Output: double* sngVals_d, singular values in device in column vector
double* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, double* mtxA_d)
{
	
	double *mtxA_cpy_d = NULL; // Need a copy to tranpose mtxZ'

	double *mtxU_d = NULL;
	double *sngVals_d = NULL;
	double *mtxVT_d = NULL;


	/*The devInfo is an integer pointer
    It points to device memory where cuSOLVER can store information 
    about the success or failure of the computation.*/
    int *devInfo = NULL;

    int lwork = 0;//Size of workspace
    //work_d is a pointer to device memory that serves as the workspace for the computation
    //Then passed to the cuSOLVER function performing the computation.
    double *work_d = NULL; // 
    double *rwork_d = NULL; // Place holder
    

    //Specifies options for computing all or part of the matrix U: = ‘A’: 
    //all m columns of U are returned in array
    signed char jobU = 'A';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'A';

	//Error cheking after performing SVD decomp
	int infoGpu = 0;

	bool debug = false;


	if(debug){
		printf("\n\n~~mtxA~~\n\n");
		print_mtx_clm_d(mtxA_d, numOfRow, numOfClm);
	}


	//(1) Allocate memeory in device
	//Make a copy of mtxZ for mtxZ'
    CHECK(cudaMalloc((void**)&mtxA_cpy_d, numOfRow * numOfClm * sizeof(double)));

	//For SVD decomposition
	CHECK(cudaMalloc((void**)&mtxU_d, numOfRow * numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&sngVals_d, numOfClm * sizeof(double)));
	CHECK(cudaMalloc((void**)&mtxVT_d, numOfClm * numOfClm * sizeof(double)));

	//(2) Copy value to device
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, numOfRow * numOfClm * sizeof(double), cudaMemcpyDeviceToDevice));
	
	
	if(debug){
		printf("\n\n~~mtxA cpy~~\n\n");
		print_mtx_clm_d(mtxA_cpy_d, numOfRow, numOfClm);
	}


	//(4) Calculate workspace for SVD decompositoin
	cusolverDnSgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork);
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));
	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

    //(3) Compute SVD decomposition
    cusolverDnDgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_cpy_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo);
	
	//(4) Check SVD decomp was successful. 
	checkCudaErrors(cudaMemcpy(&infoGpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(infoGpu != 0){
		printf("\n\n😖😖😖Unsuccessful SVD execution😖😖😖\n");
	}

	// if(debug){
	// 	printf("\n\n~~sngVals_d~~\n\n");
	// 	print_mtx_clm_d(sngVals_d, numOfClm, numOfClm);
	// }

	//(5) Free memoery
	checkCudaErrors(cudaFree(work_d));
	checkCudaErrors(cudaFree(devInfo));
	checkCudaErrors(cudaFree(mtxA_cpy_d));
	checkCudaErrors(cudaFree(mtxU_d));
	checkCudaErrors(cudaFree(mtxVT_d));


	return sngVals_d;

} // end of extractSngVals


double validateCG(const CSRMatrix &csrMtx, int numOfA, double *dnsVecX_d, double* dnsVecY_d)
{
	double residual = 0.0f;

	cublasHandle_t cublasHandler = NULL;
	checkCudaErrors(cublasCreate(&cublasHandler));
	
	den_vec_subtract_multiplly_Sprc_Den_vec(csrMtx, dnsVecX_d, dnsVecY_d);
	checkCudaErrors(cublasDdot(cublasHandler, numOfA, dnsVecY_d, 1, dnsVecY_d, 1, &residual));	

	checkCudaErrors(cublasDestroy(cublasHandler));

	return sqrt(residual);

}

//Input: CSRMatrix &csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d
//Process: perform R = B - AX, then calculate the first column vector 2 norms
//Output: double twoNorms
double validateBFBCG(const CSRMatrix &csrMtx, int numOfA, double *dnsMtxX_d, int numClmsB, double *dnsMtxC_d)
{
	bool debug = false;
	
	den_mtx_subtract_multiply_Sprc_Den_mtx(csrMtx, dnsMtxX_d, numClmsB, dnsMtxC_d);
	if(debug){
		printf("\n\nmtxR = B - AX\n");
		printf("~~mtxR~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numOfA, numClmsB);
	}
	
	cublasHandle_t cublasHandler = NULL;
	checkCudaErrors(cublasCreate(&cublasHandler));

	double twoNorms = 0.0f;
	calculateResidual(cublasHandler, dnsMtxC_d, numOfA, numClmsB, twoNorms);

	checkCudaErrors(cublasDestroy(cublasHandler));

	return twoNorms;
}




#endif // HELPER_FUNCTIONS_H