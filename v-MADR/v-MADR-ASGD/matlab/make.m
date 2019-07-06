clear all;
clc;

mex  -largeArrayDims vMADR_ASGD_train.c linear_model_matlab.c ../linear.cpp  ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
mex  -largeArrayDims vMADR_ASGD_predict.c linear_model_matlab.c ../linear.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
          

