#include <float.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../linear.h"

#include "mex.h"
#include "linear_model_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int print_null(const char *s,...) {}
int (*info)(const char *fmt,...);

double dot(struct feature_node *px, struct feature_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
        }
    }
    return sum;
}

double powi(double base, int times)
{
    int t;
    double tmp = base, ret = 1.0;
    
    for(t=times; t>0; t/=2)
    {
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

static void fake_answer(mxArray *plhs[])
{
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
//     plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
//     plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

void do_predict(mxArray *plhs[], const mxArray *prhs[], struct model *model_)
{
    int i, j, k, low, high;
    int training_instance_number, testing_instance_number;
    int elements, num_samples;
    double *ptr_label, *samples, *ptr_predict_label, *ptr_dec_values, *ptr, *w;
    mwIndex *ir, *jc;
    mxArray *pprhs[1], *pplhs[1], *instance_train_col, *instance_test_col; // instance sparse matrix in row format
    int correct = 0; 
    int total = 0;
    struct feature_node *train_space, *test_space;
    struct feature_node **training_instance, **testing_instance;
    
   
    // test instance 
    testing_instance_number = (int) mxGetM(prhs[0]); 
    plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);     
    ptr_dec_values = mxGetPr(plhs[0]);
    
    
        
    testing_instance = Malloc(struct feature_node*, testing_instance_number);
    pprhs[0] = mxDuplicateArray(prhs[0]);
    if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
    {
        mexPrintf("Error: cannot transpose testing instance matrix\n");
        fake_answer(plhs);
        return;
    }
    instance_test_col = pplhs[0];
    samples = mxGetPr(instance_test_col);
    ir = mxGetIr(instance_test_col);
    jc = mxGetJc(instance_test_col);
    num_samples = (int) mxGetNzmax(instance_test_col);
    elements = num_samples + testing_instance_number;
    test_space = Malloc(struct feature_node, elements);
    j = 0;
    for(i=0;i<testing_instance_number;i++)
    {
        testing_instance[i] = &test_space[j];
        low = (int) jc[i], high = (int) jc[i+1];
        for(k=low;k<high;k++)
        {
            test_space[j].index = (int) ir[k]+1;
            test_space[j].value = samples[k];
            j++;
        }
        test_space[j++].index = -1;
    }
    
   
    
    
    for(i=0;i<testing_instance_number;i++)
    {
       double dec_values = 0;
              
        while(testing_instance[i]->index!=-1)
        {
            dec_values += model_->w[testing_instance[i]->index-1] * testing_instance[i]->value;
            testing_instance[i]++;
        }
                
       
        ptr_dec_values[i] = dec_values;        
    }    
    
    
    free(testing_instance);
    free(test_space);
}

void exit_with_help()
{
    mexPrintf(
            "Usage: [pre_value] = vMADR_ASGD_predict(testing_instance_matrix, training_instance_matrix, model)\n"
            "Returns:\n"
            "  pre_value: predicted value.\n"
            "made by AsunaYY"
            );
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    struct model *model_;
    char cmd[CMD_LEN];
    info = &mexPrintf;
    
    if(nrhs != 3)
    {
        exit_with_help();
        fake_answer(plhs);
        return;
    }
    
    if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) ) {
        mexPrintf("Error: testing instance matrix and training instance matrix must be double\n");
        fake_answer(plhs);
        return;
    }
        
    if(mxIsStruct(prhs[2]))
    {
        const char *error_msg;
        model_ = Malloc(struct model, 1);
        error_msg = matlab_matrix_to_model(model_, prhs[2]);
        if(error_msg)
        {
            mexPrintf("Error: can't read model: %s\n", error_msg);
            free_and_destroy_model(&model_);
            fake_answer(plhs);
            return;
        }
        
                
        if(mxIsSparse(prhs[0]) && mxIsSparse(prhs[1]))
            do_predict(plhs, prhs, model_);
        else
        {
            mexPrintf("Testing_instance_matrix must be sparse; "
                    "use sparse(Testing_instance_matrix) first\n");
            fake_answer(plhs);
        }
        
        // destroy model_
        free_and_destroy_model(&model_);
    }
    else
    {
        mexPrintf("model file should be a struct array\n");
        fake_answer(plhs);
    }
    
    return;
}
