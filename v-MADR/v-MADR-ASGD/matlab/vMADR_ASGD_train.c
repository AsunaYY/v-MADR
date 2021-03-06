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
#define INF HUGE_VAL

void print_null(const char *s) {}
void print_string_matlab(const char *s) {mexPrintf(s);}


void exit_with_help()
{
    mexPrintf(
            "Usage: model = vMADR_ASGD_train(label_vector, instance_matrix, vMADR_parametes, 'vMADR_options');\n"
            "vMADR_options:\n"
            "-s solver_type: set type of solver (default 1)\n"
            "	 0 -- Coordinate Descent (dual)\n"
            "	 1 -- Average Stochastic Gradient Descent (primal)\n"
            "-k kernel_type: set type of kernel function (default 0)\n"
            "	 0 -- linear: u'*v\n"
            "-t times : set times for asgd to scan data (default 5)\n" 
            "made by AsunaYY"
            );
}

struct parameter param;		// set by parse_command_line
struct problem prob;		// set by read_problem
struct model *model_;
struct feature_node *x_space, *xi;

int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
    int i, argc = 1;
    char cmd[CMD_LEN];
    char *argv[CMD_LEN/2];
    void (*print_func)(const char *) = print_string_matlab;	// default printing to matlab display
    
    // default values
    param.solver_type = ASGD;
    param.kernel_type = LINEAR;
    param.times = 5;
    param.degree = 3;
    param.gamma = 0;
    param.coef0 = 0;
    param.eps = 0.01;
    
    // put options in argv[]
    if(nrhs == 4)
    {
        mxGetString(prhs[3], cmd,  mxGetN(prhs[3]) + 1);
        if((argv[argc] = strtok(cmd, " ")) != NULL)
            while((argv[++argc] = strtok(NULL, " ")) != NULL)
                ;
    }
    else
        return 1;
    
    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
            return 1;
        switch(argv[i-1][1])
        {
            case 's':
                param.solver_type = atoi(argv[i]);
                break;
            case 'k':
                param.kernel_type = atoi(argv[i]);
                break;
            case 'd':
                param.degree = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'c':
                param.coef0 = atof(argv[i]);
                break;
            case 't':
                param.times = atoi(argv[i]);
                break;
            case 'e':
                param.eps = atoi(argv[i]);
                break;
            default:
                mexPrintf("unknown option\n");
                return 1;
        }
    }
    
    set_print_string_function(print_func);
    
    if(param.solver_type == ASGD)
        param.kernel_type = LINEAR;
    return 0;
}

static void fake_answer(mxArray *plhs[])
{
    plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

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

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat, const mxArray *para_vec)
{
    int i, j, k, low, high;
    mwIndex *ir, *jc;
    int elements, max_index, num_samples;
    double *condNumber;
    double *samples, *labels;
    double *X, *H, *para;
    double *QQ, *invQQ, *G, *x_square;
    mxArray *instance_mat_col, *Q, *invQ, *cond;
    mxArray *prhs[1], *plhs[1];
    double *Ge = NULL, *Gy = NULL, ysum;
    double *Xe = NULL, *Xy = NULL;
    double* p = NULL;
    
        
    prhs[0] = mxDuplicateArray(instance_mat);
    mexCallMATLAB(1, plhs, 1, prhs, "transpose");
    instance_mat_col = plhs[0];
//     mexCallMATLAB(1, plhs, 1, prhs, "full");
         
    prob.num = (int) mxGetM(label_vec);
    
    // each column is one instance
    labels = mxGetPr(label_vec);
    samples = mxGetPr(instance_mat_col);
    ir = mxGetIr(instance_mat_col);
    jc = mxGetJc(instance_mat_col);
    
    num_samples = (int) mxGetNzmax(instance_mat_col);
    elements = num_samples + prob.num;
    max_index = (int) mxGetM(instance_mat_col);
    prob.y = Malloc(double, prob.num);
    prob.x = Malloc(struct feature_node*, prob.num);
    x_space = Malloc(struct feature_node, elements);
    
    j = 0;
    for(i=0;i<prob.num;i++)
    {
        prob.x[i] = &x_space[j];
        prob.y[i] = labels[i];
        low = (int) jc[i], high = (int) jc[i+1]; // 一列一列处理,jc[i]是第i列第一个非零元素
        for(k=low;k<high;k++)
        {
            x_space[j].index = (int) ir[k]+1;
            x_space[j].value = samples[k];
            j++;
        }
        x_space[j++].index = -1;
    }
    prob.dim = max_index;
    
    para = mxGetPr(para_vec);
    prob.lambda1 = para[0];
    prob.lambda2 = para[1];
    prob.C = para[2];
    prob.epsilon = para[3];
    
      

    
    return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    // Transform the input Matrix to libsvm format
    if(nrhs > 2 && nrhs < 5)
    {
        int i, j;
        int err=0;
        
        if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
            mexPrintf("Error: label vector and instance matrix must be double\n");
            fake_answer(plhs);
            return;
        }
        
        if(parse_command_line(nrhs, prhs, NULL))
        {
            exit_with_help();
            fake_answer(plhs);
            return;
        }
        
        if(mxIsSparse(prhs[1]))
            err = read_problem_sparse(prhs[0], prhs[1], prhs[2]);
        else
        {
            mexPrintf("Training_instance_matrix must be sparse; "
                    "use sparse(Training_instance_matrix) first\n");
            fake_answer(plhs);
            return;
        }
               
        if(err)
        {
            fake_answer(plhs);
            return;
        }
        
       
        model_ = train(&prob, &param);        
       
        model_to_matlab_structure(plhs, model_);
       
        free_and_destroy_model(&model_);
        free(prob.y);
        free(prob.x);
       
        free(x_space);
    }
    else
    {
        exit_with_help();
        fake_answer(plhs);
        return;
    }
}
