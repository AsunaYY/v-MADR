#include <float.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
//#include "../linear.h"
#include <algorithm>
using std::random_shuffle;
#include "mex.h"
//#include "linear_model_matlab.h"

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#ifdef __cplusplus
extern "C" {
#endif
    
    extern double dnrm2_(int *, double *, int *);
    extern double ddot_(int *, double *, int *, double *, int *);
    extern int daxpy_(int *, double *, double *, int *, double *, int *);
    extern int dscal_(int *, double *, double *, int *);
    
struct feature_node
{
    int index;
    double value;
};
    
struct problem
{
    int kernel, num, dim;
    double lambda1, lambda2, C, v;                                         //update
    struct feature_node **x; // instance
    double *y;
    double *G;
    double *H;
    double *invQG;
    double *invQp;
};

#ifdef __cplusplus
}
#endif



void print_null(const char *s) {}
void print_string_matlab(const char *s) {mexPrintf(s);}


void exit_with_help()
{
    mexPrintf(
            "Usage: alpha = trainVMADR_CD(label_vector, instance_matrix, prob, vMADR_parametes);\n"
            "prob: {H, G, invQG, invQp}\n"
            "vMDR_parameters: [lambda1,lambda2, C, v] \n"
            "made by AsunaYY"
            );
}

// struct parameter param;		// set by parse_command_line
 struct problem prob;		// set by read_problem
// struct model *model_;
 struct feature_node *x_space, *xi;

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

int read_problem_sparse(const mxArray *label_vec,    const mxArray *instance_mat, 
                                            const mxArray* prob_struct, const mxArray *para_vec)
{
    int i, j, k, low, high;
    mwIndex *ir, *jc;
    int elements, max_index, num_samples;
    double *condNumber;
    double *samples, *labels;
    double *para;
    mxArray *instance_mat_col, *cond;
    mxArray *prhs[1], *plhs[1];
    
        
    
    // process X , Y  and params
    prhs[0] = mxDuplicateArray(instance_mat);
    mexCallMATLAB(1, plhs, 1, prhs, "transpose");
    instance_mat_col = plhs[0];
    mexCallMATLAB(1, plhs, 1, prhs, "full");
         
    prob.num = (int) mxGetM(label_vec);
     
     // each column is one instance
    labels = mxGetPr(label_vec);
    samples = mxGetPr(instance_mat_col);
    ir = mxGetIr(instance_mat_col);
    jc = mxGetJc(instance_mat_col);
    
    num_samples = (int) mxGetNzmax(instance_mat_col);
    elements = num_samples + prob.num;
    prob.dim = (int) mxGetN(instance_mat_col);                             //update
    prob.y = Malloc(double, prob.num);
    prob.x = Malloc(struct feature_node*, prob.num);
    x_space = Malloc(struct feature_node, elements);
    
    j = 0;
    for(i=0;i<prob.num;i++)
    {
        prob.x[i] = &x_space[j];
        prob.y[i] = labels[i];
        low = (int) jc[i], high = (int) jc[i+1]; 
        for(k=low;k<high;k++)
        {
            x_space[j].index = (int) ir[k]+1;
            x_space[j].value = samples[k];
            j++;
        }
        x_space[j++].index = -1;
    }
   
    
    para = mxGetPr(para_vec);
    prob.lambda1 = para[0];
    prob.lambda2 = para[1];
    prob.C = para[2];
    prob.v = para[3];                                                      //update

    // read problem struct
    prob.H = mxGetPr(mxGetCell(prob_struct, 0));
    prob.G = mxGetPr(mxGetCell(prob_struct, 1));
    prob.invQG = mxGetPr(mxGetCell(prob_struct, 2));
    prob.invQp = mxGetPr(mxGetCell(prob_struct, 3));
    
    return 0;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_cd(
        const problem *prob, double *alpha, double eps, int kernel)
{
    int num = prob->num;
    int dim = prob->dim;
    double lambda1 = prob->lambda1/prob->C;
    double lambda2 = prob->lambda2/prob->C;
    double C = 1/prob->C;
    double v = prob->v;                                                    //update
    int i, s, iter = 0, inc = 1;
    double d, G;
    int max_iter = 1000;
    int *index = new int[num];
    double *beta = new double[num];
    double* y = new double[num];
    int active_size = num;
    
    // PG: projected gradient, for shrinking and stopping
    double PG;
    double PGmax_old = INF;
    double PGmin_old = -INF;
    double PGmax_new, PGmin_new;
    
    for(i=0; i<num; i++)
    {
        y[i] = prob->y[i];
        index[i] = i;
        
         //initialize \beta = C*v/2
        beta[i] = C*v/2;                                                   //update
    }
    
   
  
    // kernel
    //initialize \w = (4\lambda1)/(m^2) \Q^(-1)\G\M\y + (2\lambda2)/m \Q^(-1)\G\y
    // linear
    //initialize \w = (4\lambda1)/(m^2) \Q^(-1)\X\M\y + (2\lambda2)/m \Q^(-1)\X\y
    if(kernel == 0) //kernel
    {
        for(i=0; i<num; i++)
        {
            alpha[i] = prob->invQp[i];
        }        
    }
    else //linear
    {
        for(i=0; i<dim; i++)
        {
            alpha[i] = prob->invQp[i];            
        }       
    }
        
    while(iter < max_iter)                                                 //update
    {
        PGmax_new = 0;
        PGmin_new = 0; // by lmz
        
        for(i=0; i<active_size; i++)
        {
            int j = i+rand()%(active_size-i);
            swap(index[i], index[j]);
        }
        
        for(s=0; s<active_size; s++)
        {
            i = index[s];
            G = 0;
            double yi = y[i];
            
            // calculate gradient g[i]:
            // linear: g[i] = (X; -X) \w^top + (\epsilon - y; \epsilon + y)
            // kernel: g[i] = (G; -G) \w^top + (\epsilon - y; \epsilon + y);
            if(kernel == 0) //kernel
            {
                G = ddot_(&num, alpha, &inc, prob->G+i*num, &inc);
                
                G = 2 * (G - yi);                                          //update
                    
            }
            else
            {
                feature_node *xi = prob->x[i];
                while(xi->index!= -1)
                {
                    G += alpha[xi->index-1]*(xi->value);
                    xi++;
                }
                
                G = 2 * (G - yi);                                          //update
                            
            }
            
            PG = 0;
            if (beta[i] == 0)
            {
                if (G > PGmax_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G < 0)
                {  
                    PG = G;
                    PGmin_new = min(PGmin_new, PG);
                }
            }
            else if (beta[i] == C/num)                                     //update
            {
                if (G < PGmin_old)
                {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
                else if (G > 0)
                {
                    PG = G;
                    PGmax_new = max(PGmax_new, PG); //
                }
            }
            else
            {
                PG = G;
                PGmax_new = max(PGmax_new, PG);
                PGmin_new = min(PGmin_new, PG);
            }
            
            
            if(fabs(PG) > 1.0e-12)
            {
                double beta_old = beta[i];
                beta[i] = min(max(beta[i] - G/(4*prob->H[i*num + i]), 0.0), C/num);//update

                d = beta[i] - beta_old;
                
                if(kernel == 0)
                {
                    for(int j=0; j<num; j++)
                        alpha[j] += prob->invQG[i*num + j] * d * 2;
                }
                else
                {
                    for(int j=0; j<dim; j++)
                        alpha[j] += prob->invQG[i * dim + j] * d * 2;
                }
            }
        }
        
               
        iter++;
        
        if(PGmax_new - PGmin_new <= eps)
        {
            if(active_size == num)
                break;
            else
            {
                active_size = num;
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;

        if (PGmax_old == 0)
            PGmax_old = INF;
        if (PGmin_old == 0)
            PGmin_old = -INF;
    }
    

    
    delete [] beta;
    delete [] y;
    delete [] index;
}


// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
{
    // Transform the input Matrix to libsvm format
    if(nrhs == 4)
    {
        int i, j;
        double* alpha = NULL;
        int err=0;
        
        if(!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
            mexPrintf("Error: label vector and instance matrix must be double\n");
            fake_answer(plhs);
            return;
        }
        
       if(mxIsSparse(prhs[1]))
            err = read_problem_sparse(prhs[0], prhs[1], prhs[2], prhs[3]);
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
        
        plhs[0] = mxCreateDoubleMatrix(prob.num, 1, mxREAL);
        alpha = mxGetPr(plhs[0]);
        
        solve_cd(&prob, alpha, 0.01, 0);       
    }
    else
    {
        exit_with_help();
        fake_answer(plhs);
        return;
    }
}
