#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <algorithm>
#include "linear.h"
using std::random_shuffle;
#include "mex.h"

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
    dst = new T[n];
    memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#ifdef __cplusplus
extern "C" {
#endif
    
    extern double dnrm2_(int *, double *, int *);
    extern double ddot_(int *, double *, int *, double *, int *);
    extern int daxpy_(int *, double *, double *, int *, double *, int *);
    extern int dscal_(int *, double *, double *, int *);
    
#ifdef __cplusplus
}
#endif

static void print_string_stdout(const char *s)
{
    fputs(s,stdout);
    fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

int compare_double(const void *a, const void *b)
{
    if(*(double *)a > *(double *)b)
        return -1;
    if(*(double *)a < *(double *)b)
        return 1;
    return 0;
}

static double calculate_err(double* alpha, double* alpha_old, int m)
{
    double ret = 0.0;
    int i = 0;
    for(i = 0; i < m; i ++)
        ret += (alpha[i] - alpha_old[i]) * (alpha[i] - alpha_old[i]);
    
    return sqrt(ret);
}



#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class asgd
{
public:
    asgd(int num, int dim, double lambda1, double lambda2, double lambda3, double eps,  double eta0=0);
    ~asgd();
    void renorm();
    double wnorm();
    double anorm();
    void reset();
    double epsInsensitiveLoss(double a, double y);                 
    double dEpsInsensitiveLoss(double a, double y);             
    double hingeLoss(double a, double y);
    double dHingeLoss(double a, double y);
    double squareHingeLoss(double a, double y);
    double dSquareHingeLoss(double a, double y);
    double multiplyx(double *w, const struct feature_node *x); // 计算w^T * x
    void addx(double *w, const struct feature_node *x, double alpha); // 计算w = w + alpha * x
    void determineEta0(const struct problem *prob, int sample); // 选取eta0
    double evaluateEta(const struct problem *prob, int sample, int instanceNum, double eta);
    void trainOne(const struct feature_node *x, const double y, const feature_node *x2, const double y2, double eta, double mu);
    void testOne(const struct feature_node *x, const double y, double *ploss);
    void train_asgd(const struct problem *prob);
    double* geta();
    void clear();
private:
    int  dim;
    double  lambda1;
    double  lambda2;
    double  lambda3;
    double  eta0;
    double  mu0;
    double  tstart;
    double *w;
    double  wDivisor;
    double *a;
    double  aDivisor;
    double  wFraction;
    int  t;
    double one;
    double epsilon;
    int inc;
};

asgd::asgd(int num, int dim, double lambda1, double lambda2, double lambda3, double eps,  double eta0)
: dim(dim), lambda1(lambda1), lambda2(lambda2), lambda3(lambda3), eta0(eta0), mu0(1), tstart(num), wDivisor(1), aDivisor(1), wFraction(0), t(0), one(1.0), inc(1),epsilon(eps)
{
    w = new double[dim];
    a = new double[dim];
    for(int i = 0; i < dim; i++)
    {
        w[i] = 0;
        a[i] = 0;
    }
}

asgd::~asgd()
{
    delete w;
    delete a;
}

void asgd::reset()
{
    for(int i = 0; i < dim; i++)
    {
        w[i] = 0;
        a[i] = 0;
    }
    wDivisor=1;
    aDivisor=1;
    wFraction=0;
}

void asgd::clear()
{
    for(int i = 0; i < dim; i++)
    {
        w[i] = 0;
        a[i] = 0;
    }
}

void asgd::renorm()
{
    if (wDivisor != 1.0 || aDivisor != 1.0 || wFraction != 0)
    {
        double *w_new = new double[dim];
        double aDivisor_new = 1.0/aDivisor;
        double wDivisor_new = 1.0/wDivisor;
        memcpy(w_new, w, sizeof(double)*dim);
        
        dscal_(&dim, &wFraction, w_new, &inc);
        daxpy_(&dim, &one, w_new, &inc, a, &inc);
        dscal_(&dim, &aDivisor_new, a, &inc);
        dscal_(&dim, &wDivisor_new, w, &inc);
        
        wDivisor = 1;
        aDivisor = 1;
        wFraction = 0;
        delete w_new;
    }
}

double asgd::wnorm()
{
    double norm = dnrm2_(&dim, w, &inc);
    norm = norm / wDivisor / wDivisor;
    return norm;
}

double asgd::anorm()
{
    renorm();
    double norm = dnrm2_(&dim, a, &inc);
    return norm;
}
double asgd::epsInsensitiveLoss(double a, double y)                    //  a = \w^T \y      
{
    double z = y - a;
    double ret = 0.0;
    if (z > epsilon)
        ret = z - epsilon;
    else if (-z > epsilon)
        ret = -z - epsilon;
    else
        ret = 0.0;
    
    return ret;        
}

double asgd::dEpsInsensitiveLoss(double a, double y)       
{
    double z = y - a;
    if(z > epsilon) 
        return -1.0;
    else if(z < -epsilon)
        return 1.0;
    else
        return 0.0;
}



double asgd::multiplyx(double *w, const struct feature_node *x)
{
    double dotwtx = 0;
    while(x->index!=-1)
    {
        dotwtx += w[x->index-1] * x->value;
        x++;
    }
    return dotwtx;
}

void asgd::addx(double *w, const struct feature_node *x, double alpha)
{
    while(x->index!=-1)
    {
        w[x->index-1] += alpha * x->value;
        x++;
    }
}

void asgd::determineEta0(const problem *prob, int sample)
{
    const double factor = 2.0;
    int instanceNum = prob->num;
    double loEta = 1;
    double loCost = evaluateEta(prob, sample, instanceNum, loEta);
    double hiEta = loEta * factor;
    double hiCost = evaluateEta(prob, sample, instanceNum, hiEta);
    if(loCost < hiCost)
        while(loCost < hiCost)
        {
        hiEta = loEta;
        hiCost = loCost;
        loEta = hiEta / factor;
        loCost = evaluateEta(prob, sample, instanceNum, loEta);
        }
    else if(hiCost < loCost)
        while(hiCost < loCost)
        {
        loEta = hiEta;
        loCost = hiCost;
        hiEta = loEta * factor;
        hiCost = evaluateEta(prob, sample, instanceNum, hiEta);
        }
    eta0 = loEta;
    //info("eta0 is %f\n",eta0);
}

double asgd::evaluateEta(const problem *prob, int sample, int instanceNum, double eta)
{
    feature_node **x = prob->x;
    double *y = prob->y;
    int *index = new int[sample];
    int *index2 = new int[sample];
    for(int i=0; i<sample; i++)
    {
        index[i] = rand()%instanceNum;
        index2[i] = rand()%instanceNum;
        trainOne(x[index[i]], y[index[i]], x[index2[i]], y[index2[i]], eta, 1.0);
    }
    double ploss = 0;
    for(int i=0; i<sample; i++)
        testOne(x[i], y[i], &ploss);
    ploss = ploss / instanceNum;
    double cost1 = 0;
    double cost2 = 0;
    double cost3 = 0;
    double etXTw = 0.0; // modify
    double ety = 0.0;      // modify
    double ytXTw = 0.0;
    double *XTw = new double[sample];
    for(int i=0; i<sample; i++)
    {
        XTw[i] = multiplyx(w, x[index[i]]);
        ytXTw += XTw[i] * y[index[i]]; // y^T X w
        etXTw += XTw[i];
        ety += y[index[i]];
    }

    //update by AsunaYY!!!!!!!!!!!!!!!!!!!!!!!!!!
    cost2 = 4 *  lambda1 * ety * etXTw / (sample * sample) - (4 * lambda1 + 2 * lambda2) * ytXTw / sample;
    cost2 /= wDivisor;
    cost1 = (2 * lambda1 + lambda2) * ddot_(&sample, XTw, &inc, XTw, &inc) / sample - 2 * lambda1 * etXTw * etXTw / (sample * sample);
    cost1 /= (wDivisor * wDivisor);
    cost3 = 0.5 * lambda3 * ddot_(&dim, w, &inc, w, &inc) / (wDivisor * wDivisor);
   
    

    double cost = ploss + cost1 + cost2 + cost3;
    delete index;
    delete index2;
    delete XTw;
    reset();
    return cost;
}

void asgd::trainOne(const feature_node *x, const double y, const feature_node *x2, const double y2, double eta, double mu)
{   
    if(aDivisor > 1e5 || wDivisor > 1e5)
        renorm();
    
    double s = multiplyx(w, x) / wDivisor;
    double d;
   
    d = dEpsInsensitiveLoss(s,y);  // ldr
    
  
    d += ( (4 * lambda1 + 2 * lambda2) * s - 4 * lambda1 * multiplyx(w, x2)/ wDivisor );
    d += ( -(4 * lambda1 + 2 * lambda2) * y + 4 * lambda1 * y2 );          //update by AsunaYY!!!!!!!!!!!!!!!!!!!!!!!!!!
       
    wDivisor = wDivisor / (1 - eta * lambda3);
    double etd = - eta * wDivisor * d;
    if (etd != 0)
        addx(w, x, etd);
    if(mu >= 1)
    {
        aDivisor = wDivisor;
        wFraction = 1;
    }
    else if(mu > 0)
    {
        if(etd != 0)
            addx(a, x, - wFraction * etd);
        aDivisor = aDivisor / (1 - mu);
        wFraction = wFraction + mu * aDivisor / wDivisor;
    }
}

void asgd::testOne(const feature_node *x, const double y, double *ploss)
{
    double s = multiplyx(a,x);
    if(wFraction != 0)
        s += multiplyx(w,x) * wFraction;
    s = s / aDivisor;
   
    *ploss += epsInsensitiveLoss(s, y);
}

void asgd::train_asgd(const problem *prob)
{
    feature_node **x = prob->x;
    double *y = prob->y;
    int num = prob->num;
    int *index = new int[num];
    int *index2 = new int[num];
    for(int i=0; i < num; i++)
    {
        index[i] = i;
        index2[i] = i;
    }
    random_shuffle(index, index+num);
    random_shuffle(index2, index2+num);
    for(int i=0; i < num; i++)
    {
        //info("training on instance %d\n",index[i]);
        double eta = eta0 / pow(1 + lambda3 * eta0 * t, 0.75);
        double mu = (t < tstart) ? 1.0 : mu0 / (1 + mu0 * (t - tstart));
        trainOne(x[index[i]], y[index[i]], x[index2[i]], y[index2[i]], eta, mu);
        t += 1;
    }
    delete index;
    delete index2;    
}

double* asgd::geta()
{
    return a;
}

static void solve_asgd(const problem *prob, int times, double *w)
{
    int num = prob->num;
    int dim = prob->dim;
    double lambda1 = prob->lambda1/prob->C;
    double lambda2 = prob->lambda2/prob->C;
    double lambda3 = 1/prob->C;
    double epsilon = prob->epsilon;
    asgd md(num, dim, lambda1, lambda2, lambda3, epsilon);
    
    int sample = 1000;
    sample = min(sample, num);
    
    md.determineEta0(prob, sample);
    
    md.clear();
                
    for(int i=0; i<times; i++)
        md.train_asgd(prob);
    md.renorm();
    memcpy(w, md.geta(), sizeof(double)*dim);
}

static void train_one(const problem *prob, const parameter *param, double *w)
{
    double eps=param->eps;    
    
    int times = param->times;
    solve_asgd(prob, times, w);   
           
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{ 
    int w_size;
    
    w_size = prob->dim;
    model *model_ = Malloc(model,1);
    
    model_->nr_var = w_size;
    model_->nr_feature = prob->dim;
    model_->param = *param;
    
   
    model_->w=Malloc(double, w_size);
                
    train_one(prob, param, &model_->w[0]);
    
    // calculate the ||\w||^2
    model_->ww = 0;
    
    return model_;
}

void free_model_content(struct model *model_ptr)
{
    if(model_ptr->w != NULL)
        free(model_ptr->w);

}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
    struct model *model_ptr = *model_ptr_ptr;
    if(model_ptr != NULL)
    {
        free_model_content(model_ptr);
        free(model_ptr);
    }
}

void set_print_string_function(void (*print_func)(const char*))
{
    if (print_func == NULL)
        liblinear_print_string = &print_string_stdout;
    else
        liblinear_print_string = print_func;
}