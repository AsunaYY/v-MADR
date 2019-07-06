#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int num, dim;
    double lambda1, lambda2, C, epsilon;
	double *y; //label
	struct feature_node **x; //instance
    double **H;
    double **invQY;
    double **G;
    double *p;
};

enum { CD, ASGD }; /* solver_type */
enum { LINEAR, POLY, RBF, SIGMOID }; /* kernel_type */

struct parameter
{
	int solver_type;
    int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */
    int times;
	double eps;	        /* stopping criteria */
};

struct model
{
	struct parameter param;
    int nr_feature;
	int nr_var;
    double ww;
	double *w;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

