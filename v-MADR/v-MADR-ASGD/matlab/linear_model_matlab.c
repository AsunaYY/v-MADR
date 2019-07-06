#include <stdlib.h>
#include <string.h>
#include "../linear.h"

#include "mex.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define NUM_OF_RETURN_FIELD 5

static const char *field_names[] = {
	"Parameters",
	"nr_feature",
    "nr_var",        
    "ww",
	"w",  
};

const char *model_to_matlab_structure(mxArray *plhs[], struct model *model_)
{
	int i;
	int nr_w;
	double *ptr;
	mxArray *return_model, **rhs;
	int out_id = 0;
	int w_size;

	rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);

	// Parameters
	// for now, only solver_type is needed
	rhs[out_id] = mxCreateDoubleMatrix(7, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model_->param.solver_type;
    ptr[1] = model_->param.kernel_type;
    ptr[2] = model_->param.degree;
    ptr[3] = (double)model_->param.gamma;
    ptr[4] = (double)model_->param.coef0;
    ptr[5] = model_->param.times;
    ptr[6] = (double)model_->param.eps;
	out_id++;

   	// nr_feature
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model_->nr_feature;
	out_id++;
    
    // nr_var
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model_->nr_var;
	out_id++;
	w_size = model_->nr_var;
    
  
    // ww
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = (double)model_->ww;
	out_id++;
     
	// w
	rhs[out_id] = mxCreateDoubleMatrix(w_size, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	for(i = 0; i < w_size*1; i++)
		ptr[i]=model_->w[i];
	out_id++;
 
	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);

	/* Fill struct matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
		mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
	plhs[0] = return_model;
	mxFree(rhs);

	return NULL;
}

const char *matlab_matrix_to_model(struct model *model_, const mxArray *matlab_struct)
{
	int i, num_of_fields;
	int nr_w;
	double *ptr;
	int id = 0;
	int n, w_size;
	mxArray **rhs;

	num_of_fields = mxGetNumberOfFields(matlab_struct);
	rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);

	for(i=0;i<num_of_fields;i++)
		rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);

	nr_w=1;
	model_->nr_var=0;
	model_->w=NULL;
    
// 	model_->label=NULL;
//     model_->nr_class=0;

	// Parameters
	ptr = mxGetPr(rhs[id]);
	model_->param.solver_type = (int)ptr[0];
    model_->param.kernel_type = (int)ptr[1];
    model_->param.degree = (int)ptr[2];
    model_->param.gamma = (double)ptr[3];
    model_->param.coef0 = (double)ptr[4];
    model_->param.times = (int)ptr[5];
    model_->param.eps = (double)ptr[6];
	id++;

	// nr_feature
	ptr = mxGetPr(rhs[id]);
	model_->nr_feature = (int)ptr[0];
	id++;
    
    // nr_var
	ptr = mxGetPr(rhs[id]);
	model_->nr_var = (int)ptr[0];
	id++;
    
	w_size = model_->nr_var;
    
    // ww
	ptr = mxGetPr(rhs[id]);
	model_->ww = (double)ptr[0];
	id++;
	
    // w
	ptr = mxGetPr(rhs[id]);
	model_->w=Malloc(double, w_size*nr_w);
	for(i = 0; i < w_size*nr_w; i++)
		model_->w[i]=ptr[i];
	id++;
	mxFree(rhs);

	return NULL;
}

