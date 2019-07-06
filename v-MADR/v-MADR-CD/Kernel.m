function kernel = Kernel(Xtrain, Xtest,kerfPara,kerType)
% Construct the positive (semi-) definite and symmetric kernel matrix
% Outputs
%   kernel  : N x N (N x Nt) kernel matrix
% Inputs
%   Xtrain      : N x d matrix with the inputs of the training data
%   Xtest  : Nt x d matrix with the inputs of the test data
%   kerfPara   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   kerType : Kernel type (by default 'RBF_kernel')




[n_train, dim] = size(Xtrain);
[n_test, dim] = size(Xtest);


if strcmp(kerType,'rbf')
    dist_train = sum(Xtrain.^2,2)*ones(1,n_test);
    dist_test = sum(Xtest.^2,2)*ones(1,n_train);
    sq_dists = dist_train+dist_test' - 2*Xtrain*Xtest';
    kernel = exp(-sq_dists.*kerfPara);
  
elseif strcmp(kerType,'lin')
    kernel = Xtrain * Xtest'

    
elseif strcmp(kerType,'poly')
    kernel = (Xtrain*Xtest'+1).^kerfPara;

end