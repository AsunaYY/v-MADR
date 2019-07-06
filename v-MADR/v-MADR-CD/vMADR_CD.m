function  [preY, MSE, time] = vmadr_cd(testX, testY, trainX, trainY, params, kerType)

%testX: testset data
%testY: testset label
%trainX: trainset data
%trainY: trainset label
%params: vector: [lambda1, lambda2, C, v, [Gamma,Degree]]
%kerType: rbf/lin/poly

%% params
lambda1 = params(1);
lambda2 = params(2);
C = params(3);
v = params(4);
kerfPara = params(5);

%% train data&label
[num, dim] = size(trainX);
if(num ~= length(trainY))
    error('ldrm1_cd_learn::trainY and trainX must be the same dimension! \n') ;
end

%% G 
G = zeros(num);
G = Kernel(trainX, trainX,kerfPara,kerType); 
% processing
G = (G + G')/2 + 0.001*eye(size(G));

%% Q and p
Q = sparse(zeros(size(G)));
p = sparse(zeros(num, 1));
Ge = sum(G, 2);

Q = sparse(G) + (4*lambda1+2*lambda2)/(num) * sparse(G')*sparse(G) -(4*lambda1)/ (num * num) * sparse(Ge) * sparse(Ge') ;

p = (4 * lambda1) / (num * num) * sparse(Ge) * sum(trainY) - (4 * lambda1 + 2 * lambda2) / (num) * sparse(G) * sparse(trainY) ;   

 %% invQp and invQG
it = tic;
invQp = Q\(-p);
invQG = Q\sparse(G);
time = toc(it);
  
%% H = G * invQ * G
H = sparse(zeros(num, num));
H(1:num, 1:num) = sparse(G) * invQG(:,1:num);

%% prob
prob.H = H;
prob.G = G;
prob.invQG = invQG; 
prob.invQp = invQp; 

%% alpha
tstart = tic;
alpha = trainVMADR_CD(trainY, sparse(trainX), prob, params);
time = time + toc(tstart);

%% preY & mse
kernel = Kernel(trainX, testX,kerfPara,kerType);
preY =  kernel' * alpha;
%mse
mse = power(preY - testY,2);
MSE = sum(mse) / length(testY);

end


