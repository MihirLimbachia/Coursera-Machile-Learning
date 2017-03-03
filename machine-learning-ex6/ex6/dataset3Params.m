function [Cb, sigmab] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
maxPred=size(yval);
Cb=0.1;
sigmab=0.1;
for i=1:1:length(C)
    for j=1:1:length(sigma)
       Cval=C(i);
       sigmaval=sigma(j);
       model=svmTrain(X,y,Cval,@(x1, x2) gaussianKernel(x1, x2, sigmaval));
       Pr=svmPredict(model,Xval);
       p= mean(double(Pr~=yval));
       if(p < maxPred)
            maxPred=p;
            Cb=Cval;
            sigmab=sigmaval;
       end
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
