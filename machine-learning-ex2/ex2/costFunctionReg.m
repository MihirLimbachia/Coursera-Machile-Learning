function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
z=sigmoid(X*theta);
v1=-1.*(y.*log(z));
temp1=ones(size(y));
v2=-1.*((temp1-y).*log(1-z));
wg=ones(size(theta));
wg(1)=0;
J = (sum(v1 + v2))/m + (lambda*sum(wg.*theta.*theta))/(2*m);
grad = (X'*(sigmoid(X*theta)-y))/m + (lambda*(wg.*theta))/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
