function func = sigmoid(m)
% sigmoid
% calculates sigmoid on given input
% in case of matrix - calculates sigmoid for each matrix element separatly

func = 1./( 1 + exp(-m));