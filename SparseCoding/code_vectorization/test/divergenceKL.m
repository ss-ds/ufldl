function func = divergenceKL(sprsParam, sprsValue)
    
   %Calculates divergence using KL algorithm
   %sprsParam - target value
   %sprsValue - real, calculated value
    
    func = sprsParam * log(sprsParam./sprsValue) + (1 - sprsParam ) * log ((1 - sprsParam)./(1 - sprsValue) );
    
    
    
    
end