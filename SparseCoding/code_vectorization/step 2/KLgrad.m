function func = KLgrad(sprsParam, sprsValue)
    
   %Calculates divergence using KL algorithm
   %sprsParam - target value
   %sprsValue - real, calculated value
    
    func = ( -sprsParam./sprsValue + (1 - sprsParam ) ./ (1 - sprsValue) );
    
    
    
    
end