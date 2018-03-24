### Test 1
   
   
This is built on the vectorAdd example from the CUDA sdk, the sample simply tests the mutate operator using a mutation rate of 0 for 
validation purposes. A vector of random integers is uploaded to select the gene (bit) to mutate, and a vector of floating point numbers
between 1 and 0 contains the probability of mutation and that's compared to the mutation_rate variable to determine whether to change the
value of the bit from 0 to 1 (or visa versa, if the value is 1 then its set to zero if the test passes).

