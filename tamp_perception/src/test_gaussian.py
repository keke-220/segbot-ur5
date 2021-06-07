# import the required libraries 
import random 
import matplotlib.pyplot as plt 
    
# store the random numbers in a list 
nums = [] 
mu = 0
sigma = 0.1
    
for i in range(10000): 
    temp = random.gauss(mu, sigma) 
    nums.append(temp) 
        
# plotting a graph 
plt.hist(nums, bins = 200) 
plt.show()
