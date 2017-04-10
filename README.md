

```python
import numpy as np
import math as m
import timeit
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
```

# Options Pricing Model Paper

## Black Scholes 


```python
m=5 #time steps
K=100 #strike price
r=0.05 #risk free rate
T=20/36 #strike time
sigma=0.30 #implied volatiity

time = scipy.linspace (0.0, T , m ) #time series
S = 100 # stock price

logSoverK = scipy . log ( S / K )
n12 = (( r + sigma **2/2) *( T - time ) )
n22 = (( r - sigma **2/2) *( T - time ) )
numerd1 = logSoverK + n12 
numerd2 = logSoverK  + n22
d1 = numerd1 /( sigma * scipy . sqrt (T - time )) 
d2 = numerd2 /( sigma * scipy . sqrt (T - time ))

part1 = S * norm . cdf ( d1 )
part2 = norm.cdf(d2) * K * scipy.exp( - r *( T - time ) ) 
VC=part1-part2

#scipy.column_stack((scipy.transpose(S),scipy.transpose(VC)))


```

    C:\Users\TanayT\Anaconda3\lib\site-packages\ipykernel\__main__.py:15: RuntimeWarning: invalid value encountered in true_divide
    C:\Users\TanayT\Anaconda3\lib\site-packages\ipykernel\__main__.py:16: RuntimeWarning: invalid value encountered in true_divide
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:1731: RuntimeWarning: invalid value encountered in greater_equal
      cond2 = (x >= self.b) & cond0
    


```python
VC
```




    array([ 10.21728253,   8.70879925,   6.97153122,   4.79714021,          nan])



## Binomial Options Pricing 

My notation here is based on the original Cox, Ross and Rubinstein paper.


```python
import numpy as np
import math as m
import timeit

def bop(n,t,S,v):
    dt = t/n
    u = m.exp(v*m.sqrt(dt))
    d = 1/u
    Pm = np.zeros((n+1, n+1))
    for j in range(n+1):
        for i in range(j+1):
            Pm[i,j] = S*m.pow(d,i) * m.pow(u,j-i)
    return Pm
```

I generated the pricing tree for a few n values...


```python
n = 5
t = 200/365
S = 100
v = .3
x = bop(n,t,S,v)
n = 17
z = bop(n,t,S,v)

print('n = 5:\n',np.matrix(x.astype(int)))
print('n = 17:\n',np.matrix(z.astype(int)))
```

    n = 5:
     [[100 110 121 134 148 164]
     [  0  90 100 110 121 134]
     [  0   0  81  90 100 110]
     [  0   0   0  74  81  90]
     [  0   0   0   0  67  74]
     [  0   0   0   0   0  60]]
    n = 17:
     [[100 105 111 117 124 130 138 145 153 162 171 180 190 201 212 224 236 249]
     [  0  94 100 105 111 117 124 130 138 145 153 162 171 180 190 201 212 224]
     [  0   0  89  94 100 105 111 117 124 130 138 145 153 162 171 180 190 201]
     [  0   0   0  85  89  94  99 105 111 117 124 130 138 145 153 162 171 180]
     [  0   0   0   0  80  85  89  94 100 105 111 117 124 130 138 145 153 162]
     [  0   0   0   0   0  76  80  85  89  94 100 105 111 117 124 130 138 145]
     [  0   0   0   0   0   0  72  76  80  85  89  94  99 105 111 117 124 130]
     [  0   0   0   0   0   0   0  68  72  76  80  85  89  94  99 105 111 117]
     [  0   0   0   0   0   0   0   0  64  68  72  76  80  85  89  94 100 105]
     [  0   0   0   0   0   0   0   0   0  61  64  68  72  76  80  85  89  94]
     [  0   0   0   0   0   0   0   0   0   0  58  61  64  68  72  76  80  85]
     [  0   0   0   0   0   0   0   0   0   0   0  55  58  61  64  68  72  76]
     [  0   0   0   0   0   0   0   0   0   0   0   0  52  55  58  61  64  68]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  49  52  55  58  61]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  47  49  52  55]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  44  47  49]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  42  44]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40]]
    

After noticing the recursive pattern in the tree, I generated the set of all unique numbers in the matrix as an ordered 1d array and looped through the elements of the pricing matrix calling values from the unique set.


```python
def better_bop(n,t,S,v):
    dt = t/n
    u = m.exp(v*m.sqrt(dt))
    d = 1/u
    ups = np.zeros(n+1)
    dwns = np.zeros(n+1)
    tot = np.zeros(2*n+1)
    Pm = np.zeros((n+1, n+1))
    tmp = np.zeros((2,n+1))
    for j in range(n+1):
        tmp[0,j] = S*m.pow(d,j)
        tmp[1,j] = S*m.pow(u,j)
    tot = np.unique(tmp)
    c = n
    for i in range(c+1):
        for j in range(c+1):
            Pm[i,j-c-1] = tot[(n-i)+j]
        c=c-1
    return Pm
trial = better_bop(n,t,S,v)
print('n = 17:\n',np.matrix(trial.astype(int)))
```

    n = 17:
     [[100 105 111 117 124 130 138 145 153 162 171 180 190 201 212 224 236 249]
     [  0  94 100 105 111 117 124 130 138 145 153 162 171 180 190 201 212 224]
     [  0   0  89  94 100 105 111 117 124 130 138 145 153 162 171 180 190 201]
     [  0   0   0  85  89  94 100 105 111 117 124 130 138 145 153 162 171 180]
     [  0   0   0   0  80  85  89  94 100 105 111 117 124 130 138 145 153 162]
     [  0   0   0   0   0  76  80  85  89  94 100 105 111 117 124 130 138 145]
     [  0   0   0   0   0   0  72  76  80  85  89  94 100 105 111 117 124 130]
     [  0   0   0   0   0   0   0  68  72  76  80  85  89  94 100 105 111 117]
     [  0   0   0   0   0   0   0   0  64  68  72  76  80  85  89  94 100 105]
     [  0   0   0   0   0   0   0   0   0  61  64  68  72  76  80  85  89  94]
     [  0   0   0   0   0   0   0   0   0   0  58  61  64  68  72  76  80  85]
     [  0   0   0   0   0   0   0   0   0   0   0  55  58  61  64  68  72  76]
     [  0   0   0   0   0   0   0   0   0   0   0   0  52  55  58  61  64  68]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  49  52  55  58  61]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  47  49  52  55]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  44  47  49]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  42  44]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  40]]
    

Testing for consistency and timing...


```python
%%timeit
method1 = bop(n,t,S,v)
```

    10000 loops, best of 3: 147 µs per loop
    


```python
%%timeit
method2 = better_bop(n,t,S,v)
```

    10000 loops, best of 3: 84.7 µs per loop
    


```python
method1 = bop(n,t,S,v)
method2 = better_bop(n,t,S,v)
print('\nConsistent entries?: ' , np.allclose(method1,method2)) #tests if the matrices are equal
```

    
    Consistent entries?:  True
    

Method 2 performs much quicker giving the same results.

## Working Backwards to find the value of the option

From here, I determined the value of the option based on stike price and value at earlier nodes (as shown in the paper) was very simple to implement in this matrix.


```python
def OptionsVal(n, S, K, r, v, T, PC):
    dt = T/n                    
    u = m.exp(v*m.sqrt(dt)) 
    d = 1/u                     
    p = (m.exp(r*dt)-d)/(u-d)   
    Pm = np.zeros((n+1, n+1))   
    Cm = np.zeros((n+1, n+1))
    tmp = np.zeros((2,n+1))
    for j in range(n+1):
        tmp[0,j] = S*m.pow(d,j)
        tmp[1,j] = S*m.pow(u,j)
    tot = np.unique(tmp)
    c = n
    for i in range(c+1):
        for j in range(c+1):
            Pm[i,j-c-1] = tot[(n-i)+j]
        c=c-1
    for j in range(n+1, 0, -1):
        for i in range(j):
            if (PC == 1):                               
                if(j == n+1):
                    Cm[i,j-1] = max(K-Pm[i,j-1], 0)     
                else:
                    Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j]) 
            if (PC == 0):                               
                if (j == n + 1):
                    Cm[i,j-1] = max(Pm[i,j-1]-K, 0)     
                else:
                    Cm[i,j-1] = m.exp(-.05*dt) * (p*Cm[i,j] + (1-p)*Cm[i+1,j])  
    return [Pm,Cm]
```


```python
S = 100
k = 100
r = .05
v = .3
T = 20/36
n = 17
PC = 0
Pm,CmC = OptionsVal(n,S,k,r,v,T,PC)
PC = 1
_,CmP= OptionsVal(n, S, k, r, v, T, PC)
print('Pricing:\n',np.matrix(Pm.astype(int)))
print('Call Option:\n',np.matrix(CmC.astype(int)))
print('Put Option:\n',np.matrix(CmP.astype(int)))
```

    Pricing:
     [[100 105 111 117 124 131 138 146 154 162 172 181 191 202 213 225 238 251]
     [  0  94 100 105 111 117 124 131 138 146 154 162 172 181 191 202 213 225]
     [  0   0  89  94 100 105 111 117 124 131 138 146 154 162 172 181 191 202]
     [  0   0   0  84  89  94 100 105 111 117 124 131 138 146 154 162 172 181]
     [  0   0   0   0  80  84  89  94 100 105 111 117 124 131 138 146 154 162]
     [  0   0   0   0   0  76  80  84  89  94 100 105 111 117 124 131 138 146]
     [  0   0   0   0   0   0  72  76  80  84  89  94 100 105 111 117 124 131]
     [  0   0   0   0   0   0   0  68  72  76  80  84  89  94 100 105 111 117]
     [  0   0   0   0   0   0   0   0  64  68  72  76  80  84  89  94 100 105]
     [  0   0   0   0   0   0   0   0   0  61  64  68  72  76  80  84  89  94]
     [  0   0   0   0   0   0   0   0   0   0  58  61  64  68  72  76  80  84]
     [  0   0   0   0   0   0   0   0   0   0   0  55  58  61  64  68  72  76]
     [  0   0   0   0   0   0   0   0   0   0   0   0  52  55  58  61  64  68]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  49  52  55  58  61]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46  49  52  55]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  44  46  49]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  41  44]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  39]]
    Call Option:
     [[ 10  13  17  22  27  33  40  47  55  64  73  82  92 103 114 125 138 151]
     [  0   7   9  12  16  21  26  33  40  47  55  63  72  82  92 102 113 125]
     [  0   0   4   6   8  12  16  20  26  32  39  47  55  63  72  81  91 102]
     [  0   0   0   2   4   5   8  11  15  20  25  32  39  46  54  63  72  81]
     [  0   0   0   0   1   2   3   5   7  10  14  19  25  31  38  46  54  62]
     [  0   0   0   0   0   0   1   1   2   4   6   9  13  18  24  31  38  46]
     [  0   0   0   0   0   0   0   0   0   1   2   3   5   8  12  17  24  31]
     [  0   0   0   0   0   0   0   0   0   0   0   0   1   2   4   7  11  17]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   2   5]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
    Put Option:
     [[ 7  5  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  9  7  5  3  1  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0 12  9  6  4  2  1  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0 15 12  9  6  4  2  1  0  0  0  0  0  0  0  0]
     [ 0  0  0  0 18 15 11  8  5  3  1  0  0  0  0  0  0  0]
     [ 0  0  0  0  0 22 18 15 11  8  5  2  1  0  0  0  0  0]
     [ 0  0  0  0  0  0 26 22 18 15 11  7  4  2  0  0  0  0]
     [ 0  0  0  0  0  0  0 30 26 22 18 14 10  7  3  1  0  0]
     [ 0  0  0  0  0  0  0  0 33 30 26 22 18 14 10  6  2  0]
     [ 0  0  0  0  0  0  0  0  0 37 34 30 26 23 19 14 10  5]
     [ 0  0  0  0  0  0  0  0  0  0 40 37 34 30 27 23 19 15]
     [ 0  0  0  0  0  0  0  0  0  0  0 43 41 37 34 31 27 23]
     [ 0  0  0  0  0  0  0  0  0  0  0  0 47 44 41 38 35 31]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0 49 47 44 41 38]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 52 50 47 44]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 55 53 50]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 57 55]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 60]]
    

## Error Accumulation Between Two Methods


```python
type(CmC[1,1])
```




    numpy.float64




```python
CmC.shape[1]
```




    18




```python
for i in range(CmC.shape[1]):
    print(CmC[i,i])
```

    10.3431748302
    7.13944312261
    4.6226679354
    2.7588220853
    1.48054109593
    0.688577830495
    0.261364819933
    0.0723485357168
    0.0110219039948
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    


```python
(CmC[0,0]-VC[0])/(VC[0])*100
```




    1.2321505181400323



The conclusion of this error analysis is that if we take the Black Scholes value to be the accepted value of the option at time t=0, the method developed has 1.23% error.

## Time Differences between the two methods


```python
%%timeit
m=5 #time steps
K=100 #strike price
r=0.05 #risk free rate
T=20/36 #strike time
sigma=0.30 #implied volatiity

time = scipy.linspace (0.0, T , m ) #time series
S = 100 # stock price

logSoverK = scipy . log ( S / K )
n12 = (( r + sigma **2/2) *( T - time ) )
n22 = (( r - sigma **2/2) *( T - time ) )
numerd1 = logSoverK + n12 
numerd2 = logSoverK  + n22
d1 = numerd1 /( sigma * scipy . sqrt (T - time )) 
d2 = numerd2 /( sigma * scipy . sqrt (T - time ))

part1 = S * norm . cdf ( d1 )
part2 = norm.cdf(d2) * K * scipy.exp( - r *( T - time ) ) 
VC=part1-part2
```

    C:\Users\TanayT\Anaconda3\lib\site-packages\ipykernel\__main__.py:271: RuntimeWarning: invalid value encountered in true_divide
    C:\Users\TanayT\Anaconda3\lib\site-packages\ipykernel\__main__.py:272: RuntimeWarning: invalid value encountered in true_divide
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:875: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    C:\Users\TanayT\Anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:1731: RuntimeWarning: invalid value encountered in greater_equal
      cond2 = (x >= self.b) & cond0
    

    1000 loops, best of 3: 202 µs per loop
    

Conclusion? The Black Scholes Computation takes more time than our matrix generation by 60 microseconds. Neither method has yet been optimized, both calculate more values than necessary. However, we can take this result to heart that our efforts are going somewhere.


```python

```
