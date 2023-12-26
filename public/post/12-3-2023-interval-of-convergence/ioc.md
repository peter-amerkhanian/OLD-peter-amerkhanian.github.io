``` python
import numpy as np
import matplotlib.pyplot as plt
import sympy
```

``` python
x, n = sympy.symbols('x, n')
```

``` python
a_n = x**n / (n*5**n)
a_n
```

$\displaystyle \frac{5^{- n} x^{n}}{n}$

``` python
sympy.limit(a_n.subs(n, n+1) / a_n, n, sympy.oo)
```

$\displaystyle \frac{x}{5}$

``` python
series = sympy.Sum(
    a_n,
    (n, 1, sympy.oo)
    )
series
```

$\displaystyle \sum_{n=1}^{\infty} \frac{5^{- n} x^{n}}{n}$

Let \$\_{n=1}^a_n be a series with nonzero terms. Let  
$$
\rho = \lim_{n\rightarrow \infty} \left|  \frac{a_{n+1}}{a_n} \right |  
$$
- if
