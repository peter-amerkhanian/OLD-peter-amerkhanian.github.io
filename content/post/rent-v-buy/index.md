---
title: "Buy vs. Rent, A Financial Modeling Workflow in Python"
date: 2022-08-06T14:14:01-07:00
draft: false
categories: ['Python']
tags: ['Python', 'Statistics', 'Finance', 'Simulation']
math: true
---
### Summary
This post goes through the following exercises:
- Use `numpy-financial` to build a [loan amortization calculator](https://en.wikipedia.org/wiki/Amortization_calculator) for a home mortgage
- Use said table as well as simulated home and stock equity returns over time to compare year-to-year wealth resulting from the following strategies:  
  - 1.) buying a residential living space  
  - 2.) renting one instead and investing the dollar amount that would have been your down-payment

### A Note on `numpy-financial`
At one point in time, `numpy`, the popular Python numerical analysis library, included 10 specialized functions for financial analysis. Given their specific nature, they were eventually removed from `numpy`, I think in 2019 ([learn about why that is here](https://numpy.org/neps/nep-0032-remove-financial-functions.html)) and are now available in the separate library, `numpy-financial`. The library still seems focused on the same [10 core functions](https://numpy.org/numpy-financial/latest/), which handle tasks like cacluating loan payment amounts given some inputs, and applied financial economics tasks like calculating time value of money. Cool... Anyways, I'll use it to create an amortization schedule for a mortgage.



### Environment/Packages
I built this notebook in a Google Colab instance, which seems to include most major Python libraries ([more info](https://stackoverflow.com/questions/47109539/what-are-the-available-libraries-within-google-colaboratory)).  

You'll probably have to download `numpy-financial` (it's not included in Anaconda as far as I know), which you can accomplish within any notebook-like environment using the following command:


```python
!pip install numpy-financial
```

    Requirement already satisfied: numpy-financial in c:\users\peteramerkhanian\anaconda3\lib\site-packages (1.0.0)
    Requirement already satisfied: numpy>=1.15 in c:\users\peteramerkhanian\anaconda3\lib\site-packages (from numpy-financial) (1.20.1)
    

You'll want to load the usual suspects - `pandas`, `numpy`, `seaborn`, `matplotlib`. I also run `from datetime import datetime` since we will be working with ranges of dates, and I run `sns.set_style()` to get my seaborn plots looking a bit more aesthetically pleasing - read more on themes [here](https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme).  


```python
import pandas as pd
import numpy as np
import numpy_financial as npf
from datetime import datetime

import seaborn as sns
# set seaborn style
sns.set_style("white")

import matplotlib.pyplot as plt
# Set Matplotlib font size
plt.rcParams.update({'font.size': 14})
```

### Definining Constants
I'll run this as a comparison between buying an apartment that costs $<b></b>700,000 with a 20% downpayment, versus renting a home for \$2,600 a month. This is meant to approximate buying versus renting a two-bed one-bath apartment.  

Buying fees are defined at 4%, the homeowners association fees are defined as \$700 monthly.


```python
# Buying Constants
interest_rate = 0.065
cost = 700000
hoa = 700
down_payment = cost * .2
principal = cost - down_payment
buying_fees = principal*.04

# Renting Constants
rent = 2600
```

`npf.pmt()` can be used to generate a monthly mortgage payment given those buying constants:


```python
npf.pmt(interest_rate/12, 12*30, principal)
```




    -3539.580931560606



alternatively, we can use `npf.ppt()` to see how much of the payment goes towards the principal, and use `npf.ipmt()` to see how much goes towards interest (see below for applications of those functions).

### Defining Randon Variables
I'll make the simplifying assumption that both "annual home appreciation" and "annual stock appreciation" are generated from normal distributions. This is a kind of strong assumption, but one that seems to be routinely made at least with regards to stock market returns, even if there might be better distribution choices out there ([more here](https://arxiv.org/ftp/arxiv/papers/1906/1906.10325.pdf)). I could alternatively get historical data for these variables (yahoo finance has historical stock data, Zillow's ZHVI index has home appreciation for the past 20 years) and use [the bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) to approximate the true sampling distribution for each, but we'll just use normal distributions and keep this as more of a back-of-the-envelope calculator for now.  

Here's a look at how we'll draw from a normal distribution. Given an average annual return, $\mu = 0.0572$ ($\mu$, or, mu, is a common variable name for average) and a standard deviation $\sigma = 0.1042$ ($\sigma$, or, sigma, is the common variable name for standard deviation), we can draw one sample from a normal distribution as follows:


```python
# Set a random seed for stability of results
np.random.seed(30)

mean = .0572
standard_deviation = .1042
samples = 1

# Draw the sample
np.random.normal(mean, standard_deviation, samples)
```




    array([-0.07451429])



We now simulate market returns for every month by supplying mean and standard deviation values for both home and stock market appreciation and drawing 360 samples (360 months in 30 years). For simplicity, we'll just use world-wide aggregate values from ["The Rate of Return on Everything, 1870-2015"](https://www.frbsf.org/economic-research/wp-content/uploads/sites/4/wp2017-25.pdf).


```python
mu_stock = .1081
sigma_stock = .2267

mu_home = .0572
sigma_home = .1042
```

Given that stock and home appreciation is probably correlated, I'll have ti sample from a bivariate normal distribution using `numpy.random.Generator.multivariate_normal` - documentation [here](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html), rather than the univariate distribution draw shows above. I am going to assume a correlation coefficient, $\rho_{stock,home}$ of 0.5 - a fairly clear correlation.  
In order to use that numpy function, I'll need to translate my correlation statistic into a covariance statistic, and I'll use the following formula ([source](https://en.wikipedia.org/wiki/Correlation)):  
$$ \begin{align*}
cov_{stock,home} &= \rho_{stock,home} \times \sigma_{stock} \sigma_{home} \\\
cov_{stock,home} &= 0.5 \times .2267 \times .1042 \end{align*} $$

I calculate covariance and confirm that the covariance and correlations match up below:


```python
cov = 0.5 * sigma_stock * sigma_home
print("Covariance:", cov)
print("Back to correlation:", cov / (sigma_stock * sigma_home))
```

    Covariance: 0.01181107
    Back to correlation: 0.5
    

Now that I have the covariance, I'll be able to sample from a bivariate normal distribution of the form shown below ([source](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case_2)).    
$$ \begin{pmatrix} Stock \\\\ Home\end{pmatrix} \sim \mathcal{N} \left[ \begin{pmatrix} \mu_{s} \\\ \mu_{h}\end{pmatrix}, \begin{pmatrix} \sigma_{s}^2 & cov_{s,h} \\\ cov_{s,h} & \sigma_{h}^2\end{pmatrix} \right]$$
*Note, $s$ is shorthand for stock and $h$ is shorthand for home.*  
  
Now I'll code that in Python and confirm that the means and standard deviations of our samples match what we expect:


```python
cov_matrix = [[sigma_stock**2, cov],
              [cov, sigma_home**2]]

returns_df = pd.DataFrame(np.random
                          .default_rng(30)
                          .multivariate_normal([mu_stock, mu_home],
                                               cov_matrix,
                                               360) ,
                          columns=["Stock_Appreciation", "Home_Appreciation"])
print("Means:", returns_df.mean(axis=0).values)
print("Std. Devs:", returns_df.std(axis=0).values)

returns_df = (returns_df / 12)
```

    Means: [0.10764063 0.05970695]
    Std. Devs: [0.22544095 0.10543034]
    

Plotting the simulated values, we can see that stock market returns are typically higher than home value appreciation.


```python
returns_df.cumsum().plot(figsize=(9,5))
plt.xlabel("Months")
plt.ylabel("Money Multiplier")
plt.title("Simulated Home/Stock Returns")
sns.despine();
```


    
![png_one](images/output_18_0.png)
    



```python
home_performance = returns_df.cumsum()['Home_Appreciation'] + 1
stock_performance = returns_df.cumsum()['Stock_Appreciation'] + 1
```

Now we can define two spread-sheet-like dataframes:
- one that shows a mortgage amortization schedule for if you were to buy the \$600,000 home, along with the home's appreciation over time.
- one that shows a table of rent payments and the stock market growth of what would have been your down payment (you can invest the down payment since you didn't end up purchasing a house).
### Ownership Table


```python
# Buying Table
df_own = pd.DataFrame()
df_own["Period"] =  pd.Series(range(12*30)) + 1
df_own["Date"] = pd.date_range(start=datetime.today(),
                           periods=12*30,
                           freq='MS',
                           name="Date")
df_own["Principal_Paid"] = npf.ppmt(interest_rate/12,
                                    df_own["Period"],
                                    12*30,
                                    principal)
df_own["Interest_Paid"] = npf.ipmt(interest_rate/12,
                                   df_own["Period"],
                                   12*30,
                                   principal)
df_own["HOA_Paid"] = hoa
df_own["HOA_Paid"] = df_own["HOA_Paid"].cumsum()
df_own["Balance_Remaining"] = principal + df_own["Principal_Paid"].cumsum()
df_own["Home_Value"] = round(cost * home_performance, 2)
df_own["PropTax_Paid"] = (df_own["Period"]
                          .apply(lambda x:
                                 (cost * 1.02**((x-1)/12) * 0.01)
                                 if (x-1) in list(range(0, 12*1000, 12))
                                 else 0)
                          .cumsum())
df_own["Sale_Fees"] = df_own["Home_Value"] * .07
df_own["Own_Profit"] = (df_own["Home_Value"] -
                              df_own["HOA_Paid"] -
                              df_own["Balance_Remaining"] -
                              (buying_fees + df_own["Sale_Fees"]) -
                              df_own["PropTax_Paid"])
df_own = round(df_own, 2)
```

Note this code, which is a bit of a monster:
```python
df_own["PropTax_Paid"] = (df_own["Period"]
                          .apply(lambda x:
                                 (cost * 1.02**((x-1)/12) * 0.01)
                                 if (x-1) in list(range(0, 12*1000, 12))
                                 else 0)
                          .cumsum())
```
What is happening here is a calculation of property assessed value and property tax according to California's property assessment/tax regime ([more here)](https://www.boe.ca.gov/proptaxes/pdf/pub29.pdf). We'll look at this in two pieces, first, assessed value. In California, once you purchase a property, its assessed value is set at the purchase price, then increases annually by the lower of 2% or the rate of inflation according to the California Consumer Price Index (CCPI). You could write out an equation for this as follows:
$$  \begin{align*}
AnnualFactor_y =
\begin{cases}
        1 + CCPI_y, & \text{if } CCPI_y < 0.02 \\\
        1.02, & \text{otherwise}
\end{cases}
\end{align*} $$
$AnnualFactor$ is the amount that the assessed value of a home will appreciate (expressed as a multiplier) in a given year, $y$. We define $y^\*$ as the year of initial purchase and 
$PurchasePrice$ as the amount that the home was purchased for. Given that, $AssessedValue$ is defined as follows:

$$ \begin{align*}
AssessedValue_y =
    \begin{cases}
        PurchasePrice, & \text{if } y = y^\* \\\
        AssessedValue_{y-1} \times AnnualFactor_y, & \text{otherwise }
    \end{cases}
\end{align*} $$
In our code, we will simplify this calculation by excluding the CCPI and just always using 1.02 as our annual factor. Therefore, we get:

$$ \begin{align*}
  AssessedValue_y = PurchasePrice \times 1.02^y
\end{align*} $$

and once we factor in taxes (1%), we get:  
$$
\begin{equation*}
  PropertyTax_y = 0.01(PurchasePrice \times 1.02^y)
\end{equation*}
$$
and finally we look at the the cumulative total property tax you've paid in a given year $y$, which is `df_own["PropTax_Paid"] `:
$$
\begin{equation*}
  PropertyTaxPaid_y = \sum_{y=1}^{30} 0.01(PurchasePrice \times 1.02^y)
\end{equation*}
$$
There's some elements added to the code to work between years and months, but that equation is the gist of it.  
We end up with the following table for property ownership:


```python
df_own
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Date</th>
      <th>Principal_Paid</th>
      <th>Interest_Paid</th>
      <th>HOA_Paid</th>
      <th>Balance_Remaining</th>
      <th>Home_Value</th>
      <th>PropTax_Paid</th>
      <th>Sale_Fees</th>
      <th>Own_Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2022-09-01 14:11:08.272596</td>
      <td>-506.25</td>
      <td>-3033.33</td>
      <td>700</td>
      <td>559493.75</td>
      <td>701405.73</td>
      <td>7000.00</td>
      <td>49098.40</td>
      <td>62713.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-10-01 14:11:08.272596</td>
      <td>-508.99</td>
      <td>-3030.59</td>
      <td>1400</td>
      <td>558984.76</td>
      <td>707143.89</td>
      <td>7000.00</td>
      <td>49500.07</td>
      <td>67859.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2022-11-01 14:11:08.272596</td>
      <td>-511.75</td>
      <td>-3027.83</td>
      <td>2100</td>
      <td>558473.02</td>
      <td>723148.91</td>
      <td>7000.00</td>
      <td>50620.42</td>
      <td>82555.47</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2022-12-01 14:11:08.272596</td>
      <td>-514.52</td>
      <td>-3025.06</td>
      <td>2800</td>
      <td>557958.50</td>
      <td>736621.91</td>
      <td>7000.00</td>
      <td>51563.53</td>
      <td>94899.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2023-01-01 14:11:08.272596</td>
      <td>-517.31</td>
      <td>-3022.28</td>
      <td>3500</td>
      <td>557441.19</td>
      <td>744559.68</td>
      <td>7000.00</td>
      <td>52119.18</td>
      <td>102099.31</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>355</th>
      <td>356</td>
      <td>2052-04-01 14:11:08.272596</td>
      <td>-3445.26</td>
      <td>-94.33</td>
      <td>249200</td>
      <td>13968.65</td>
      <td>1937424.41</td>
      <td>283976.55</td>
      <td>135619.71</td>
      <td>1232259.49</td>
    </tr>
    <tr>
      <th>356</th>
      <td>357</td>
      <td>2052-05-01 14:11:08.272596</td>
      <td>-3463.92</td>
      <td>-75.66</td>
      <td>249900</td>
      <td>10504.74</td>
      <td>1940547.12</td>
      <td>283976.55</td>
      <td>135838.30</td>
      <td>1237927.53</td>
    </tr>
    <tr>
      <th>357</th>
      <td>358</td>
      <td>2052-06-01 14:11:08.272596</td>
      <td>-3482.68</td>
      <td>-56.90</td>
      <td>250600</td>
      <td>7022.06</td>
      <td>1944357.20</td>
      <td>283976.55</td>
      <td>136105.00</td>
      <td>1244253.59</td>
    </tr>
    <tr>
      <th>358</th>
      <td>359</td>
      <td>2052-07-01 14:11:08.272596</td>
      <td>-3501.54</td>
      <td>-38.04</td>
      <td>251300</td>
      <td>3520.51</td>
      <td>1949518.54</td>
      <td>283976.55</td>
      <td>136466.30</td>
      <td>1251855.18</td>
    </tr>
    <tr>
      <th>359</th>
      <td>360</td>
      <td>2052-08-01 14:11:08.272596</td>
      <td>-3520.51</td>
      <td>-19.07</td>
      <td>252000</td>
      <td>-0.00</td>
      <td>1953845.89</td>
      <td>283976.55</td>
      <td>136769.21</td>
      <td>1258700.12</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 10 columns</p>
</div>



### Rental Table
This one is a but more simple, only examining the total rent you've paid in a given month and simulated stock returns at that point.


```python
# Rental Table
df_rent = pd.DataFrame()
df_rent["Period"] =  pd.Series(range(12*30)) + 1
df_rent["Date"] = pd.date_range(start=datetime.today(),
                           periods=12*30,
                           freq='MS',
                           name="Date")
df_rent["DownPayment_Invested"] =  stock_performance * down_payment
df_rent["Rent_Paid"] = rent
df_rent["Rent_Paid"] = df_rent["Rent_Paid"].cumsum()
df_rent["Rent_Profit"] = df_rent["DownPayment_Invested"] - df_rent["Rent_Paid"]
df_rent = round(df_rent, 2)
```


```python
df_rent
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Period</th>
      <th>Date</th>
      <th>DownPayment_Invested</th>
      <th>Rent_Paid</th>
      <th>Rent_Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2022-09-01 14:11:08.352336</td>
      <td>136919.68</td>
      <td>2600</td>
      <td>134319.68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2022-10-01 14:11:08.352336</td>
      <td>140789.47</td>
      <td>5200</td>
      <td>135589.47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2022-11-01 14:11:08.352336</td>
      <td>142175.79</td>
      <td>7800</td>
      <td>134375.79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2022-12-01 14:11:08.352336</td>
      <td>145552.44</td>
      <td>10400</td>
      <td>135152.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2023-01-01 14:11:08.352336</td>
      <td>146217.26</td>
      <td>13000</td>
      <td>133217.26</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>355</th>
      <td>356</td>
      <td>2052-04-01 14:11:08.352336</td>
      <td>594635.22</td>
      <td>925600</td>
      <td>-330964.78</td>
    </tr>
    <tr>
      <th>356</th>
      <td>357</td>
      <td>2052-05-01 14:11:08.352336</td>
      <td>591339.68</td>
      <td>928200</td>
      <td>-336860.32</td>
    </tr>
    <tr>
      <th>357</th>
      <td>358</td>
      <td>2052-06-01 14:11:08.352336</td>
      <td>589319.15</td>
      <td>930800</td>
      <td>-341480.85</td>
    </tr>
    <tr>
      <th>358</th>
      <td>359</td>
      <td>2052-07-01 14:11:08.352336</td>
      <td>588221.21</td>
      <td>933400</td>
      <td>-345178.79</td>
    </tr>
    <tr>
      <th>359</th>
      <td>360</td>
      <td>2052-08-01 14:11:08.352336</td>
      <td>592090.64</td>
      <td>936000</td>
      <td>-343909.36</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 5 columns</p>
</div>



### Results
At this point, I'll merge the ownership and rental tables and plot out what happened in this simulation


```python
merged = pd.merge(df_own, df_rent, on="Period")
merged = merged.melt(value_vars = ["Rent_Profit", "Own_Profit"], id_vars='Period')
```


```python
plt.figure(figsize=(14, 6))
plt.title("Wealth Outcomes for Owning vs. Renting a 2b1br Apt")
sns.lineplot(data=merged, x="Period", y="value", hue="variable")
for x in range(0, 350, 12):
    if x == 0:
        plt.axvline(x, color="grey", linestyle=":", alpha=1, label="Year")
    else:
        plt.axvline(x, color="grey", linestyle=":", alpha=0.7)
    plt.text(x+1, -100000, str(int(x/12)), alpha=0.8)
plt.axhline(0, color="red", linestyle="--", alpha=0.5, label="Zero")
plt.legend()
sns.despine()
```


    
![png_two](images/output_29_0.png)
    


We can quickly seee that ownership will clearly build more wealth in the medium and long run:


```python
years = 5
print(f"Owner after {years} years:", df_own.loc[12*years-1, "Own_Profit"])
print(f"Renter after {years} years:", df_rent.loc[12*years-1, "Rent_Profit"])
```

    Owner after 5 years: 215177.81
    Renter after 5 years: 40500.11
    

However, we can see that, in the unlikely case that the home is sold within the first year or so, it's the renter that has more wealth, likely due to the owner contending with buying/selling fees:


```python
years = 1
print(f"Owner after {years} years:", df_own.loc[12*years-1, "Own_Profit"])
print(f"Renter after {years} years:", df_rent.loc[12*years-1, "Rent_Profit"])
```

    Owner after 1 years: 114027.92
    Renter after 1 years: 115865.99
    

A possible takeaway here is that, as long as you can be confident ou'll be able to hold onto the house for more than a year, it's probably better to purchase it. Uncertainty estimates would be useful here, and could be obtained by running the simulation under a wide variety of randomly generated market conditions.

[Back to top]()