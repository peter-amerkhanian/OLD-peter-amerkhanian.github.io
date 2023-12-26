<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js" integrity="sha512-c3Nl8+7g4LMSTdrm621y7kf9v3SDPnhxLNhcjFJbKECVnmZHTdo+IRO05sNLTH/D3vA6u1X32ehoLC7WFVdheg==" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
<script type="application/javascript">define('jquery', [],function() {return window.jQuery;})</script>


``` python
import pandas as pd
import numpy as np
```

I recently came across a list of [10 theorems/proofs](https://twitter.com/causalinf/status/1259448663270658050) that you "need to know" if you do econometrics. These were compiled by [Jeffrey Wooldridge](https://econ.msu.edu/faculty/wooldridge/), an economist and textbook author whose introductory textbook has been fundamental to my interest in econometrics. As an exercise, I'm working through these 10 items, compiling resources, textbook passages, and data exercises that I think can make them easier to understand. The first item I'm trying to write my notes on is the Law of Iterated Expectations, but I'll be prefacing/augmenting the notes with some discussion of basic probability for completeness.

To start, I'll simulate some data.

``` python
# Set a random seed for reproducability
np.random.seed(42)
# Define the number of people in the dataset
num_people = 100
# Generate random ages - X ~ Uniform(min, max)
ages = np.random.randint(67, 80, num_people)
# Create DataFrame
data = {'Person_ID': range(1, num_people + 1), 'Age': ages}
people_df = pd.DataFrame(data).set_index("Person_ID")

people_df.head()
```

  <div id="df-5443f1d1-012e-42c4-a006-4b285a925643" class="colab-df-container">
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
      <th>Age</th>
    </tr>
    <tr>
      <th>Person_ID</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77</td>
    </tr>
    <tr>
      <th>5</th>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5443f1d1-012e-42c4-a006-4b285a925643')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5443f1d1-012e-42c4-a006-4b285a925643 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5443f1d1-012e-42c4-a006-4b285a925643');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-add2dd9a-39b2-457a-90d3-bd0036af875c">
  <button class="colab-df-quickchart" onclick="quickchart('df-add2dd9a-39b2-457a-90d3-bd0036af875c')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-add2dd9a-39b2-457a-90d3-bd0036af875c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

Let's say that these data represent life spans, thus $\text{age}_i$ is an individual's lifespan, e.g. $\text{age}_2=$

``` python
people_df.loc[2]
```

    Age    70
    Name: 2, dtype: int64

## Expectation

The mean of a random variable, like age above, is also referred to as its "expected value," denoted $E(\text{age})$.

``` python
people_df['Age'].mean()
```

    73.27

The mean above is specifically called an *arithmetic mean*, defined mathematically as follows:  
$$ \bar{x} = \frac{1}{n} \sum_i^n x_i$$

``` python
(1/len(people_df)) * people_df['Age'].sum()
```

    73.27

But the arithmetic mean is just a special case of the more general weighted mean, defined as follows:
$$
\begin{align*}
\text{weighted-mean}(x) &= \sum_i^n x_i p_i \\
\end{align*}
$$
Where the weights, $p_1, p_2, ...,p_n$ are non-negative numbers that sum to 1. We can see that the arithmetic mean is the specific case of the weighted mean where all weights are equal
$$
\begin{align*}
\text{If } [p_1=p_2=...=p_n] &\text{ And } [\sum_i^n p_i =1]\\
\text{weighted-mean}(x) &= \sum_i^n x_i \frac{1}{n} \\
\text{weighted-mean}(x) &= \frac{1}{n} \sum_i^n x_i = \bar{x}\\
\end{align*}
$$
We use the more general weighted mean when we define expectation.

> the expected value of $X$ is a weighted average of the possible values that
> $X$ can take on, weighted by their probabilities
> -- <cite>(Blitzstein & Wang, 2019)</cite>

More formally, given a random variable, $X$, with distinct possible values, $x_1, x_2, ... x_n$, the *expected value* $X$ is defined as:
$$
\begin{align*}
E(X) &= x_1P(X = x_1) + x_2P(X = x_2) + ... + x_nP(X = x_n) \\
&= \sum_{i=1}^n x_iP(X = x_i)
\end{align*}
$$

Now we'll demonstrate this formula on our data. It's useful here to move from our individual-level dataset, where each row is a person, to the following, where each row is a lifespan, which the probability that an individual has that lifespan.

``` python
prob_table = people_df['Age'].value_counts(normalize=True)
prob_table = prob_table.sort_index()
prob_table
```

    67    0.03
    68    0.10
    69    0.06
    70    0.09
    71    0.09
    72    0.06
    73    0.08
    74    0.10
    75    0.06
    76    0.09
    77    0.05
    78    0.13
    79    0.06
    Name: Age, dtype: float64

Adapting the formula above to our data, we must solve the following:
$$
\begin{align*}
E(\text{Age}) &= \text{Age}_1P(\text{Age}=\text{Age}_1) + \text{Age}_2P(\text{Age}=\text{Age}_2) + ... + \text{Age}_3P(\text{Age}=\text{Age}_3) \\
&= \sum_{i=1}^n \text{Age}_iP(\text{Age}=\text{Age}_i)
\end{align*}
$$
Which we can do transparently using a for-loop:

``` python
summation = 0
for i in range(len(prob_table)):
  summation += prob_table.index[i] * prob_table.values[i]
summation
```

    73.27

As a quick aside -- this can also be expressed as the dot product of two vectors, where the dot product is defined as follows:
\$ P()=\_1P(=\_1) + \_2P(=\_2) + ... + \_3P(=\_3)\$

``` python
prob_table.index.values @ prob_table.values
```

    73.27

Though we will stick to the summation notation paired with python for-loops for consistency

## Conditional Expectation

We often have more than one variable available to us in an analysis. Below I simulate the variable gender:

``` python
np.random.seed(45)
people_df['Gender'] = np.random.choice(['Female', 'Male'], len(people_df))
people_df.head()
```

  <div id="df-5c54ae23-f1a1-46f4-b913-00e0737938b5" class="colab-df-container">
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
      <th>Age</th>
      <th>Gender</th>
    </tr>
    <tr>
      <th>Person_ID</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>73</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>74</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5c54ae23-f1a1-46f4-b913-00e0737938b5')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5c54ae23-f1a1-46f4-b913-00e0737938b5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5c54ae23-f1a1-46f4-b913-00e0737938b5');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-4e34fd79-91c9-4bf4-9b1a-e5198cfea544">
  <button class="colab-df-quickchart" onclick="quickchart('df-4e34fd79-91c9-4bf4-9b1a-e5198cfea544')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-4e34fd79-91c9-4bf4-9b1a-e5198cfea544 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

Each row in our dataset represents an individual person, and we now have access to both their gender and their life-span. It follows that we may be interested in how life-span varies across gender. In code, this entails a groupby operation, grouping on gender before calculting the mean age:

``` python
people_df.groupby('Gender')['Age'].mean()
```

    Gender
    Female    73.236364
    Male      73.311111
    Name: Age, dtype: float64

The code in this case resembles the formal notation of a conditional expectation:
$E(\text{Age} \mid \text{Gender}=\text{Gender}_j)$, where each $\text{Gender}=\text{Gender}_j$ is a distinct event.

If we are interested specifically in the mean life-span given the event that gender is equal to male (a roundabout way of saying the average life-span for males in the data), we could calculate the following

$E(\text{Age} \mid \text{Gender}=\text{Male})$

``` python
people_df.groupby('Gender')['Age'].mean()['Male']
```

    73.31111111111112

These groupby operations in `pandas` obscure some of the conceptual stuff happening inside the conditional expectation, which we'll delve deeper into now.

So what exactly is the conditional expectation, $E(X \mid Y=y)$?

Before answering this, it will be useful to refresh the related concept of conditional probability:  
\> If $X=x$ and $Y=y$ are events with $P(Y=y)>0$, then the conditional probability of $X=x$ given $Y=y$ is denoted by $P(X=x \mid Y=y)$, defined as  
$$
P(X=x \mid Y=y) = \frac{P(X=x , Y=y)}{P(Y=y)}
$$

This formula specifically describes the probability of the event, $X=x$, given the *evidence*, an observed event $Y=y$.

We want to shift to describing a mean conditional on that evidence, and we include that information via the weights in the expectation.
\>Recall that the expectation $E(X)$ is a weighted average of the possible values of $X$, where the weights are the PMF values $P(X = x)$. After learning that an event $Y=y$ occurred, we want to **use weights that have been updated to reflect this new information**.  
\>-- <cite>(Blitzstein & Wang, 2019)</cite>

The key point here is that **just the weights that each $x_i$ gets multiplied by will change**, going from the probability $P(X=x)$ to the **conditional probability $P(X=x \mid Y=y)$**.

Armed with conditional probability formula above, we can define how to compute the conditional expected value
$$\begin{align*}
E(X \mid Y=y) &= \sum_{x} x P(X=x \mid Y=y) \\
&= \sum_{x} x \frac{P(X=x , Y=y)}{P(Y=y)}
\end{align*}
$$

Returning to our example with data, we substitute terms to find the following:
$$\begin{align*}
E(\text{Age} \mid \text{Gender}=\text{Male}) &= \sum_{i=1}^n \text{Age}_iP(\text{Age}=\text{Age}_i \mid \text{Gender}=\text{Male}) \\
&= \sum_{i=1}^n \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Male})}{P(\text{Gender}=\text{Male})}
\end{align*}
$$

We can explicitly compute this with a for-loop in python, as we did for $E(X)$, but this time we will need to do a little up front work and define components we need for calculating the weights, $\frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Male})}{P(\text{Gender}=\text{Male})}$
\### Components
1. The conditional probability distribution: $P(\text{Age}=\text{Age}_i, \text{Gender} = \text{Male})$
2. The probability of the event, $P(\text{Gender}=\text{Male})$

Where 1.) is the following:

``` python
P_Age_Gender = pd.crosstab(people_df['Age'],
                           people_df['Gender'],
                           normalize='all')
P_Age_Gender['Male']
```

    Age
    67    0.02
    68    0.02
    69    0.03
    70    0.05
    71    0.04
    72    0.02
    73    0.06
    74    0.05
    75    0.03
    76    0.03
    77    0.01
    78    0.05
    79    0.04
    Name: Male, dtype: float64

and 2.) is:

``` python
P_Gender = people_df['Gender'].value_counts(normalize=True)
P_Gender.loc['Male']
```

    0.45

With those two pieces, we'll convert the following into a for-loop:
$$
\sum_{i=1}^n \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Male})}{P(\text{Gender}=\text{Male})}
$$

``` python
E_age_male = 0
n = len(P_Age_Gender['Male'])
for i in range(n):
  weight = P_Age_Gender['Male'].values[i] / P_Gender.loc['Male']
  E_age_male += P_Age_Gender['Male'].index[i] * weight
E_age_male
```

    73.31111111111112

We confirm that this is equal to the result of the more direct groupby:

``` python
people_df.groupby('Gender')['Age'].mean()['Male']
```

    73.31111111111112

## The Law of Iterated Expectations

The law of iterated expectations, also referred to as the law of total expectation, the tower property, Adam's law, or, my favorite, LIE, states the following:
$$E(X) = E(E(X \mid Y))$$
Which is to say, **the weighted average of $X$ is equal to the weighted average of the weighted averages of $X$ conditional on each value of $Y$**. This isn't a particularly useful sentence, so let's return to our example data. We plug in our values as follows:
$$E(\text{Age}) = E(E(\text{Age} \mid \text{Gender}))$$
Now it is useful to break this into some components that we've seen before. We previously found
$$
E(\text{Age} \mid \text{Gender}=\text{Male}) =  \sum_{i=1}^n \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Male})}{P(\text{Gender}=\text{Male})}
$$
Over all $Gender_j$, we have the more generalizable expression:
$$
E(\text{Age} \mid \text{Gender}=\text{Gender}_j)
$$
Which can tell us about any gender, not just $\text{Gender}=\text{Male}$. This is equivalent to the expression:
$$
E(\text{Age} \mid \text{Gender}=\text{Gender}_j) = \sum_{i=1}^n \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j) }{P(\text{Gender}=\text{Gender}_j)}
$$

Given this, let's return to the informal definition of the LIE, but break it into parts. The weighted average of $X$ is equal to:  
1. The weighted average of
2. the weighted averages of $X$ conditional on each value of $Y$".

The expression above, $E(\text{Age} \mid \text{Gender}=\text{Gender}_j)$ is equivalent to 2.) "the weighted averages of $X$ conditional on each value of $Y$."
So what we need to do now is find the weighted average of that expression. We'll set up in the next few lines

$$\begin{align*}
E(\text{Age}) &=E( \underbrace{E(\text{Age} \mid \text{Gender}=\text{Gender}_j)}_{\text{weighted averages conditional on each gender}} ) \\
&=E(\sum_{i} \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j) }{P(\text{Gender}=\text{Gender}_j)}) \\
\end{align*}
$$
With that set up, we'll now write out the last weighted average explicity. Note that the variation in $\text{Age}_i$ has been accounted for -- we are now averaging over gender, $\text{Gender}_j$.
$$
\begin{align*}
&=\sum_j (\sum_{i} \text{Age}_i \frac{P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j) }{P(\text{Gender}=\text{Gender}_j)}) P(\text{Gender}=\text{Gender}_j) \\
&=\sum_j \sum_{i} \text{Age}_i P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j) \\
\end{align*}
$$
Since $j$ only appears in one of these two terms, we can rewrite this as follows:  
$$
\begin{align*}
&=  \sum_{i} \text{Age}_i \sum_j P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j)
\end{align*}
$$
Here I'll pause, because the next steps can be clarified with code. $P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j)$ is the joint probability distribution of age and gender, and it helps to take a look at exactly what it is in pandas:

``` python
P_Age_Gender
```

  <div id="df-3d4fa049-b225-4663-ae29-4025411ed6a0" class="colab-df-container">
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
      <th>Gender</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.08</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.04</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.04</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.02</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.05</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.06</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.04</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.08</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.02</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3d4fa049-b225-4663-ae29-4025411ed6a0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3d4fa049-b225-4663-ae29-4025411ed6a0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3d4fa049-b225-4663-ae29-4025411ed6a0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-57b54436-df31-4dad-910e-06004a3fc39e">
  <button class="colab-df-quickchart" onclick="quickchart('df-57b54436-df31-4dad-910e-06004a3fc39e')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-57b54436-df31-4dad-910e-06004a3fc39e button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

Let's compute the summation of $P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j)$over $\text{Gender}_j$ and see what we get.

``` python
P_Age_Gender["Male"] + P_Age_Gender["Female"]
```

    Age
    67    0.03
    68    0.10
    69    0.06
    70    0.09
    71    0.09
    72    0.06
    73    0.08
    74    0.10
    75    0.06
    76    0.09
    77    0.05
    78    0.13
    79    0.06
    dtype: float64

Interestingly, that is the exact same thing we get if we simply compute the probability of each age, $P(\text{Age}=\text{Age}_i)$

``` python
people_df['Age'].value_counts(normalize=True).sort_index()
```

    67    0.03
    68    0.10
    69    0.06
    70    0.09
    71    0.09
    72    0.06
    73    0.08
    74    0.10
    75    0.06
    76    0.09
    77    0.05
    78    0.13
    79    0.06
    Name: Age, dtype: float64

So when you sum $P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j)$ only over $j$, you're just left with $P(\text{Age}=\text{Age}_i)$. This result stems from the definition of the *Marginal PMF*:  
\>For the discrete random variables $X$ and $Y$, the marginal PMF of $X$ is:  
\>$$P(X=x) = \sum_y P(X=x, Y=y)$$
(322).

and with this definition in mind we can finish the proof for the LIE:
$$
\begin{align*}
E(\text{Age})  &= \sum_{i} \text{Age}_i \sum_j P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j) \\
&= \sum_{i} \text{Age}_i  P(\text{Age}=\text{Age}_i) \\
&= E(\text{Age})
\end{align*}
$$

We can directly show the last bit, $E(\text{Age}) = \sum_j \sum_{i} \text{Age}_i P(\text{Age}=\text{Age}_i, \text{Gender}=\text{Gender}_j)$ using the joint probability distribution object from before:

``` python
P_Age_Gender.head()
```

  <div id="df-8191b3c0-0d5d-4bbe-96c6-caeb649a7587" class="colab-df-container">
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
      <th>Gender</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>Age</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.08</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.04</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.05</td>
      <td>0.04</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8191b3c0-0d5d-4bbe-96c6-caeb649a7587')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8191b3c0-0d5d-4bbe-96c6-caeb649a7587 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8191b3c0-0d5d-4bbe-96c6-caeb649a7587');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9947ea9c-48a0-433c-9281-8a71dc01194a">
  <button class="colab-df-quickchart" onclick="quickchart('df-9947ea9c-48a0-433c-9281-8a71dc01194a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9947ea9c-48a0-433c-9281-8a71dc01194a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

``` python
(P_Age_Gender
 .sum(axis=1) # sum over j
 .reset_index() # bring out Age_i
 .product(axis=1) # Age_i * P(Age=Age_i)
 .sum() # sum over i
 )
```

    73.27

``` python
people_df['Age'].mean()
```

    73.27

## The Law of Iterated Expectations with Nested Conditioning

https://www.math.arizona.edu/\~tgk/464_07/cond_exp.pdf

https://stats.stackexchange.com/questions/95947/a-generalization-of-the-law-of-iterated-expectations

$$
E(X \mid Y) = E(E(X \mid Z, Y) \mid Y)
$$

``` python
np.random.seed(48)
people_df['Smoker'] = np.random.choice(['Yes', 'No'], len(people_df))
people_df.head()
```

  <div id="df-526f40f5-eb89-4133-a0f7-737c607e979b" class="colab-df-container">
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
      <th>Age</th>
      <th>Gender</th>
      <th>Smoker</th>
    </tr>
    <tr>
      <th>Person_ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>73</td>
      <td>Male</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>Female</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79</td>
      <td>Male</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77</td>
      <td>Female</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>74</td>
      <td>Female</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-526f40f5-eb89-4133-a0f7-737c607e979b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-526f40f5-eb89-4133-a0f7-737c607e979b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-526f40f5-eb89-4133-a0f7-737c607e979b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-13950bca-f62c-4fc3-958c-81f3dba8a687">
  <button class="colab-df-quickchart" onclick="quickchart('df-13950bca-f62c-4fc3-958c-81f3dba8a687')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-13950bca-f62c-4fc3-958c-81f3dba8a687 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

$$
\begin{align*}
E(\text{Age} \mid \text{Gender}) = E(E(\text{Age} \mid \text{Smoker}, \text{Gender}) \mid \text{Gender})
\end{align*}
$$

``` python
people_df.groupby('Gender')['Age'].mean()
```

    Gender
    Female    73.236364
    Male      73.311111
    Name: Age, dtype: float64

$$
\begin{align*}
E(\text{Age} \mid \text{Gender}) &= E(E(\text{Age} \mid \text{Smoker}, \text{Gender}) \mid \text{Gender}) \\
&= \sum_{\text{Smoker}_s} E(\text{Age} \mid \text{Smoker}=\text{Smoker}_s, \text{Gender}) P(\text{Smoker}=\text{Smoker}_s \mid \text{Gender}) \\
\end{align*}
$$

``` python
people_df.groupby(['Smoker', 'Gender'])['Age'].mean()
```

    Smoker  Gender
    No      Female    73.500000
            Male      72.444444
    Yes     Female    73.060606
            Male      73.888889
    Name: Age, dtype: float64

``` python
people_df.groupby(['Smoker', 'Gender'])['Age'].mean().reset_index().groupby('Gender')['Age'].mean()
```

    Gender
    Female    73.280303
    Male      73.166667
    Name: Age, dtype: float64

``` python
people_df.groupby(['Gender', 'Smoker'])['Age'].agg(['mean', 'count'])
```

  <div id="df-e32f2c8d-438c-4d53-865a-12810c23caae" class="colab-df-container">
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
      <th></th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>Smoker</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">Female</th>
      <th>No</th>
      <td>73.500000</td>
      <td>22</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>73.060606</td>
      <td>33</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Male</th>
      <th>No</th>
      <td>72.444444</td>
      <td>18</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>73.888889</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e32f2c8d-438c-4d53-865a-12810c23caae')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e32f2c8d-438c-4d53-865a-12810c23caae button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e32f2c8d-438c-4d53-865a-12810c23caae');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f412b558-98a5-4285-b819-fc2957e47672">
  <button class="colab-df-quickchart" onclick="quickchart('df-f412b558-98a5-4285-b819-fc2957e47672')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f412b558-98a5-4285-b819-fc2957e47672 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

``` python
(22 / (22 + 33)) *  73.500000 + (33 / (22 + 33)) *  73.060606
```

    73.2363636

``` python
P_smoke_gender = pd.crosstab(people_df['Smoker'], people_df['Gender'], normalize='columns')
P_smoke_gender
```

  <div id="df-95fc5e0f-9ce2-4e26-b18d-cfcde7bdb324" class="colab-df-container">
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
      <th>Gender</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>Smoker</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>0.4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>0.6</td>
      <td>0.6</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-95fc5e0f-9ce2-4e26-b18d-cfcde7bdb324')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-95fc5e0f-9ce2-4e26-b18d-cfcde7bdb324 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-95fc5e0f-9ce2-4e26-b18d-cfcde7bdb324');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-7d77f747-53a2-4ca5-b489-abb2cfabef70">
  <button class="colab-df-quickchart" onclick="quickchart('df-7d77f747-53a2-4ca5-b489-abb2cfabef70')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-7d77f747-53a2-4ca5-b489-abb2cfabef70 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>

``` python
total = 0
for smoker in P_smoke_gender.index:
  total += (
      # E(Age | Smoker, Gender)
      people_df.groupby(['Smoker', 'Gender'])['Age'].mean()[smoker]['Female'] *
      # P(Smoker | Gender)
      P_smoke_gender.loc[smoker, "Female"]
      )
total
```

    73.23636363636363

$$
\begin{align*}
E(X \mid Y) &= E(E(X \mid Z, Y) \mid Y) \\
E(\text{Age} \mid \text{Gender}) &= E(E(\text{Age} \mid \text{Smoker}, \text{Gender}) \mid \text{Gender}) \\
&= \sum_{\text{Smoker}_s} E(\text{Age} \mid \text{Smoker}=\text{Smoker}_s, \text{Gender}) P(\text{Smoker}=\text{Smoker}_s \mid \text{Gender}) \\
&= \sum_{\text{Smoker}_s} \sum_{\text{Age}_i} \text{Age}_i P(\text{Age}=\text{Age}_i \mid \text{Smoker}=\text{Smoker}_s, \text{Gender}) P(\text{Smoker}=\text{Smoker}_s \mid \text{Gender}) \\
&= \sum_{\text{Smoker}_s} \sum_{\text{Age}_i} \frac{P(\text{Smoker}=\text{Smoker}_s, \text{Gender})}{P(\text{Gender})} \frac{P(\text{Age}=\text{Age}_i , \text{Smoker}=\text{Smoker}_s, \text{Gender})}{P(\text{Smoker}=\text{Smoker}_s, \text{Gender})} \text{Age}_i \\
&= \sum_{\text{Smoker}_s} \sum_{\text{Age}_i} \frac{P(\text{Smoker}=\text{Smoker}_s, \text{Gender})}{P(\text{Gender})} \frac{P(\text{Age}=\text{Age}_i , \text{Smoker}=\text{Smoker}_s, \text{Gender})}{P(\text{Smoker}=\text{Smoker}_s, \text{Gender})} \text{Age}_i \\
&= \sum_{\text{Smoker}_s} \sum_{\text{Age}_i} \frac{P(\text{Age}=\text{Age}_i , \text{Smoker}=\text{Smoker}_s, \text{Gender})}{P(\text{Gender})} \text{Age}_i \\
&=  \sum_{\text{Age}_i} \frac{\text{Age}_i}{P(\text{Gender})} \sum_{\text{Smoker}_s} P(\text{Age}=\text{Age}_i , \text{Smoker}=\text{Smoker}_s, \text{Gender}) \\
&=  \sum_{\text{Age}_i} \frac{\text{Age}_i}{P(\text{Gender})} P(\text{Age}=\text{Age}_i , \text{Gender})(1) \\
&=  \sum_{\text{Age}_i} \text{Age}_i \frac{P(\text{Age}=\text{Age}_i , \text{Gender})}{P(\text{Gender})} \\
&=  \sum_{\text{Age}_i} \text{Age}_i P(\text{Age}=\text{Age}_i  \mid \text{Gender})
\end{align*}
$$

Can be arbitrarily extended
$$
\begin{align*}
E(X \mid Y) &= E(E(X \mid Z, Y) \mid Y) \\
&= \sum_z E(X \mid Z=z, Y=y) P[E(X \mid Z, Y) = E(X \mid Z=z, Y=y) \mid Y=y] \\
&= \sum_z E(X \mid Z=z, Y=y) P(Z=z \mid Y=y) \\
&= \sum_z \sum_x x P(X=x \mid Z=z, Y=y) P(Z=z \mid Y=y) \\
&= \sum_z \sum_x x  \frac{P(X=x , Z=z, Y=y)}{P(Z=z, Y=y)} \times \frac{P(Z=z, Y=y)}{P(Y=y)} \\
&= \sum_z \sum_x x  \frac{P(X=x , Z=z, Y=y)}{P(Y=y)} \\
&= \sum_x \frac{x}{P(Y=y)} \sum_z P(X=x , Z=z, Y=y) \\
&= \sum_x \frac{x}{P(Y=y)}  P(X=x , Y=y) \\
&= \sum_x x\frac{ P(X=x , Y=y)}{P(Y=y)}\\
&= \sum_x x P(X=x \mid Y=y)\\
&= E(X \mid Y=y)\\
&= E(X \mid Y)\\
\end{align*}
$$
