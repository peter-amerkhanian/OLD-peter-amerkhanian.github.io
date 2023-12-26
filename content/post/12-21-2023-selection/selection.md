ok
================
2023-11-03

``` r
library(tidyverse)
```

    ## Warning: package 'forcats' was built under R version 4.3.0

    ## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ## ✔ dplyr     1.1.1     ✔ readr     2.1.4
    ## ✔ forcats   1.0.0     ✔ stringr   1.5.0
    ## ✔ ggplot2   3.4.2     ✔ tibble    3.2.1
    ## ✔ lubridate 1.9.2     ✔ tidyr     1.3.0
    ## ✔ purrr     1.0.1     
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()
    ## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(broom)
```

    ## Warning: package 'broom' was built under R version 4.3.0

``` r
if (!requireNamespace("MatchIt", quietly = TRUE)) {
  install.packages("MatchIt")
}
library("MatchIt")
data("lalonde")

head(lalonde)
```

    ##      treat age educ   race married nodegree re74 re75       re78
    ## NSW1     1  37   11  black       1        1    0    0  9930.0460
    ## NSW2     1  22    9 hispan       0        1    0    0  3595.8940
    ## NSW3     1  30   12  black       0        0    0    0 24909.4500
    ## NSW4     1  27   11  black       0        1    0    0  7506.1460
    ## NSW5     1  33    8  black       0        1    0    0   289.7899
    ## NSW6     1  22    9  black       0        1    0    0  4056.4940

``` r
# custom function to transpose while preserving names
transpose_df <- function(df) {
  t_df <- data.table::transpose(df)
  colnames(t_df) <- rownames(df)
  rownames(t_df) <- colnames(df)
  t_df <- t_df %>%
    tibble::rownames_to_column(.data = .) %>%
    tibble::as_tibble(.)
  return(t_df)
}
```

``` r
model <- lm(re78~treat, data=lalonde)
tidy_model <- tidy(model, conf.int = TRUE)
tidy_model
```

    ## # A tibble: 2 × 7
    ##   term        estimate std.error statistic  p.value conf.low conf.high
    ##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>    <dbl>     <dbl>
    ## 1 (Intercept)    6984.      361.    19.4   1.64e-65    6276.     7693.
    ## 2 treat          -635.      657.    -0.966 3.34e- 1   -1926.      655.
