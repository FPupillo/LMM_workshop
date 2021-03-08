``` r
# load the packages and source functions
library(dplyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(RLRsim)
source("helper_functions/simulateData.R")
# suppress scientific notation
options(scipen=5)
```

Simulate data

``` r
# set.seed
set.seed(1234)
df<-my_sim_data(   
  SubN = 30,
  beta_0 = 1.28, # grand mean (fixed intercept)
  PEmean = 0.30,# PEmean
  beta_PE = 0.55, # effect of PE (fixed slope)
  n_items = 300, # number of trials
  tau_0 = 1.80, # by-subject random intercept sd .Check if it is sd, because than it will be squared into variance
  tau_1 = 0.5, # by-subject random slope sd
  rho = 0.3, # correlation between intercept and slope
  sigma = 0.3, # residual standard deviation
  RTorACC = 1 # Reaction times or accuracy (1 = RT, 2 = accuracy)
  )

# show first row of simulated data
head(df)
```

    ## # A tibble: 6 x 8
    ##   subj_id  T_0s  T_1s item_id     PE     e_si     RT PElevel
    ##     <int> <dbl> <dbl>   <int>  <dbl>    <dbl>  <dbl>   <dbl>
    ## 1       1 -2.21 0.203       1 -0.390  0.391   -0.829       2
    ## 2       1 -2.21 0.203       2  0.492 -0.196   -0.752       3
    ## 3       1 -2.21 0.203       3  0.226 -0.00506 -0.762       2
    ## 4       1 -2.21 0.203       4  0.249 -0.170   -0.910       2
    ## 5       1 -2.21 0.203       5  0.121 -0.338   -1.17        2
    ## 6       1 -2.21 0.203       6 -0.554 -0.555   -1.90        2

How can we deal with such a dataset? We can aggregate values at the
participant level, then run OLS

``` r
df_agg<-df %>%
  group_by(subj_id) %>%
  summarise(RT=mean(RT), PE=mean(PE)) 

# plot 
ggplot(df_agg, aes(y=PE, x = RT))+
  geom_point()+
  geom_smooth(method="lm")
```

    ## `geom_smooth()` using formula 'y ~ x'

![](lmmworkshop_files/figure-markdown_github/aggregate-1.png)

``` r
linearmod<-lm(RT~PE, data = df_agg)

summary(linearmod)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PE, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -5.4377 -1.3886 -0.3138  2.2284  5.4544 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)   0.3229     9.4950   0.034    0.973
    ## PE            4.5753    65.6630   0.070    0.945
    ## 
    ## Residual standard error: 2.576 on 28 degrees of freedom
    ## Multiple R-squared:  0.0001734,  Adjusted R-squared:  -0.03553 
    ## F-statistic: 0.004855 on 1 and 28 DF,  p-value: 0.9449

``` r
#linearmod<-lmer(RT~PE+(1+PE|subj_id), data = df)


# get f value
summary.aov(linearmod)
```

    ##             Df Sum Sq Mean Sq F value Pr(>F)
    ## PE           1   0.03   0.032   0.005  0.945
    ## Residuals   28 185.78   6.635

``` r
# first, let's run an intercept only, unconditional model
mixmod_unc<-lmer(RT~1+(1|subj_id), data = df)
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, :
    ## Model failed to converge with max|grad| = 0.00397025 (tol = 0.002, component 1)

``` r
summary(mixmod_unc)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ 1 + (1 | subj_id)
    ##    Data: df
    ## 
    ## REML criterion at convergence: 12017.4
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -5.8101 -0.5957  0.0395  0.6287  4.7928 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev.
    ##  subj_id  (Intercept) 6.4019   2.5302  
    ##  Residual             0.2159   0.4647  
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)  
    ## (Intercept)   0.9837     0.4620 29.0321   2.129   0.0418 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## optimizer (nloptwrap) convergence code: 0 (OK)
    ## Model failed to converge with max|grad| = 0.00397025 (tol = 0.002, component 1)

``` r
# test significance
# model without random intercept. We can create a random intercept that is constant at 1
df$constint<-rep(1, nrow(df))

mod_unc<-lm(RT~1, data=df)
mixmod_unc<-lmer(RT~1+(1|subj_id), data = df)
```

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, :
    ## Model failed to converge with max|grad| = 0.00397025 (tol = 0.002, component 1)

``` r
anova(mixmod_unc, mod_unc)
```

    ## refitting model(s) with ML (instead of REML)

    ## Data: df
    ## Models:
    ## mod_unc: RT ~ 1
    ## mixmod_unc: RT ~ 1 + (1 | subj_id)
    ##            npar   AIC   BIC   logLik deviance Chisq Df Pr(>Chisq)    
    ## mod_unc       2 42264 42278 -21130.1    42260                        
    ## mixmod_unc    3 12024 12045  -6008.8    12018 30243  1  < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# use simulation based-test
exactLRT(mixmod_unc,mod_unc)
```

    ## No restrictions on fixed effects. REML-based inference preferable.

    ## Using likelihood evaluated at REML estimators.

    ## Please refit model with method="ML" for exact results.

    ## 
    ##  simulated finite sample distribution of LRT. (p-value based on 10000
    ##  simulated values)
    ## 
    ## data:  
    ## LRT = 30243, p-value < 2.2e-16
