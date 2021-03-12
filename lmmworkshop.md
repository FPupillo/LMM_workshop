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
  beta_0 = 0.60, # grand mean (fixed intercept)
  PEmean = 0.30,# PEmean
  beta_PE = 1.28, # effect of PE (fixed slope)
  n_items = 300, # number of trials
  tau_0 = 1.80, # by-subject random intercept sd .Check if it is sd, because than it will be squared into variance
  tau_1 = 2.50, # by-subject random slope sd
  rho = 0.3, # correlation between intercept and slope
  sigma = 0.3, # residual standard deviation
  RTorACC = 1 # Reaction times or accuracy (1 = RT, 2 = accuracy)
  )

# show first row of simulated data
head(df)
```

    ## # A tibble: 6 x 8
    ##   subj_id  T_0s  T_1s item_id     PE     e_si    RT PElevel
    ##     <int> <dbl> <dbl>   <int>  <dbl>    <dbl> <dbl>   <dbl>
    ## 1       1  2.42  2.44       1 -0.390  0.391    8.68       2
    ## 2       1  2.42  2.44       2  0.492 -0.196   11.4        3
    ## 3       1  2.42  2.44       3  0.226 -0.00506 10.6        2
    ## 4       1  2.42  2.44       4  0.249 -0.170   10.5        2
    ## 5       1  2.42  2.44       5  0.121 -0.338    9.85       2
    ## 6       1  2.42  2.44       6 -0.554 -0.555    7.13       2

How can we deal with such a dataset? We can aggregate values at the
participant level, then run OLS

``` r
# We are assuming that RT are normally distributed

# aggregate RT at the subject level
df_agg<-df %>%
  group_by(subj_id) %>%
  summarise(RT=mean(RT), PE=mean(PE)) 

# plot 
ggplot(df_agg, aes(y=RT, x = PE))+
  geom_point()+
  geom_smooth(method="lm")
```

    ## `geom_smooth()` using formula 'y ~ x'

![](lmmworkshop_files/figure-markdown_github/aggregate-1.png)

``` r
# simple regression
linearmod<-lm(RT~PE, data = df_agg)

# get the summary
summary(linearmod)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PE, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.7819 -1.5765  0.0868  0.9321  5.3083 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)    8.233      7.024   1.172    0.251
    ## PE            -3.730     48.573  -0.077    0.939
    ## 
    ## Residual standard error: 1.905 on 28 degrees of freedom
    ## Multiple R-squared:  0.0002105,  Adjusted R-squared:  -0.0355 
    ## F-statistic: 0.005897 on 1 and 28 DF,  p-value: 0.9393

``` r
# first, let's run an intercept only, unconditional model
mixmod_unc<-lmer(RT~1+(1|subj_id), data = df, control=lmerControl(optimizer="bobyqa"))#,optCtrl=list(maxfun=100000)))

summary(mixmod_unc)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ 1 + (1 | subj_id)
    ##    Data: df
    ## Control: lmerControl(optimizer = "bobyqa")
    ## 
    ## REML criterion at convergence: 38252
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.5976 -0.5385  0.0698  0.5965  3.5325 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev.
    ##  subj_id  (Intercept) 3.493    1.869   
    ##  Residual             4.030    2.008   
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)   7.6947     0.3419 29.0000   22.51   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# calculate the intraclass correlation
ICC<-VarCorr(mixmod_unc)$subj_id[1]/(VarCorr(mixmod_unc)$subj_id[1]+summary(mixmod_unc)$sigma^2)

# paste the results
paste("The IntraClass Correlation is",  ICC)
```

    ## [1] "The IntraClass Correlation is 0.464271789380495"

``` r
# test significance
# model without random intercept. We can create a random intercept that is constant at 1
df$constint<-rep(1, nrow(df))

mod_unc<-lm(RT~1, data=df)
mixmod_unc<-lmer(RT~1+(1|subj_id), data = df)

anova(mixmod_unc, mod_unc)
```

    ## refitting model(s) with ML (instead of REML)

    ## Data: df
    ## Models:
    ## mod_unc: RT ~ 1
    ## mixmod_unc: RT ~ 1 + (1 | subj_id)
    ##            npar   AIC   BIC logLik deviance  Chisq Df Pr(>Chisq)    
    ## mod_unc       2 43566 43580 -21781    43562                         
    ## mixmod_unc    3 38258 38279 -19126    38252 5310.3  1  < 2.2e-16 ***
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
    ## LRT = 5310.3, p-value < 2.2e-16

``` r
# plot
  ggplot(df, aes(y=RT, x = PE))+
  geom_smooth(method="lm", se=F)+aes(colour = factor(subj_id))+
    geom_smooth(method="lm", colour="black")
```

    ## `geom_smooth()` using formula 'y ~ x'
    ## `geom_smooth()` using formula 'y ~ x'

![](lmmworkshop_files/figure-markdown_github/maximal-1.png)

``` r
maxMod<-lmer(RT~PE+(1+PE|subj_id), data = df, control=lmerControl(optimizer="bobyqa"))
summary(maxMod)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ PE + (1 + PE | subj_id)
    ##    Data: df
    ## Control: lmerControl(optimizer = "bobyqa")
    ## 
    ## REML criterion at convergence: 4179.7
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3829 -0.6809  0.0004  0.6715  3.6142 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev. Corr
    ##  subj_id  (Intercept)  2.50371 1.5823       
    ##           PE          12.62806 3.5536   0.44
    ##  Residual              0.08771 0.2962       
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)   7.4325     0.2889 28.9999  25.726  < 2e-16 ***
    ## PE            1.8175     0.6488 29.0001   2.801  0.00897 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##    (Intr)
    ## PE 0.438

``` r
anova(maxMod)
```

    ## Type III Analysis of Variance Table with Satterthwaite's method
    ##     Sum Sq Mean Sq NumDF DenDF F value   Pr(>F)   
    ## PE 0.68828 0.68828     1    29  7.8469 0.008972 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# grand mean centring and person mean centring
df<- df %>%
  # Grand mean centering (GMC)
  mutate (PE.gmc = PE-mean(PE)) %>%
  # Person mean centering (centering withing clusters - Participants)
  group_by(subj_id) %>%
  mutate(PE.cm = mean(PE),
         PE.cwc = PE-PE.cm ) %>%
  ungroup %>%
  # grand mean centering of the aggregated variable
  mutate(PE.cmc= PE.cm-mean(PE.cm))


maxModCent<-lmer(RT~PE.cwc+PE.cmc+(1+PE.cwc|subj_id), data = df, control=lmerControl(optimizer="bobyqa"))
summary(maxModCent)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ PE.cwc + PE.cmc + (1 + PE.cwc | subj_id)
    ##    Data: df
    ## Control: lmerControl(optimizer = "bobyqa")
    ## 
    ## REML criterion at convergence: 4170.9
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3829 -0.6809  0.0003  0.6716  3.6142 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev. Corr
    ##  subj_id  (Intercept)  3.57840 1.8917       
    ##           PE.cwc      12.62807 3.5536   0.64
    ##  Residual              0.08771 0.2962       
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)   7.6947     0.3454 28.6405  22.279  < 2e-16 ***
    ## PE.cwc        1.8175     0.6488 29.0001   2.801  0.00897 ** 
    ## PE.cmc       -2.1428    37.0993 28.0000  -0.058  0.95435    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##        (Intr) PE.cwc
    ## PE.cwc 0.639        
    ## PE.cmc 0.000  0.000

``` r
# fit a model with covariance of random effects set at zero
maxModZeroCov<-lmer(RT~PE+PE+(PE||subj_id), data = df, control=lmerControl(optimizer="bobyqa"))

summary(maxModZeroCov)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ PE + PE + (PE || subj_id)
    ##    Data: df
    ## Control: lmerControl(optimizer = "bobyqa")
    ## 
    ## REML criterion at convergence: 4185.9
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3830 -0.6811  0.0001  0.6721  3.6143 
    ## 
    ## Random effects:
    ##  Groups    Name        Variance Std.Dev.
    ##  subj_id   (Intercept)  2.50377 1.5823  
    ##  subj_id.1 PE          12.62840 3.5536  
    ##  Residual               0.08771 0.2962  
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)   7.4325     0.2889 28.9999  25.726  < 2e-16 ***
    ## PE            1.8175     0.6488 29.0000   2.801  0.00897 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##    (Intr)
    ## PE 0.000

``` r
# let's compare
anova(maxMod, maxModZeroCov )
```

    ## refitting model(s) with ML (instead of REML)

    ## Data: df
    ## Models:
    ## maxModZeroCov: RT ~ PE + PE + (PE || subj_id)
    ## maxMod: RT ~ PE + (1 + PE | subj_id)
    ##               npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)  
    ## maxModZeroCov    5 4196.2 4231.7 -2093.1   4186.2                       
    ## maxMod           6 4191.8 4234.4 -2089.9   4179.8 6.3997  1    0.01141 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
# create a categorical predictor between, simulating that we are randomly assigning participants
# to different groups of PE
PEbet<-rep(c("HighPE", "LowPE", "MediumPE"), each= nrow(df_agg)/3)

# created the levels in the aggregated dataset
  # pick random subject
  df_agg$PEbw<-sample(PEbet, nrow(df_agg), replace = F)
  
  # we want it as factor
  df_agg$PEbw<-as.factor(df_agg$PEbw)

  # descriptives
df_agg %>%
  group_by(PEbw) %>%
  summarise(mean=mean(RT), sd = sd(RT))
```

    ## # A tibble: 3 x 3
    ##   PEbw      mean    sd
    ## * <fct>    <dbl> <dbl>
    ## 1 HighPE    8.09  1.47
    ## 2 LowPE     7.03  2.07
    ## 3 MediumPE  7.96  2.03

``` r
# plot 
ggplot(df_agg, aes(PEbw, RT))+
  geom_bar(aes(PEbw, RT, fill = PEbw),
           position="dodge",stat="summary")+
  geom_point()+
  stat_summary(fun.data = "mean_cl_boot", size = 0.8, geom="errorbar", width=0.2 )# this line adds error bars
```

    ## No summary function supplied, defaulting to `mean_se()`

![](lmmworkshop_files/figure-markdown_github/categorical%20predictor%20between-1.png)

``` r
# create an Anova between participant
bwlm<-lm(RT~PEbw, data=df_agg)

summary(bwlm)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PEbw, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.8281 -1.3797 -0.1673  0.8697  5.0103 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)    8.0887     0.5934  13.632 1.27e-13 ***
    ## PEbwLowPE     -1.0539     0.8392  -1.256     0.22    
    ## PEbwMediumPE  -0.1279     0.8392  -0.152     0.88    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.876 on 27 degrees of freedom
    ## Multiple R-squared:  0.06506,    Adjusted R-squared:  -0.004194 
    ## F-statistic: 0.9394 on 2 and 27 DF,  p-value: 0.4033

``` r
# re-level in order to have medium as the reference level
df_agg$PEbw<-relevel(df_agg$PEbw, ref = "MediumPE")

# refit
bwlm<-lm(RT~PEbw, data=df_agg)

summary(bwlm)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PEbw, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.8281 -1.3797 -0.1673  0.8697  5.0103 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   7.9608     0.5934  13.416 1.86e-13 ***
    ## PEbwHighPE    0.1279     0.8392   0.152     0.88    
    ## PEbwLowPE    -0.9260     0.8392  -1.104     0.28    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.876 on 27 degrees of freedom
    ## Multiple R-squared:  0.06506,    Adjusted R-squared:  -0.004194 
    ## F-statistic: 0.9394 on 2 and 27 DF,  p-value: 0.4033

``` r
anova(bwlm)
```

    ## Analysis of Variance Table
    ## 
    ## Response: RT
    ##           Df Sum Sq Mean Sq F value Pr(>F)
    ## PEbw       2  6.615  3.3077  0.9394 0.4033
    ## Residuals 27 95.066  3.5210

``` r
# method 1: contrast poly
contrasts(df_agg$PEbw)<-contr.poly(3) # we have three levels in our categorical predictor

# refit the model
bwlmContr<-lm(RT~PEbw, data=df_agg)

summary(bwlmContr)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PEbw, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.8281 -1.3797 -0.1673  0.8697  5.0103 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   7.6947     0.3426  22.461   <2e-16 ***
    ## PEbw.L       -0.6548     0.5934  -1.104    0.280    
    ## PEbw.Q       -0.4825     0.5934  -0.813    0.423    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.876 on 27 degrees of freedom
    ## Multiple R-squared:  0.06506,    Adjusted R-squared:  -0.004194 
    ## F-statistic: 0.9394 on 2 and 27 DF,  p-value: 0.4033

``` r
# set our own contrasts
# linear
contrast1<-c(-1,0,1)
# quadratic
contrast2<-c(1,-2,1)


contrasts(df_agg$PEbw)<-cbind(contrast1, contrast2)
  
# refit the model
bwlmContrCust<-lm(RT~PEbw, data=df_agg)

summary(bwlmContrCust)               
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PEbw, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.8281 -1.3797 -0.1673  0.8697  5.0103 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     7.6947     0.3426  22.461   <2e-16 ***
    ## PEbwcontrast1  -0.4630     0.4196  -1.104    0.280    
    ## PEbwcontrast2  -0.1970     0.2422  -0.813    0.423    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.876 on 27 degrees of freedom
    ## Multiple R-squared:  0.06506,    Adjusted R-squared:  -0.004194 
    ## F-statistic: 0.9394 on 2 and 27 DF,  p-value: 0.4033

``` r
# what if we convert the categorical into continuous?
df_agg$PEbwC<-as.vector(NA)

for (n in 1:nrow(df_agg)){
  if (df_agg$PEbw[n]=="MediumPE"){
  df_agg$PEbwC[n]<-0.33
  } else if (df_agg$PEbw[n] == "HighPE"){
    df_agg$PEbwC[n]<-0.80
  }  else if (df_agg$PEbw[n] == "LowPE"){
    df_agg$PEbwC[n]<-0.20
  }
}

ggplot(df_agg, aes(y=RT, x = PEbwC))+
  geom_point()+
  geom_smooth(method="lm")
```

    ## `geom_smooth()` using formula 'y ~ x'

![](lmmworkshop_files/figure-markdown_github/continuous%20predictor%20between-1.png)

``` r
# fit lm
bwlmC<-lm(RT~PEbwC, data=df_agg)

summary(bwlmC)
```

    ## 
    ## Call:
    ## lm(formula = RT ~ PEbwC, data = df_agg)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.9191 -1.6143  0.1484  1.0757  5.4305 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   7.0919     0.6796  10.435 3.73e-11 ***
    ## PEbwC         1.3597     1.3253   1.026    0.314    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.871 on 28 degrees of freedom
    ## Multiple R-squared:  0.03623,    Adjusted R-squared:  0.001813 
    ## F-statistic: 1.053 on 1 and 28 DF,  p-value: 0.3137

``` r
# we have a categorical variable in the dataset, which is PElevel
df$PElevel<-as.factor(df$PElevel)

# let's inspect that
  ggplot(df, aes(y=RT, x = PElevel))+
  geom_boxplot( )
```

![](lmmworkshop_files/figure-markdown_github/categorical%20predictor%20within%20ez-1.png)

``` r
# eazy anova on the aggregate dataset
df_aggbw<-df %>%
          group_by(subj_id,PElevel) %>%
          summarise(RT=mean(RT))
```

    ## `summarise()` has grouped output by 'subj_id'. You can override using the `.groups` argument.

``` r
head(df_aggbw)
```

    ## # A tibble: 6 x 3
    ## # Groups:   subj_id [2]
    ##   subj_id PElevel    RT
    ##     <int> <fct>   <dbl>
    ## 1       1 1        6.93
    ## 2       1 2        8.76
    ## 3       1 3       11.9 
    ## 4       2 1        7.68
    ## 5       2 2        6.35
    ## 6       2 3        4.12

``` r
# ezanova
library(ez)
```

    ## Registered S3 methods overwritten by 'car':
    ##   method                          from
    ##   influence.merMod                lme4
    ##   cooks.distance.influence.merMod lme4
    ##   dfbeta.influence.merMod         lme4
    ##   dfbetas.influence.merMod        lme4

``` r
ezModel<-ezANOVA(data = df_aggbw, # dataframe
                 dv = .(RT), # dependent variable. This functions requires to place the name of each variable within .() 
                 wid = .(subj_id), # variable that identifies participants )
                 within = .(PElevel), # independent variable
                 detailed = T
                 )
```

    ## Warning: Converting "subj_id" to factor for ANOVA.

``` r
ezModel
```

    ## $ANOVA
    ##        Effect DFn DFd        SSn      SSd          F            p p<.05
    ## 1 (Intercept)   1  29 4649.02600 182.0397 740.617464 3.377350e-22     *
    ## 2     PElevel   2  58   93.06356 334.9127   8.058349 8.162931e-04     *
    ##         ges
    ## 1 0.8999314
    ## 2 0.1525592
    ## 
    ## $`Mauchly's Test for Sphericity`
    ##    Effect           W            p p<.05
    ## 2 PElevel 0.002235943 7.806401e-38     *
    ## 
    ## $`Sphericity Corrections`
    ##    Effect       GGe       p[GG] p[GG]<.05       HFe       p[HF] p[HF]<.05
    ## 2 PElevel 0.5005596 0.008167513         * 0.5006196 0.008165235         *

``` r
# 
ModCateg<-lmer(RT~PElevel+(PElevel|subj_id), data = df, control=lmerControl(optimizer="bobyqa"))
```

    ## boundary (singular) fit: see ?isSingular

``` r
summary(ModCateg)
```

    ## Linear mixed model fit by REML. t-tests use Satterthwaite's method [
    ## lmerModLmerTest]
    ## Formula: RT ~ PElevel + (PElevel | subj_id)
    ##    Data: df
    ## Control: lmerControl(optimizer = "bobyqa")
    ## 
    ## REML criterion at convergence: 25596
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.0859 -0.5149 -0.0259  0.4480  5.1094 
    ## 
    ## Random effects:
    ##  Groups   Name        Variance Std.Dev. Corr       
    ##  subj_id  (Intercept)  6.070   2.4638              
    ##           PElevel2     3.960   1.9899   -0.81      
    ##           PElevel3    22.960   4.7916   -0.82  1.00
    ##  Residual              0.963   0.9813              
    ## Number of obs: 9000, groups:  subj_id, 30
    ## 
    ## Fixed effects:
    ##             Estimate Std. Error      df t value Pr(>|t|)    
    ## (Intercept)   6.0170     0.4520 29.0720  13.311 6.73e-14 ***
    ## PElevel2      1.0267     0.3664 29.0676   2.802  0.00894 ** 
    ## PElevel3      2.4814     0.8761 29.0106   2.832  0.00832 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##          (Intr) PElvl2
    ## PElevel2 -0.814       
    ## PElevel3 -0.819  0.996
    ## optimizer (bobyqa) convergence code: 0 (OK)
    ## boundary (singular) fit: see ?isSingular

``` r
# ModCateg<-lmer(RT~PElevel+(1|subj_id/PElevel), data = df, control=lmerControl(optimizer="bobyqa"))
# summary(ModCateg)
# 
# anova (ModCateg)
# ModCateg2<-lmer(RT~PElevel+(-1+PElevel|subj_id)+(1|subj_id),data = df, control=lmerControl(optimizer="bobyqa"))
# 
# anova(ModCateg, ModCateg2)
# 
# summary(rePCA(ModCateg))
# 
# anova(maxModZeroCov)
# # let's compare
# anova(maxMod, maxModZeroCov )
# 
# # set contrasts
# contrasts(df$PElevel)<-contr.poly(3)
# ModCateg<-lmer(RT~PElevel+(1|subj_id/PElevel), data = df, control=lmerControl(optimizer="bobyqa"))
# 
# summary(ModCateg)
# 
# anova(ModCateg)
# # bayesian
# fit_b<-brm(RT~PElevel +(PElevel|subj_id),   #similar to lmwr
#            
#            warmup = 500, 
#            iter = 2000, 
#            chains = 2, 
#            #prior = prior2,
#            inits = "0", 
#            cores=1,
#            data=df)
# try nlme
library(nlme)
```

    ## 
    ## Attaching package: 'nlme'

    ## The following object is masked from 'package:lme4':
    ## 
    ##     lmList

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     collapse

``` r
test.lme<-lme(fixed = RT~PElevel, 
              random = ~ PElevel| subj_id, 
              data = df, 
              method = "REML", 
              control= lmeControl( opt = "optim", optimMethod ="BFGS" ))

summary(test.lme)
```

    ## Linear mixed-effects model fit by REML
    ##   Data: df 
    ##        AIC      BIC    logLik
    ##   25617.41 25688.46 -12798.71
    ## 
    ## Random effects:
    ##  Formula: ~PElevel | subj_id
    ##  Structure: General positive-definite, Log-Cholesky parametrization
    ##             StdDev    Corr         
    ## (Intercept) 2.4384871 (Intr) PElvl2
    ## PElevel2    1.9914401 -0.814       
    ## PElevel3    4.8093095 -0.820  1.000
    ## Residual    0.9813231              
    ## 
    ## Fixed effects:  RT ~ PElevel 
    ##                Value Std.Error   DF   t-value p-value
    ## (Intercept) 6.017379 0.4474548 8968 13.448017  0.0000
    ## PElevel2    1.026334 0.3666576 8968  2.799163  0.0051
    ## PElevel3    2.481034 0.8793245 8968  2.821523  0.0048
    ##  Correlation: 
    ##          (Intr) PElvl2
    ## PElevel2 -0.815       
    ## PElevel3 -0.819  0.996
    ## 
    ## Standardized Within-Group Residuals:
    ##         Min          Q1         Med          Q3         Max 
    ## -4.09022135 -0.51442517 -0.02559548  0.44716378  5.11028780 
    ## 
    ## Number of Observations: 9000
    ## Number of Groups: 30

``` r
test.lme
```

    ## Linear mixed-effects model fit by REML
    ##   Data: df 
    ##   Log-restricted-likelihood: -12798.71
    ##   Fixed: RT ~ PElevel 
    ## (Intercept)    PElevel2    PElevel3 
    ##    6.017379    1.026334    2.481034 
    ## 
    ## Random effects:
    ##  Formula: ~PElevel | subj_id
    ##  Structure: General positive-definite, Log-Cholesky parametrization
    ##             StdDev    Corr         
    ## (Intercept) 2.4384871 (Intr) PElvl2
    ## PElevel2    1.9914401 -0.814       
    ## PElevel3    4.8093095 -0.820  1.000
    ## Residual    0.9813231              
    ## 
    ## Number of Observations: 9000
    ## Number of Groups: 30
