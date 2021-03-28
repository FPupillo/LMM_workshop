  # set up the custom data simulation function
  # In order to get more explanation please check the file "power_analysis1.R" in the same folder
  library(dplyr)
  library(tidyr)
  library(lme4)
  #library(tmvtnorm) # generate values from the truncate normaal distribution
  library(truncnorm)
  #library(foreach)
  #library(doParallel)
  
  my_sim_data <- function(n_subj,
   # number of subjects
   SubN = 60,
    beta_0 = 1.28, # grand mean memory
    # PEmean
    PEmean = 0.30,
    beta_PE = -0.29, # effect of PE
    n_items = 300, # number of item
    tau_0 = 1.80, # by-subject random intercept sd .Check if it is sd, because than it will be squared into variance
    tau_1 = 0.50, # by-subject random slope sd
    rho = 0.3, # correlation between intercept and slope
    sigma = 0.3, # residual standard deviation
   
   RTorACC = 1 # RT or Accuracy. 1 is reaction time (normal skewed), 2 accuracy (binary)
    ) { 
  
  #autocorrelation
  #generate a matrix
  #read https://stats.stackexchange.com/questions/183406/how-to-generate-series-of-pseudorandom-autocorrelated-numbers for a reference
  tmp.r <- matrix(0.1, n_items, n_items)
  tmp.r <-tmp.r^abs(row(tmp.r)-col(tmp.r))

    
    # total number of items = n_ingroup + n_outgroup
    items <- data.frame(
      #item_id = seq_len(n_items), PE = MASS::mvrnorm(1, rep(PEmean,n_items), tmp.r) # the autocorrelation is between 3 consecutiive assessments
      # item_id = seq_len(n_items), PE = rtmvnorm(n=n_items, mean = rep(PEmean,n_items),
      #                                 lower=rep(-1,n_items), 
      #                                 upper=rep(1, n_items)) # the autocorrelation is between 3 consecutiive assessments 
      item_id = seq_len(n_items), PE = rtruncnorm(n_items, a = -0.5, b=0.5, mean=PEmean, sd=0.5) 
       )
    

    
    # variance-covariance matrix
    cov_mx <- matrix(
      c(tau_0^2, rho * tau_0 * tau_1,
        rho * tau_0 * tau_1, tau_1^2 ),
      nrow = 2, byrow = TRUE)
    
    subjects <- data.frame(subj_id = seq_len(SubN),
                           MASS::mvrnorm(n = SubN,
                                         mu = c(T_0s = 0, T_1s = 0),
                                         Sigma = cov_mx))
  
    trials<- crossing(subjects, items) %>%
      mutate(e_si = rnorm(nrow(.), mean = 0, sd = sigma))# %>%
      #select(subj_id, item_id, valence ,everything())
    
    # add random noise
    # in order to keep the direction of PE, we are drawing the random noise from the uniform distribution
    # if pe is <0, we make it more negative, otherwise, we make if more positive (adding or subtracting a number
    # that is betwen 0 an d0.5)
    for (n in 1:nrow(trials)){
      if (trials$PE[n]<0){
      trials$PE[n]<-trials$PE[n]-runif(1, min=0, max=0.5)
      } else{
        trials$PE[n]<-trials$PE[n]+runif(1, min=0, max=0.5)
      }
    }
    

    
     #select(subj_id, item_id, valence, PM)
    
    if (RTorACC==1){
      dat_sim<- trials %>%
        mutate(RT = beta_0 + T_0s + (beta_PE + T_1s)*PE  + e_si)# %>% 
      # add the smallest value to make it positive (absolute number)
      dat_sim$RT<-dat_sim$RT+abs(min(dat_sim$RT))
      
      # shuffle trials to create some variability among them
      dat_sim<-dat_sim %>%
        group_by(subj_id)%>%
        mutate(item_id = sample(item_id))
     # hist(dat_sim$RT)
      #dat_sim$RT<-dexp(dat_sim$RT, rate=0.25)
      # add 500 ms
      #dat_sim$RT<-dat_sim$RT+0.5
      #dat_sim$RT<-dsn(dat_sim$RT, xi=1, omega=6, alpha=5)
      #hist(dat_sim$RT)
      

      
    } else{
      dat_sim<- trials %>%
        mutate(rec_acc = beta_0 + T_0s + (beta_PE + T_1s)*PE  + e_si)
      # convert the log odds into proportions
      # read https://aosmith.rbind.io/2020/08/20/simulate-binomial-glmm/#a-single-simulation-for-a-binomial-glmm for a referencep 
      dat_sim$rec_acc  <-plogis(dat_sim$rec_acc)
      # convert into binomial
      for (n in 1:nrow(dat_sim)){
        dat_sim$rec_acc[n]<-rbinom(1, 1, prob = dat_sim$rec_acc[n])
      }
    
    }
    
    # create fictitious levels of PE
    dat_sim$PElevel<-as.vector(NA)
    for (n in 1:nrow(dat_sim)){
      if (dat_sim$PE[n]<(-0.66)){
        dat_sim$PElevel[n]<-1
      } else if(dat_sim$PE[n]>= (-0.66) & dat_sim$PE[n] < 0.33 ) {
        dat_sim$PElevel[n]<-2
      } else if(dat_sim$PE[n]>= 0.33) {
        dat_sim$PElevel[n]<-3}
    }
  
    return(dat_sim)
  }
  