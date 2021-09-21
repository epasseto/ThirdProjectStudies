#import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import udacourse3 #my library

#graphs
import matplotlib.patches as mpatches
import matplotlib.style as mstyles
import matplotlib.pyplot as mpyplots

#from matplotlib.pyplot import hist
#from matplotlib.figure import Figure
import seaborn as sns

from statsmodels.stats import proportion as proptests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from time import time
#% matplotlib inline

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_experiment_size(p_null, 
                       p_alt, 
                       alpha=0.05, 
                       beta=0.20,
                       verbose=False):
    '''This function takes a size of effect and returns the minimum number of 
    samples needed to achieve the desired power
    Inputs:
      - p_null (mandatory) - null hypothesis success rate (base) - (numpy Float)
      - p_alt (mandatory) - success rate (desired) - what we want to detect -
        (numpy Float)
      - alpha (optional) - Type-I (false positive) rate of error - (numpy Float -
        default = 5%)
      - beta (optional) - Type-II (false negative) rate of error - (numpy Fload -
        default = 20%)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - n - required number of samples for each group, in order to obtain the 
        desired power
    '''
    if verbose:
        print('###function experiment size started - Analytic solution')        
    begin = time()

    #takes z-scores and st dev -> 1 observation per group!
    z_null = stats.norm.ppf(1 - alpha)
    z_alt  = stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
    
    #calculate the minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
    
    n_max = np.ceil(n)
    
    end = time()
    if verbose:
        print('elapsed time: {:.5f}s'.format(end-begin))
        print('experiment size:', n_max)

    return n_max

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_peeking_sim(alpha=0.05, 
                   p=0.5, 
                   n_trials=1000,
                   n_blocks=2,
                   n_sims=10_000,
                   verbose=False):
    '''This function aims to simulate the rate of Type I error (false positive) 
    produced by the early stopping decision. It is based on a significant result
    when peeking ahead.
    Inputs:
        - alpha (optional) - Type I error rate that was supposed
        - p (optional) - probability of individual trial success
        - n_trials (optional) - number of trials in a full experiment
        - n_blocks (optional) - number of times data is looked at (including end)
        - n_sims: Number of simulated experiments run
    Output:
        p_sig_any: proportion of simulations significant at any check point, 
        p_sig_each: proportion of simulations significant at each check point
    '''
    if verbose:
        print('###function peeking sim started')        
    begin=time()
    
    #generate the data
    trials_per_block = np.ceil(n_trials / n_blocks).astype(int)
    data = np.random.binomial(trials_per_block, p, [n_sims, n_blocks])
    
    #put the data under a standard
    data_cumsum = np.cumsum(data, axis = 1)
    block_sizes = trials_per_block * np.arange(1, n_blocks+1, 1)
    block_means = block_sizes * p
    block_sds   = np.sqrt(block_sizes * p * (1-p))
    data_zscores = (data_cumsum - block_means) / block_sds
    
    #results
    z_crit = stats.norm.ppf(1-alpha/2)
    sig_flags = np.abs(data_zscores) > z_crit
    p_sig_any = (sig_flags.sum(axis = 1) > 0).mean()
    p_sig_each = sig_flags.mean(axis = 0)
    
    tuple = (p_sig_any, p_sig_each)
    
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return tuple

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot(first_graph, 
            second_graph=False,
            aux=False,
            type='none',
            verbose=False):
    '''This function plots the distribution for Power
    Inputs:
      - null_dist (mandatory) - 
      - alt_dist (mandatory) -
      - p_crit (mandatory) - 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output - True, if everything goes well - this is a plot function only!
    '''
    if verbose:
        print('###function plot started')        
    
    #making the plot
    mstyles.use('seaborn-darkgrid')#ggplot') #dark_background')
    fig_zgen = mpyplots.figure() #creating the object    
    axis_zgen = fig_zgen.add_axes([0,0,1,1]) #X0 y0 width height
    
    if type == 'htest': #histogram for h0 h1 test
        #assertions
        assert aux > 0. #aux receives p_critical
        #assert second_graph
        if verbose:
            print('plotting hypothesis test')
        #preprocessing
        low_bound = first_graph.ppf(.01) #null hypothesis distribution
        high_bound = second_graph.ppf(.99) #alternative hypothesis distribution
        x = np.linspace(low_bound, high_bound, 201)
        y_null = first_graph.pdf(x) #null
        y_alt = second_graph.pdf(x) #alternative
        #plotting
        axis_zgen.plot(x, y_null)
        axis_zgen.plot(x, y_alt)
        axis_zgen.vlines(aux, 
                         0, 
                         np.amax([first_graph.pdf(aux), second_graph.pdf(aux)]),
                         linestyles = '--', color='red')
        axis_zgen.fill_between(x, y_null, 0, where = (x >= aux), alpha = .5)
        axis_zgen.fill_between(x, y_alt , 0, where = (x <= aux), alpha = .5)
        axis_zgen.legend(labels=['null hypothesis','alternative hypothesis'], fontsize=12)
        title = 'Hypothesis Test'
        x_label = 'difference'
        y_label = 'density'
        
    elif type == 'hist': #time count histogram
        if verbose:
            print('plotting data histogram')
        n_bins = np.arange(0, first_graph.max()+400, 400)
        mpyplots.hist(first_graph, 
                      bins = n_bins)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
    
    elif type == '2hist':
        #assertions
        #assert second_graph
        #assert aux == data['time'] #aux receives data['time']
        counts1 = first_graph
        counts2 = second_graph
        if verbose:
            plot('plotting test (control and experiment) histograms')
        #plotting
        borders = np.arange(0, aux.max()+400, 400)
        mpyplots.hist(counts1, alpha=0.5, bins=borders)
        mpyplots.hist(counts2, alpha=0.5, bins=borders)
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
        
    elif type == 'stest':
        #assertions
        #assert second_graph
        #assert aux == data['day'] 
        if verbose:
            print('plotting signal test (control and experiment) graphs')
        #preprocessing
        x=aux
        y_control=first_graph
        y_experiment=second_graph
        #plotting
        axis_zgen.plot(x, y_control)
        axis_zgen.plot(x, y_experiment)      
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Signal Test'
        x_label = 'day of experiment'
        y_label = 'success rate'
        
    else:
        raise Exception('type of graph invalid or not informed')
    
    fig_zgen.suptitle(title, fontsize=14, fontweight='bold')
    mpyplots.xlabel(x_label, fontsize=14)
    mpyplots.ylabel(y_label, fontsize=14)
    mpyplots.show()
    
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_power(p_null, 
             p_alt, 
             n, 
             alpha=0.05, 
             plot=False,
             verbose=False):
    '''This function takes an alpha rate and computes the power of detecting the 
    difference in two populations.The populations can have different proportion 
    parameters.
    Inputs:
      - p_null (mandatory) - rate of success (base) under the Null hypothesis
        (numpy Float) 
      - p_alt (mandatory) -  rate of sucess (desired) must be larger than the
        first parameter - (numpy Float)
      - n (mandatory) - number of observations for each group - (integer)
        alpha (optional) - rate of Type-I error (false positive-normally the
        more dangerous) - (numpy Float - default 5%)
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
        power - the power for detection of the desired difference under the 
        Null Hypothesis.
    '''
    if verbose:
        print('###function power started - works by Trial  & Error')        
    begin = time()
    
    #the idea: start with the null hypothesis. Our main target is to find 
    #Type I errors (false positive) trigger (critical value is given by
    #Alpha parameter - normally 5%).
    
    #se_null → standard deviation for the difference in proportions under the
    #null hypothesis for both groups
    #-the base probability is given by p_null
    #-the variance of the difference distribution is the sum of the variances for
    #-the individual distributions
    #-for each group is assigned n observations.
    se_null = np.sqrt((p_null * (1-p_null) + p_null * (1-p_null)) / n)
    #null_dist → normal continuous random variable (form Scipy doc)
    null_dist = stats.norm(loc=0, scale=se_null)

    #p_crit: Compute the critical value of the distribution that would cause us 
    #to reject the null hypothesis. One of the methods of the null_dist object 
    #will help you obtain this value (passing in some function of our desired 
    #error rate alpha). The power is the proportion of the distribution under 
    #the alternative hypothesis that is past that previously-obtained critical value.
    p_crit = null_dist.ppf(1-alpha) #1-alpha=95%
    
    #se_alt: Now it's time to make computations in the other direction. 
    #This will be standard deviation of differences under the desired detectable 
    #difference. Note that the individual distributions will have different variances 
    #now: one with p_null probability of success, and the other with p_alt probability of success.
    se_alt  = np.sqrt((p_null * (1-p_null) + p_alt  * (1-p_alt)) / n)

    #alt_dist: This will be a scipy norm object like above. Be careful of the 
    #"loc" argument in this one. The way the power function is set up, it expects 
    #p_alt to be greater than p_null, for a positive difference.
    alt_dist = stats.norm(loc=p_alt-p_null, scale=se_alt)

    #beta → Type-II error (false negative) - I fail to reject the null for some
    #non-null states
    beta = alt_dist.cdf(p_crit)    
    
    if plot:
        fn_plot(first_graph=null_dist, 
                second_graph=alt_dist,
                aux=p_crit,
                type='htest',
                verbose=verbose)
        
    power = (1 - beta)
    end = time()
    if verbose:
        print('power coefficient: {:.4f}'.format(power))
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return power

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_ci(data, 
                   q, 
                   c=0.95, 
                   n_trials=1000,
                   verbose=False):
    '''This function takes a quartile for a data and returns a confidence 
    interval, using Bootstrap method.
    Inputs:
      - data (mandatory) - a series of numpy Float data to be processed - it
        can be a Pandas Series - (numpy Array)
      - q: quantile to be estimated, must be between 0 and 1
      - c: confidence interval width
      - n_trials (optional) - the number of samples that bootstrap will perform
        (default=1000)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ci: upper an lower bound for the confidence interval (Tuple of numpy Float)
    '''
    if verbose:
        print("###function quantile ci started - Bootstrapping method")        
    begin=time()

    #sample quantiles for bootstrap
    n_points = data.shape[0]
    sample_qs = []
    
    #loop for each bootstrap element
    for _ in range(n_trials):
        #random sample for the data (with replacement)
        sample = np.random.choice(data, n_points, replace = True)
        
        #desired quantile
        sample_q = np.percentile(sample, 100 * q)
        
        #append to the list of sampled quantiles
        sample_qs.append(sample_q)
        
    #confidence interval bonds
    lower_limit = np.percentile(sample_qs, (1 - c)/2 * 100)
    upper_limit = np.percentile(sample_qs, (1 + c)/2 * 100)
    
    tuple = (lower_limit, upper_limit)
    
    end = time()
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))

    return (lower_limit, upper_limit)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_permutation_test(x, 
                                 y, 
                                 q, 
                                 alternative='less',
                                 n_trials=10_000,
                                 verbose=False):
    '''this function takes a vector of independent feature, another of dependent
    feature and calculates a confidence interval for a quantile of a dataset.
    It uses a Bootstrap method.
    Inpus:
      - x (mandatory) - a vector containing zeroes and 1 values, for the 
        independent (to be grouped) feature - (Boolean)
      - y (mandatory) - a vector containing zeroes and 1 values, for the 
        dependent (output) feature
      - q (mandatory) - a verctor containing zeroes and 1 valures for the output
        quantile
      - alternative (optional) - please inform the type of test to be performed
        (possible: 'less' and 'greater') - (default='less')
      - n_trials (optional) number of permutation trials to perform  
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - p - the estimated p-value of the test (numpy Float)
    '''
    if verbose:
        print("###function quantile permutation test - Bootstrapping method")        
    begin=time()
    
    #initialize list for bootstrapped sample quantiles
    sample_diffs = []
    
    #loop on trials
    for _ in range(n_trials):
        #permute the grouping labels
        labels = np.random.permutation(y)
        
        #difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q  = np.percentile(x[labels == 1], 100 * q)
        
        #add to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)
    
    #observed statistic for the difference
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q  = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q
    
    #p-value for the result
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()
    
    p = hits / n_trials
    
    end = time()
    if verbose:
        print('estimated p for the test: {:.4f}'.format(p))
        print('elapsed time: {:.3f}s'.format(end-begin))
    
    return p

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_ranked_sum(x, 
                  y, 
                  alternative='two-sided',
                  verbose=False):
    '''This function returns a p-value for a ranked-sum test. It is presumed 
    that there are no ties.
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory)  - a vector of numpy Float, as the second entry
      - alternative (optional) - the test to be performed (options:'two-sided', 
        'less', 'greater') (default='two-sided')
    Output:
      - an estimative for p-value for the ranked test
    '''
    if verbose:
        print('###function ranked sum started')        
    begin=time()
    
    #U
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties
    
    #z-score
    n_1 = x.shape[0]
    n_2 = y.shape[0]
    mean_u = n_1 * n_2 / 2
    sd_u = np.sqrt( n_1 * n_2 * (n_1 + n_2 + 1) / 12 )
    z = (u - mean_u) / sd_u
    
    #rules for the p-value, according to the test
    if alternative == 'two-sided':
        p = 2 * stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p = stats.norm.cdf(z)
    elif alternative == 'greater':
        p = stats.norm.cdf(-z)
        
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return p

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_read_data(filepath, 
                 index_col='id',
                 verbose=False):
    '''This function reads a .csv file
    Inputs:
      - filepath (mandatory) - String containing the full path for the data to
        oppened
      - index_col (optional) - String containing the name of the index column
        (default='id')
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - Pandas Dataframe with the data
    *this function came from my library udacourse2.py and was adapted for this
    project
    '''
    if verbose:
        print('*subfunction read_data started')
    
    #reading the file
    df = pd.read_csv(filepath)
    df.set_index(index_col)
    
    if verbose:
        print('file readed as Dataframe')

    #testing if Dataframe exists
    #https://stackoverflow.com/questions/39337115/testing-if-a-pandas-dataframe-exists/39338381
    if df is not None: 
        if verbose:
            print('dataframe created from', filepath)
            #print(df.head(5))
    else:
        raise Exception('something went wrong when acessing .csv file', filepath)
    
    #setting a name for the dataframe (I will cound need to use it later!)
    ###https://stackoverflow.com/questions/18022845/pandas-index-column-title-or-name?rq=1
    #last_one = filepath.rfind('/')
    #if last_one == -1: #cut only .csv extension
    #    df_name = filepath[: -4] 
    #else: #cut both tails
    #    df_name = full_path[last_one+1: -4]   
    #df.index.name = df_name
    #if verbose:
    #    print('dataframe index name setted as', df_name)

    return df

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_sign_test(x, 
                 y, 
                 alternative='two-sided',
                 verbose=False):
    '''This function returns a p-value for a ranked-sum test. It is presumed
    that there are no ties.
    Input parameters:
      - x  1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory)  - a vector of numpy Float, as the second entry
      - alternative (optional) - the test to be performed (options:'two-sided', 
        'less', 'greater') (default='two-sided')
    Output:
      - an estimative for p-value for the sign test
    '''
    if verbose:
        print('###function sign test started')
    begin=time()
   
    # compute parameters
    n = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p = min(1, 2 * stats.binom(n, 0.5).cdf(min(k, n-k)))
    if alternative == 'less':
        p = stats.binom(n, 0.5).cdf(k)
    elif alternative == 'greater':
        p = stats.binom(n, 0.5).cdf(n-k)

    end = time()
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))
   
    return p
