Kristine karimi  Data science and machine learning projects
   Statistics Tutorial for Data Science
      Statistics Tutorial with Python

      One of the basic conditions  to become a data scientist is the basic 

understanding  of statistics. From simple concepts such as mean ,median,standard 

deviations to complex statistical concepts such as  probability mass functions,

Normal distributions , uniform distribution and may more. But  lucky you have 
 
spent the last three years studying those concepts as undergraduate student   doing 

cats and main exams for those concepts. So  honesty speaking ,you are at the right 

hand if you want to master those concepts. Let  get started!



A discrete variable is a variable that can take only the countable numbers of the 

values. One example of the discrete variable is the outcome of the dice. That is ,if 
 
the outcome of events is 1 to 10 it   shows it  a discrete  variable which ranges

 between    1 to 10.

Continuous variable  are variables that take  uncountable number of values. A 

good example of this is  length .


In statistic we represent  discrete distribution with PMF(Probability mass function)

and CDF(cumulative Distribution function).

While continuous function we represent with PDF (probability density function) .




PMF defines the probability of all possible values of y of a random number while

PDF represent  probability of all possible values of continuous values.

To understand this concepts better let visualize it . 

PDF of  normal distribution with mean o and standard 

deviation 1

#importing the requires libraries

import os
import numpy as np
import pandas as pd
from math import sqrt
from pylab import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
# Statistics
from statistics import median
from scipy import signal
from scipy.special import factorial
import scipy.stats as stats
from scipy.stats import sem, binom, lognorm, poisson, bernoulli, spearmanr
from scipy.fftpack import fft, fftshift


# PDF of  normal distribution with mean o and standard 

deviation 1

# Plot normal distribution
mu = 0
variance = 1
sigma = sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.figure(figsize=(16,5))
plt.plot(x, stats.norm.pdf(x, mu, sigma), label='Normal Distribution')
plt.title('Normal Distribution with mean = 0 and std = 1')
plt.legend(fontsize='xx-large')
plt.grid()
plt.show()

From the above code we have our normal distribution 

with mean 0 and std 1 of pdf.

PMF (Probability Mass Function)

Let visualize the pmf  of binomial distribution  for

number of values between 52 and 57

# PMF Visualization
n = 200
p = 0.5
plt.style.use('dark_background')
fig, ax = plt.subplots(1, 1, figsize=(17,5))
x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='Binomial PMF')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)
rv = binom(n, p)
#ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1, label='frozen PMF')
ax.legend(loc='best', frameon=False, fontsize='xx-large')
plt.title('PMF of a binomial distribution (n=200, p=0.5)', fontsize='xx-large')
plt.grid()
plt.show()

With those two visualizations we got a better

understanding of the PDF and PMF.

I understand you are a bit  lost if you are new to 

statistic and  you got no idea ,what the heck is normal 

distribution and binomial distribution, but let me 

give you some clarity .

Normal Distribution
Normal distribution is also called  Gaussian distribution or bell curve 
 
has my lecturer used to call it back in  1.2 . In normal distribution

 the data is symmetrically distributed with no skew has  you have

 seen in the above diagram of  of pdf of normal.  If you have plotted

 it. Which  I highly recommend you to  do . In normal distribution  

most of the  values  cluster around the central  region with values

 tapering  off  as they go further  away from the center.  A prefect

 example of this is how the marks of any exam  are distributed in a

 given class .Very few get very good marks in this case

 70+  ,majority lies between 40 to 69 and very few get 39 and

 below . And that how  normal distribution  operates .The measure

 of central tendency are mean  mode an  median.

Let plot a scatter plot for normal distribution.

# Generate Normal Distribution
normal_dist = np.random.randn(200)
normal_df = pd.DataFrame({'value' : normal_dist})
# Create a Pandas Series for easy sample function
normal_dist = pd.Series(normal_dist)

normal_dist2 = np.random.randn(200)
normal_df2 = pd.DataFrame({'value' : normal_dist2})
# Create a Pandas Series for easy sample function
normal_dist2 = pd.Series(normal_dist)

normal_df_total = pd.DataFrame({'value1' : normal_dist, 
                                'value2' : normal_dist2})


 #Scatterplot
plt.figure(figsize=(15,5))
sns.scatterplot(data=normal_df)
plt.legend(fontsize='xx-large')
plt.title('Scatterplot of a Normal Distribution', fontsize='xx-large')

To show you the bell curve my lecturer  used to  talk 

about let plot  distplot  for better understanding.
# Normal Distribution as a Bell Curve
plt.figure(figsize=(18,5))
sns.distplot(normal_df)
plt.title('Normal distribution (n=1000)', fontsize='xx-large')
plt.grid()
plt.show()

If you have plotted distplot above ,you now know why my

 lecturer loved to call it  bell curve.

Binomial Distribution.
Binomial distribution has a countable number of outcomes and thus
making it discrete has we saw above in pmf.


Binomial distribution must meet three condition which are :
    I.   The number of observations are limited. 
    II.  Each of the observations are  independent.
    III.  The probability of success is the same for all the trials.
Another distribution worthy mentioning here is  Bernoulli
 Distribution.
Bernoulli distribution is a special kind of binomial  distribution with 0 
and 1 has it values.

Let plot them 
# Change of heads (outcome 1)
p = 0.6

# Create Bernoulli samples
bern_dist = bernoulli.rvs(p, size=1000)
bern_df = pd.DataFrame({'value' : bern_dist})
bern_values = bern_df['value'].value_counts()
# Plot Distribution
plt.figure(figsize=(18,4))
bern_values.plot(kind='bar', rot=0)
plt.annotate(xy=(0.85,300), 
             s='Samples that came up Tails\nn = {}'.format(bern_values[0]), 
             fontsize='large', 
             color='white')
plt.annotate(xy=(-0.2,300), 
             s='Samples that came up Heads\nn = {}'.format(bern_values[1]), 
             fontsize='large', 
             color='white')
plt.title('Bernoulli Distribution: p = 0.6, n = 1000')
plt.grid()
plt.plot()

Poisson Distribution
Poisson distribution  is as discrete probability distribution of a given  
 number of events  occurring  in a given time period. 
And here is the plot.
x = np.arange(0, 20, 0.1)
y = np.exp(-5)*np.power(5, x)/factorial(x)

plt.figure(figsize=(15,8))
plt.title('Poisson distribution with lambda=5', fontsize='xx-large')
plt.plot(x, y, 'bs')
plt.show()

To check out the code  visit : https://github.com/tinajs2018/-Statistics-Tutorial-for-Data-Science-Statistics-Tutorial-with-Python

With that ,let practice more on those distributions .In  part two of 
basic statistic with python we shall dig deeper .
Stay safe and don’t forget to follow me for weekly  updates.










Statistics Tutorial for Data Science part two
Statistics Tutorial with Python

Part  1 we looked at various  distribution with their 

source  code .Today we shall explore more and look at

1 Summary Statistics and Moments

2    Sampling methods

3   Covariance

4  Bias, MSE and SE

5  Correlation

6 Linear regression

This article might a bit longer   so fasten your seatbelts  and let take a ride into statistic with

python.

Summary Statistics
In summary statistic we shall talk about mean  median  and mode. All these three are central 

Tendency measures.

Mean is also also called the first moment .Mean is  balance point has statistician like to call it .

Median is the  middle value when ordered. N+1/2 postion.

Mode  is  most frequent values .

When you are doing  data analysis you  like to check the statistic summary of the data .

Which can be obtained my describe method

data.describe()

Moments
A moment is a quantitative measure that says something about the
 shape of a distribution. There are central moments and non-central
 moments. But in this  today we shall focus on central moments.
The 0th central moment is the total probability and is always equal to 1.
The 1st moment is the mean (expected value).
The 2nd central moment is the variance. Variance is average of the
 squared distance of the mean. Having have mentioned  variance ,
 standard deviation is The square root of the variance and this
 measures the spread of the distribution.
The 3rd central moment is the skewness. Skweness  is a measure
 that describes the contrast of one tail versus the other tail. For
 example  if our revenue  distribution is more spread to the right of 
normal curve we say the revenue distribution is skwened  to the
right.
The 4th central moment is the kurtosis. Kurtosis   is  a  statistical
measures used to describe the degree to which the score cluster in
 the tails or the peak of distributions.

 print('The first four calculated moments of a normal distribution')
# Mean
mean = data['revenue'].mean()
print('Mean: ', mean)
# Variance
var = np.var(data['revenue'])
print('Variance: ', var)
# Return unbiased skew normalized by N-1
skew = data['revenue'].skew()
print('Skewness: ', skew)
# Return unbiased kurtosis over requested axis using Fisher's definition of kurtosis 
# (kurtosis of normal == 0.0) normalized by N-1
kurt = data['revenue'].kurtosis()
print('Kurtosis: ', kurt)

Covariance
Covariance is a measure  of the relationship  between two random 
variables. It show how much two random variables vary together.
If two variables are independent, their covariance is 0. However, a
 covariance of 0 does not imply that the variables are independent.

# Covariance between Age and Income
print('Covariance between revenue and rank: ')

data[['revenue', 'rank']].cov()

Sampling methods.

Their are two types of sampling methods 
1 Non-Representative Sampling.
2 Representative Sampling.
In most  of statistic course they focus more on representative
 sampling. Hence we shall start with it.
Representative sampling include:
1 Simple Random Sampling  entails  picking  samples (psuedo)randomly.

2 Cluster Sampling  entails  dividing the population into groups (clusters) and pick samples from those groups.

3 Stratified Sampling  entails  picking the same amount of samples from different groups (strata) in the population.

4 Systematic Sampling entails picking samples with a fixed interval. For example every 5th sample (0, 5, 10, 15 etc.).

print('---Representative samples:---\n')
# Simple (pseudo)random sample
rand_samples = data['revenue'].sample(5)
print('Random samples:\n\n{}\n'.format(rand_samples))

# Make random clusters of ten people (Here with replacement)
p1 = data['revenue'].sample(5)
p2 = data['revenue'].sample(5)
p3 = data['revenue'].sample(5)
p4 =data['revenue'].sample(5)
p5 = data['revenue'].sample(5)

# Take sample from every cluster (with replacement)
clusters = [p1,p2,p3,p4,p5]
cluster_samples = []
for c in clusters:
    clus_samp = c.sample(1)
    cluster_samples.extend(clus_samp)
print('Cluster samples:\n\n{}'.format(cluster_samples))  

# Stratified Sampling
# We will get 1 revenue from every city in the dataset
# We have 8 companies so that makes a total of 8 samples


strat_samples = []

for company in data['company'].unique():
    samp = data[data['company'] == company].sample(1)
    strat_samples.append(samp['revenue'].item())
    
print('Stratified samples:\n\n{}\n'.format(strat_samples))
# Systematic sample (Every 2000th value)
sys_samples = data[data.index % 2000 == 0]
print('Systematic samples:\n\n{}\n'.format(sys_samples))

Non-Representative Sampling includes:
1 Purposive Sampling  which entails picking samples for a specific
 purpose.
2  Convenience Sampling  entails picking samples that are most convenient.
3 Haphazard Sampling  entails picking samples without thinking about them. 

print('---Non-Representative samples:---\n')

# Purposive samples (Pick samples for a specific purpose)
purp_samples = data['revenue'].nlargest(n=5)
print('Purposive samples:\n\n{}\n'.format(purp_samples))

# Convenience samples
con_samples = data[0:5]
print('Convenience samples:\n\n{}\n'.format(con_samples))

# Haphazard samples (Picking out some numbers)
hap_samples = [data['revenue'][16], data['revenue'][55], data['revenue'][58], data['revenue'][21], data['revenue'][20]]
print('Haphazard samples:\n\n{}\n'.format(hap_samples))


Bias, MSE and SE

Bias  is a measure of how far the sample mean deviates from the population mean. The sample mean is also called expected  mean.

# # Take sample
# sample1 = data['profit'].sample(100)

#  Calculate Expected Value (EV), population mean and bias
ev = sample1.mean()[0]
pop_mean = data['profit'].mean()[0]
bias = ev - pop_mean

print('Sample mean (Expected Value): ', ev)
print('Population mean: ', pop_mean)
print('Bias: ', bias)

M S E  measures how much estimators deviate from the true distribution.

Standard Error (SE) measures how spread the distribution is from
 the sample mean.
# Standard Error (SE)
uni_sample = data['profit'].sample(100)
norm_sample = data['revenue'].sample(100)

Correlation
Correlation is  a statistical relationship,whether causal or not between  two  random or bivariate  data.

# Correlation between two revenue and ceo_founder
# Using Pearson's correlation
print('Pearson: ')
data[['ceo_founder', 'revenue']].corr(method='pearson')

Linear regression
It is a linear approach to modelling the relationship between a scalar 
response  and one or more explanatory variables.

# Plot data with Linear Regression
plt.style.use('classic')
plt.figure(figsize=(16,5))
plt.title('Well fitted but not well fitting: Linear regression plot on quadratic data', fontsize='xx-large')
sns.regplot(data['prev_rank'], data['profit']

With that on statistic for data science you are at a better position  to 
mastering data science.

To check out the code  and data  , 
 visit :   https://github.com/tinajs2018/Statistics-Tutorial-for-Data-Science-part-two

If there is topic you would like me to cover on statistic for data science let me know  via the comment section

Stay safe and don’t forget to follow me for weekly  updates.





All the pandas functions  In data exploration you need to know.
Introduction to Pandas 
The Pandas library is built on NumPy and provides easy-to-use
data structures and data analysis tools for the Python
programming language.
Pandas  is used in multiple stages  of data analysis starting from data manipulations  to data analysis.
Each dataset in pandas is represented  in a tabular format know as  data frames.
In this tutorials we shall   look at all the pandas functions  from the basic function to advance .

First let import the vital libraries

#importing of the pandas 
 import pandas as pd.

#Read and Write to CSV
data=pd.read_csv('Billionaire.csv')
head() fuction is used  to see the first few  rows of the dataset.
data.head()

You can read the dataset  from various format such as  Excel

#Read and Write to xlsx
data=pd.read_csv('Billionaire.xlsx)
data.head()

tail () fuction is used to view the last rows of the data set.

Selection
Pandas as a function which enables selections of various rows and columns of the data set.
1  selection of one   element in a dataset.
   data[ 'Name']
2  selecting a subset of dataframe
    data[1:]
3  Select single value by row &column
   data.iloc[0,1]

  4  Select single value by row &column labels
   data.loc [[0],['NetWorth']]

Dropping.
Sometime in data analysis  we may wish to drop some rows and columns from 
our dataset for various reasons such as lots of missing values  and may more
They are two  ways you can drop the  values.
1   Drop values from rows (axis=0)
   data.drop([4, 1])

2   Drop values from columns(axis=1)
 data.drop('Country', axis=1)

Sorting and ranking of the data
1  Sort by labels along an axis.
data.sort_index()
2 Sort by the values along an axis.
data.sort_values(by='Source')

3  Assign ranks to entries
 data.rank()

Gaining  information of the data.
1  Getting the number of rows and columns in a data set
  data.shape
2  describe the index of the data frame
  data.index

3 Describe DataFrame columns available in the dataset
 data.columns
4  General information of the dataframe 
 data.info()
5 checking for Number of non-NA values
  data.count()

Getting the Summary of the data set
1 Sum of values
 data.sum()
2  Cummulative sum of values
  data['Rank'].cumsum()
3  Mimimum values
 data['NetWorth'].min()
4  # Maximum values
data['NetWorth'].max()
5  Summary statistics of the data
 data.describe()
6 Mean of values
data['Rank'].mean()

# Median of values
data['Rank'].median()

Missing values
1  checking for missing  values in the data frame
 data.isnull().sum()
2 Filing the missing a values
data.fillna('Unknown')

Renaming 
This function helps in changing the index name and the column name of the data
data.rename(columns={'Name':'Fullnames'},inplace=True)
data.head()

Combing  the data
pd.concat([‘state’,’region’)

Others
 1 checking the datatypes of the data
  data.dtypes
2  checking out the  unique distributions of the data
 data['Source'].unique
 3  Grouping the data in a dataframe
Best=data.groupby(['Rank']).sum()
Best
4 sorting data
 data.sort_values(by=(['NetWorth','Source']))
5Converting columns form one dataset to another.
 data.Rank.astype(int)

To check out the code  and data  , 
 visit :  https://github.com/tinajs2018/pandas-tutorial

Stay safe and don’t forget to follow me for weekly  updates


ALL Topics  for mathematics for machine learning.

Aspiring  to become  a guru in machine learning and data science,  fascinated by how 

data is being leverage in this era?  Well, you choose    the sexist career in the universal.

One of the  basic fundamentals of machine learning and data science is mathematics.

          No matter what kind of love- hate relationship you have  with mathematics ,you have 

to ace  mathematics  to  understand to    logic behind  machine learning algorithms.

The core concepts  used in mathematics and statistic  are  vital in  making   strategic  

decisions while  designing  machine learning models.

        One thing to keep in mind is that the company want someone who can solve  their 

problems with machine learning rather than someone  who  knows  mathematics concepts  

behind machine learning models. So ,just  know enough  to enable  you to navigate through 

the  various   problems.

 Below is  all the topics  you need  for machine learning.

1 Probability distibutions
                1  Discrete and continuous probability
          2  Gaussian distribution
          3 Construction of  probability space
          4 Summary  statistic   and independent

         5  Bayes Theorem
         6  Sum  Rule
         7 product rule
     
2 Analytical Geometry 

  1   Norms
   2  Inner products
   3 Orthogonality
   4 Orthogonality Basic
   5 Orthogonal Complement
   6 Orthogonal Projections.
   7Lenght and Distances
   8 inner products of functions

3 Linear Regression 

    1 Problem formulation

    2  Parameter Estimation
    3 Bayesian Regression

4 Linear Algebra

 1 Matrices
 2 Vector spaces 
3  Basic and Rank
4 Linear Equations
5 Linear independence
6 Linear mapping


5 Density Estimation
   1  Parameter learning
    2 Gaussian Mixture model
    3 Latent -Variables  perspective

6 Vector calculus:
 1 Gradients of matrices
  2 Partial Differentiation
  3  Back propagation and Automatic
 4 Useful  indentities for Computing  Gradient
 5 Higher -Order Derivatives

7 Principal Component Analysis
1 Problem Setting
2 Eigenvector computation and low Rank approximation
3 Maximum variance perspective
4 Projection Perspective 
5 PCA in high Dimensions

8 Matrix  Decomposition 

1 Determinant and Trace
2 Eigenvalues and Eigenvectors
3 Singular  Value Decomposition
4 matrix Approximation
5 Eigen composition  and Diagonalozation.


With that Your are one step closer to your dream  job. Just know the basics and move on to

 the  practical part of using the knowledge  acquired to solve real life problems.


Stay safe and don’t forget to follow me for weekly  updates and enjoy the process.


  


   









.
