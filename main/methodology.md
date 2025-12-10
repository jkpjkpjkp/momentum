2.2. Methodology
In this paper, we consider stock returns and their decomposition in every intraday period. Such intervals are
common in intraday studies such as Heston et al. (2010)
and many others. Our first period (i � 1) is the overnight
interval from the market close of the previous day to
1000 hours, where the ending time is chosen to ensure
that almost all securities are traded at least once by then.
The next period (i � 2) is from 1000 to 1030 hours and so
on until the last period (i � 13), which is between 1530
and 1600 hours. We obtain systematic decomposition
from running the following cross-sectional regression:
RETs, d,i � αd,i +
X
p
j�1
Cs, d�1,jθd,i,j + ɛs, d,i, (2)
where RETs, d,i is the return of stock s on day d in intraday period i, αd, i is the intercept on day d in period i,
Cs, d�1,j is the standardized anomaly j of stock s observable at the market close of day d � 1, and ɛs, d,i is the
residual. The unknown slope estimates θd,i,j can be
interpreted as factor returns (Fama and French 2020),
and so the second term captures the total systematic
component of the stocks at intraday frequency.12
Using the estimated coefficients θˆ
d,i,j, we can then
decompose the raw return of stock s on d in intraday
period i into two parts:
RETs, d,i � SYSs, d,i + RESs, d,i, (3)
where
SYSs, d,i �
X
p
j�1
Cs, d�1,jθˆ
d,i,j, (4)
is the estimated systematic component explained by common systematic factors, which is simply referred to as
SYS.
13
It is important to point out that for simplicity we refer
to the total systematic component, the right-hand side
of Equation (4), as SYS. This is consistent with Fama
and French (1993) and Stambaugh et al. (2012), although
they focus on the time series version of the factors. Our
definition of SYS is quite general and can include any
cross-sectional tradable or nontradable factors that contribute systematically to expected returns. In particular,
SYS can include both risk-based and behavioral factors.
As argued by Kozak et al. (2018), factor covariances
should explain cross-sectional variation in expected
returns even in a model of sentiment-driven asset
prices, because time-varying investor sentiment can
give rise to an Intertemporal Capital Asset Pricing
Model-like Stochastic discount factor. As a result, the
SYS component can reflect both compensation for risk
and exposure to (systematic) mispricing correction.
The residual part RESs, d,i � RETs, d,i � SYSs, d,i captures the return component unexplained by the factors,
including the intercept, which can be viewed as alpha
relative to cross-sectional factors. Although there is an
extensive literature on various properties of alpha from
time series regressions, there is little analysis on SYS, a
measure that reflects the total contribution of various
systematic factors.
Our objective is to study the properties of SYS and its
predictive power on future returns. We exploit its predictive power as follows. At the beginning of each intraday period i (i � 1,:: :, 13) on each day d, we sort stocks
into 10 portfolios based on their realized SYS values
available at the time (i.e., the systematic component of
returns in the previous intraday period). We buy stocks
in the top decile with high SYS values and short those in
the bottom decile with low SYS values. We hold this
long-short value-weighted portfolio during period i on
day d, resulting in the systematic portfolio return:
RMd,i � Rd,i
10 � Rd,i
1 , i � 1, 2, :::, 13, (5)
where Rd,i
1 and Rd,i
10 are the returns of decile portfolios 1
and 10 during period i on day d, respectively. As a result,
we have 13 long-short portfolios per day corresponding
to the 13 intraday intervals. Each of these portfolios
enters position at the beginning of the corresponding
intraday period and exits position at the end of the corresponding intraday period (i.e., rebalancing once per
day). Then for each intraday period i, we have a time
series of such systematic long-short portfolio over trading days. Besides the 13 long-short portfolios that are
rebalanced once per day at a fixed time of the day and
are held over one intraday period, we also consider
a systematic long-short portfolio investing in all 13
SYS signals and obtain its time series returns. For
Li, Yuan, and Zhou: Systematic Momentum
Management Science, Articles in Advance, pp. 1–25, © 2025 INFORMS 5
Downloaded from informs.org by [144.48.80.31] on 08 December 2025, at 23:08 . For personal use only, all rights reserved. 
each long-short portfolio, we further decompose the
holding-period return into systematic and residual components. These time series would allow us to examine
whether the systematic component itself exhibits momentum and if such systematic momentum can imply a
systematic return momentum.
We use each of the three anomaly sets as Cs, d�1, j and
estimate the slope coefficients in Equation (2) to obtain
the corresponding SYS. Panels A and B of Table A.3 in
the Online Appendix, respectively, report the correlations
among the 15 RP anomalies and the 15 EL anomalies. The
correlations are generally low, suggesting that multicollinearity is unlikely an issue when we use 15 anomalies
jointly to explain the cross-sectional returns of Russell 1000
stocks. Thus, for the first two anomaly sets, we run simple Ordinary Least Squares regressions to obtain the systematic decomposition. To improve the efficiency of the
slope estimators, we purposely use inverse variances as
regression weights.14
The third and larger set of 60 anomalies potentially contains more predictive information about future returns
while raising challenges of efficiently estimating the
increased number of unknown parameters. To alleviate
concerns about overfitting, we apply two solutions based
on machine learning dimension reductions. First, following Kozak et al. (2020) and Ehsani and Linnainmaa (2022),
we use 15 principal components (PCs) to reduce the
dimensionality. Specifically, we first fit the PCA model
with anomaly data from January 1970 to December 1992
and then construct 15 out-of-sample PCs between January
1993 and December 2020 matching the sample period in
our main analysis. Second, we use a penalized regression
with the LASSO method that encourages sparse estimates
of regression coefficients by introducing the L1 penalty.
Following Dong et al. (2022), we fit the LASSO model
period by period using the Akaike information criterion
(AIC). Such criteria are useful for selecting the value of the
regularization parameter by making a tradeoff between
the goodness of fit and the complexity of the model.15