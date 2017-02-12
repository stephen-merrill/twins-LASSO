import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

twinData = pd.read_csv("C:/Users/J40311/Documents/School/495R/twins.txt")
twinData.head()

twinData.dtypes

twinData.columns

### make sure missing data read in as missing
twinData[['DLHRWAGE','EDUCH']]
twinData[['DLHRWAGE']]
twinData = twinData.dropna()
twinData[['DLHRWAGE']]

# remove rows with missing data (regression will fail to run with missing data)
twinData = pd.read_csv("C:/Users/J40311/Documents/School/495R/twins.txt", na_values=["."])
twinData = twinData.dropna()
twinData[['DLHRWAGE']]
twinData.DLHRWAGE

# check normality of response variable (need to drop missing data to generate)
import matplotlib.pyplot as plt
plt.hist(twinData.DLHRWAGE.dropna())
plt.hist(twinData.DLHRWAGE,50)
plt.show()

### basic linear regression (without variable selection)
import statsmodels.api as sm

# if I needed to convert one of my variables to factors, could do so
twinData.MALEL
twinData['MALEL'] = pd.Categorical(twinData.MALEL).codes
X = twinData.drop('DLHRWAGE',axis=1)
X.columns
y = twinData[['DLHRWAGE']]

# include intercept in model
X1 = sm.add_constant(X)

model = sm.OLS(y, X1).fit()
model.summary()

recode1 = {-1:0, 0:0, 1:1}
recode2 = {-1:0, 0:1, 1:0}
recode3 = {-1:1, 0:0, 1:0}

# transform levels of categorical variablies into 0/1 dummy variables
twinData['DMARRIED1'] = twinData.DMARRIED.map(recode1)
twinData['DMARRIED0'] = twinData.DMARRIED.map(recode2)
twinData.head(10)

twinData['DUNCOV1'] = twinData.DUNCOV.map(recode1)
twinData['DUNCOV0'] = twinData.DUNCOV.map(recode2)
twinData.head(10)

#select predictor variables and target variable as separate data sets  
predvar= twinData[['DEDUC1','AGE','AGESQ','WHITEH','MALEH','EDUCH','WHITEL','MALEL','EDUCL','DEDUC2','DTEN','DMARRIED0','DMARRIED1','DUNCOV0','DUNCOV1']]

target = twinData.DLHRWAGE
 
# standardize predictors to have mean=0 and sd=1 (required for LASSO)
predictors=predvar.copy()
from sklearn import preprocessing
predictors['DEDUC1']=preprocessing.scale(predictors['DEDUC1'].astype('float64'))
predictors['AGE']=preprocessing.scale(predictors['AGE'].astype('float64'))
predictors['AGESQ']=preprocessing.scale(predictors['AGESQ'].astype('float64'))
predictors['WHITEH']=preprocessing.scale(predictors['WHITEH'].astype('float64'))
predictors['MALEH']=preprocessing.scale(predictors['MALEH'].astype('float64'))
predictors['EDUCH']=preprocessing.scale(predictors['EDUCH'].astype('float64'))
predictors['WHITEL']=preprocessing.scale(predictors['WHITEL'].astype('float64'))
predictors['MALEL']=preprocessing.scale(predictors['MALEL'].astype('float64'))
predictors['EDUCL']=preprocessing.scale(predictors['EDUCL'].astype('float64'))
predictors['DEDUC2']=preprocessing.scale(predictors['DEDUC2'].astype('float64'))
predictors['DTEN']=preprocessing.scale(predictors['DTEN'].astype('float64'))
predictors['DMARRIED0']=preprocessing.scale(predictors['DMARRIED0'].astype('float64'))
predictors['DMARRIED1']=preprocessing.scale(predictors['DMARRIED1'].astype('float64'))
predictors['DUNCOV0']=preprocessing.scale(predictors['DUNCOV0'].astype('float64'))
predictors['DUNCOV1']=preprocessing.scale(predictors['DUNCOV1'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, resp_train, resp_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)

# specify the lasso regression model
# precompute=True helpful for large data sets
model=LassoLarsCV(cv=10, precompute=True).fit(pred_train,resp_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca() # set up axes
plt.plot(m_log_alphas, model.coef_path_.T) # alpha on x axis, change in regression coefficients on y axis
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV') 
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.savefig('Fig01')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
plt.savefig('Fig02')
         
# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(resp_train, model.predict(pred_train))
test_error = mean_squared_error(resp_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,resp_train)
rsquared_test=model.score(pred_test,resp_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
