import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

boston=load_boston()
boston

print(boston.DESCR)

boston.keys()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

boston_df['Price']=boston.target
boston_df
boston_df.info()

boston_df.columns
boston_df.corr

from sklearn.model_selection import train_test_split

X=boston_df.drop(["AGE","Price"],axis=1)
y=boston_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
lm.coef_

coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=["coffecient"])
coeff_df

y_pre=lm.predict(X_test)
y_test-y_pre

import matplotlib
import seaborn as sns

sns.distplot((y_test-y_pre))

from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pre))
print(metrics.mean_squared_error(y_test,y_pre))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pre)))
