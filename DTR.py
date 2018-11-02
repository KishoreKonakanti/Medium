import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.datasets import load_diabetes

diaData = load_diabetes()
df = pd.DataFrame(data=diaData['data'], columns=diaData['feature_names'])

features = list(diaData['feature_names'])

X = df[features]
Y= diaData.target

lscores = list()
dscores = list()
ascores = list()
bscores = list()
rscores = list()

num_iters = 1000

for i in range(num_iters):
    print('Iteration :',i)
    (XTrain, XTest, YTrain, YTest) = train_test_split(X, Y, test_size=20, random_state=123)    
# Linear Regressions
    lreg = LinearRegression()
    lreg.fit(XTrain, YTrain)
    
    # Decision Tree Regression
    dtr = DecisionTreeRegressor(max_leaf_nodes=34)
    dtr.fit(XTrain, YTrain)
       
    # ADA BOOST REGRESSION
    sen = AdaBoostRegressor(n_estimators=200)
    sen.fit(XTrain, YTrain)         
    
    # Bagging Regression
    breg = BaggingRegressor(n_estimators=100)
    breg.fit(XTrain, YTrain)
    
    # Random Forest
    rfreg = RandomForestRegressor(n_estimators=10)
    rfreg.fit(XTrain, YTrain)
    
    dscores.append(dtr.score(XTest, YTest) * 100)
    bscores.append(breg.score(XTest, YTest) * 100)
    ascores.append(sen.score(XTest, YTest) * 100)
    rscores.append(rfreg.score(XTest, YTest) * 100)
    lscores.append(lreg.score(XTest, YTest) * 100)

plt = matplotlib.pyplot
plt.figure(figsize=(15,15))
plt.scatter(range(num_iters), dscores, color='k', label='Decision Tree Regressor')
plt.scatter(range(num_iters), lscores, color='b', label='Linear Regressor')
plt.scatter(range(num_iters), rscores, color='r', label='Random Forests')
plt.scatter(range(num_iters), ascores, color='g', label='ADA Boost')
plt.scatter(range(num_iters), bscores, marker=r'$\clubsuit$', label='Bagged Regressor')
plt.legend(loc='upper right', shadow=True)
plt.title('Comparison of Regressors')

plt.scatter(range(num_iters), dscores, linewidths=0.3)
plt.scatter(range(num_iters), ascores, marker='-')
plt.scatter(range(num_iters), bscores, marker='^')
plt.scatter(range(num_iters), rscores, marker='*')
plt.scatter(range(num_iters), lscores, marker='#')
            
plt.show()