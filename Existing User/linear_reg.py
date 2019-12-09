#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:57:45 2019

@author: liujiyuan
"""

def linear_regression_for_all (train,test,p,q,bi,bit,bu,pp,mu):
    predictors_test = np.zeros((test.shape[0],4))
    predictors_test = pd.DataFrame(predictors_test)
    predictors_test.columns = np.array(['pq','bi+bibint','bu','post_processing'])
    
    predictors_train = np.zeros((train.shape[0],4))
    predictors_train = pd.DataFrame(predictors_train)
    predictors_train.columns = np.array(['pq','bi+bibint','bu','post_processing'])
    
    
  
    for i in range(test.shape[0]):
        obs = test.iloc[i,:]
        predictors_test.iloc[i,0] = float(np.array(p.loc[:,[obs.userId]].T.dot(q.loc[:,[obs.movieId]])))
        predictors_test.iloc[i,1] = float(np.array(bi.loc[:,[obs.movieId]])) + float(np.array(bit.loc[(bit['movieId']== obs.movieId) & (bit['bin'] == obs.bin),'bias']))
        predictors_test.iloc[i,2] = float(np.array(bu.loc[:,[obs.userId]]))
        predictors_test.iloc[i,3] = float(np.array(pp.iloc[int(np.array(obs.userId-1)),int(obs['index'])]))
    
    for i in range(train.shape[0]):
        obs = train.iloc[i,:]
        predictors_train.iloc[i,0] = float(np.array(p.loc[:,[obs.userId]].T.dot(q.loc[:,[obs.movieId]])))
        predictors_train.iloc[i,1] = float(np.array(bi.loc[:,[obs.movieId]])) + float(np.array(bit.loc[(bit['movieId']== obs.movieId) & (bit['bin'] == obs.bin),'bias']))
        predictors_train.iloc[i,2] = float(np.array(bu.loc[:,[obs.userId]]))
        predictors_train.iloc[i,3] = float(np.array(pp.iloc[int(np.array(obs.userId-1)),int(obs['index'])]))
    
     
     #predictors_test =  test.apply(get_predictors,1)
    y_train = train['rating'] - mu
    #y_train = np.array(y_train)
    y_test = test['rating']
    y_test = np.array(y_test)
     
    regressor = LinearRegression()
    regressor.fit(predictors_train.iloc[:,], y_train)
       
    print(regressor.intercept_)
    print(regressor.coef_)
     
    y_train_estimate = regressor.predict(predictors_train.iloc[:,]) 
    y_train_estimate[y_train_estimate > 5] = 5
    y_train_estimate[y_train_estimate < 0] = 0
     
    y_test_estimate = regressor.predict(predictors_test.iloc[:,]) + mu
    y_test_estimate[y_test_estimate > 5] = 5
    y_test_estimate[y_test_estimate < 0] = 0
     
    def RMSE(estimation, truth):
    ###Root Mean Square Error
        estimation = np.float64(estimation)
        truth = np.float64(truth)
        num_sample = estimation.shape[0]
    
    # sum square error
        sse = np.sum(np.square(truth - estimation))
        return np.sqrt(np.divide(sse, num_sample - 1))
    
    train_RMSE_final = RMSE(y_train_estimate,y_train)
    
    test_RMSE_final = RMSE(y_test_estimate,y_test)
    
    print("train RMSE:", train_RMSE_final, "\t")
    print("test RMSE:", test_RMSE_final, "\t")    