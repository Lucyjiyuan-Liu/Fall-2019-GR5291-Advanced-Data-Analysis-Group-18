#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:12:55 2019

@author: liujiyuan
"""



    
def RMSE(estimation, data):
###Root Mean Square Error
    import numpy as np
    
    
    def sqr_err(obs):
        sqr_error=(obs['rating']-estimation[int(obs['bin']-1),int(obs['userId']-1),int(obs['index'])])**2
        return(sqr_error)
            
    return(np.sqrt(np.mean(data.apply(sqr_err,1))))

     
def ALSTDfit(f, l,iteration_max, train, test):
##Alternating Least Squares with Temperal Dynamics
    

    
    ## train test: must contain all the possible userID, movieID and timebin
    import numpy as np
    import pandas as pd
    
        
    ## Get the dimension and global mean of the data set 
    n_user = np.unique(train.userId).shape[0]
    # n_user = 610
    n_item = np.unique(train.movieId).shape[0]
    # n_item = 3268
    n_timebin = np.unique(train.bin).shape[0]
    # n_timebin = 15
    mu = np.mean(train.loc[:,'rating'])
    #min_rating = 0
    #max_rating = train.rating.max()
        
    # data state
    q = np.array(np.random.random((f,n_item))) # Movie matrix
    p = np.array(np.random.random((f,n_user))) # User matrix
    bi = np.zeros((1,n_item)) # Movie bias
    bu = np.zeros((1,n_user)) # User bias
    bit = np.zeros((n_timebin*n_item,3)) # Movie time variate bias
  
  
    ## Step 1: Initialize Movie matrix (q), User matrix (p), Movie bias(bi) and User bias(bu)
    #q = np.array(np.random.random((f,n_item)))
    q = pd.DataFrame(q)
    q.columns = [np.unique(train.movieId)]
    
    #p = np.array(np.random.random((f,n_user)))
    p = pd.DataFrame(p)
    p.columns = [np.unique(train.userId)]
    
    #bi = np.zeros((1,n_item))
    bi = pd.DataFrame(bi)
    bi.columns = [np.unique(train.movieId)]
    
    #bu = np.zeros((1,n_user))
    bu=pd.DataFrame(bu)
    bu.columns = [np.unique(train.userId)]
    
    #bit = np.zeros((n_timebin*n_item,3))
    bit = pd.DataFrame(bit)
    bit.columns = ['movieId','bin','bias']
    bit.movieId = np.unique(train.movieId).repeat(n_timebin)
    bit.bin = np.tile(np.unique(train.bin),n_item)
    bit.bias = np.zeros((n_timebin * n_item))
    
    
    # global mean 
    # mu=np.mean(train.loc[:,'rating'])
    
    # sort movie id
    movie_id=np.unique(train.movieId)
    
    # sort the data by userid then by movie id
    train = train.sort_values(['userId','movieId'],ascending=[True,True])
    
    #train and test rmse
    #train.insert(3,'estimate',0)
    #test.insert(3,'estimate',0)
    
    # RMSE
    
    test_RMSE = np.zeros(iteration_max)
    train_RMSE = np.zeros(iteration_max)


    for itera in range(iteration_max):

        # step2: fix q, solve p
        # define new factors
        ones_movie=pd.DataFrame(np.repeat(1,n_item)).T
        ones_movie.columns=[np.unique(train.movieId)]
        q.tilde=ones_movie.append(q,ignore_index=True)
        p.tilde=bu.append(p,ignore_index=True)

        for u in range(n_user):
            # find the moives rated by user u, and time_bin of the rating
            rating_i=train.loc[train['userId']==u+1,['movieId','bin']]
            bi_bin_u = np.zeros((1,rating_i.shape[0]))
            
            # get bibin(t)
            for s in range(rating_i.shape[0]):
                bi_bin_u[0,s] = bit.loc[(bit['movieId'] == rating_i.movieId.iloc[s]) & (bit['bin'] == rating_i.bin.iloc[s]),'bias']
            
            #update p.tilde
            M1=np.linalg.inv(np.array(q.tilde.loc[:,rating_i.movieId].dot(q.tilde.loc[:,rating_i.movieId].T))+l*np.eye(f+1))
            M2=np.array(q.tilde.loc[:,rating_i.movieId])
            M3=np.array(train.loc[train['userId']==u+1,'rating'])- mu - bi.loc[:,rating_i.movieId] - bi_bin_u
            p.tilde.iloc[:,u]=M1.dot(M2).dot(M3.T)
    
        #update bu and p
        bu=pd.DataFrame(p.tilde.iloc[0,:]).T
        p=p.tilde.drop([0])

        
        #step3:fix p, solve q
        #update p.tilde
        ones_user=pd.DataFrame(np.repeat(1,n_user)).T
        ones_user.columns=[np.unique(train.userId)]
        p.tilde=ones_user.append(p,ignore_index=True)
        q.tilde=bi.append(q,ignore_index=True)

        for i in range(n_item):
            # find the users who rate movie i, and time_bin of the rating
            rating_u=train.loc[train['index']==i,['movieId','userId','bin']]
            bi_bin_i = np.zeros((1,rating_u.shape[0]))
            
            # get bibin(t)
            for o in range(rating_u.shape[0]):
                bi_bin_i[0,o] = bit.loc[(bit['movieId'] == rating_u.movieId.iloc[o]) & (bit['bin'] == rating_u.bin.iloc[o]),'bias']
            
            #update q.tilde
            M1=np.linalg.inv(np.array(p.tilde.loc[:,rating_u.userId].dot(p.tilde.loc[:,rating_u.userId].T)) + l*np.eye(f+1))
            M2=np.array(p.tilde.loc[:,rating_u.userId])
            M3=np.array(train.loc[train['index']==i,'rating']) - mu - bu.loc[:,rating_u.userId] - bi_bin_i
            q.tilde.iloc[:,i]=M1.dot(M2).dot(M3.T)
   
    
        #update bi and q
        bi=pd.DataFrame(q.tilde.iloc[0,:]).T
        q=q.tilde.drop([0])
        
        
        #step4:fix others, solve bit
        for t in range(n_timebin):
            for n in range(n_item):
        
                # find the number of users who rate movie i, in time_bin t
                Nit = len(train[(train['index']==n) &(train['bin'] == t+1)])
            
                # find total rating of movie i, in time_bin t
                rit = np.array(train[(train['index']==n) & (train['bin'] == t+1)].rating) - mu - \
np.array(bu.loc[0,train[(train['index']==n) & (train['bin'] == t+1)].userId]) - \
float(bi.loc[0,movie_id[n]]) - \
np.array(p.loc[:,train[(train['index']==n) & (train['bin'] == t+1)].userId].T.dot(q.iloc[:,n]))
                rit = rit.sum()
            

                #update bit
                bit.loc[(bit['movieId'] == movie_id[n]) & (bit['bin'] == t+1),'bias'] = rit/(Nit+l)


        
        # Summerize
        print("iter:", itera+1, "\t")
        
        
        # Estimate 
        est_rating = np.zeros((n_timebin, n_user, n_item))
        est_rating_1 = np.array(p.T.dot(q)) + mu + np.array(bu.T)
        est_rating_1 = np.add(est_rating_1, np.array(bu.T))
        est_rating_1 = np.add(est_rating_1, np.array(bi))
        
        for m in range(n_timebin):
            bt = bit[bit['bin'] == m + 1].bias
            est_rating_1 = np.add(est_rating_1, np.array(bt))
            est_rating[m,:,:] = est_rating_1
        
    
        
        # RMSE 
        train_RMSE[itera] = RMSE(est_rating,train) ## first time: 1.150437954121489
        test_RMSE[itera] = RMSE(est_rating,test) ## first time: 1.4897433028093507
        print("train RMSE:", train_RMSE[itera], "\t")
        print("test RMSE:", test_RMSE[itera], "\t")

    return mu, q, p, bi, bu, bit, train_RMSE, test_RMSE   


    
def ALSTD_CV (data_train,K,f,l,iteration):
    import numpy as np
    import pandas as pd
    from sklearn.utils import shuffle
## train set 
## K-fold
## iteration
## lamda
## iteration


    #initialize train and test data and cv result 
    n_col=data_train.shape[1]

    #cv_result_mat=np.zeros((k,n_users,n_movies))
    n = data_train.shape[0] 
    n_fold=int(n/K)  

    data = np.zeros((K,n_fold,n_col))         
    data_train = shuffle(data_train) 

    train_rmse = np.zeros((K,iteration))
    test_rmse = np.zeros((K,iteration))

    for i in range(K):
        data[i] = data_train[i*n_fold:(i+1)*n_fold]
        vali = data[i]
        vali = pd.DataFrame(vali)
        train_new = data_train.drop(vali.index,axis=0)

        train_rmse[i,] = ALSTDfit(f, l, iteration, train_new, vali)[6] 
        test_rmse[i,] = ALSTDfit(f, l, iteration, train_new, vali)[7]
    
    ## result 
    mean_train_rmse = train_rmse.mean(axis=0)
    mean_test_rmse = test_rmse.mean(axis=0)
    sd_train_rmse = train_rmse.std(axis=0)
    sd_test_rmse = test_rmse.std(axis=0)
    
    return mean_train_rmse, mean_test_rmse, sd_train_rmse, sd_test_rmse


