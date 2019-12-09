########################################
### Alternating Least Square with Bias ###
########################################

#Define a function to calculate RMSE
RMSE <- function(rating, est_rating){
  sqr_err <- function(obs){
    sqr_error <- (obs[3] - est_rating[as.character(obs[1]), as.character(obs[2])])^2
    return(sqr_error)
  }
  return(sqrt(mean(apply(rating, 1, sqr_err))))  
}

# Alternating least squares
# a function returns a list containing factorized matrices p and q, training and testing RMSEs.

ALS.R1R2 <- function(f = 10, lambda = 0.3, max.iter, data, train, test){
  
  # Step 1: Initialize Movie matrix (q), User matrix (p), Movie bias(bi) and User bias(bu)
  q <- matrix(runif(f*I, -10, 10), ncol = I)
  colnames(q) <- levels(as.factor(data$movieId))
  
  p <- matrix(runif(f*U, -10, 10), ncol = U) 
  colnames(p) <- levels(as.factor(data$userId))
  
  bi <- matrix(rep(0, I), ncol = I)
  colnames(bi) <- levels(as.factor(data$movieId))

  bu <- matrix(rep(0, U), ncol=U)
  colnames(bu) <- levels(as.factor(data$userId))
  
  # grand mean
  mu <- mean(train$rating)
  
  # sort movie id
  movie.id <- sort(unique(data$movieId))
  
  # record train and test rmse
  train_RMSE <- c()
  test_RMSE <- c()
  
  
  for (l in 1:max.iter){
    # Step 2: Fix q, solve p
    # define new factors
    q.tilde <- rbind(rep(1,I), q)
    colnames(q.tilde) <- levels(as.factor(data$movieId))
    p.tilde <- rbind(bu, p)
    
    for (u in 1:U) {
      # find the moives rated by user u
      i.rated.by.u <- as.character(train[train$userId==u,]$movieId)
      
      # update p.tilde
      p.tilde[,u] <- solve(q.tilde[,i.rated.by.u] %*% t(q.tilde[,i.rated.by.u]) + lambda * diag(f+1)) %*%
                          q.tilde[,i.rated.by.u] %*% (train[train$userId==u,]$rating - mu -bi[,i.rated.by.u])
    }
    
    # update bu and p
    bu[1,] <- p.tilde[1, ]
    p <- p.tilde[-1, ]
    
    # Step 3: Fix p, solve q
    # define new factors
    p.tilde <- rbind(rep(1,U), p)
    colnames(p.tilde) <- levels(as.factor(data$userId))
    q.tilde <- rbind(bi, q)
    
    for (i in 1:I) {
      # find the users who rate movie i
      u.rated.i <- as.character(train[train$movieId==movie.id[i],]$userId)
      q.tilde[,i] <- solve(p.tilde[,u.rated.i] %*% t(p.tilde[,u.rated.i]) + lambda* diag(f+1)) %*%
                          p.tilde[,u.rated.i] %*% (train[train$movieId==movie.id[i],]$rating - mu - bu[,u.rated.i])
      
    }
    
    # update bi and q
    bi[1,] <- q.tilde[1,]
    q <- q.tilde[-1,]
    
    # Summerize
    cat("iter:", l, "\t")
    est_rating <- t(p) %*% q + mu + bu[1,] + rep(bi[1,], each = ncol(p))
    colnames(est_rating) <- levels(as.factor(data$movieId))
    
    train_RMSE_cur <- RMSE(train, est_rating)
    cat("training RMSE:", train_RMSE_cur, "\t")
    train_RMSE <- c(train_RMSE, train_RMSE_cur)
    
    test_RMSE_cur <- RMSE(test, est_rating)
    cat("test RMSE:",test_RMSE_cur, "\n")
    test_RMSE <- c(test_RMSE, test_RMSE_cur)
    
    
  }
  
  return(list(p = p, q = q, bi = bi, bu = bu, mu= mu, train_RMSE = train_RMSE, test_RMSE = test_RMSE, ALS.rating=est_rating))
}



