###############################
### Kernel Ridge Regression ###
###############################


## Function to Normalize Each Row
norm.row <- function (m) {
  std <- function (vec){
    return (vec/sqrt(sum(vec^2)))
  }
  return (t(apply(m, 1,std)))
}

## Kernel Ridge Regression
## a function returns a list containing estimated rating matrix, training and testing RMSEs.

KRR.Post <- function (lambda = 10, data, train, test) {
  
  ## Identify Movie Matrix (X), Normalized Movie Matrix (norm.X), and ratings (r) for each user, save in lists
  X <- vector(mode = "list", length = U)
  norm.X <- vector(mode = "list", length = U)
  r <- vector(mode = "list", length = U)
  
  for (u in 1:U) {
    ## get movie numbers rated by user u
    i.rated.by.u <- as.character(train[train$userId==u,]$movieId)
    
    X[[u]] <- result$q[,i.rated.by.u]
    norm.X[[u]] <- norm.row(t(X[[u]]))
    norm.X[[u]][is.na(norm.X[[u]])] <- 0
    r[[u]] <- train[train$userId==u,]$rating - result$mu - result$bu[1,u] - result$bi[1,i.rated.by.u]
  }
  
  ## save krr model for each user
  model <- vector(mode = "list", length = U)
  
  for (u in 1:U) {
    model[[u]] <- krr(norm.X[[u]],r[[u]], lambda = lambda)
  }
  
  ## get estimating matrix
  est_rating <- matrix(NA, ncol = I, nrow=U)
  colnames(est_rating) <- levels(as.factor(data$movieId))
  rownames(est_rating) <- levels(as.factor(data$userId))
  
  for (u in 1:U) {
    est_rating[u,] <- predict(model[[u]], norm.row(t(result$q))) 
    est_rating[u,][is.na(est_rating[u, ])] <- 0
    est_rating[u,] <- est_rating[u,] + result$mu + result$bi[1,] + result$bu[1,u]
  }
  
  # Summerize
  train_RMSE <- RMSE(train, est_rating)
  cat("training RMSE:", train_RMSE, "\t")
  
  
  test_RMSE <- RMSE(test, est_rating)
  cat("test RMSE:",test_RMSE, "\n")
  
  return(list(krr.rating=est_rating, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
  
}