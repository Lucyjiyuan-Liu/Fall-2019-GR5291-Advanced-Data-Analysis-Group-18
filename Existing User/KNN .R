##########################
### KNN Postprocessing ###
##########################


KNN.Post <- function (data, train, test){
  
  ## get estimating matrix
  knn.rating <- matrix(NA, ncol = I, nrow=U)
  colnames(knn.rating) <- levels(as.factor(data$movieId))
  rownames(knn.rating) <- levels(as.factor(data$userId))
  
  ## calculate movie mean
  movie.id <- sort(unique(data$movieId))
  movie.mean <- train %>% group_by(movieId) %>% arrange(movieId) %>% summarize (rate.mean=mean(rating))
  
  for (i in 1:I){
    close.movieid <- names(sort(similarity.matrix[i, ],decreasing = TRUE)[2])
    knn.rating[,i] <- movie.mean[movie.mean$movieId==as.numeric(close.movieid),]$rate.mean
  }
  
  # Summerize
  train_RMSE <- RMSE(train, knn.rating)
  cat("training RMSE:", train_RMSE, "\t")
  
  
  test_RMSE <- RMSE(test, knn.rating)
  cat("test RMSE:",test_RMSE, "\n")
  
  return(list(knn.rating=knn.rating, train_RMSE = train_RMSE, test_RMSE = test_RMSE))
}