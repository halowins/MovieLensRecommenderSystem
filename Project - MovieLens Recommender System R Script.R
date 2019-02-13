#############################################################
# Create edx set, validation set, and submission file
#############################################################
# Note: this process could take a couple of minutes
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(cvTools)) install.packages("cvTools", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(Hmisc)
library(cvTools)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#unzipped the downloaded file and read ratings.dat and store in ratings variable
ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),col.names = c("userId", "movieId", "rating", "timestamp"))

#unzipped the downloaded file and read movies.dat and store in movies variable
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

#assign appropriate data type to the attributes
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],title = as.character(title),genres = as.character(genres))

#join both data frames ratings and movies by movieId
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# edx dataset exploration
#############################################################
#observe the structure of the data and its content
head(edx)
str(edx)
describe(edx)


#check no duplicate of user and movie id combination in the dataset
edx %>% group_by(userId,movieId) %>% summarise(count=n()) %>% filter(count>1)

#Plot Users Count
edx %>% count(userId) %>% ggplot(aes(n))+ geom_histogram(bins = 30, colour = "black", position = "dodge",alpha=1)+ theme_bw()+ggtitle("Users Count") +xlab("n")+ylab("Users")+scale_x_log10()

#Plot Movies Count
edx %>% count(movieId) %>% ggplot(aes(n))+ geom_histogram(bins = 30, colour = "black", position = "dodge",alpha=1)+ theme_bw()+ggtitle("Movies Count") +xlab("n")+ylab("Movies")+scale_x_log10()

#Plot Count By Ratings
options(scipen=10000)
edx %>% group_by(rating) %>% summarise(count=n()) %>% ggplot(aes(y=count, x=rating, fill=count))+geom_bar(stat="identity") +  ylab("Count") + theme_bw()+ggtitle("Count By Ratings")+scale_y_sqrt()

#Same Title and Year But Different MovieId
edx %>% group_by(movieId,title) %>% summarise(count=n()) %>% group_by(title) %>% summarise(count=n()) %>% filter(count>1) %>% nrow
edx %>% filter(title=='War of the Worlds (2005)') %>% distinct(movieId)

#break down the genres into rows
edx %>% separate_rows(genres, sep = "\\|") %>% group_by(genres) %>% summarise(count = n()) %>% arrange(desc(count))


#############################################################
#1. The Simple Average
#############################################################

#Calculate the average of rating in train set
mu <- mean(edx$rating)
#Calculate RMSE
rmse <- RMSE(validation$rating, mu)
rmse

#Store the results
rmse_results <- data_frame(Model = "The Simple Average", RMSE = rmse)
rmse_results %>% knitr::kable()

#############################################################
#2. Movie Effect Model
#############################################################

#Calculate movie bias in train set, grouping by movieId and get the average delta value of rating and average rating
movie_eff <- edx %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu))

#Plotting the distribution of b_i and observe the variability
movie_eff %>% ggplot(aes(b_i)) + geom_histogram(bins=40,colour = "black", position = "dodge",alpha=1)

#Assign the movie bias for validation set
b_i <-  validation %>% left_join(movie_eff, by='movieId') %>% .$b_i

#Enhanced model
predicted_ratings <- mu + b_i

#Calculate RMSE
rmse <- RMSE(predicted_ratings, validation$rating)

#Append the results to rmse_results
rmse_results <- bind_rows(rmse_results, data_frame(Model="Movie Effect Model", RMSE = rmse))
rmse_results %>% knitr::kable()


#############################################################
#3. Regularized Movie Effect Model
#############################################################

#Top 10 best movies from the Movie Effect model
edx %>% left_join(movie_eff, by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(desc(b_i))  %>% head(.,10) %>% knitr::kable()

#Top 10 worst movies from the Movie Effect model
edx %>% left_join(movie_eff, by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(b_i)  %>% head(.,10) %>% knitr::kable()

#create an empty list
rmse_cv <- list()
set.seed(1)

#set a sequence of lambdas values
lambdas <- seq(0, 10, 0.25)

#set the number of fold
k<-10

#perform the cross validation
folds <- cvFolds(NROW(edx), K=k)

for(i in 1:k)
{ 
  #split to train and test set
  train_cv <- edx[folds$subsets[folds$which != i], ] #Set the training set
  temp_cv <- edx[folds$subsets[folds$which == i], ] #Set the test set
  
  test_cv <- temp_cv %>% semi_join(train_cv , by = "movieId") %>% semi_join(train_cv , by = "userId")
  
  # Add rows removed from validation set back into train set
  removed_cv <- anti_join(temp_cv, test_cv)
  train_cv <- rbind(train_cv, removed_cv)
  
  #get average rating value from train set
  mu <- mean(train_cv$rating)
  
  #summing bi_i and count n by movieId
  summ<- train_cv %>% group_by(movieId) %>% summarise(s = sum(rating - mu), n_i = n())
  
  #calculate rmses for each lambda sequence
  rmses <- sapply(lambdas, function(l){
    predicted_ratings <- test_cv %>% left_join(summ, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>% mutate(pred = mu + b_i) %>% .$pred
    return(RMSE(predicted_ratings, test_cv$rating))
  })
  
  #store RMSE value into a list
  rmse_cv[[i]]<-rmses
}

#convert list to dataframe
rmse_cv_df <- as.data.frame(do.call(rbind, rmse_cv))
rmse_cv_df

#calculate average RMSE
mean_rmses <- colMeans(rmse_cv_df)

#plot lambda and rmse
qplot(lambdas, mean_rmses)  

#find lambda value with the lowest average RMSE
lambda_cv <- lambdas[which.min(mean_rmses)]
lambda_cv

#Regularised Movie Effect Model
lambda <- lambda_cv

#Calculate b_i with lambda penalty term
reg_movie_eff <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#Top 10 best movies from the Movie Effect model after regularized
edx %>% left_join(reg_movie_eff , by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(desc(b_i))  %>% head(.,10) %>% knitr::kable()

#Top 10 worst movies from the Movie Effect model after regularized
edx %>% left_join(reg_movie_eff , by="movieId") %>% group_by(title,movieId, b_i) %>% summarise(count=n()) %>% arrange(b_i)  %>% head(.,10) %>% knitr::kable()

#Enhanced Model
predicted_ratings <- validation %>% left_join(reg_movie_eff, by='movieId') %>%
  mutate(pred = mu + b_i) %>% .$pred

#Calculate RMSE
rmse <- RMSE(predicted_ratings, validation$rating)

#Append the results to rmse_results 
rmse_results <- bind_rows(rmse_results,data_frame(Model="Regularized Movie Effect Model",RMSE = rmse))
rmse_results %>% knitr::kable()

#############################################################
#4. Movie+User Effect Model
#############################################################

#Calculate user bias in train set, grouping by userId and get the average delta value of (rating-b_i) and average rating
user_eff <- edx %>% left_join(movie_eff, by='movieId') %>% group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

#Plotting the distribution of b_u and observe the variability
user_eff %>% ggplot(aes(b_u)) + geom_histogram(bins=40,colour = "black", position = "dodge",alpha=1)

#Assign the user bias for validation set
b_u <-  validation %>% left_join(movie_eff, by='movieId') %>% left_join(user_eff, by='userId') %>% .$b_u

#Enhanced model
predicted_ratings <- mu + b_i + b_u

#Calculate RMSE
rmse <- RMSE(predicted_ratings, validation$rating)

#Append the results to rmse_results 
rmse_results <- bind_rows(rmse_results,data_frame(Model="Movie + User Effects Model",RMSE = rmse))

rmse_results %>% knitr::kable()

#############################################################
#5. Regularized Movie+User Effect Model
#############################################################


#create an empty list
rmse_cv <- list()
set.seed(1)

#set a sequence of lambdas values
lambdas <- seq(0, 10, 0.25)

#set the number of fold
k<-10

#perform the cross validation
folds <- cvFolds(NROW(edx), K=k)

for(i in 1:k)
{ 
  #split to train and test set
  train_cv <- edx[folds$subsets[folds$which != i], ] #Set the training set
  temp_cv <- edx[folds$subsets[folds$which == i], ] #Set the test set
  
  test_cv <- temp_cv %>% semi_join(train_cv , by = "movieId") %>% semi_join(train_cv , by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed_cv <- anti_join(temp_cv, test_cv)
  train_cv <- rbind(train_cv, removed_cv)
  
  #calculate rmses for each lambda sequence  
  rmses <- sapply(lambdas, function(l){
    
  #get average rating value from train set
  mu <- mean(train_cv$rating)
    
  #calculate b_i value with lambda
  b_i <- train_cv %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu)/(n()+l))
  
  #calculate b_u value with lambda
  b_u <- train_cv %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu)/(n()+l))
    
  #predict rating with test set
  predicted_ratings <- test_cv%>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% .$pred
    
  return(RMSE(predicted_ratings, test_cv$rating))
  })
  
  #store RMSE value into a list
  rmse_cv[[i]]<-rmses
}
#convert list to dataframe
rmse_cv_df <- as.data.frame(do.call(rbind, rmse_cv))
rmse_cv_df 

#calculate average RMSE
mean_rmses <- colMeans(rmse_cv_df)

#plot lambda and rmse
qplot(lambdas, mean_rmses)

#find lambda value with the lowest average RMSE
lambda_cv <- lambdas[which.min(mean_rmses)]

#Regularised Movie+User Effect Model
lambda <- lambda_cv
lambda

#Calculate b_i with lambda penalty term
b_i <- edx %>% group_by(movieId) %>% summarise(b_i = sum(rating - mu)/(n()+lambda))

#Calculate b_u with lambda penalty term
b_u <- edx %>% left_join(b_i, by="movieId") %>% group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu)/(n()+lambda))

#predict ratings with the validation set
predicted_ratings <- validation%>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% .$pred

#Calculate RMSE   
rmse <- RMSE(predicted_ratings, validation$rating)

#Append the results to rmse_results
rmse_results <- bind_rows(rmse_results,data_frame(Model="Regularized Movie + User Effect Model",RMSE = rmse ))
rmse_results %>% knitr::kable()




