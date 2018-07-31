######
# Adaboost Classifier
######

require(rpart) # for decision stump
require(caret)
require(mlbench)

# set seed to ensure reproducibility
set.seed(100)

###
# calculate the alpha value using epsilon
# Input: 
# epsilon: value from calculate_epsilon 
# output: alpha value (single value) 
###
calculate_alpha <- function(epsilon){
  alpha <- (0.5)*(log2((1-epsilon)/epsilon))
  return(alpha)
}

###
# calculate the epsilon value  
# input:
# weights: weights generated at the end of the previous iteration
# y_true: actual labels (ground truth)
# y_pred: predicted labels (from your newly generated decision stump)
# n_elements: number of elements in y_true or y_pred
# output:
# just the epsilon or error value 
###
calculate_epsilon <- function(weights, y_true, y_pred, n_elements){
  errors <- which(y_true != y_pred)
  epsilon = 0
  for(i in 1:length(errors)){
    epsilon <- epsilon + unlist(weights[errors[i]])
  }
  epsilon <- (1/n_elements) * epsilon
  return(epsilon)
}

###
# Calculate the weights 
# Input:
# old_weights: weights from previous iteration
# alpha: current alpha value 
# y_true: actual class labels
# y_pred: predicted class labels
# n_elements: number of values in y_true or y_pred
# Output:
# a vector of size n_elements containing updated weights
###
calculate_weights <- function(old_weights, alpha, y_true, y_pred, n_elements){
  weights <- old_weights
  errors <- which(y_true != y_pred)
  for (i in 1:length(weights)){
    if (i %in% errors) {weights[i] <- unlist(weights[i]) * exp(alpha)}
    else {weights[i] <- unlist(weights[i]) * exp(-alpha)}
  }
  for (i in 1:length(weights)){
    weights[i] = 1/n_elements
  }
  return(weights)
}

###
# Implementation of myadaboost - simple adaboost classification
# The 'rpart' method from 'rpart' package is used to create a decision stump 
# Input: 
# train: training dataset (attributes + class label)
# k: number of iterations of adaboost
# n_elements: number of elements in 'train'
# Output:
# a vector of predicted values for 'train' after all the iterations of adaboost are completed
###
myadaboost <- function(train, k, n_elements){
  
  #initializing weights
  weights = list()
  for (i in 1:nrow(train)){
    weights[i] = 1/nrow(train)
  }
  
  #creating a temp list for stroing predicted values
  temp = list()
  for(i in 1:nrow(train)){
    temp[i] = 0
  }
  
  #iterating the boosting algorithm
  for (i in 1:k){
    #sampling the train data with replacement
    smp_size <- floor(nrow(train))
    train_ind <- sample(seq_len(nrow(train)), size = smp_size, replace = TRUE, prob = weights)
    train_data <- train[train_ind, ]
    
    wt_df <- unlist(weights)
    
    #training the decision stump model using rpart package
    tree <- rpart(Label ~ .,data = train_data,  weights = wt_df, method = "class", maxdepth=1 )
    # require(rpart.plot)
    # rpart.plot(tree)
    # print(tree)
    
    #predicting the class for the Ionosphere dataset using the tree generated
    pred <- predict(tree, train, type = "class")
    
    #calculating epsilon
    y_true <- train$Label
    epsilon <- calculate_epsilon(weights,y_true, pred,n_elements)
    # print(epsilon)
    
    #calculating alpha
    alpha <- calculate_alpha(epsilon)
    
    #calculating updated weights
    weights <- calculate_weights(weights,alpha,y_true,pred,n_elements)
    # print(do.call(sum,weights))
    
    #storing the predicted values in temp
    for (i in 1:nrow(train)){
      temp2 <- as.numeric(levels(pred[i])[pred[i]])
      temp3<- unlist(temp[i])
      temp[i] = temp3+temp2
    }
    
  }
  # finding the overall prediction
  for (i in 1:nrow(train)){
    if(temp[i]>0){temp[i] = 1}
    else {temp[i] = -1}
  }
  temp <- unlist(temp)
  return(temp)
}
###end of function


# Preprocess the data and then call the adaboost function
data("Ionosphere")
Ionosphere <- Ionosphere[,-c(1,2)]
# converting the class labels into format that is easy to interpret
# -1 for bad, 1 for good (create a column named 'Label' which will serve as class variable)
Ionosphere$Label[Ionosphere$Class == "good"] = 1
Ionosphere$Label[Ionosphere$Class == "bad"] = -1
# remove unnecessary columns
Ionosphere <- Ionosphere[,-(ncol(Ionosphere)-1)]
# class variable
cl <- Ionosphere$Label
# train and predict on training data using adaboost
predictions <- myadaboost(Ionosphere, 5, nrow(Ionosphere))
# generate confusion matrix
print(table(cl, predictions))

