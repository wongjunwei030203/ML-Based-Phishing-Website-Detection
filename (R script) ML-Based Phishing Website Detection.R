# setting working directory
setwd("/Users/junwei/Desktop/3152 A2")

install.packages("tree")
install.packages("rpart")
install.packages("adabag")
install.packages("e1071")
install.packages("randomForest")
install.packages("pROC")
install.packages("ROCR")
install.packages("neuralnet")
install.packages("caret")

library(neuralnet)
library(caret)
library(tree)
library(rpart)
library(adabag)
library(e1071)
library(randomForest)
library(pROC)
library(ROCR)

rm(list = ls())
Phish <- read.csv("PhishingData.csv")
set.seed(32845650)
L <- as.data.frame(c(1:50))
L <- L[sample(nrow(L), 10, replace = FALSE),]
Phish <- Phish[(Phish$A01 %in% L),]
PD <- Phish[sample(nrow(Phish), 2000, replace = FALSE),] # sample of 2000 rows

# Question 1

dim(PD)
# observing the structure of the data
str(PD)

# proportion of the variable "Class" that is non-phishing
sum(PD$Class == 0, na.rm = TRUE) / nrow(PD)

# proportion of the variable "Class" that is phishing
sum(PD$Class == 1, na.rm = TRUE) / nrow(PD)

summary(PD) # summary of data

# Question 2

# converting the class variable to type "factor"
PD$Class <- as.factor(PD$Class)

# number of NA values in the PD dataset
sum(is.na(PD))

# omitting rows containing NA values
PD_omit = na.omit(PD)

# Question 3
set.seed(32845650)
train.row = sample(1:nrow(PD_omit), 0.7 * nrow(PD_omit))

# creating the training and testing datasets
PD.train = PD_omit[train.row,]
PD.test = PD_omit[-train.row,]

# Question 4

# in this question, i fit the 5 models

tree_fit <- tree(Class ~., data=PD.train)

nb_fit <- naiveBayes(Class ~. -Class, data=PD.train)

set.seed(32845650)
bag_fit <- bagging(Class ~., data=PD.train, mfinal=5)

set.seed(32845650)
boost_fit <- boosting(Class ~., data=PD.train, mfinal=10)

set.seed(32845650)
rf_fit <- randomForest(Class ~. -Class, data=PD.train)

# Question 5

# calculating the accuracy for the decision tree model
tree_pred <- predict(tree_fit, PD.test, type="class")
tree_conf <- confusionMatrix(data=tree_pred, reference=PD.test$Class)
tree_accuracy <- tree_conf$overall["Accuracy"]

# calculating the accuracy for naive bayes model
nb_pred <- predict(nb_fit, PD.test)
nb_conf <- confusionMatrix(data=nb_pred, reference=PD.test$Class)
nb_accuracy <- nb_conf$overall["Accuracy"]

# this function is used to calculate the accuracy of the bagging and boosting models
calculate_accuracy = function(conf_matrix) {
  True_Positive <- conf_matrix[2, 2]
  True_Negative <- conf_matrix[1, 1]
  False_Positive <- conf_matrix[1, 2]
  False_Negative <- conf_matrix[2, 1]
  
  acc <- (True_Positive + True_Negative) / (True_Positive + True_Negative + False_Positive + False_Negative)
  return(acc)
}

# calculating the accuracy for the bagging model
bag_pred <- predict.bagging(bag_fit, PD.test)

# accuracy for the boosting model
boost_pred <- predict.boosting(boost_fit, PD.test)

# accuracy for random forest model
rf_pred <- predict(rf_fit, PD.test, type = "class")
rf_conf <- confusionMatrix(data=rf_pred, reference=PD.test$Class)
rf_accuracy <- rf_conf$overall["Accuracy"]

# list to keep track of the accuracy scores of every model
model_accuracy = list()
model_accuracy = append(model_accuracy, tree_accuracy)
model_accuracy = append(model_accuracy, nb_accuracy)
model_accuracy = append(model_accuracy, calculate_accuracy(bag_pred$confusion))
model_accuracy = append(model_accuracy, calculate_accuracy(boost_pred$confusion))
model_accuracy = append(model_accuracy, rf_accuracy)

# creating a dataframe that stores all the model accuracies
Q5AccuracyTable <- as.data.frame(t(model_accuracy))
colnames(Q5AccuracyTable) <- c("Decision Tree Model", "Naive Bayes Model", "Bagging Model", "Boosting Model", "Random Forest Model")
Q5AccuracyTable


# Question 6

# calculating auc for decision tree
tree_predict <- predict(tree_fit, PD.test, type = "vector")
tree_predictions <- prediction(tree_predict[,2], PD.test$Class)

# calculating auc for naive bayes
nb_predict <- predict(nb_fit, PD.test, type='raw')
nb_predictions <- prediction(nb_predict[,2], PD.test$Class)

# calculating auc for bagging
bag_predictions <- prediction(bag_pred$prob[,2], PD.test$Class)

# calculating auc for boosting
boost_predictions <- prediction(boost_pred$prob[,2], PD.test$Class)

# calculating auc for random forest
rf_predict = predict(rf_fit, PD.test, type="prob")
rf_predictions <- prediction(rf_predict[,2], PD.test$Class)

# adding every model's AUC value to a list to keep track of them
model_auc = list()
model_auc = append(model_auc, as.numeric(performance(tree_predictions, "auc")@y.values))
model_auc = append(model_auc, as.numeric(performance(nb_predictions,"auc")@y.values))
model_auc = append(model_auc, as.numeric(performance(bag_predictions, "auc")@y.values))
model_auc = append(model_auc, as.numeric(performance(boost_predictions, "auc")@y.values))
model_auc = append(model_auc, as.numeric(performance(rf_predictions, "auc")@y.values))

# creating a dataframe that stores all the model AUC values
Q6AUCTable <- as.data.frame(t(model_auc))
colnames(Q6AUCTable) <- c("Decision Tree Model", "Naive Bayes Model", "Bagging Model", "Boosting Model", "Random Forest Model")
Q6AUCTable

# plotting the ROC curves to visualize the performance of the models
plot(performance(tree_predictions, "tpr","fpr"), main="Comparing ROC curves of all the models", col="red") # tree ROC curve
plot(performance(nb_predictions,"tpr","fpr"), add=TRUE, col="forestgreen") # naive bayes ROC
plot(performance(bag_predictions,"tpr","fpr"), add = TRUE, col="orange") # bagging ROC
plot(performance(boost_predictions,"tpr","fpr"), add=TRUE, col ="royalblue") # boosting ROC
plot(performance(rf_predictions, "tpr","fpr"), add=TRUE, col="violet") # random forest ROC
abline(0,1)

legend("bottomright",legend=c("Decision Tree","Naive Bayes", "Bagging","Boosting","Random Forest"),
       col=c("red","forestgreen","orange","royalblue","violet"),
       lty = 1)

# Question 7

# this dataframe stores performance measures of each model (accuracy and AUC)
pm_df = cbind(model_accuracy, model_auc)
pm_df = cbind(c("Decision Tree","Naive Bayes", "Bagging","Boosting","Random Forest"), pm_df)
pm_df <- data.frame(pm_df[,-1], row.names = pm_df[,1])
pm_df

# Question 8

# identifying the important attributes for the models
summary(tree_fit)
# the naive bayes model does not identify most important attributes
bag_fit$importance
boost_fit$importance
rf_fit$importance

# Question 9

#cross validation test to find which value to use to prune
cvtest = cv.tree(tree_fit, FUN = prune.misclass)
cvtest

pruned_treefit = prune.misclass(tree_fit, best = 3)
summary(pruned_treefit)
plot(pruned_treefit)
text(pruned_treefit, pretty = 0)

# calculating accuracy
pruned_tree.predict = predict(pruned_treefit, PD.test, type = "class")
pruned_tree_pred <- table(actual = PD.test$Class, predicted = pruned_tree.predict)
calculate_accuracy(pruned_tree_pred)


# calculating the area under the curve
pruned_tree_conf = predict(pruned_treefit, PD.test, type = "vector")
prunedtree_pred_obj <- prediction(pruned_tree_conf[,2], PD.test$Class)
auc = performance(prunedtree_pred_obj, "auc")
as.numeric(auc@y.values)

# Question 10

set.seed(32845650)

# performing cross validation on the random forest model
rf_cv <- rfcv(PD.train[,1:25], PD.train$Class)
rf_cv

# checking the importance of predictors to see which can be removed
rf_fit$importance

# removing the lowest 5 predictors with the lowest importance scores
PD.train_reduced = PD.train[,-c(3,5,7,13,25)]

set.seed(32845650)
# fitting the new random forest model
rf_fit <- randomForest(Class~., data = PD.train_reduced, ntree = 2500, max.depth = 150, min.node.size = 2)
rf_pred <- predict(rf_fit, newdata = PD.test, type = "class")

# calculating the accuracy of this new random forest model
rf_conf <- table(observed = PD.test$Class, predicted=rf_pred)
calculate_accuracy(rf_conf)

# calculating the area under the curve
rf_cv_conf = predict(rf_fit, PD.test, type = "prob")
rf_pred_obj <- prediction(rf_cv_conf[,2], PD.test$Class)
auc = performance(rf_pred_obj, "auc")
as.numeric(auc@y.values)


# Question 11

# performing some preprocessing
PD_ANN = PD[complete.cases(PD),]
PD$Class = as.numeric(PD$Class) # changing Class to numeric data type

set.seed(32845650)
PD_ANN[, sapply(PD_ANN, is.numeric)] <- scale(PD_ANN[, sapply(PD_ANN, is.numeric)])

set.seed(32845650)
train.row = sample(1:nrow(PD_ANN), 0.7*nrow(PD_ANN))

# training and testing datasets for the NN model
PD_ANN.train = PD_ANN[train.row,]
PD_ANN.test = PD_ANN[-train.row,]

# indicators for the training dataset
Class = PD_ANN.train[,26]
PD_ANN.train = PD_ANN.train[,1:25]
dummy <- dummyVars(" ~ .", data = PD_ANN.train)
newPD_ANN.train <- data.frame(predict(dummy, newdata = PD_ANN.train))
PD_ANN.train = cbind(newPD_ANN.train, Class)

# indicators for the testing set
Class = PD_ANN.test[,26]
PD_ANN.test = PD_ANN.test[,1:25]
dummy <- dummyVars(" ~ .", data = PD_ANN.test)
newPD_ANN.test <- data.frame(predict(dummy, newdata = PD_ANN.test))
PD_ANN.test = cbind(newPD_ANN.test, Class)

# fitting the neural network model with all the predictors (A01 to A25)
set.seed(32845650)
PD.nn = neuralnet(Class ~ A01 + A02 + A03 + A04 + A05 + A06 + A07 + A08 + A09+ A10 +
                          A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + 
                          A21 + A22 + A23 + A24 + A25, 
                          data=PD_ANN.train, hidden=3, linear.output=TRUE)

# plotting the neural network
plot(PD.nn, rep="best")

# setting a threshold value of 0.5
PD.pred = compute(PD.nn, PD_ANN.test[,1:25])
PD.prob <- PD.pred$net.result
pred <- ifelse(PD.prob>0.5, 1, 0)

# calculating the accuracy
nnConf <- table(observed = PD_ANN.test$Class, predicted=pred[,2])
nnConf
calculate_accuracy(nnConf)

# calculating the AUC
auc <- roc(PD_ANN.test$Class, pred[,2])$auc
auc

# Question 12

# fitting a support vector machine
set.seed(32845650)
svm_fit <- svm(Class ~., data = PD.train, kernel="linear", probability = TRUE)

svm_pred <- predict(svm_fit, newdata = PD.test)
svm_conf <- confusionMatrix(data=svm_pred, reference=PD.test$Class)

# calculating the accuracy
svm_accuracy <- svm_conf$overall["Accuracy"]
svm_accuracy

# calculating the AUC
svm_conf <- predict(svm_fit, PD.test, probability = TRUE)
svm_conf <- attr(svm_conf, "probabilities")
svm_pred <- ROCR::prediction(svm_conf[, 1], PD.test$Class)
auc.svm <- performance(svm_pred, "auc")@y.values[[1]]
auc.svm



