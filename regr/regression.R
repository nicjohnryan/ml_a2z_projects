#=======================================
# A-Z ML
# Regression Practice
# Wine Quality Dataset
#=======================================


# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names
# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/


# Preliminaries
#===================

library(caret)
library(dplyr)
library(ggplot2)


# Import Practice Data
#===================

dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")


# Simple EDA
#===================

# we will treat quality as a numeric response
pairs(dataset)

lapply(dataset, function(var) { ggplot(dataset, aes(x=var)) + geom_histogram() })

summary(dataset)


# Data Preprocessing
#===================

# train test split based on outcome
set.seed(1234)

trainIndex <- createDataPartition(dataset$quality, p=0.8, list=FALSE)

train <- dataset[trainIndex,]
test <- dataset[-trainIndex,]

# center and scale 
scaledValues <- preProcess(train[,-12], method=c("center", "scale"))
trainScaled <- predict(scaledValues, train[-12])
testScaled <- predict(scaledValues, test[-12])

# append outcome variable back to train and test
train <- cbind(trainScaled, quality=train[,12])
test <- cbind(testScaled, quality=test[,12])

lapply(train, function(var) { ggplot(train, aes(x=var)) + geom_histogram() })


# Compare Models
#===================

# helper model function
fitMods <- function(method) { 
            train(quality ~ ., data=train,
             method=method,
             trControl = fitControl)}


set.seed(1234)
fitControl <- trainControl(method="repeatedcv",
                number=10,
                repeats=10)

# linear stepwise regression
linReg <- fitMods(method="glmStepAIC")  

# lasso regression
lassoReg <- fitMods(method="lasso") 

# svr
svrRadial <- fitMods(method="svmRadial") 
svrPoly <- fitMods(method="svmPoly") 
svrLinear <- fitMods(method="svmLinear") 


# decision tree
dTree <- fitMods(method="rpart") 

# random forest
forest <- fitMods(method="rf")  


# Check Model Results
#===================

results <- resamples(list(linReg=linReg, lassoReg=lassoReg, svrRadial=svrRadial,
                          svrPoly=svrPoly,svrLinear=svrLinear,dTree=dTree,forest=forest))
summary(results)

dotplot(results)


# helper functions for model results
checkAccuracy <- function(classifier) {
    pred <- predict(classifier, test[,-12])
    
    # print plot of actual vs expected
    print(plot(test$quality, pred))
    
    print("accuracy table")
    print(table(test$quality, round(pred,0)))
    
    print("accuracy statistic")
    print(sum(round(pred,0) == test$quality) / nrow(test))
    return(pred)
}

checkVisualFit <- function(var) {
    ggplot() +
        geom_point(aes(x=test$alcohol, y=test$quality), color="red") +
        geom_line(aes(x=test$alcohol, y=var), color="blue") + 
        ggtitle("Actual vs Expected for Alcohol Content")
}

forest_preds <- checkAccuracy(forest)
checkVisualFit(forest_preds)

# final model, 500 trees is probably plenty.
forest
forest$finalModel

# most important variables
varImpPlot(forest$finalModel,type=2)

# correls
library(corrplot)
corrplot(cor(train), method="number", type="upper")




