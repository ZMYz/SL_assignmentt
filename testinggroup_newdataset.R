

#load data
install.packages("MASS")
install.packages("ISLR")
install.packages("lattice")
install.packages("caret")
install.packages("kernlab")
install.packages("corrplot")
install.packages("e1071")
install.packages('ggplot2')
install.packages("kernlab")
install.packages("Matrix")

library(Matrix)
library(MASS)
library(ISLR)
library(ggplot2)
library(lattice)
library(caret)
library(kernlab)
library(corrplot)
library(e1071)

#data() # show all the datasets we have
US <- read.csv("D:/Learning/Statistical learning/Assignment/2/acs2017_county_data_used.csv",
    header = TRUE)

#drop three first column which is ID and name of county/region (unique data)
USMatrix <- data.matrix(US)[, -(1:3)]
cor(USMatrix)

#find the high correlation variables
corrplot(cor(USMatrix), order = "hclust")
highCor <- findCorrelation(cor(USMatrix), cutoff = 0.8)
length(highCor)
names(data.frame(USMatrix[,as.vector(highCor)]))
head(USMatrix)

#set control for training process with 10 times 10-folds cross-validation
set.seed(2)
US_myControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 10,
                          verboseIter = T)

USMatrix <- data.frame(USMatrix)
set.seed(1)
intrain <- createDataPartition(y = USMatrix$Unemployment,
                              p = 0.8,
                              list = FALSE)
US_training <- USMatrix[intrain, ]
US_testing <- USMatrix[-intrain, ]
dim(US_training)
dim(US_testing)
anyNA(USMatrix)
head(US_training)

#linear regression
USMatrix <- data.matrix(USMatrix)
set.seed(5)

US_lm <- train(Unemployment ~ .,
            US_training,
            method = 'lm',
            preProcess = c('center', 'scale'),
            trControl = myControl)
summary(US_lm)

plot(varImp(US_lm), main = "importance of variables in linear regression model for US cencus dataset", 
     asp = 0.5, type = "p", cex.lab = 1, las = 2)
US_lm_test <- predict(US_lm, US_testing)
summary(US_lm_test)


###################regularization#####################

#################ridge regression####################
US_myGrid <- expand.grid(alpha = 0, lambda = seq(0, 1, length = 10))
US_ridge <- train(
  Unemployment ~ .,
  data = US_training,
  method = 'glmnet',
  preProcess = c('center', 'scale'),
  tuneGrid = US_myGrid,
  trControl = US_myControl
)

US_ridge_test <- predict(US_ridge, newdata = US_testing)

US_ridge
plot(US_ridge)
plot(US_ridge$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(US_ridge))  # the importance rank of variables

###remove insignificant predictors and rerun the model, and compare

US_ridge2 <- train(Unemployment ~ . - Carpool - FamilyWork - Pacific - Production,
                data = US_training,
                method = 'glmnet',
                preProcess = c('center', 'scale'),
                tuneGrid = US_myGrid,
                trControl = US_myControl)
US_ridge_test2 <- predict(US_ridge, newdata = US_testing)

US_ridge2
plot(US_ridge2)
plot(US_ridge2$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(US_ridge2))  # the importance rank of variables
mean(US_ridge2$resample$RMSE)
mean(US_ridge$resample$RMSE)

###############lasso regression####################
set.seed(3)
US_myGrid2 <- expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 10))
US_lasso <- train(Unemployment ~ .,
                  data = training,
                  method = 'glmnet',
                  preProcess = c('center', 'scale'),
                  tuneGrid = US_myGrid2,
                  trControl = US_myControl)

US_lasso_test <- predict(US_lasso, US_testing)
US_lasso
plot(US_lasso)
plot(US_lasso$finalModel, xvar = 'lambda', label = TRUE)
plot(US_lasso$finalModel, xvar = 'dev', label = TRUE)   ##right side: overfitting occurs
plot(varImp(US_lasso))
# check what does it mean
US_lasso
#############Elastic Net######################
set.seed(4)
US_myGrid3 <- expand.grid(alpha = seq(0, 1, length = 10),
                          lambda = seq(0.0001, 0.2, length = 5))
#alpha=0:1: choose 0 or 1
US_ElasticNet <- train(Unemployment ~ .,
                       data = US_training,
                       method = 'glmnet',
                       preProcess = c('center', 'scale'),
                       tuneGrid = US_myGrid3,
                       trControl = US_myControl)

US_EN_test <- predict(US_ElasticNet, US_testing)
plot(US_ElasticNet)


# compare 3 types of models by RMSE

US_model_list <- list(ridge = US_ridge,
                   lasso = US_lasso,
                   ElasticNet = US_ElasticNet,
                   lm = US_lm)
US_resamples <- resamples(US_model_list)
summary(US_resamples)
bwplot(US_resamples, metric = "RMSE")
dotplot(US_resamples, metric = 'RMSE')


####################Multiple Layer perceptron############
install.packages("ANN2")
library(ANN2)
US_mlp_10cv <- function(data, l1, l2, hiddenlayers, learnrate, iterations) {
  score = list()
  data <- data[sample(nrow(data)), ]
  folds <- cut(seq(1, nrow(data)), breaks = 10, label = F)
  for (i in 1:10) {
    testindex <- which(folds == i, arr.ind = T)
    test <- data[testindex, ]
    train <- data[-testindex, ]
    ytrain <- train$Unemployment
    xtrain <- model.matrix(Unemployment ~ ., data = US_train)[, -34]
    ytest <- test$Unemployment
    xtest <- model.matrix(Unemployment ~ ., data = US_test)[, -34]
    mlp_l1 <- neuralnetwork(xtrain, ytrain,
                            hidden.layers = hiddenlayers,
                            regression = T,
                            loss.type = 'squared',
                            standardize = T,
                            L1 = l1, L2 = l2,
                            learn.rates = learnrate,
                            verbose = F,
                            val.prop = 0.2,
                            n.epochs = iterations)
    mlp_l1_test <- predict(mlp_l1, xtest)
    pred <- unlist(mlp_l1_test) #convert into vector
    score[[i]] = RMSE(ytest, pred)
  }
  score <- unlist(score)
  meanRMSE <- mean(score)
  return(mean(score))
}

US_mlp_10cv(USMatrix, 0, 1, 1, 0.01, 75)
#no regularisation
set.seed(111)
US_rmse_mlp <- US_mlp_10cv(Boston,0,0,2,0.01,75)
#l1-lasso regularisation
set.seed(222)
US_rmse_l1_mlp <- US_mlp_10cv(Boston,1,0,2,0.01,75)
#l2-ridge regularisation
set.seed(333)
US_rmse_l2_mlp <- US_mlp_10cv(Boston,0,1,1,0.01,75)
#Elastic net
set.seed(444)
US_rmse_en_mlp <- US_mlp_10cv(Boston,1,1,1,0.01,75)



#this function adds 10-fold cv with MLP together, and we can check the RMSE for the test set.
####################support vector regression########
#no-regularisation
US_svr_nr<-train(Unemployment~.,
                data = US_training,
                method='svmRadial',
                preProcess = c('center','scale'),
                trControl = US_myControl)

US_svr_nr_test <- predict(US_svr_nr, US_testing)
US_rmse<-mean(US_svr_nr$resample$RMSE)

#L2
library(LiblineaR)
US_svr <- LiblineaR(x_training, y_training,svr_eps=0.5,type=11,cross=0,verbose=T)
pred_SVR<-predict(svr,newx = x_testing)

mse<-LiblineaR(x_training, y_training,svr_eps=0.5,type=11,cross=10,verbose=T)
rmse_l2_svr<-sqrt(mse)

#check the average RMSE of different techniques
Performance<-matrix(data=c(RMSE_lm,RMSE_lasso,RMSE_ridge,RMSE_Eln,rmse_svr_nr,rmse_l2_svr,rmse_mlp,rmse_l1_mlp,rmse_l2_mlp,rmse_en_mlp),ncol=1)
rownames(Performance)<-c('LinearRegres','LinearLasso','LinearRidge','LinearElastic','SVM','SVMl2','MLP','MLPl1','MLPl2','MLPElastic')
colnames(Performance)<-c('RMSE')
Performance
