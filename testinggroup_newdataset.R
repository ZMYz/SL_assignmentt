#load packages
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

# set work directory to the right place that include the dataset
US <- read.csv("D:/Learning/Statistical learning/Assignment/2/acs2017_county_data_used.csv",
    header = TRUE)                 

#drop three first column which is ID and name of county/region (unique data)
USMatrix <- data.matrix(US)[, -(1:3)]
cor(USMatrix)

#find the high correlation variables
corrplot(cor(USMatrix), order = "hclust")
highCor <- findCorrelation(cor(USMatrix), cutoff = 0.8)

names(data.frame(USMatrix[,as.vector(highCor)]))
head(USMatrix)

#cv is for elastic net linear regression
US_myControl2<-trainControl(method='cv',number=10,verboseIter = F)

#set control for training process with 10 times 10-folds cross-validation
set.seed(2)
US_myControl <- trainControl(method = 'repeatedcv',
                          number = 10,
                          repeats = 10,
                          verboseIter = T)

USFrame <- data.frame(USMatrix)
set.seed(1)
intrain <- createDataPartition(y = USFrame $Unemployment,
                              p = 0.8,
                              list = FALSE)
US_training <- USFrame [intrain, ]
US_testing <- USFrame[-intrain, ]
dim(US_training)
dim(US_testing)
anyNA(USMatrix)
head(US_training)

# Function to calculate Rsquared

US_r2<-function(actual,predict){
            predict<-unlist(predict)
            rss<-sum((predict-actual)^2)
            tss<-sum((actual-mean(actual))^2)
            rsq<-1-rss/tss
            }



#linear regression
set.seed(5)

US_lm <- train(Unemployment ~ .,
              US_training,
              method = 'lm',
              preProcess = c('center', 'scale'),
              trControl = US_myControl2)
summary(US_lm)

plot(varImp(US_lm), main = "importance of variables in linear regression model for US cencus dataset", 
     asp = 0.5, type = "p", cex.lab = 1, las = 2)
US_lm_test <- predict(US_lm, US_testing)
summary(US_lm_test)


###################regularization#####################

#################ridge regression####################
US_myGrid <- expand.grid(alpha = 0, lambda = seq(0, 1, length = 10))
US_ridge <- train(Unemployment ~ .,
                  data = US_training,
                  method = 'glmnet',
                  preProcess = c('center', 'scale'),
                  tuneGrid = US_myGrid,
                  trControl = US_myControl2
)

US_ridge_test <- predict(US_ridge, newdata = US_testing)

US_ridge
plot(US_ridge)
plot(US_ridge$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(US_ridge))  # the importance rank of variables

###remove insignificant predictors and rerun the model, and compare

US_ridge2 <- train(Unemployment ~ . - Carpool - Women - TotalPop - IncomePerCapErr - Men - Asian - Production,
                  data = US_training,
                  method = 'glmnet',
                  preProcess = c('center', 'scale'),
                  tuneGrid = US_myGrid,
                  trControl = US_myControl2)
US_ridge_test2 <- predict(US_ridge, newdata = US_testing)

US_ridge2
plot(US_ridge2)
plot(US_ridge2$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(US_ridge2))  # the importance rank of variables

# extract the RMSE from predictions on training set

mean(US_ridge2$resample$RMSE)
mean(US_ridge$resample$RMSE)

###############lasso regression####################
set.seed(3)
US_myGrid2 <- expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 10))
US_lasso <- train(Unemployment ~ .,
                  data = US_training,
                  method = 'glmnet',
                  preProcess = c('center', 'scale'),
                  tuneGrid = US_myGrid2,
                  trControl = US_myControl2)
  # alpha = 1, lambda = 1e-04
US_lasso_test <- predict(US_lasso, US_testing)
US_lasso
plot(US_lasso)
plot(US_lasso$finalModel, xvar = 'lambda', label = TRUE)
plot(US_lasso$finalModel, xvar = 'dev', label = TRUE)   ##right side: overfitting occurs ???
plot(varImp(US_lasso))
# check what does it mean

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
                       trControl = US_myControl2)

US_EN_test <- predict(US_ElasticNet, US_testing)
plot(US_ElasticNet)


# compare 3 types of models by RMSE, not working," "There are different numbers of resamples in each model"

#US_model_list <- list(ridge = US_ridge,
 #                     lasso = US_lasso,
  #                    ElasticNet = US_ElasticNet,
   #                   lm = US_lm)
#US_resamples <- resamples(US_model_list)
#summary(US_resamples)
#bwplot(US_resamples, metric = "RMSE")
#dotplot(US_resamples, metric = 'RMSE')


####################Multiple Layer perceptron############

install.packages("ANN2")
library(ANN2)
US_finalmodel <- list(0)
US_mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  #US_RMSE=list()
  US_R2=list()
  US_r2 <- 0
  set.seed(1111)
  fold10 <- createFolds(data$Unemployment,k=10,list=F)
  set.seed(123)
  for(i in 1:10){
    US_testindex<-which(fold10==i,arr.ind=T)
    US_test <- data[US_testindex,]
    US_train <- data[-US_testindex,]
    US_ytrain <- US_train$Unemployment
    US_xtrain <- model.matrix(Unemployment~.,data = US_train)[,-34]
    US_ytest <- US_test$Unemployment
    US_xtest<- model.matrix(Unemployment~.,data = US_test)[,-34]
    #split data
    US_mlp <- neuralnetwork(US_xtrain, US_ytrain, 
                         hidden.layers = hiddenlayers, 
                         regression=T, loss.type='squared',
                         standardize = T,
                         L1 = l1, L2 = l2,
                         learn.rates = learnrate,
                         verbose=F, val.prop=0.2, 
                         n.epochs = iterations)
    #train the model
    US_mlp_test <- predict(US_mlp,US_xtest)
    #prediction
    US_pred <- unlist(US_mlp_test) #convert into vector
    #US_RMSE[[i]] = RMSE(US_ytest, US_pred)
    US_rss <- sum((US_pred - US_ytest)^2)
    US_tss <- sum((US_ytest - mean(US_ytest))^2)
    US_rsq <- 1-US_rss/US_tss
    US_R2[i] <- US_rsq
    #select the predictive model with the highest r2
    if(US_rsq >= US_r2){
      US_finalmodel <<- US_mlp
      US_r2<-US_rsq
    }
  }
  #RMSE<-unlist(RMSE)
  #meanRMSE<-mean(RMSE)
  #print(unlist(R2))
  #print(RMSE)
  print(US_finalmodel)
  print(US_r2)
  print(US_R2)
}
US_mlp
anyNA(USFrame)
#no regularisation
set.seed(111)
US_rmse_mlp <- US_mlp_10cv(US_training,0,0,2,0.01,75)
#l1-lasso regularisation
set.seed(222)
US_rmse_l1_mlp <- US_mlp_10cv(US_training,1,0,2,0.01,75)
#l2-ridge regularisation
set.seed(333)
US_rmse_l2_mlp <- US_mlp_10cv(US_training,0,1,1,0.01,75)
#Elastic net
set.seed(444)
US_rmse_en_mlp <- US_mlp_10cv(US_training,1,1,1,0.01,75)


##
#US_finalmodel <- list(0)   ###not sure what does it mean?? 
US_mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  #US_RMSE=list()
  US_R2=list()
  US_r2 <- 0
  set.seed(1111)
  fold10 <- createFolds(data$Unemployment,k=10,list=F)
  set.seed(123)
  for(i in 1:10){
    testindex<-which(fold10==i,arr.ind=T)
    test <- data[testindex,]
    train <- data[-testindex,]
    ytrain <- train$Unemployment
    xtrain <- model.matrix(Unemployment~.,data = train)[,-34]
    ytest <- test$Unemployment
    xtest<- model.matrix(Unemployment~.,data = test)[,-34]
    #split data
    US_mlp <- neuralnetwork(xtrain, ytrain, 
                            hidden.layers = hiddenlayers, 
                            regression=T, loss.type='squared',
                            standardize = T,
                            L1 = l1, L2 = l2,
                            learn.rates = learnrate,
                            verbose=F, val.prop=0.2, 
                            n.epochs = iterations)
    #train the model
    US_mlp_test <- predict(US_mlp,xtest)
    #prediction
    pred <- unlist(US_mlp_test) #convert into vector
    #US_RMSE[[i]] = RMSE(ytest, pred)
    rss <- sum((pred-ytest)^2)
    tss <- sum((ytest-mean(ytest))^2)
    rsq <- 1-rss/tss
    US_R2[[i]]=rsq
    #select the predictive model with the highest r2
    if(rsq >= US_r2){
      US_finalmodel <<- US_mlp
      US_r2<-rsq
    }
  }
  #RMSE<-unlist(RMSE)
  #meanRMSE<-mean(RMSE)
  #print(unlist(R2))
  #print(RMSE)
  print(US_finalmodel)
  print(US_r2)
  print(US_R2)
}

####################support vector regression########
#no-regularisation
US_svr_nr<-train(Unemployment~.,
                data = US_training,
                method='svmRadial',
                preProcess = c('center','scale'),
                trControl = US_myControl2)

US_svr_nr_test <- predict(US_svr_nr, US_testing)
US_rmse<-mean(US_svr_nr$resample$RMSE)
summary(US_svr_nr_test)
#L2
library(LiblineaR)

US_svr_l2<-train(Unemployment~.,
                 data = US_training,
                 method='svmLinear3',
                 preProcess = c('center','scale'),
                 trControl = US_myControl2)
US_svr_l2_test <- predict(US_svr_l2, US_testing)
US_rmse<-mean(US_svr_l2$resample$RMSE)
summary(US_svr_l2_test)

#US_y_training<-US_training$Unemployment
#US_x_training<-as.matrix(US_training)[,-34]
#US_x_testing<-as.matrix(US_testing)[,-34]
#!US_y_testing<-US_testing$Unemployment
#dim(US_x_testing)
#dim(US_x_training)


#US_svr <- LiblineaR(US_x_training, US_y_training,svr_eps=0.5,type=11,cross=0,verbose=T)
#US_pred_SVR<-predict(US_svr,newx = US_x_testing)

#US_svr_mse<-LiblineaR(US_x_training, US_y_training,svr_eps=0.5,type=11,cross=10,verbose=T)
#US_rmse_l2_svr<-sqrt(US_svr_mse)            #  ??? does it right?

#check the average RMSE of different techniques
Performance<-matrix(data=c(US_RMSE_lm,US_RMSE_lasso,US_RMSE_ridge,RMSE_Eln,rmse_svr_nr,rmse_l2_svr,rmse_mlp,rmse_l1_mlp,rmse_l2_mlp,rmse_en_mlp),ncol=1)
rownames(Performance)<-c('LinearRegres','LinearLasso','LinearRidge','LinearElastic','SVM','SVMl2','MLP','MLPl1','MLPl2','MLPElastic')
colnames(Performance)<-c('RMSE')
Performance
