#load data
install.packages("ISLR")
install.packages('ggplot2')
install.packages('kernlab')

library(MASS)
library(ISLR)
library(ggplot2)
library(lattice)
library(caret)
library(kernlab)
library(Matrix)

#data() # show all the datasets we have
fix(Boston) #show data like spreadsheet
dim(Boston)  # # of obervations and cols
str(Boston)  #check the type of data
names(Boston)  # variables/cols

# Creating the split of data into training and test set
set.seed(1)
intrain <- createDataPartition(y = Boston$crim, p = 0.8, list = FALSE)
training <- Boston[intrain,]
testing <- Boston[-intrain,]

# Checking the number of variables in both the training and test datasets
dim(training)
dim(testing)

# Checking if there are any missing values in the dataset
anyNA(Boston)

###################regularization#####################

set.seed(2)
# Cross validation controls being set - to be used in all 4 methods for consistancy
myControl <- trainControl(method = 'repeatedcv', repeats = 10, number = 10, verboseIter = T)

###################linear regression##################

set.seed(5)

lm <- train(crim ~ .,
          training,
          method = 'lm',
          preProcess = c('center','scale'),
          trControl = myControl)

lm

lm_test <- predict(lm, testing)

plot(testing$crim)
plot(lm_test)

plot(x = lm_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)
#abline(lm)

# Doesn't work #
#sn_train <- names(training)
#n_train <- n_train[2:14]
#ggplot(training, aes(y = crim, x = n_train)) +
#  geom_point() +
#  geom_smooth(method = "lm")

par(mfrow = c(2,2))
plot(lm$finalModel)
par(mfrow = c(1,1))

#### LM without regularising the data ####
set.seed(14)

lm2 <- train(crim ~ .,
            training,
            method = 'lm',
            trControl = myControl)

lm2

lm_test2 <- predict(lm2, testing)

plot(testing$crim)
plot(lm_test2)

plot(x = lm_test2, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#### LM with 50 Repititions ####

lm50 <- train(crim ~ .,
            training,
            method = 'lm',
            preProcess = c('center','scale'),
            trControl = trainControl(method = 'repeatedcv', repeats = 100, number = 10, verboseIter = F))

lm50

lm_test50 <- predict(lm50, testing)

plot(testing$crim)
plot(lm_test50)

plot(x = lm_test50, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#################ridge regression####################

myGrid <- expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 10))

set.seed(5)
ridge <- train(crim ~ ., 
             data = training,
             method = 'glmnet',
             preProcess = c('center','scale'),
             tuneGrid = myGrid,
             trControl = myControl)

ridge_pred <- predict(ridge, newdata = testing)

ridge

plot(ridge)
plot(ridge$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(ridge))  # the importance rank of variables

plot(testing$crim)
plot(ridge_pred)

plot(x = ridge_pred, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#### Ridge Regression without regulaising the data ####

ridgeNR <- train(crim ~ ., 
               data = training,
               method = 'glmnet',
               tuneGrid = myGrid,
               trControl = myControl)

ridge_pred_NR <- predict(ridgeNR, newdata = testing)

ridgeNR

plot(ridgeNR)
plot(ridgeNR$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(ridgeNR))  # the importance rank of variables

plot(testing$crim)
plot(ridge_pred_NR)

plot(x = ridge_pred_NR, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

####remove insignificant predictors and rerun the model, and compare####

ridge_train <- subset(training, select = -c(age, chas, rm, ptratio))
ridge_test <- subset(testing, select = -c(age, chas, rm, ptratio))

ridge2 <- train(crim ~ ., 
               data = ridge_train,
               method = 'glmnet',
               preProcess = c('center','scale'),
               tuneGrid = myGrid,
               trControl = myControl)

ridge_pred2 <- predict(ridge2, newdata = ridge_test)

ridge2
plot(ridge2)
plot(ridge2$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(ridge2))  # the importance rank of variables

plot(ridge_test$crim)
plot(ridge_pred2)

plot(x = ridge_pred2, y = ridge_test$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

###############lasso regression####################

set.seed(3)

myGrid2 <- expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 10))

lasso <- train(crim ~ ., 
             data = training,
             method = 'glmnet',
             preProcess = c('center','scale'),
             tuneGrid = myGrid2,
             trControl = myControl)

lasso_test <- predict(lasso, testing)

plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label = TRUE)
plot(lasso$finalModel, xvar = 'dev', label = TRUE)   ##right side: overfitting occurs
# check what does it mean

plot(x = lasso_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0, b=1)

#### Lasso Regression without regularising the data ####

lasso_NR <- train(crim ~ ., 
               data = training,
               method = 'glmnet',
               tuneGrid = myGrid2,
               trControl = myControl)

lasso_test_NR <- predict(lasso_NR, testing)

plot(lasso_NR)
plot(lasso_NR$finalModel, xvar = 'lambda', label = TRUE)
plot(lasso_NR$finalModel, xvar = 'dev', label = TRUE)   ##right side: overfitting occurs
# check what does it mean

plot(x = lasso_test_NR, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#############Elastic Net######################

set.seed(4)

myGrid3 <- expand.grid(alpha = seq(0, 1, length = 10), lambda = seq(0, 1, length = 10))

#alpha=0:1: choose 0 or 1
ElasticNet <- train(crim ~ ., 
                  data = training,
                  method = 'glmnet',
                  preProcess = c('center','scale'),
                  tuneGrid = myGrid3,
                  trControl = myControl)

EN_test <- predict(ElasticNet, testing)
plot(ElasticNet)

set.seed(4)
ElasticNet1 <- train(crim ~ ., 
                    data = training,
                    method = 'glmnet',
                    preProcess = c('center','scale'),
                    tuneGrid = myGrid3,
                    trControl = trainControl(method = 'cv', number = 10, verboseIter = T))

EN_test1 <- predict(ElasticNet1, testing)
plot(ElasticNet1)

#### Elastic Net without reguarising the data ####

ElasticNet_NR <- train(crim ~ ., 
                    data = training,
                    method = 'glmnet',
                    tuneGrid = myGrid3,
                    trControl = trainControl(method = 'cv', number = 10, verboseIter = T))

EN_test_NR <- predict(ElasticNet, testing)
plot(ElasticNet_NR)

####################Approach Comparison##################

model_list <- list(ridge = ridge, lasso = lasso, ElasticNet = ElasticNet, lm = lm)
lapply(model_list, summary)

resamp <- resamples(model_list)
summary(resamp)
str(resamp)

# Looking at the RMSE
bwplot(resamp, metric="RMSE")
dotplot(resamp, metric='RMSE')

# Looking at the R-squared
bwplot(resamp, metric="Rsquared")
dotplot(resamp, metric='Rsquared')

Group_RMSE <- data.frame(Ridge = resamp$values$`ridge~RMSE`, Lasso = resamp$values$`lasso~RMSE`, ElasticNet = resamp$values$`ElasticNet~RMSE`,LM = resamp$values$`lm~RMSE`)
GR <- cbind(Group_RMSE, ID = rownames(Group_RMSE))

summary(Group_RMSE)
Mean_RMSE <- data.frame(colMeans(Group_RMSE))
Mean_RMSE

RMSE_sorted <- apply(Group_RMSE, 2, sort, decreasing = F)
RMSE_sorted <- data.frame(RMSE_sorted)

RMSE_sorted <- cbind(RMSE_sorted, ID = rownames(RMSE_sorted))

plot(RMSE_sorted[,1:4])

RMSE_sorted$ID <- factor(RMSE_sorted$ID, levels = RMSE_sorted$ID[order(1:100)])

ggplot(RMSE_sorted, aes(x = order(ID))) + 
  geom_line(aes(y = Ridge), colour = "blue") + 
  geom_line(aes(y = Lasso), colour = "grey") + 
  geom_line(aes(y = ElasticNet), colour = "red") + 
  geom_line(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 100, 10)) +
  ylab("RMSE") + 
  xlab("Iteration") +
  ggtitle("Ordered plot of RMSE per Regularisation Approch")

ggplot(GR, aes(x = order(ID))) + 
  geom_line(aes(y = Ridge), colour = "blue") + 
  geom_line(aes(y = Lasso), colour = "grey") + 
  geom_line(aes(y = ElasticNet), colour = "red") + 
  geom_line(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 100, 10)) +
  ylab("RMSE") + 
  xlab("Iteration") +
  ggtitle("Plot of RMSE per Regularisation Approch")

####################Multiple Layer perceptron############
install.packages("ANN2")
library(ANN2)

mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  #RMSE=list()
  R2=list()
  r2<-0
  set.seed(1111)
  fold10<-createFolds(data$crim,k=10,list=F)
  set.seed(123)
  for(i in 1:10){
    testindex<-which(fold10==i,arr.ind=T)
    test<-data[testindex,]
    train<-data[-testindex,]
    ytrain<-train$crim
    xtrain<-model.matrix(crim~.,data=train)[,-1]
    ytest<-test$crim
    xtest<-model.matrix(crim~.,data=test)[,-1]
    #split data
    mlp<-neuralnetwork(xtrain,ytrain,hidden.layers = hiddenlayers,regression=T,loss.type='squared',standardize = T,L1=l1,L2=l2,learn.rates=learnrate,verbose=F,val.prop=0.2,n.epochs = iterations)
    #train the model
    mlp_test<-predict(mlp,xtest)
    #prediction
    pred<-unlist(mlp_test) #convert into vector   
    #RMSE[[i]]=RMSE(ytest,pred)
    #r-squared calculation
    rss<-sum((pred-ytest)^2)
    tss<-sum((ytest-mean(ytest))^2)
    rsq<-1-rss/tss
    R2[[i]]=rsq
    #select the predictive model with the highest r2
    if(rsq >= r2){
      finalmodel<<-mlp
      r2<-rsq
    }
  }
  #RMSE<-unlist(RMSE)
  #meanRMSE<-mean(RMSE)
  #print(unlist(R2))
  #print(RMSE)
  print(finalmodel)
  print(r2)
  print(R2)
}

#no regularisation
r2_mlp<-mlp_10cv(training,0,0,2,0.01,75)
finalpred_noreg<-predict(finalmodel,x_testing)

#l1-lasso regularisation
r2_l1_mlp<-mlp_10cv(training,1,0,2,0.01,75)
finalpred_l1<-predict(finalmodel,x_testing)
#l2-ridge regularisation
set.seed(333)
r2_l2_mlp_l2<-mlp_10cv(training,0,1,1,0.01,75)
#Elastic net
set.seed(444)
r2_en_mlp<-mlp_10cv(training,1,1,1,0.01,75)
finalpred_l3<-predict(finalmodel,x_testing)


#this function adds 10-fold cv with MLP together, and we can check the RMSE for the test set.

####################support vector regression########
#no-regularisation
svr_nr<-train(crim~.,
              data=training,
              method='svmRadial',
              preProcess=c('center','scale'),
              trControl=myControl)

svr_nr_test<-predict(svr_nr,testing)
rmse<-mean(svr_nr$resample$RMSE)

#L2
library(LiblineaR)
svr<-LiblineaR(x_training, y_training,svr_eps=0.5,type=11,cross=0,verbose=T)
pred_SVR<-predict(svr,newx = x_testing)

mse<-LiblineaR(x_training, y_training,svr_eps=0.5,type=11,cross=10,verbose=T)
rmse_l2_svr<-sqrt(mse)

#check the average RMSE of different techniques
Performance<-matrix(data=c(RMSE_lm,RMSE_lasso,RMSE_ridge,RMSE_Eln,rmse_svr_nr,rmse_l2_svr,rmse_mlp,rmse_l1_mlp,rmse_l2_mlp,rmse_en_mlp),ncol=1)
rownames(Performance)<-c('LinearRegres','LinearLasso','LinearRidge','LinearElastic','SVM','SVMl2','MLP','MLPl1','MLPl2','MLPElastic')
colnames(Performance)<-c('RMSE')
Performance

#https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
#SVR for boston
# New dataset: 
# https://www.kaggle.com/muonneutrino/us-census-demographic-data: just few N/A values, have the income column with continuous value which could use for regression               
# https://www.kaggle.com/karangadiya/fifa19             : some N/A values, too much variables
