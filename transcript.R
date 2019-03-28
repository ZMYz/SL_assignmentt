#load data
library(MASS)
library(ISLR)
#data() # show all the datasets we have
fix(Boston) #show data like spreadsheet
dim(Boston)  # # of obervations and cols
str(Boston)  #check the type of data
names(Boston)  # variables/cols

install.packages('ggplot2')
library(ggplot2)
library(lattice)
library(caret)
library(kernlab)
set.seed(1)
intrain<-createDataPartition(y=Boston$crim,p=0.8,list=FALSE)
training<-Boston[intrain,]
testing<-Boston[-intrain,]
dim(training)
dim(testing)
anyNA(Boston)

###################regularization#####################
library(Matrix)

set.seed(2)
myControl<-trainControl(method='cv',number=10,verboseIter = T)

#################ridge regression####################
myGrid<-expand.grid(alpha=0,lambda=seq(0.0001,1,length=10))
ridge<-train(crim~., 
                    data=training,
                    method='glmnet',
                    preProcess=c('center','scale'),
                    tuneGrid=myGrid,
                    trControl=myControl)

ridge_test<-predict(ridge,newdata=testing)

ridge
plot(ridge)
plot(ridge$finalModel,xvar='lambda',label=TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(ridge))  # the importance rank of variables

###remove insignificant predictors and rerun the model, and compare
###############lasso regression####################
set.seed(3)
myGrid2<-expand.grid(alpha=1,lambda=seq(0.0001,1,length=10))
lasso<-train(crim~., 
             data=training,
             method='glmnet',
             preProcess=c('center','scale'),
             tuneGrid=myGrid2,
             trControl=myControl)

lasso_test<-predict(lasso,testing)

plot(lasso)
plot(lasso$finalModel,xvar='lambda',label=TRUE)
plot(lasso$finalModel,xvar='dev',label=TRUE)   ##right side: overfitting occurs
# check what does it mean

#############Elastic Net######################
set.seed(4)
myGrid3<-expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5))
#alpha=0:1: choose 0 or 1
ElasticNet<-train(crim~., 
             data=training,
             method='glmnet',
             preProcess=c('center','scale'),
             tuneGrid=myGrid3,
             trControl=myControl)

EN_test<-predict(ElasticNet,testing)
plot(ElasticNet)

#linear regression

set.seed(5)

lm<-train(crim~.,
          training,method='lm',
          preProcess=c('center','scale'),
          trControl=myControl)

lm_test<-predict(lm,testing)

model_list<-list(ridge=ridge,lasso=lasso,ElasticNet=EN,lm=lm)
resamples<-resamples(model_list)
summary(resamples)
bwplot(resamples,metric="RMSE")
dotplot(resamples,metric='RMSE')

####################Multiple Layer perceptron############
################L1 regularization
library(snnR)

y=training$crim
x=model.matrix(crim~.,training)[,-1]
nHidden <- matrix(c(5,5,15,5,5),1,5)
l1_mlp<-snnR(x,y,
             nHidden=nHidden,
             normalize=TRUE,
             verbose=TRUE,
             lambda=c(0.001,0.01,0.1,1))

l1_mlp_test<-predict(l1_mlp,testing)
l1_mlp$Mse

###############L1 regularization
library(ANN2)
mlp_l1<-autoencoder(training,hidden.layers = 3,standardize = T,L1=1,verbose=T)
#at layers = 3, get the minimal
mlp_l1
mlp_l1_test<-predict(mlp_l1,testing)
#best layers
#weight

#############L2 regularization
mlp_l2<-autoencoder(training,hidden.layers = 1,standardize = T,L2=1,verbose=T)
#at layers = 1, get the minimal
mlp_l2
mlp_l2_test<-predict(mlp_l2,testing)
##############elastic net
mlp_eln<-autoencoder(training,hidden.layers = 1,standardize = T,L1=0.6,L2=0.4,verbose=T)
#at layers = 1, get the minimal
mlp_eln
mlp_eln_test<-predict(mlp_eln,testing)

####################support vector regression########

myGrid<-expand.grid(alpha=0:1,lambda=seq(0.001,0.1,length=10))
svr<-train(crim~.,data=training, method='svmRadialSigma',trControl=myControl,preProcess=c('center','scale'),tuneLength=10)

test_pred<-predict(svr,newdata=testing)
test_pred

plot(test_pred)
plot(testing$crim)
#out-of-sample RMSE
sqrt(mean((testing$crim-test_pred)^2))


#ridge regression
library(glmnet)
x=model.matrix(crim~.,data=boston)[,-1]
y=boston$crim
set.seed(3)
train=sample(1:nrow(x),3*nrow(x)/4)
test=(-train)
y.test=y[test]

cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlambda=cv.out$lambda.min
bestlambda

ridge=glmnet(x[train,],y[train],alpha=0,lambda=bestlambda)
ridge.pred=predict(ridge,lambda=bestlambda,x[test,])
mean((ridge.pred-y.test)^2)

ridge.coef=predict(ridge,lambda=bestlambda,type='coefficient')
ridge.coef

#https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
