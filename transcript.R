#load data
install.packages("MASS")
install.packages("ISLR")
library(MASS)
library(ISLR)
#data() # show all the datasets we have
fix(Boston) #show data like spreadsheet
dim(Boston)  # # of obervations and cols
str(Boston)  #check the type of data
names(Boston)  # variables/cols

install.packages('ggplot2')
install.packages("lattice")
install.packages("caret")
install.packages("kernlab")
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

r2<-function(actual,predict){
  predict<-unlist(predict)
  rss<-sum((predict-actual)^2)
  tss<-sum((actual-mean(actual))^2)
  rsq<-1-rss/tss
}

###################regularization#####################
install.packages("Matrix")
library(Matrix)

set.seed(2)
myControl<-trainControl(method='cv',number=10,verboseIter = F)

#repeated cv is for elastic net linear regression
myControl2<-trainControl(method='repeatedcv',number=10,repeats=10,verboseIter = F)

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
ridge2<-train(crim~.-age - chas, 
             data=training,
             method='glmnet',
             preProcess=c('center','scale'),
             tuneGrid=myGrid,
             trControl=myControl)
ridge_test2<-predict(ridge,newdata=testing)

ridge2
plot(ridge2)
plot(ridge2$finalModel,xvar='lambda',label=TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(ridge2))  # the importance rank of variables
# RMSE decreases when remove redundance variables

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

plot(varImp(lasso))
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
install.packages("ANN2")
library(ANN2)

mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  RMSE=list()
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
    RMSE[[i]]=RMSE(ytest,pred)
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

y_training<-training$crim
x_training<-as.matrix(training)[,-1]
x_testing<-as.matrix(testing)[,-1]
y_testing<-testing$crim

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
 
