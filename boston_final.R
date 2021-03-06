#load data
install.packages("MASS")

library(MASS)

#data() # show all the datasets we have
data(Boston)

install.packages('ggplot2')
install.packages("lattice")
install.packages("caret")
install.packages("kernlab")

library(ggplot2)
library(lattice)
library(caret)
library(kernlab)

# Creating the split of data into training and test set
set.seed(4)
intrain<-createDataPartition(y=Boston$crim,p=0.8,list=FALSE)

training<-Boston[intrain,]
testing<-Boston[-intrain,]

# Scaling both the training adn the test data
#training <- data.frame(scale(training))
#testing <- data.frame(scale(testing))

# Checking the number of variables in both the training and test datasets
dim(training)
dim(testing)

# Checking if there are any missing values in the dataset
anyNA(Boston)

###################regularization#####################

set.seed(2)
myControl<-trainControl(method='cv',number=10,verboseIter = F)

#repeated cv is for elastic net linear regression
myControl2<-trainControl(method='repeatedcv',number=10,repeats=10,verboseIter = F)

###################linear regression#####################

set.seed(5)

lm <-train(crim~.,
           training,method='lm',
           preProcess=c('center','scale'),
           trControl=myControl)

lm_test <-predict(lm,testing)

lm

plot(testing$crim)
plot(lm_test)
plot(varImp(lm))

plot(x = lm_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

par(mfrow = c(2,2))
plot(lm$finalModel)

#################ridge regression####################

set.seed(4)

myGrid<-expand.grid(alpha=0,lambda=seq(0,1,length=10))

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

plot(testing$crim)
plot(ridge_test)

plot(x = ridge_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

plot(ridge$finalModel)

###############lasso regression####################
set.seed(3)
myGrid2<-expand.grid(alpha=1,lambda=seq(0,1,length=10))
lasso<-train(crim~., 
             data=training,
             method='glmnet',
             preProcess=c('center','scale'),
             tuneGrid=myGrid2,
             trControl=myControl)

lasso_test<-predict(lasso,testing)

lasso

plot(lasso)
plot(lasso$finalModel,xvar='lambda',label=TRUE)
plot(lasso$finalModel,xvar='dev',label=TRUE)   ##right side: overfitting occurs

plot(varImp(lasso))
# check what does it mean

plot(x = lasso_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0, b=1)

plot(lasso$finalModel)

#############Elastic Net######################
set.seed(4)
myGrid3<-expand.grid(alpha=seq(0,1,length=10),lambda=seq(0,1,length=10))
#alpha=0:1: choose 0 or 1
ElasticNet<-train(crim~., 
                  data=training,
                  method='glmnet',
                  preProcess=c('center','scale'),
                  tuneGrid=myGrid3,
                  trControl=myControl2)

EN_test<-predict(ElasticNet,testing)

ElasticNet

plot(ElasticNet)
plot(varImp(ElasticNet))

plot(x = EN_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0, b=1)

plot(ElasticNet$finalModel)

####################Regression Approach Comparison##################

#model_list <-list(ridge=ridge,lasso=lasso,ElasticNet=ElasticNet,lm=lm)
model_list <-list(ridge=ridge,lasso=lasso,lm=lm)
summary(model_list)

resamp <- resamples(model_list)
summary(resamp)

EN_Samp <- ElasticNet$resample
EN_Samp
bwplot(resamp, metric="RMSE")
bwplot(EN_Samp$RMSE, ylab = "Elastic Net", las = 0)

dotplot(resamp, metric='RMSE')
dotplot(EN_Samp$RMSE)

# Looking at the R-squared
bwplot(resamp, metric="Rsquared")
bwplot(EN_Samp$Rsquared)
dotplot(resamp, metric='Rsquared')
dotplot(EN_Samp$Rsquared)

####################Regression Model Comparison##################

#ord_EN_Samp <- EN_Samp[order(EN_Samp$Resample),]

index <- rep(1:10, 10)
Grouped_EN_Samp <- aggregate(x=EN_Samp[,1:3], by = list(index), FUN=mean)

#Group_R2 <- data.frame(Ridge = resamp$values$`ridge~Rsquared`, Lasso = resamp$values$`lasso~Rsquared`, ElasticNet = Grouped_EN_Samp$Rsquared, LM = resamp$values$`lm~Rsquared`)
Group_R2 <- data.frame(Ridge = resamp$values$`ridge~RMSE`, Lasso = resamp$values$`lasso~RMSE`, ElasticNet = Grouped_EN_Samp$RMSE, LM = resamp$values$`lm~RMSE`)
GR2 <- cbind(Group_R2, ID = rownames(Group_R2))

summary(Group_R2)
Mean_R2 <- data.frame(colMeans(Group_R2))
Mean_R2

R2_sorted <- apply(Group_R2, 2, sort, decreasing = F)
R2_sorted <- data.frame(R2_sorted)

R2_sorted <- cbind(R2_sorted, ID = rownames(R2_sorted))

plot(R2_sorted[,1:4])

R2_sorted$ID <- factor(R2_sorted$ID, levels = R2_sorted$ID[order(1:100)])

ggplot(R2_sorted, aes(x = order(ID))) + 
  geom_line(aes(y = Ridge, colour = "Ridge"), size = 1) + #Blue
  geom_line(aes(y = Lasso, colour = "Lasso"), size = 1) + #Green
  geom_line(aes(y = ElasticNet, colour = "ElasticNet"), size = 1) + #Red
  geom_line(aes(y = LM, colour = "LM"), size = 1) + #Orange
  scale_x_discrete(limits = seq(0, 10)) +
  ylab("R-Squared") + 
  xlab("Iteration") +
  theme(legend.position="bottom", legend.box = "horizontal") +
  scale_colour_manual("", values = c(Ridge = "blue", Lasso = "green", ElasticNet = "red", LM = "orange")) +
  ggtitle("Ordered plot of R-2 per Regularisation Approch")

# Non-sorted data line plot
ggplot(GR2, aes(x = order(ID))) + 
  geom_line(aes(y = Ridge), colour = "blue") + 
  geom_line(aes(y = Lasso), colour = "green") + 
  geom_line(aes(y = ElasticNet), colour = "red") + 
  geom_line(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 10)) +
  ylab("R-Squared") + 
  xlab("Iteration") +
  ggtitle("Plot of R-2 per Regularisation Approch")

# Non-sorted data dot plot
ggplot(GR2, aes(x = order(ID))) + 
  geom_point(aes(y = Ridge), colour = "blue") + 
  geom_point(aes(y = Lasso), colour = "green") + 
  geom_point(aes(y = ElasticNet), colour = "red") + 
  geom_point(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 100, 10)) +
  ylab("R-Squared") + 
  xlab("Iteration") +
  ggtitle("Dot Plot of R-2 per Regularisation Approch")

# Density Plot
ggplot(GR2) + 
  geom_density(aes(Ridge, colour = "Ridge")) + #Blue
  geom_density(aes(Lasso, colour = "Lasso")) + #Green
  geom_density(aes(ElasticNet, colour = "ElasticNet")) + #Red
  geom_density(aes(LM, colour = "LM"))

# Density plots - using the melt function
library(reshape2)

melt_GR2 <- melt(GR2)

ggplot(melt_GR2) + 
  geom_density(aes(value, colour = variable)) + 
  facet_grid(.~variable)

ggplot(melt_GR2) + 
  geom_density(aes(value, colour = variable))

ggplot(melt_GR2) + 
  geom_jitter(aes(value, 0, colour = variable)) + 
  facet_grid(.~variable)

# Combined plot - density and points
ggplot(melt_GR2) + 
  geom_density(aes(value, colour = variable)) + 
  geom_point(aes(value, 0, colour = variable)) +
  facet_grid(.~variable)

####################Regression Prediction versus Actual############

lm_pred_list <- list(actual = testing$crim, ridge = ridge_test, lasso = lasso_test, ElasticNet = EN_test, lm = lm_test)
lm_pred_df <- data.frame(lm_pred_list)

cor_accuracy <- cor(lm_pred_df)
cor_accuracy[1,]

sort_lm_pred_df <- lm_pred_df[order(lm_pred_df$actual),]
slpd <- cbind(sort_lm_pred_df, ID = 1:100)

lmpd <- cbind(lm_pred_df, ID = 1:100)

# Plotting the four comparison plots of Actual versus predicted
ggplot(lmpd, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("No Regularisation") +
  theme_bw()

ggplot(lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = lasso), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lasso), shape = 1) +  # Points of predicted values
  ggtitle("Lasso") +
  theme_bw()

ggplot(lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("Ridge") +
  theme_bw()

ggplot(lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ElasticNet), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ElasticNet), shape = 1) +  # Points of predicted values
  ggtitle("Ealsticnet") +
  theme_bw()

# Accuracy of the prediction versus actual
postResample(pred = lm_pred_df$lm, obs = lm_pred_df$actual)
postResample(pred = lm_pred_df$lasso, obs = lm_pred_df$actual)
postResample(pred = lm_pred_df$ridge, obs = lm_pred_df$actual)
postResample(pred = lm_pred_df$ElasticNet, obs = lm_pred_df$actual)

reg_nr <- postResample(pred = lm_pred_df$lm, obs = lm_pred_df$actual)
reg_l1 <- postResample(pred = lm_pred_df$lasso, obs = lm_pred_df$actual)
reg_l2 <- postResample(pred = lm_pred_df$ridge, obs = lm_pred_df$actual)
reg_EN <- postResample(pred = lm_pred_df$ElasticNet, obs = lm_pred_df$actual)
####################support vector regression##############

#no-regularisation
svr_nr<-train(crim~.,
              data=training,
              method='svmRadial',
              preProcess=c('center','scale'),
              trControl=myControl)

svr_nr_test<-predict(svr_nr,testing)
#rmse<-mean(svr_nr$resample$RMSE)

svr_nr
plot(svr_nr)
plot(varImp(svr_nr))

plot(testing$crim)
plot(svr_nr_test)

plot(x = svr_nr_test, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#################L2 SVM#######################
library(LiblineaR)

y_training<-training$crim
x_training<-as.matrix(training)[,-1]
x_testing<-as.matrix(testing)[,-1]
y_testing<-testing$crim

svr_l2<-train(crim~.,
                 data = training,
                 method='svmLinear3',
                 preProcess = c('center','scale'),
                 trControl = myControl)

pred_SVR2 <- predict(svr_l2, x_testing)

rmse_svr_l2<-mean(svr_l2$results$RMSE)

summary(svr_l2_test)

#svr

plot(testing$crim)

plot(pred_SVR2)

plot(x = pred_SVR2, y = testing$crim,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)


####################Comparison of SVR Predictions########

svr_pred_list <- list(actual = testing$crim, lm = svr_nr_test, ridge = pred_SVR2)
svr_pred_df <- data.frame(svr_pred_list)

svr_cor_accuracy <- cor(svr_pred_df)
svr_cor_accuracy[1,]

sort_svr_pred_df <- svr_pred_df[order(svr_pred_df$actual),]
sort_svr <- cbind(sort_svr_pred_df, ID = 1:100)

svr_id <- cbind(svr_pred_df, ID = 1:100)

# Plotting the two comparison plots of Actual versus predicted
ggplot(svr_id, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("No Regularisation") +
  theme_bw()

ggplot(svr_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("Ridge") +
  theme_bw()

# Accuracy of the prediction versus actual
postResample(pred = svr_pred_df$lm, obs = svr_pred_df$actual)
postResample(pred = svr_pred_df$ridge, obs = svr_pred_df$actual)

svm_nr <- postResample(pred = svr_pred_df$lm, obs = svr_pred_df$actual)
svm_l2 <- postResample(pred = svr_pred_df$ridge, obs = svr_pred_df$actual)

####################Multiple Layer perceptron############
install.packages("ANN2")
library(ANN2)

set.seed(125)
fold10<- createFolds(training$crim,k=10,list=F)

mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  #R2=list()
  #r2<-0
  score = list()
  rmse_val <- 50
  #fold10<-createFolds(data$crim,k=10,list=F)
  for(i in 1:10){
    testindex<-which(fold10==i,arr.ind=T)
    test<-data[testindex,]
    train<-data[-testindex,]
    ytrain<-train$crim
    xtrain<-model.matrix(crim~.,data=train)[,-1]
    ytest<-test$crim
    xtest<-model.matrix(crim~.,data=test)[,-1]
    #split data
    mlp<-neuralnetwork(xtrain,ytrain,hidden.layers = hiddenlayers,regression=T,loss.type='squared',standardize = T, activ.functions = "relu", random.seed = 5,batch.size = 25,L1=l1,L2=l2,learn.rates=learnrate,verbose=F,val.prop=0.2,n.epochs = iterations)
    #train the model
    mlp_test<-predict(mlp,xtest)
    #prediction
    pred<-unlist(mlp_test) #convert into vector
    #r-squared calculation
    #rss<-sum((pred-ytest)^2)
    #tss<-sum((ytest-mean(ytest))^2)
    #rsq<-1-rss/tss
    #R2[[i]]=rsq
    pred_RMSE = RMSE(ytest,pred)
    score[[i]] = pred_RMSE
    #select the predictive model with the highest r2
    if(pred_RMSE <= rmse_val){
      finalmodel<<-mlp   #  <<- set global variable 
      rmse_val<-pred_RMSE
      B_rmse <<- rmse_val
    }
  }
  print(finalmodel)
  print(rmse_val)
  print(score)
}

#no regularisation
r2_mlp<- mlp_10cv(training,0,0,c(5),0.01,100)
noreg_mlp_model <- finalmodel
MLP_NR_RMSE <- B_rmse
finalpred_noreg<-predict(noreg_mlp_model,x_testing)
#postResample(pred = finalpred_noreg$`predictions`, obs = testing$crim)

#l1-lasso regularisation
r2_l1_mlp<-mlp_10cv(training,1,0,c(5),0.01,100)
l1_mlp_model <- finalmodel
MLP_L1_RMSE <- B_rmse
finalpred_l1<-predict(l1_mlp_model,x_testing)

#l2-ridge regularisation
r2_l2_mlp_l2<-mlp_10cv(training,0,1,c(5),0.01,100)
l2_mlp_model <- finalmodel
MLP_L2_RMSE <- B_rmse
finalpred_l2 <- predict(l2_mlp_model,x_testing)

#Elastic net
r2_en_mlp<-mlp_10cv(training,1,1,c(5),0.01,100)
en_mlp_model <- finalmodel
MLP_EN_RMSE <- B_rmse
finalpred_l3<-predict(en_mlp_model,x_testing)

# Plotting the training Models for MLP
plot(noreg_mlp_model)
plot(l1_mlp_model)
plot(l2_mlp_model)
plot(en_mlp_model)

################ MLP Plotting and Analysis ########################

mlp_pred_list <- list(actual = testing$crim, ridge = finalpred_l2$predictions[,1], lasso = finalpred_l1$predictions[,1], ElasticNet = finalpred_l3$predictions[,1], lm = finalpred_noreg$predictions[,1])
mlp_pred_df <- data.frame(mlp_pred_list)

mlp_cor_accuracy <- cor(mlp_pred_df)
mlp_cor_accuracy[1,]

sort_mlp_pred_df <- mlp_pred_df[order(mlp_pred_df$actual),]
sort_mlp <- cbind(sort_mlp_pred_df, ID = 1:100)

mlp_id <- cbind(lm_pred_df, ID = 1:100)

# Plotting the four comparison plots of Actual versus predicted
ggplot(mlp_id, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("MLP No Regularisation") +
  theme_bw()

ggplot(mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = lasso), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lasso), shape = 1) +  # Points of predicted values
  ggtitle("MLP Lasso") +
  theme_bw()

ggplot(mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("MLP Ridge") +
  theme_bw()

ggplot(mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ElasticNet), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ElasticNet), shape = 1) +  # Points of predicted values
  ggtitle("MLP Ealsticnet") +
  theme_bw()

# Accuracy of the prediction versus actual
postResample(pred = mlp_pred_df$lm, obs = mlp_pred_df$actual)
postResample(pred = mlp_pred_df$lasso, obs = mlp_pred_df$actual)
postResample(pred = mlp_pred_df$ridge, obs = mlp_pred_df$actual)
postResample(pred = mlp_pred_df$ElasticNet, obs = mlp_pred_df$actual)

mlp_nr <- postResample(pred = mlp_pred_df$lm, obs = mlp_pred_df$actual)
mlp_l1 <- postResample(pred = mlp_pred_df$lasso, obs = mlp_pred_df$actual)
mlp_l2 <- postResample(pred = mlp_pred_df$ridge, obs = mlp_pred_df$actual)
mlp_EN <- postResample(pred = mlp_pred_df$ElasticNet, obs = mlp_pred_df$actual)

######## Overall Comparison of Models #######
#check the average RMSE of different techniques

a_en <- ElasticNet$results[ElasticNet$results$alpha == ElasticNet$bestTune$alpha,]
l_ln <- a_en[a_en$lambda == ElasticNet$bestTune$lambda,]

lasso_val <- lasso$result[lasso$results$lambda == lasso$bestTune$lambda,]
ridge_val <- ridge$results[ridge$results$lambda == ridge$bestTune$lambda,]

RMSE_lm <- lm$results$RMSE
RMSE_lasso<- lasso_val$RMSE
RMSE_ridge<- ridge_val$RMSE
RMSE_Eln<- l_ln$RMSE
rmse_svr_nr <- svr_nr$results$RMSE[3]
rmse_l2_svr <- rmse_l2_svr
MLP_NR_RMSE
MLP_L1_RMSE
MLP_L2_RMSE
MLP_EN_RMSE

Performance<-matrix(data=c(RMSE_lm,RMSE_lasso,RMSE_ridge,RMSE_Eln,rmse_svr_nr,rmse_l2_svr,MLP_NR_RMSE,MLP_L1_RMSE,MLP_L2_RMSE,MLP_EN_RMSE),ncol=1)
rownames(Performance)<-c('LinearRegres','LinearLasso','LinearRidge','LinearElastic','SVM','SVMl2','MLP','MLPl1','MLPl2','MLPElastic')
colnames(Performance)<-c('RMSE')
Performance

# Prediction RMSE
reg_nr[1]
reg_l1[1]
reg_l2[1]
reg_EN[1]
svm_nr[1]
svm_l2[1]
mlp_nr[1]
mlp_l1[1]
mlp_l2[1]
mlp_EN[1]

Pred_Performance <- matrix(data=c(reg_nr[1],
                                  reg_l1[1],
                                  reg_l2[1],
                                  reg_EN[1],
                                  svm_nr[1],
                                  svm_l2[1],
                                  mlp_nr[1],
                                  mlp_l1[1],
                                  mlp_l2[1],
                                  mlp_EN[1]), ncol = 1)
rownames(Pred_Performance)<-c('LinearRegres','LinearLasso','LinearRidge','LinearElastic','SVM','SVMl2','MLP','MLPl1','MLPl2','MLPElastic')
colnames(Pred_Performance)<-c('RMSE')
Pred_Performance
