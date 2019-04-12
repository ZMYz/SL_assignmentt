# switch the working directory before runing the code

#load packages

install.packages("lattice")
install.packages("caret")
install.packages("kernlab")
install.packages("corrplot")
install.packages("e1071")
install.packages('ggplot2')
install.packages("kernlab")
install.packages("Matrix")

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
anyNA(US)
anyNA(USMatrix)

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
anyNA(USFrame)

set.seed(1)
intrain <- createDataPartition(y = USFrame$Unemployment,
                              p = 0.8,
                              list = FALSE)
US_training <- USFrame[intrain, ]
US_testing <- USFrame[-intrain, ]
dim(US_training)
dim(US_testing)

head(US_training)

#linear regression
set.seed(5)

US_lm <- train(Unemployment ~ .,
              US_training,
              method = 'lm',
              preProcess = c('center', 'scale'),
              trControl = US_myControl2)
# appear warning because of using too many predictors for the lm function
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
plot(varImp(US_ridge), main = "importance of variables in ridge regularized linear regression model for US cencus dataset")  # the importance rank of variables

###remove insignificant predictors and rerun the model, and compare

US_ridge2 <- train(Unemployment ~ . - Carpool - Women - TotalPop - IncomePerCapErr - Men - Asian - Production,
                  data = US_training,
                  method = 'glmnet',
                  preProcess = c('center', 'scale'),
                  tuneGrid = US_myGrid,
                  trControl = US_myControl2)
US_ridge_test2 <- predict(US_ridge, newdata = US_testing)

US_ridge2
plot(US_ridge2, main = "RMSE value for ridge regression model when dropout insignificant variables")
plot(US_ridge2$finalModel, xvar = 'lambda', label = TRUE)  #increasing lambda helps to reduce the size of coefficients
plot(varImp(US_ridge2))  # the importance rank of variables

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
                          lambda = seq(0.0001, 0.2, length = 10))
#alpha=0:1: choose 0 or 1
US_ElasticNet <- train(Unemployment ~ .,
                       data = US_training,
                       method = 'glmnet',
                       preProcess = c('center', 'scale'),
                       tuneGrid = US_myGrid3,
                       trControl = US_myControl)

US_EN_test <- predict(US_ElasticNet, US_testing)
plot(US_ElasticNet)
plot(US_ElasticNet$finalModel, xvar = 'lambda', label = TRUE)
plot(US_ElasticNet$finalModel, xvar = 'dev', label = TRUE)
# compare 3 types of models by RMSE, not working," "There are different numbers of resamples in each model"

US_model_list <- list(ridge = US_ridge,
                      lasso = US_lasso,
                      ElasticNet = US_ElasticNet,
                      lm = US_lm)
US_resamples <- resamples(US_model_list)
summary(US_resamples)
bwplot(US_resamples, metric = "RMSE")
dotplot(US_resamples, metric = 'RMSE')
#####Plotting and comparing
##########################
####################Regression Approach Comparison##################

#model_list <-list(ridge=ridge,lasso=lasso,ElasticNet=ElasticNet,lm=lm)
US_model_list <-list(US_ridge=US_ridge,US_lasso = US_lasso,US_lm = US_lm)
summary(US_model_list)

US_resamp <- resamples(US_model_list)
summary(US_resamp)

US_EN_Samp <- US_ElasticNet$results

bwplot(US_resamp, metric="RMSE")
bwplot(US_EN_Samp$RMSE)

dotplot(US_resamp, metric='RMSE')
dotplot(US_EN_Samp$RMSE)

# Looking at the R-squared
bwplot(US_resamp, metric="Rsquared")
bwplot(US_EN_Samp$Rsquared)
dotplot(US_resamp, metric='Rsquared')
dotplot(US_EN_Samp$Rsquared)

####################Regression Model Comparison##################

#ord_EN_Samp <- EN_Samp[order(EN_Samp$Resample),]

US_index <- rep(1:10, 10)
US_Grouped_EN_Samp <- aggregate(x=US_EN_Samp[,1:3], by = list(US_index), FUN=mean)

#Group_R2 <- data.frame(Ridge = resamp$values$`ridge~Rsquared`, Lasso = resamp$values$`lasso~Rsquared`, ElasticNet = Grouped_EN_Samp$Rsquared, LM = resamp$values$`lm~Rsquared`)
US_Group_R2 <- data.frame(Ridge = US_resamp$values$`US_ridge~RMSE`, 
                       Lasso = US_resamp$values$`US_lasso~RMSE`, 
                       ElasticNet = US_Grouped_EN_Samp$RMSE, 
                       LM = US_resamp$values$`US_lm~RMSE`)
US_GR2 <- cbind(US_Group_R2, ID = rownames(US_Group_R2))

summary(US_Group_R2)
US_Mean_R2 <- data.frame(colMeans(US_Group_R2))
US_Mean_R2

US_R2_sorted <- apply(US_Group_R2, 2, sort, decreasing = F)
US_R2_sorted <- data.frame(US_R2_sorted)

US_R2_sorted <- cbind(US_R2_sorted, ID = rownames(US_R2_sorted))

plot(US_R2_sorted[,1:4])

US_R2_sorted$ID <- factor(US_R2_sorted$ID, levels = US_R2_sorted$ID[order(1:100)])

ggplot(US_R2_sorted, aes(x = order(ID))) + 
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
ggplot(US_GR2, aes(x = order(ID))) + 
  geom_line(aes(y = Ridge), colour = "blue") + 
  geom_line(aes(y = Lasso), colour = "green") + 
  geom_line(aes(y = ElasticNet), colour = "red") + 
  geom_line(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 10)) +
  ylab("R-Squared") + 
  xlab("Iteration") +
  ggtitle("Plot of R-2 per Regularisation Approch")

# Non-sorted data dot plot
ggplot(US_GR2, aes(x = order(ID))) + 
  geom_point(aes(y = Ridge), colour = "blue") + 
  geom_point(aes(y = Lasso), colour = "green") + 
  geom_point(aes(y = ElasticNet), colour = "red") + 
  geom_point(aes(y = LM), colour = "orange") +
  scale_x_discrete(limits = seq(0, 100, 10)) +
  ylab("R-Squared") + 
  xlab("Iteration") +
  ggtitle("Dot Plot of R-2 per Regularisation Approch")

# Density Plot
ggplot(US_GR2) + 
  geom_density(aes(Ridge, colour = "Ridge")) + #Blue
  geom_density(aes(Lasso, colour = "Lasso")) + #Green
  geom_density(aes(ElasticNet, colour = "ElasticNet")) + #Red
  geom_density(aes(LM, colour = "LM"))

# Density plots - using the melt function
library(reshape2)

US_melt_GR2 <- melt(US_GR2)

ggplot(US_melt_GR2) + 
  geom_density(aes(value, colour = variable)) + 
  facet_grid(.~variable)

ggplot(US_melt_GR2) + 
  geom_density(aes(value, colour = variable))

ggplot(US_melt_GR2) + 
  geom_jitter(aes(value, 0, colour = variable)) + 
  facet_grid(.~variable)

# Combined plot - density and points
ggplot(US_melt_GR2) + 
  geom_density(aes(value, colour = variable)) + 
  geom_point(aes(value, 0, colour = variable)) +
  facet_grid(.~variable)

####################Regression Prediction versus Actual############

US_lm_pred_list <- list(actual = US_testing$Unemployment, ridge = US_ridge_test, lasso = US_lasso_test, ElasticNet = US_EN_test, lm = US_lm_test)
US_lm_pred_df <- data.frame(US_lm_pred_list)

US_cor_accuracy <- cor(US_lm_pred_df)
US_cor_accuracy[1,]

US_sort_lm_pred_df <- US_lm_pred_df[order(US_lm_pred_df$actual),]
US_slpd <- cbind(US_sort_lm_pred_df, ID = 1:nrow(US_testing))

US_lmpd <- cbind(US_lm_pred_df, ID = 1:nrow(US_testing))

# Plotting the four comparison plots of Actual versus predicted
  # no regularization
ggplot(US_lmpd, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("No Regularisation") +
  theme_bw()
  # lasso
ggplot(US_lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = lasso), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lasso), shape = 1) +  # Points of predicted values
  ggtitle("Lasso") +
  theme_bw()
  # ridge
ggplot(US_lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("Ridge") +
  theme_bw()
  #Elastic net
ggplot(US_lmpd, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ElasticNet), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ElasticNet), shape = 1) +  # Points of predicted values
  ggtitle("Ealsticnet") +
  theme_bw()

# Accuracy of the prediction versus actual
US_reg_nr <- postResample(pred = US_lm_pred_df$lm, obs = US_lm_pred_df$actual)
US_reg_l1 <- postResample(pred = US_lm_pred_df$lasso, obs = US_lm_pred_df$actual)
US_reg_l2 <- postResample(pred = US_lm_pred_df$ridge, obs = US_lm_pred_df$actual)
US_reg_EN <- postResample(pred = US_lm_pred_df$ElasticNet, obs = US_lm_pred_df$actual)
US_reg_nr
US_reg_l1
US_reg_l2
US_reg_EN


####################support vector regression########

US_y_training<- US_training$Employment
US_x_training<-as.matrix(US_training)[,-34]
US_x_testing<-as.matrix(US_testing)[,-34]
US_y_testing<-US_testing$Employment

#no-regularisation
US_svr_nr<-train(Unemployment~.,
                 data = US_training,
                 method='svmRadial',
                 preProcess = c('center','scale'),
                 trControl = US_myControl2)

US_svr_nr_test <- predict(US_svr_nr, US_testing)

US_svr_nr
plot(US_svr_nr)
plot(varImp(US_svr_nr))

plot(x = US_svr_nr_test, y = US_testing$Unemployment,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)

#L2
library(LiblineaR)
US_svr_l2<-train(Unemployment~.,
                 data = US_training,
                 method='svmLinear3',
                 preProcess = c('center','scale'),
                 trControl = US_myControl2)


US_svr_l2
US_pred_SVR2 <- predict(US_svr_l2, US_x_testing)
summary(US_pred_SVR2)

#svr

plot(US_testing$Unemployment)
plot(US_pred_SVR2)

plot(x = US_pred_SVR2, y = US_testing$Unemployment,
     xlab = "predicted", ylab = "actual")
abline(a=0,b=1)


####################Comparison of SVR Predictions########

US_svr_pred_list <- list(actual = US_testing$Unemployment, lm = US_svr_nr_test, ridge = US_pred_SVR2)
US_svr_pred_df <- data.frame(US_svr_pred_list)

US_svr_cor_accuracy <- cor(US_svr_pred_df)
US_svr_cor_accuracy[1,]

US_sort_svr_pred_df <- US_svr_pred_df[order(US_svr_pred_df$actual),]
US_sort_svr <- cbind(US_sort_svr_pred_df, ID = 1:nrow(US_testing))

US_svr_id <- cbind(US_svr_pred_df, ID = 1:nrow(US_testing))

# Plotting the two comparison plots of Actual versus predicted
ggplot(US_svr_id, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("No Regularisation") +
  theme_bw()

ggplot(US_svr_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("Ridge") +
  theme_bw()

# Accuracy of the prediction versus actual
postResample(pred = US_svr_pred_df$lm, obs = US_svr_pred_df$actual)
postResample(pred = US_svr_pred_df$ridge, obs = US_svr_pred_df$actual)

US_svm_nr <- postResample(pred = US_svr_pred_df$lm, obs = US_svr_pred_df$actual)
US_svm_l2 <- postResample(pred = US_svr_pred_df$ridge, obs = US_svr_pred_df$actual)


####################Multiple Layer perceptron############

install.packages('BBmisc')
library(BBmisc)
scaledUSframe<-normalize(USFrame,method='scale',range=c(0,60))

US_training2 <- data.frame(scaledUSframe[intrain, ])
US_testing2 <- data.frame(scaledUSframe[-intrain, ])

US_ytrain2 <- US_training2$Unemployment
US_xtrain2 <- model.matrix(Unemployment~.,data = US_training2)[,-34]
US_ytest2 <- US_testing2$Unemployment
US_xtest2<- model.matrix(Unemployment~.,data = US_testing2)[,-34]

install.packages("ANN2")
library(ANN2)
fold10 <- createFolds(US_training2$Unemployment,k=10,list=F)

US_mlp_10cv<-function(data,l1,l2,hiddenlayers,learnrate,iterations){
  #US_RMSE=list()
  US_R2=list()
  rmse_val <- 5000000
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
                            activ.functions = "relu", random.seed = 5,batch.size = 500,   # don't set batch.size too small
                            standardize = F,
                            L1 = l1, L2 = l2,
                            learn.rates = learnrate,
                            verbose=F, val.prop=0.2, 
                            n.epochs = iterations)
    #train the model
    US_mlp_test <- predict(US_mlp,US_xtest)
    #prediction
    US_pred <- unlist(US_mlp_test) #convert into vector
    pred_RMSE = RMSE(US_ytest, US_pred)
    #select the predictive model with the highest r2
    if(pred_RMSE <= rmse_val){
      US_finalmodel <<- US_mlp
      rmse_val<-pred_RMSE
      B_rmse <<- rmse_val
    }
  }
  print(US_finalmodel)
  print(rmse_val)
  #print(US_R2)
}

#no regularisation
set.seed(111)
US_rmse_mlp <- US_mlp_10cv(US_training2,0,0,c(23,15),0.01,100)
US_noreg_mlp_model <- US_finalmodel
US_MLP_NR_RMSE <- B_rmse
US_finalpred_noreg<-predict(US_noreg_mlp_model,US_xtest2)
postResample(pred=US_finalpred_noreg$predictions, obs = US_ytest2)


#l1-lasso regularisation
set.seed(222)
US_rmse_l1_mlp <- US_mlp_10cv(US_training2,1,0,c(23,15),0.01,100)
US_l1_mlp_model <- US_finalmodel
US_MLP_L1_RMSE <- B_rmse
US_finalpred_l1<-predict(US_l1_mlp_model,US_xtest2)
postResample(pred=US_finalpred_l1$predictions, obs = US_ytest2)

#l2-ridge regularisation
set.seed(333)
US_rmse_l2_mlp <- US_mlp_10cv(US_training2,0,1,c(23,15),0.01,100)
US_l2_mlp_model <- US_finalmodel
US_MLP_L2_RMSE <- B_rmse
US_finalpred_l2 <- predict(US_l2_mlp_model,US_xtest2)
postResample(pred=US_finalpred_l2$predictions, obs = US_ytest2)

#Elastic net
set.seed(444)
US_rmse_en_mlp <- US_mlp_10cv(US_training2,1,1,c(23,15),0.01,100)
US_en_mlp_model <- US_finalmodel
US_MLP_EN_RMSE <- B_rmse
US_finalpred_l3<-predict(US_en_mlp_model,US_xtest2)
postResample(pred=US_finalpred_l3$predictions, obs = US_ytest2)

# Plotting the training Models for MLP
plot(US_noreg_mlp_model)
plot(US_l1_mlp_model)
plot(US_l2_mlp_model)
plot(US_en_mlp_model)


################ MLP Plotting and Analysis ########################

US_mlp_pred_list <- list(actual = US_testing$Unemployment, 
                         ridge = US_finalpred_l2$predictions[,1], 
                         lasso = US_finalpred_l1$predictions[,1], 
                         ElasticNet = US_finalpred_l3$predictions[,1], 
                         lm = US_finalpred_noreg$predictions[,1])
US_mlp_pred_df <- data.frame(US_mlp_pred_list)

US_mlp_cor_accuracy <- cor(US_mlp_pred_df)
# standard deviation is zero
US_mlp_cor_accuracy[1,]

US_sort_mlp_pred_df <- US_mlp_pred_df[order(US_mlp_pred_df$actual),]
US_sort_mlp <- cbind(US_sort_mlp_pred_df, ID = 1:nrow(US_testing))

US_mlp_id <- cbind(US_mlp_pred_df, ID = 1:nrow(US_testing))

# Plotting the four comparison plots of Actual versus predicted
ggplot(US_mlp_id, aes(x = ID, y = actual)) +
  #geom_smooth(method = "lm", se = FALSE, color = "lightgrey") + # Add the regression line
  geom_segment(aes(xend = ID, yend = lm), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lm), shape = 1) +  # Points of predicted values
  ggtitle("No Regularisation") +
  theme_bw()

ggplot(US_mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = lasso), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = lasso), shape = 1) +  # Points of predicted values
  ggtitle("Lasso") +
  theme_bw()

ggplot(US_mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ridge), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ridge), shape = 1) +  # Points of predicted values
  ggtitle("Ridge") +
  theme_bw()

ggplot(US_mlp_id, aes(x = ID, y = actual)) +
  geom_segment(aes(xend = ID, yend = ElasticNet), alpha = .2) +  # Lines to connect points
  geom_point() +  # Points of actual values
  geom_point(aes(y = ElasticNet), shape = 1) +  # Points of predicted values
  ggtitle("Ealsticnet") +
  theme_bw()

# Accuracy of the prediction versus actual
postResample(pred = US_mlp_pred_df$lm, obs = US_mlp_pred_df$actual)
postResample(pred = US_mlp_pred_df$lasso, obs = US_mlp_pred_df$actual)
postResample(pred = US_mlp_pred_df$ridge, obs = US_mlp_pred_df$actual)
postResample(pred = US_mlp_pred_df$ElasticNet, obs = US_mlp_pred_df$actual)

US_mlp_nr <- postResample(pred = US_mlp_pred_df$lm, obs = US_mlp_pred_df$actual)
US_mlp_l1 <- postResample(pred = US_mlp_pred_df$lasso, obs = US_mlp_pred_df$actual)
US_mlp_l2 <- postResample(pred = US_mlp_pred_df$ridge, obs = US_mlp_pred_df$actual)
US_mlp_EN <- postResample(pred = US_mlp_pred_df$ElasticNet, obs = US_mlp_pred_df$actual)

######## Overall Comparison of Models #######
#check the average RMSE of different techniques

US_a_en <- US_ElasticNet$results[US_ElasticNet$results$alpha == US_ElasticNet$bestTune$alpha,]
US_l_ln <- US_a_en[US_a_en$lambda == US_ElasticNet$bestTune$lambda,]

US_lasso_val <- US_lasso$results[US_lasso$results$lambda == US_lasso$bestTune$lambda,]
US_ridge_val <- US_ridge$results[US_ridge$results$lambda == US_ridge$bestTune$lambda,]

US_svr2_rmse_prep <- US_svr_l2$results[US_svr_l2$results$cost == US_svr_l2$bestTune$cost,]
US_svr2_rmse <- US_svr2_rmse_prep[US_svr_l2$results$Loss == US_svr_l2$bestTune$Loss,]

US_RMSE_lm <- US_lm$results$RMSE
US_RMSE_lasso<- US_lasso_val$RMSE
US_RMSE_ridge<- US_ridge_val$RMSE
US_RMSE_Eln<- US_l_ln$RMSE
US_rmse_svr_nr <- US_svr_nr$results$RMSE[3]
US_rmse_l2_svr <- US_svr2_rmse$RMSE[1]
US_MLP_NR_RMSE
US_MLP_L1_RMSE
US_MLP_L2_RMSE
US_MLP_EN_RMSE

US_Performance<-matrix(data=c(US_RMSE_lm,US_RMSE_lasso,
                           US_RMSE_ridge,US_RMSE_Eln,
                           US_rmse_svr_nr,
                           US_rmse_l2_svr,
                           US_MLP_NR_RMSE,
                           US_MLP_L1_RMSE,
                           US_MLP_L2_RMSE,
                           US_MLP_EN_RMSE),ncol=1)
rownames(US_Performance)<-c('LinearRegres','LinearLasso',
                         'LinearRidge','LinearElastic',
                         'SVM','SVMl2','MLP','MLPl1','MLPl2',
                         'MLPElastic')
colnames(US_Performance)<-c('RMSE')
US_Performance

# Prediction RMSE
US_reg_nr[1]
US_reg_l1[1]
US_reg_l2[1]
US_reg_EN[1]
US_svm_nr[1]
US_svm_l2[1]
US_mlp_nr[1]
US_mlp_l1[1]
US_mlp_l2[1]
US_mlp_EN[1]

US_Pred_Performance <- matrix(data=c(US_reg_nr[1],
                                     US_reg_l1[1],
                                     US_reg_l2[1],
                                     US_reg_EN[1],
                                     US_svm_nr[1],
                                     US_svm_l2[1],
                                     US_mlp_nr[1],
                                     US_mlp_l1[1],
                                     US_mlp_l2[1],
                                     US_mlp_EN[1]), ncol = 1)
rownames(US_Pred_Performance)<-c('LinearRegres','LinearLasso',
                                 'LinearRidge','LinearElastic',
                                 'SVM','SVMl2','MLP','MLPl1','MLPl2',
                                 'MLPElastic')
colnames(US_Pred_Performance)<-c('RMSE')
US_Pred_Performance
