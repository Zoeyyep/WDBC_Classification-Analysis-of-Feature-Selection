#读取包
library(dplyr)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(scorecard)
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(VIM)
library(corrplot)
library(e1071)
library(xgboost)
library(Matrix)
library(neuralnet)
library(pROC)
set.seed(1024)
#Set the file path (Note: you need to replace it with the directory where the current file is located)
setwd("F:/course/Nottingham/Data Modelling and Analysis/coursework2")

#Read data (source:http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
data=read.csv("~/Desktop/DM_CW/breast-cancer-wisconsin.csv",header = FALSE) #For Macbook
data=read.csv("breast-cancer-wisconsin.csv",header = FALSE) #For windows

#Assigning column attribute names to data
col_names=c("Id","Cl.thickness","Cell.size","Cell.shape","Marg.adhesion",
            "Epith.c.size","Bare.nuclei","Bl.cromatin","Normal.nucleoli",
            "Mitoses","Class")
colnames(data)=col_names

# Step 1 : Data Pre-processing
View(data)
sapply(data,class)
str(data)
head(data,1)
dim(data)
summary(data)
data=data[,2:11]  #remove column of ID

#observe the class attribute
ggplot(data,aes(x=Class))+geom_bar(alpha=0.5, fill='blue')+ 
  xlab("benign(2)        vs.       malignant(4)")+ylab("Frequency")+
  theme_bw()+theme_minimal()

#transfere data type
data[,1:9]= as.numeric(unlist(data[,1:9]))
data$Class=factor(data$Class,levels = c(2,4),labels = c(0,1))

#Visualisation of missing values
aggr(data,prop=FALSE, numbers=TRUE) 
sapply(data, function(x) sum(is.na(x))) #Observe the number of missing the missing value of Bare.nuclei is 16
sapply(data, function(x) 100*sum(is.na(x))/length(x)) #Proportion of missing values

#Missing value deletion (Hint: you just need only choose one method to fill the missing data)
data=drop_na(data)

#Missing values to fill median
data[is.na(data[,"Bare.nuclei"]),"Bare.nuclei"]=median(data[,"Bare.nuclei"],na.rm = TRUE)

#Missing values to fill in the average
data[is.na(data[,"Bare.nuclei"]),"Bare.nuclei"]=mean(data[,"Bare.nuclei"],na.rm = TRUE)

#Check the data again
sapply(data, function(x) 100*sum(is.na(x))/length(x))


# Step 2 Exploratory Data Analysis
# Correlation analysis of variables
cor<-cor(data[,1:9])#Calculating the correlation matrix
corrplot(cor, method = "pie",shade.col = NA, tl.col ="black", tl.srt = 45, order = "AOE") 

#Observe the distribution of attributes
ggplot(stack(data[,1:9]),aes(x=ind,y=values))+
  geom_boxplot(position=position_dodge(0.6),
               size=0.5,
               width=0.3,
               color="blue",
               outlier.color = "red",
               notchwidth = 0.5)+xlab("Attributes")+ylab("Values")+
  ggtitle("Boxplot of Attributes")

#Removal of outliers
rmv_oulier=function(df){
  fields=names(df)
  for (field in fields){
    qnt=quantile(df[,field],probs = c(0.25,0.75))
    limits=quantile(df[,field],probs = c(0.1,0.9))
    H=1.5*IQR(df[,field])
    df[df[,field] < (qnt[1] - H), field] =limits[1]
    df[df[,field] > (qnt[2] + H),field] = limits[2]
  }
  return(df)
}


data[,1:9]=rmv_oulier(data[,1:9]) #remove outliers

#Observe the distribution of attributes again
ggplot(stack(data[,1:9]),aes(x=ind,y=values))+
  geom_boxplot()+xlab("Attributes")+ylab("Values")+
  ggtitle("Boxplot of Attributes")

#Data scaled
data[,1:9]=scale(data[,1:9])

#Feature selection (method:rfe)
results=rfe(x=data[,1:9],y=data[,10],
      sizes = c(1:9),
    rfeControl = rfeControl(functions=rfFuncs, method="cv", number=10))
results  

#create 5 folds
folds=createFolds(data$Class,5)

#Method: Random Forest
#find the best parameters
rate=1
for (j in 1:9){
  set.seed(1024)
  model=randomForest(Class~.,data=data,importance=TRUE,mtry=j,ntree=1000)
  rate[j]=mean(model$err.rate)
}
rate  #first find the best mtry and the result is mtry=1

model.randomforest=randomForest(Class~.,data=data,importance=TRUE,mtry=1,ntree=1000)
plot(model.randomforest) #In this plot, it seems the results appear station after 200 trees 

randomforest.auc=1
randomforest.accuracy=1
randomforest.sensitivities=1
randomforest.specificities=1

for (i in 1:5){
  test=data[folds[[i]],]  #create test dataset
  train=data[-folds[[i]],]  #create train dataset
  model.randomforest=randomForest(Class~.,data=train,importance=TRUE,mtry=1,ntree=200) #create model
  randomforest.predict=predict(model.randomforest,test)
  #auc
  rf.roc=roc(as.numeric(test$Class),as.numeric(randomforest.predict)) #create roc for random forest
  randomforest.auc[i]=auc(rf.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,randomforest.predict)
  randomforest.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  randomforest.sensitivities[i]=rf.roc$sensitivities[2]
  #randomforest.specificities
  randomforest.specificities[i]=rf.roc$specificities[2]
}
randomforest.auc=mean(randomforest.auc) 
randomforest.auc

randomforest.accuracy=mean(randomforest.accuracy) 
randomforest.accuracy

randomforest.sensitivities=mean(randomforest.sensitivities) 
randomforest.sensitivities

randomforest.specificities=mean(randomforest.specificities)  
randomforest.specificities

#Final result
c(randomforest.accuracy,randomforest.sensitivities,randomforest.specificities,randomforest.auc)
#ROC of randomforest
plot(rf.roc, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)


#Method: logistic regression

#Fit logistic Regression Model
fit.logit <- glm(Class~.,data = train,family = binomial())
summary(fit.logit)  #Descriptive statistics for variables

#fit after removing variables
fit.logit2 <- glm(Class ~ Cl.thickness + Marg.adhesion +Bare.nuclei +Bl.cromatin,
                  data = train,family = binomial())
summary(fit.logit2)

#function comparison
anova(fit.logit2,fit.logit,test = 'Chisq')

coef(fit.logit2)  #View regression coefficients
exp (coef(fit.logit2)) #The result is indexed


logistic.auc=1
logistic.accuracy=1
logistic.sensitivities=1
logistic.specificities=1
for (i in 1:5){
  set.seed(1024)
  test=data[folds[[i]],]  #create test dataset
  train=data[-folds[[i]],]  #create train dataset
  model.logistic=fit.logit2
  logistic.predict=predict(model.logistic,test,type = 'response')
  logistic.predict=factor(logistic.predict >0.5 ,levels = c(FALSE,TRUE),labels = c(0,1))
  #auc
  logistic.roc=roc(as.numeric(test$Class),as.numeric(logistic.predict))
  logistic.auc[i]=auc(rf.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,logistic.predict)
  logistic.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  logistic.sensitivities[i]=logistic.roc$sensitivities[2]
  #randomforest.specificities
  logistic.specificities[i]=logistic.roc$specificities[2]
}
logistic.auc=mean(logistic.auc)
logistic.auc

logistic.accuracy=mean(logistic.accuracy)
logistic.accuracy

logistic.sensitivities=mean(logistic.sensitivities)
logistic.sensitivities

logistic.specificities=mean(logistic.specificities)
logistic.specificities

#Final result
c(logistic.accuracy,logistic.sensitivities,logistic.specificities,logistic.auc)
#ROC of logistic model
plot(logistic.roc, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)



#Method: Support Vector Machine SVM
svm.auc=1
svm.accuracy=1
svm.sensitivities=1
svm.specificities=1


model.svm=function(x){
  svm.auc=1
  svm.accuracy=1
  for (i in 1:5){
    set.seed(1024)
    test=data[folds[[i]],]  #create test dataset
    train=data[-folds[[i]],]  #create train dataset
    model=svm(formula=Class ~ ., data=train,kernel= x)  #we can choose different kernel for svm
    svm.predict=predict(model,test)
    #roc
    svm.roc=roc(as.numeric(test$Class),as.numeric(svm.predict))
    svm.auc[i]=auc(svm.roc)
    svm.roc$sensitivities
    svm.roc$specificities
    #accuracy svm
    cf_matrix=confusionMatrix(test$Class,svm.predict)
    svm.accuracy[i]=cf_matrix$overall[1]
    #sensitivities
    svm.sensitivities[i]=svm.roc$sensitivities[2]
    #randomforest.specificities
    svm.specificities[i]=svm.roc$specificities[2]
  }
  return(c(mean(svm.accuracy),mean(svm.sensitivities),mean(svm.specificities),mean(svm.auc)))
}
svm.radial=model.svm("radial")  
svm.linear=model.svm("linear")  
svm.polynomial=model.svm("polynomial") 
svm.sigmoid=model.svm("sigmoid") 
rbind(svm.radial,svm.linear,svm.polynomial,svm.sigmoid)

#compares results of different kernel
plot(1:4,svm.radial,col="black",type="l",ylim=c(0.875,1),xaxt="n",ylab="")
axis(1,1:4,c("accuracy","sensitivities","specificities","auc"),col.axis="blue")
lines(1:4,svm.linear,col="blue")
lines(1:4,svm.polynomial,col="green")
lines(1:4,svm.sigmoid,col="red")
legend("bottomright",c("radial","linear","polynomial","sigmoid"),col=c("black","blue","green","red"),pch=16)
  
#Methods: Neural Networks
train$benign=train$Class==0
train$malignant=train$Class==1
ann.auc=1
ann.accuracy=1
ann.sensitivities=1
ann.specificities=1
#Finde the best number of hidden layer
for (i in 1:3){
  set.seed(1024)
  network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+Marg.adhesion+
                      Epith.c.size+Bare.nuclei+Bl.cromatin+Normal.nucleoli+
                      Mitoses+Cell.size,train,hidden = i)
  net.predict=compute(network,test[,1:9])$net.result
  net.predict=c(0,1)[apply(net.predict,1,which.max)]
  ann.roc=roc(as.numeric(test$Class),as.numeric(net.predict))
  ann.auc[i]=auc(ann.roc)
}
ann.auc  #1 hidden layer is the best
network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+Marg.adhesion+
                    Epith.c.size+Bare.nuclei+Bl.cromatin+Normal.nucleoli+
                    Mitoses,train,hidden = 1)
plot(network)  #plot of ann model

for (i in 1:5){
  set.seed(1024)
  test=data[folds[[i]],] #create test dataset
  train=data[-folds[[i]],]  #create train dataset
  train$benign=train$Class==0
  train$malignant=train$Class==1
  network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+Marg.adhesion+
                      Epith.c.size+Bare.nuclei+Bl.cromatin+Normal.nucleoli+
                      Mitoses+Cell.size,train,hidden = 1,linear.output = FALSE)
  net.predict=compute(network,test[,1:9])$net.result         
  net.predict=c(0,1)[apply(net.predict,1,which.max)]
  #auc
  ann.roc=roc(as.numeric(test$Class),as.numeric(net.predict))
  ann.auc[i]=auc(ann.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,as.factor(net.predict))
  ann.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  ann.sensitivities[i]=ann.roc$sensitivities[2]
  #randomforest.specificities
  ann.specificities[i]=ann.roc$specificities[2]
}
ann.auc=mean(ann.auc)
ann.accuracy=mean(ann.accuracy)
ann.sensitivities=mean(ann.sensitivities)
ann.specificities=mean(ann.specificities)
#Final result
c(ann.accuracy,ann.sensitivities,ann.specificities,ann.auc)

#Method: XGBOOST
#read data again
data=read.csv("breast-cancer-wisconsin.csv",header = FALSE)
col_names=c("Id","Cl.thickness","Cell.size","Cell.shape","Marg.adhesion",
            "Epith.c.size","Bare.nuclei","Bl.cromatin","Normal.nucleoli",
            "Mitoses","Class")
colnames(data)=col_names
data=data[,2:11]
data$Class=factor(data$Class,levels = c(2,4),labels = c(0,1))
data=drop_na(data)
data[,1:9]=rmv_oulier(data[,1:9])
data[,1:9]=scale(data[,1:9])


xgb.auc=1
xgb.accuracy=1
xgb.sensitivities=1
xgb.specificities=1
for (i in 1:5){
  set.seed(1024)
  test=data[folds[[i]],]
  train=data[-folds[[i]],]
  train_matrix=sparse.model.matrix(Class~.-1,train)  #transform data into sparse matrix
  test_matrix=sparse.model.matrix(Class~.-1,test)
  train_label=as.numeric(train$Class==1)
  test_label=as.numeric(test$Class==1)
  train1=list(data=train_matrix,label=train_label)
  test1=list(data=test_matrix,label=test_label)
  new_train=xgb.DMatrix(data=train1$data,label=train1$label)
  new_test=xgb.DMatrix(data=test1$data,label=test1$label)
  xgb=xgboost(data=new_train,eta=0.2,objective="binary:logistic",nround=100) #create model
  xgb.predict=round(predict(xgb,newdata = new_test))
  #auc
  xgb.roc=roc(as.numeric(test$Class),as.numeric(xgb.predict))
  xgb.auc[i]=auc(xgb.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,as.factor(xgb.predict))
  xgb.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  xgb.sensitivities[i]=xgb.roc$sensitivities[2]
  #randomforest.specificities
  xgb.specificities[i]=xgb.roc$specificities[2]
}
xgb.accuracy=mean(xgb.accuracy)
xgb.sensitivities=mean(xgb.sensitivities)
xgb.specificities=mean(xgb.specificities)
xgb.auc=mean(xgb.auc)
#Final result
c(xgb.accuracy,xgb.sensitivities,xgb.specificities,xgb.auc)
plot(xgb.roc, 
     auc.polygon=T, 
     auc.polygon.col="skyblue",
     grid=c(0.1, 0.2),
     grid.col=c("green", "red"), 
     max.auc.polygon=T,
     print.thres=T)

#Data after feature selection
data=read.csv("breast-cancer-wisconsin.csv",header = FALSE)
col_names=c("Id","Cl.thickness","Cell.size","Cell.shape","Marg.adhesion",
            "Epith.c.size","Bare.nuclei","Bl.cromatin","Normal.nucleoli",
            "Mitoses","Class")
colnames(data)=col_names
data=data[,2:11]
data$Class=factor(data$Class,levels = c(2,4),labels = c(0,1))
data=drop_na(data)
data[,1:9]=rmv_oulier(data[,1:9])
data[,1:9]=scale(data[,1:9])
data=data[,c("Bare.nuclei", "Cl.thickness", "Cell.size", "Cell.shape", "Bl.cromatin","Class")] #5 attributes are selected

#After Feature selection (Hint: you can go back to the random forest part to train model)

#svm
model.svm=function(x){
  svm.auc=1
  svm.accuracy=1
  for (i in 1:5){
    set.seed(1024)
    test=data[folds[[i]],]
    train=data[-folds[[i]],]
    model=svm(formula=Class ~ ., data=train,kernel= x)
    svm.predict=predict(model,test)
    #roc
    svm.roc=roc(as.numeric(test$Class),as.numeric(svm.predict))
    svm.auc[i]=auc(svm.roc)
    svm.roc$sensitivities
    svm.roc$specificities
    #accuracy svm
    cf_matrix=confusionMatrix(test$Class,svm.predict)
    svm.accuracy[i]=cf_matrix$overall[1]
    #sensitivities
    svm.sensitivities[i]=svm.roc$sensitivities[2]
    #randomforest.specificities
    svm.specificities[i]=svm.roc$specificities[2]
  }
  return(c(mean(svm.accuracy),mean(svm.sensitivities),mean(svm.specificities),mean(svm.auc)))
}
svm.radial=model.svm("radial") 
svm.linear=model.svm("linear")  
svm.polynomial=model.svm("polynomial") 
svm.sigmoid=model.svm("sigmoid") 
rbind(svm.radial,svm.linear,svm.polynomial,svm.sigmoid)
#ANN
train$benign=train$Class==0
train$malignant=train$Class==1
ann.auc=1
ann.accuracy=1
ann.sensitivities=1
ann.specificities=1
for (i in 1:3){
  set.seed(1024)
  #only 5 attributes are used
  network=network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+
                              Bare.nuclei+Bl.cromatin+
                              Cell.size,train,hidden = i,act.fct = "tanh",linear.output = FALSE)
  net.predict=compute(network,test[,1:5])$net.result
  net.predict=c(0,1)[apply(net.predict,1,which.max)]
  ann.roc=roc(as.numeric(test$Class),as.numeric(net.predict))
  ann.auc[i]=auc(ann.roc)
}
ann.auc
network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+
                            Bare.nuclei+Bl.cromatin+
                            Cell.size,train,hidden = 1,act.fct = "tanh",linear.output = FALSE)
plot(network)
for (i in 1:5){
  set.seed(1024)
  test=data[folds[[i]],]
  train=data[-folds[[i]],]
  train$benign=train$Class==0
  train$malignant=train$Class==1
  network=neuralnet(benign+malignant~Cl.thickness+Cell.shape+
                      Bare.nuclei+Bl.cromatin+
                     Cell.size,train,hidden = 1,act.fct = "tanh",linear.output = FALSE)
  net.predict=compute(network,test[,1:5])$net.result         
  net.predict=c(0,1)[apply(net.predict,1,which.max)]
  #auc
  ann.roc=roc(as.numeric(test$Class),as.numeric(net.predict))
  ann.auc[i]=auc(ann.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,as.factor(net.predict))
  ann.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  ann.sensitivities[i]=ann.roc$sensitivities[2]
  #randomforest.specificities
  ann.specificities[i]=ann.roc$specificities[2]
}
ann.auc=mean(ann.auc)
ann.accuracy=mean(ann.accuracy)
ann.sensitivities=mean(ann.sensitivities)
ann.specificities=mean(ann.specificities)
c(ann.accuracy,ann.sensitivities,ann.specificities,ann.auc)

#xgboost
for (i in 1:5){
  set.seed(1024)
  test=data[folds[[i]],]
  train=data[-folds[[i]],]
  train_matrix=sparse.model.matrix(Class~.-1,train)
  test_matrix=sparse.model.matrix(Class~.-1,test)
  train_label=as.numeric(train$Class==1)
  test_label=as.numeric(test$Class==1)
  train1=list(data=train_matrix,label=train_label)
  test1=list(data=test_matrix,label=test_label)
  new_train=xgb.DMatrix(data=train1$data,label=train1$label)
  new_test=xgb.DMatrix(data=test1$data,label=test1$label)
  xgb=xgboost(data=new_train,eta=0.15,objective="binary:logistic",nround=100)
  xgb.predict=round(predict(xgb,newdata = new_test))
  #auc
  xgb.roc=roc(as.numeric(test$Class),as.numeric(xgb.predict))
  xgb.auc[i]=auc(xgb.roc)
  #accuracy
  cf_matrix=confusionMatrix(test$Class,as.factor(xgb.predict))
  xgb.accuracy[i]=cf_matrix$overall[1]
  #sensitivities
  xgb.sensitivities[i]=xgb.roc$sensitivities[2]
  #randomforest.specificities
  xgb.specificities[i]=xgb.roc$specificities[2]
}
xgb.accuracy=mean(xgb.accuracy)
xgb.sensitivities=mean(xgb.sensitivities)
xgb.specificities=mean(xgb.specificities)
xgb.auc=mean(xgb.auc)
c(xgb.accuracy,xgb.sensitivities,xgb.specificities,xgb.auc)
