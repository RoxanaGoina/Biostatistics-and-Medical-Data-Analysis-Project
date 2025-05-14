# Mini project

# upload the dataset from the github repository

# the github page 
medicalDataSetRawValues = read.csv("https://raw.githubusercontent.com/RoxanaGoina/Biostatistics-and-Medical-Data-Analysis-Project/refs/heads/main/Hypertension_dataset.CSV")

medicalDataSetRawValues

medicalDataSetRawValues$HTN = as.factor(medicalDataSetRawValues$HTN)


# import library for adasyn and under sampling
suppressPackageStartupMessages(c(library(caret),library(corrplot),library(smotefamily)))

#check for the na values
sum(is.na(medicalDataSetRawValues)) 

#it contains na values, so apply imputation technique

# a function used for imputation technique
getModeColumn = function(v) {
  uniqv = unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]  # the most frequent mode
}
# identify type of variable
categoricalVariable = names(Filter(is.factor, medicalDataSetRawValues))
categoricalVariable
continuousVariable= names(Filter(is.numeric, medicalDataSetRawValues))
continuousVariable

# use libraries to facilitate the imputation technique
library(dplyr)
library(tidyr)
# apply the imputation technique
cleanData = medicalDataSetRawValues %>%
  mutate(across(all_of(continuousVariable), ~replace_na(as.double(.), mean(., na.rm = TRUE)))) %>%  # for numerical values column, replace the NA values with mean value (for double records)
  mutate(across(all_of(categoricalVariable), ~replace_na(., getModeColumn(.))))  #for categorical values column, replace NA values with the more frequent mode


#statistical analysis

library(dplyr)
# remove the target (HTN) from predictors
continuousVariable = setdiff(continuousVariable, "HTN")

# categorical: freqcy table in %
categoricalVariableSummary = lapply(categoricalVariable, function(var) {
  proportionTable = prop.table(table(cleanData[[var]])) * 100
  data.frame(Variable = var,
             Level = names(proportionTable),
             Percentage = round(as.numeric(proportionTable), 1))
})
categoricalVairabileSummaryDataFrame = bind_rows(categoricalVariableSummary)
categoricalVairabileSummaryDataFrame
# continuous mean ± standard deviation
continuousVariableSummary = lapply(continuousVariable, function(var) {
  values = cleanData[[var]]
  meanValues = mean(values, na.rm = TRUE)
  standardDeviationValues = sd(values, na.rm = TRUE)
  data.frame(Variable = var,
             Mean = round(meanValues, 2),
             SD = round(standardDeviationValues, 2),
             MeanStandardDeviation = paste0(round(meanValues, 2), " ± ", round(standardDeviationValues, 2)))
})
continuousVariableSummaryDataFrame= bind_rows(continuousVariableSummary) 

continuousVariableSummaryDataFrame


#pearson chi^2 test 

chiTests = lapply(categoricalVariable, function(var) {
  table = table(cleanData[[var]], cleanData$HTN)
  test = chisq.test(table)
  data.frame(Variable = var,
             Test = "chi^2",
             p_value = round(test$p.value, 4))
})
chiResultDataFrame = bind_rows(chiTests)
chiResultDataFrame


#create partition for machine learning alg
index = createDataPartition(cleanData$HTN, p = 0.7, list = FALSE)

trainData = cleanData[index, ]
testData = cleanData[-index, ]

table(trainData$HTN)

table(testData$HTN)

# use boruta for feature selection

library(Boruta)

trainData$HTN= as.factor(trainData$HTN)
# apply boruta on trainData 
borutaResult = Boruta(HTN ~ ., data = trainData, doTrace = 3)
finalBorutaAnalys = TentativeRoughFix(borutaResult)
finalBorutaAnalys
#save the selected features
selectedFeatures = getSelectedAttributes(finalBorutaAnalys, withTentative = FALSE)
selectedFeatures = c(selectedFeatures, "HTN")
selectedFeatures
#
#obtained a dataframe which contains only the selected features
trainDataSelected= trainData[, selectedFeatures]
trainDataSelected
#obtain a test data frame with selected features
testDataSelectedFeatures = testData[,selectedFeatures]
#!!!!!!!!!!!
#cleanData =cleanData[, !names(trainDataSelected) %in% c("ID")]

#check for missing values
sum(is.na(trainDataSelected))

# create the sub-data frame: adasyn and under sampling

# Adasyn and under-sampling

xTrain = trainDataSelected[, -which(names(trainDataSelected) == "HTN")]
yTrain = trainDataSelected$HTN
adasynData = ADAS(xTrain, yTrain, K = 5)

#transform into a data frame 

adasynData = data.frame(adasynData$data)
colnames(adasynData)
ncol(adasynData)

#rename the column name class in order to contains HTN

names(adasynData)[names(adasynData) == "class"] = "HTN"

underSamplingData = downSample(x = xTrain,
                               y = yTrain,
                               yname = "HTN")
colnames(underSamplingData)


# train machine learning model;


ctrl = trainControl(
  method = "cv",
  number = 10,
  search = "grid"
)

# create grid for hyperparameter

#grid for random forest
gridRandomForest = expand.grid(
  mtry = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))

#grid for neural network
gridAnn = expand.grid( size=c(1,2,3,4,5,6,7,8, 9, 10), 
                       decay = c(0, 0.1, 0.01, 0.001, 0.001))
# grid for logistical regression
gridLogisticRegression =  expand.grid(
  alpha = c(0,1),
  lambda = c(1/0.001, 1/0.01, 1/0.1, 1/1, 1/10, 1/100, 1/1000)
)
#grid for xGBoost
gridXGBoost =expand.grid(nrounds=c(100,500),
                         max_depth=c(10, 15,20, 25, 30),
                         colsample_bytree = seq (0.5, 0.9, length.out = 5),eta = 0.1,
                         gamma = 0, min_child_weight = 1, subsample = 1)

#applies grids on models train

# ON ADASYN

#randomForest
modelrandomforest = train(
  HTN ~ ., data = adasynData, method = "rf",
  trControl = ctrl, tuneGrid = gridRandomForest
)
modelrandomforest

#neuronalNetwork
modelANN = train(
  HTN ~ ., data = adasynData, method = "nnet",
  trControl = ctrl, tuneGrid = gridAnn,
  trace = FALSE)

modelANN


#logisticalRegression 
modelLogisticalRegression = train(
  HTN ~ ., data = adasynData, method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = gridLogisticRegression
)
modelLogisticalRegression
# XGBoost
modelXGBoost = train(
  HTN ~ ., data = adasynData, method = "xgbTree",
  trControl = ctrl, tuneGrid = gridXGBoost,verbose=0
)
modelXGBoost
 
#predictions and confusion matrix
predictionRandomForest = predict(modelrandomforest, newdata = testDataSelectedFeatures)
predictionRandomForest

confusionMatrixRandomForest = confusionMatrix(predictionRandomForest, testDataSelectedFeatures$HTN)
confusionMatrixRandomForest

predictionANN = predict(modelANN, newdata = testDataSelectedFeatures)
predictionANN

confusionMatrixANN = confusionMatrix(predictionANN, testDataSelectedFeatures$HTN)
confusionMatrixANN

predictionLogisticRegression= predict(modelLogisticalRegression, newdata = testDataSelectedFeatures)
predictionLogisticRegression

confusionMatrixLogisticalRegression = confusionMatrix(predictionLogisticRegression, testDataSelectedFeatures$HTN)
confusionMatrixLogisticalRegression

predictionXGBoost= predict(modelXGBoost, newdata = testDataSelectedFeatures)
predictionXGBoost

confustionMatrixXGBoost = confusionMatrix(predictionXGBoost, testDataSelectedFeatures$HTN)
confustionMatrixXGBoost 


# On undersampling data frame


#randomforest
modelrandomforestUS = train(
  HTN ~ ., data = underSamplingData, method = "rf",
  trControl = ctrl, tuneGrid = gridRandomForest
)
modelrandomforestUS

#ann
modelANN_UnderSampling = train(
  HTN ~ ., data = underSamplingData, method = "nnet",
  trControl = ctrl, tuneGrid = gridAnn,
  trace = FALSE)

modelANN_UnderSampling

#logisticRegression 
modelLogisticalRegressionUs = train(
  HTN ~ ., data = underSamplingData, method = "glmnet",
  family = "binomial",
  trControl = ctrl,
  tuneGrid = gridLogisticRegression
)
modelLogisticalRegressionUs

# XGBoost
modelXGBoostUs = train(
  HTN ~ ., data = underSamplingData, method = "xgbTree",
  trControl = ctrl, tuneGrid = gridXGBoost,verbose=0
)
modelXGBoostUs


#predction and confusion matrix (US = under-sampling)
predictionRandomForest_UndersamplingData = predict(modelrandomforestUS, newdata = testDataSelectedFeatures)
predictionRandomForest_UndersamplingData

confustionMatrixRandomForestUS = confusionMatrix(predictionRandomForest_UndersamplingData, testDataSelectedFeatures$HTN)
confustionMatrixRandomForestUS 

predictionANN_UndersamplingData = predict(modelANN_UnderSampling, newdata = testDataSelectedFeatures)
predictionANN_UndersamplingData

confusionMatrixUnderSamplingANN = confusionMatrix(predictionANN_UndersamplingData, testDataSelectedFeatures$HTN)
confusionMatrixUnderSamplingANN 


predictionLogisticRegression_UndersamplingData= predict(modelLogisticalRegressionUs, newdata = testDataSelectedFeatures)
predictionLogisticRegression_UndersamplingData

confusionMatrixLogisticalRegressionUS = confusionMatrix(predictionLogisticRegression_UndersamplingData, testDataSelectedFeatures$HTN)
confusionMatrixLogisticalRegressionUS 


predictionXGBoostUS= predict(modelXGBoostUs, newdata = testDataSelectedFeatures)
predictionXGBoostUS

confusionMatrixXGBoostUs = confusionMatrix(predictionXGBoostUS, testDataSelectedFeatures$HTN)
confusionMatrixXGBoostUs

#function to get the metrics for the classifiers
getClassificationMetrics = function(confusionMatrix) {
  accuracy = confusionMatrix$overall["Accuracy"]
  precision = confusionMatrix$byClass["Precision"]
  recall = confusionMatrix$byClass["Recall"]
  f1 = confusionMatrix$byClass["F1"]
  specificity = confusionMatrix$byClass["Specificity"]

    metrics = data.frame(
    Accuracy = round(accuracy, 4),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    Specificity = round(specificity, 4),
    F1_Score = round(f1, 4))
  return(metrics)
}



### OBTAINING METRICS

#ADASYN DATA

metricsAdasynRandomForest = getClassificationMetrics(confusionMatrixRandomForest)
metricsAdasynRandomForest

metricsAdasynLogisticalRegression =getClassificationMetrics(confusionMatrixLogisticalRegression)
metricsAdasynLogisticalRegression


metricsAdasynANN= getClassificationMetrics(confusionMatrixANN)
metricsAdasynANN

metricsAdasynXGBoost= getClassificationMetrics(confustionMatrixXGBoost)
metricsAdasynXGBoost


# UNDERSAMPLING DATA 

metricsUndersamplingANN = getClassificationMetrics(confusionMatrixUnderSamplingANN)
metricsUndersamplingANN

metricsUndersamplingRandomForest = getClassificationMetrics(confustionMatrixRandomForestUS)
metricsUndersamplingRandomForest

metricsUnderSamplingLogisticRegression = getClassificationMetrics(confusionMatrixLogisticalRegressionUS)
metricsUnderSamplingLogisticRegression

metricsUndersamplingXGBoost = getClassificationMetrics(confusionMatrixXGBoostUs)
metricsUndersamplingXGBoost

### ROC Curve
#import libraries for ROC curve
library(PRROC)
library(pROC)

# ADASYN
rocLogisticalRegression = roc(testDataSelectedFeatures$HTN, predict(modelLogisticalRegression, testDataSelectedFeatures, type = "prob")[, "1"])
rocANN =  roc(testDataSelectedFeatures$HTN, predict(modelANN, testDataSelectedFeatures, type = "prob")[, "1"])
rocRandomForest =roc(testDataSelectedFeatures$HTN, predict(modelrandomforest, testDataSelectedFeatures, type = "prob")[, "1"])
rocXGBoost = roc(testDataSelectedFeatures$HTN, predict(modelXGBoost, testDataSelectedFeatures, type = "prob")[, "1"])

aucLogisticalRegression = auc(rocLogisticalRegression)
aucLogisticalRegression

aucANN = auc(rocANN)
aucANN

aucRandomForest = auc(rocRandomForest)
aucRandomForest

aucXGBoost = auc(rocXGBoost)
aucXGBoost

#plot the roc curves
plot(rocLogisticalRegression, col = "deepskyblue", main = "ROC Curves ADASYN", lwd = 2)
lines(rocANN, col = "green", lwd = 2)
lines(rocRandomForest, col = "deeppink", lwd = 2)
lines(rocXGBoost, col = "darkviolet", lwd = 2)
legend("bottomright", legend = c("LR", "ANN", "RF", "XGB"),
       col = c("deepskyblue", "green", "deeppink", "darkviolet"), lwd = 2)

legend("bottomright",
       legend = c(
         paste0("LR (AUC = ", round(auc(rocLogisticalRegression), 2), ")"),
         paste0("ANN (AUC = ", round(auc(rocANN), 2), ")"),
         paste0("RF (AUC = ", round(auc(rocRandomForest), 2), ")"),
         paste0("XGB (AUC = ", round(auc(rocXGBoost), 2), ")")
       ),
       col = c("deepskyblue", "green", "deeppink", "darkviolet"),
       lwd = 2, cex = 0.9)


#obtaining probability score for positive class
probsLogisticalRegression= predict(modelLogisticalRegression, testDataSelectedFeatures, type = "prob")[, "1"]
probsANN = predict(modelANN, testDataSelectedFeatures, type = "prob")[, "1"]
probsRandomForest= predict(modelrandomforest, testDataSelectedFeatures, type = "prob")[, "1"]
probsXGboost = predict(modelXGBoost, testDataSelectedFeatures, type = "prob")[, "1"]

labels = as.numeric(as.character(testDataSelectedFeatures$HTN))

#precision-Recall curves
prLogisticalRegression = pr.curve(scores.class0 = probsLogisticalRegression[labels == 1],
                  scores.class1 = probsLogisticalRegression[labels == 0], curve = TRUE)
prANN = pr.curve(scores.class0 = probsANN[labels == 1],
                   scores.class1 = probsANN[labels == 0], curve = TRUE)
prRandomForest = pr.curve(scores.class0 = probsRandomForest[labels == 1],
                  scores.class1 = probsRandomForest[labels == 0], curve = TRUE)
prXGBoost = pr.curve(scores.class0 = probsXGboost[labels == 1],
                   scores.class1 = probsXGboost[labels == 0], curve = TRUE)

# plot the precision recall curves
plot(prLogisticalRegression$curve[,1], prLogisticalRegression$curve[,2], type="l", col="deepskyblue", lwd=2, 
     xlab="Recall", ylab="Precision", main="Precision-Recall Curves ADASYN", ylim=c(0,1))
lines(prANN$curve[,1], prANN$curve[,2], col="green", lwd=2)
lines(prRandomForest$curve[,1], prRandomForest$curve[,2], col="deeppink", lwd=2)
lines(prXGBoost$curve[,1], prXGBoost$curve[,2], col="mediumorchid2", lwd=2)
legend("bottomleft", legend = c("LR", "ANN", "RF", "XGB"),
       col = c("deepskyblue", "green", "deeppink", "mediumorchid2"), lwd = 2)


###SHAP VALUES

#install package
install.packages("xgboost")
#load the package
library(xgboost)
library(SHAPforxgboost)


XTrainMatrix = as.matrix(xTrain)
shapValues=shap.values(xgb_model = modelXGBoost$finalModel, X_train = XTrainMatrix)
#obtaining the shap values
shapLong= shap.prep(shap_contrib = shapValues$shap_score, X_train = XTrainMatrix)
#plot the shap values based on impact on model output
shap.plot.summary(shapLong)

#A

shapLong$abs_shap_values = abs(shapLong$value)
# calculate the absolute mean value for every characteristics from shap values

meanAbsoluteShapValues= aggregate(abs_shap_values ~ variable, data = shapLong, FUN = mean)
# sort variable based on SHAP absolute mean value  
meanAbsoluteShapValues=meanAbsoluteShapValues[order(meanAbsoluteShapValues$abs_shap_values, decreasing = TRUE), ]
meanAbsoluteShapValues
# plot a grafic based on the importance of features
ggplot(meanAbsoluteShapValues[1:26, ], aes(x = reorder(variable, abs_shap_values), y = abs_shap_values)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = " Mean absolute SHAP values based on the importance of features",
       x = "Risk factor", y = " SHAP absolute mean value ") +
  theme_minimal()

# plot the selected featues by boruta
library(ggplot2)
library(reshape2)
selectedFeatures
#drop the "HHTN" column
featuresOnly= setdiff(selectedFeatures, "HTN")
#select the variable from trainDataSElected
dataframePlot=trainDataSelected[, featuresOnly]
#transform into long 
dataFrameLong=melt(dataframePlot, variable.name = "Feature", value.name = "Value")
#boxplot for selected features
ggplot(dataFrameLong, aes(x = Feature, y = Value, fill = Feature)) +
  geom_boxplot(outlier.alpha = 0.2) +
  theme_minimal() +
  labs(title = "Boxplotfor selected features)",
       x = "FEatures", y = "Values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

