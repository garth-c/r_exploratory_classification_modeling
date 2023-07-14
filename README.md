# R exploratory classaification modeling

The objective is to explore the customer churn data set from Kaggle to identify potentially effective predictor variables as well as explore relationship between the predictors. 
Since this data was already put together in the same file, my base assumption is that the predictor variables are reasonably valid predictors for the response variable. Thus, the
point of this exercise is to validate this assumption.
```
The source data description is:
source -> Kaggle.com

file name -> Telco-Customer_Churn.csv

reponse variable -> Churn {binary: Yes/No}

A description of the source data is below and the dimensions 7032X20 match the source data file.
Note that seniorcitizan is converted to a factor along with all other chr data. There were no
data quality issues noted - I used skimr for the data quality testing.
```
Step 1: import the source data file:
#data ingest and prep process

```
#load the needed libs
library(readxl) #read in excel files
library(tidyverse) #data wrangling
library(skimr) #data quality eval

###~~~
#read in source data set
###~~~

#customer churn data
cust_churn <- read_excel('cust_churn.xlsx', 
                         sheet = 'churn')

#validate the input
View(cust_churn)
glimpse(cust_churn)
dim(cust_churn) #must match the input file dimensions
str(cust_churn)

#recast senior citizen as character 
cust_churn$seniorcitizen <-  as.character(cust_churn$seniorcitizen)

#recast all characters to factors
cust_churn <- cust_churn %>% mutate_if(is.character, as.factor)

#validate that this worked
glimpse(cust_churn)
str(cust_churn)

#skim the input to look for data quality issues
skimr::skim(cust_churn)

#produce a summary of the data set
summary(cust_churn)
```
The data types and other info:
![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/21a3a75c-1065-4414-b197-8af40bc5a02a)

Produce a summary of the source data set:
<img width="734" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/7033ec51-9fd3-47de-a481-049e16ecaa0d">

Step 2: Explore the data set

look for class imbalance in the response variable:
```
#quick plot of the class balancing
windows()
ggplot(cust_churn) +
  geom_bar(aes(x = churn),
           fill = 'blue') +
  ggtitle('Response Variable Counts')

#get the class balance situation for the response variable
prop.table(table(cust_churn$churn))
```
there is a roughly 73% to 27% split between the Yes and No values for churn which is imbalanced. With such an imbalance, I will explore oversamling techniques
such as ROSE.

This is the specific percentage split:

<img width="273" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/5e2d9d67-3d9f-492b-84ea-81936d7f3de3">

This is a bar chart showing the magnitude of the split:

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/6b315545-dc11-47e8-8b81-c6c22f60584c)

Exploring the numeric data for correlations, it looks like tenure is definitely correlated to total charges and the other numeric values are not
very correlated. This may result in removing either tenure or totalcharges in order to avoid any multi-colinearity issues in the model.
```
#put numeric values into a holding data frame
numerics <- cust_churn[, c(6,19,20)]

#quick plot for the relationships
windows()
PerformanceAnalytics::chart.Correlation(numerics, 
                                        histogram = TRUE, 
                                        method = 'spearman')

#look for outliers in the numeric values
windows()
boxplot(numerics)
```

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/5546e290-e998-461c-bc94-4c91746a28f5)

Looking for outliers in the numeric values shows that only totalcharges has any amount of spread and possible outliers:

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/d3adbc96-f915-4170-b2aa-d01e88075bb7)

explore the categorical data potential association with a Cramer's V function. For this example, I compared the predictors: 'contract'
to 'paymentmethod' to see if there was any significant association with the same concern of potential multicolinearity:
```
#use cramer's V for the test of association between suspect variables
rcompanion::cramerV(x = cust_churn$contract,
                    y = cust_churn$paymentmethod)
```
The Cramer's V metric = 0.2667. Since Cramer's V measures association between 0 and 1 (similar to a correlation coefficient), this
result is not large enough to be concerned with. All of the potential categorical to categorical potential associations were tested
and only a few were found to be somewhat strong. I will wait to complete the EDA portion before removing highly associated predictor
variables. Below is a interpretation chart for Cramer's V metrics to add the needed context:

<img width="866" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/ecb0b76e-2370-4379-9449-099d3b0004f1">

Explore the potential numeric to categorical associations for potential multi-colinearity:

```
windows()
boxplot(totalcharges ~ deviceprotection,
        data = cust_churn,
        main = 'Compare numeric distribution between groups')

#Ho: mean ranks of the groups are statistically the same
#Ha: at least one sample mean rank differs from the rest
#a non-significant result means the two are weakly correlated
kruskal.test(totalcharges ~ deviceprotection,
             data = cust_churn)
```

A box plot by categorical group gives a quick visual to see any obvious correlation:

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/3597a2dd-0905-47a2-b516-64ea7d481889)
The idea with this test is that if the mean ranks are statistically significant between the groups (categorical variables) for the numeric
variables then the correlation between the variables is strong and should be considered for potential removal due to multicolinearity concerns.  
Note that categorical data does not 100% meet the underlying data assumption for a Kruskal test, I am considering this to be an acceptable
'off-label' usage for this test.

Step 3: develop an exploratory correlation funel model:
Develop a correlation funnel plot to see which predictors and which levels within the predictors are best to use for predicting
the churn response variable.

```
#load the needed libs
library(ppsr) #probability scoring
library(correlationfunnel) #correlation funnel with binning
library(tidyverse) #data wrangling

#set the rando seed
set.seed(12345)

summary(cust_churn)

###~~~
#data prep ~ USE THE WIDE DATA SET FOR THESE FUNCTIONS
###~~~

#all inputs need to be factors or numerics
glimpse(cust_churn)


#iterate over data and change type if needed
cust_churn <- cust_churn %>%
                      mutate_if(is.character, as.factor) %>%
                      mutate_if(is.numeric, as.integer)

#check for significant class imbalance
prop.table(table(cust_churn$churn))

#make a copy
cust_churn_adj <- cust_churn

###~~~
#correlation funnel method
###~~~

#change response variables to numeric 1/0
cust_churn_adj$churn <- ifelse(cust_churn_adj$churn == 'Yes',1,0)

#make sure this worked
glimpse(cust_churn_adj)

#check for significant class imbalance
prop.table(table(cust_churn_adj$churn)) #needs to be more than 15% for minority class

#remove blanks
cust_churn_adj <- na.omit(cust_churn_adj)

#make sure this worked & check for minority class
glimpse(cust_churn_adj) #response variable needs to be numeric

#binarize the predictors and response variables ~ data prep for funnel
termed_ml_funnel <- cust_churn_adj %>%
                            correlationfunnel::binarize()

#plot the correlation funnel
windows()
termed_ml_funnel %>% correlationfunnel::correlate(target = churn__1) %>%
                              correlationfunnel::plot_correlation_funnel(interactive = FALSE,
                                                                         limits = c(-1, 1),
                                                                         alpha = 1) +
                                                 geom_point(size = 3,
                                                            color = '#2c3e50')
```
The results are below. Yes = 1 and No = 0. From this plot I can see that the predictors contract and online security are good candidate
predictors for the final model. Specifically for contract, it looks like the factor level 'Month_to_month' is more associated with 
'No' churn and 'Two_year' is more associated with 'Yes' churn. The rest of the predictors are roughly interpreted in the same way.
The business value for this plot is high and various course corrections are able to made depending on the context of the decision
around customer churn being made.

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/bba5760c-0c75-4035-b3a9-c193a37d0979)

Step 4: Potential predictor feature selection
For this step, I use the Boruta library to help identify more potent predictors from less potent ones.
```
###~~~
#boruta section
###~~~

#all predictors need to be factors
glimpse(cust_churn)

#feature Selection identification
key_predictors <- Boruta::Boruta(churn ~ .,
                                 data = cust_churn,
                                 doTrace = 3,
                                 pValue = 0.05,
                                 mcAdj = TRUE,
                                 holdHistory = TRUE,
                                 getImp = getImpRfGini,
                                 maxRuns = 20)

#look at the output
print(key_predictors)
attributes(key_predictors) #see what is available

#create a df to hold the results
key_predictors_df <- as.data.frame(key_predictors$finalDecision)
View(key_predictors_df)

#plot of the key predictor variables
windows()
plot(key_predictors,
     las = 2,
     cex.axis = 0.7)

#plot the history - in case this is needed
windows()
plotImpHistory(key_predictors)

#tentative fix function ~ use only if there are undecided predictors
rough_fixes <- TentativeRoughFix(key_predictors,
                                 averageOver = Inf)

#look at the rough fixes
print(rough_fixes)

#create a df to hold the data
rough_fixes_df <- as.data.frame(attStats(rough_fixes))
```

The Boruta output is below. From this plot, I am able to see that the most likely best predictors are:
tenure, totalcharges, monthlycharges, and contract. The other predictors don't seem to be very strong
and/or rely on multi-colinearity for predictive power

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/77e82305-5a4b-43b8-b5aa-564ddd894a76)

Step 5: adjust reponse variable for class imbalance and use this adjusted data set for the exploratory models

The first thing that I do is use a 70/30 split for training and testing data sets. From there, I used ROSE oversampling 
to get the response variable class balance closer to a 50/50 mix so the models will be more effective. However, note that
the models will predict using the test set which has not been adjusted for the imbalance. This process will be a good test 
of how effective a prediction model will be when it used data from the wild to process. 

```
#set the environmentals
set.seed(12345)
library(ROSE) #oversampling
library(caret) #train test split

#check the data type
glimpse(cust_churn$churn)
str(cust_churn$churn)

###~~~
#train / test split
###~~~

#use caret to create a training and testing partition
#use a 70/30 split based on the response variable
inTrain_adj <- createDataPartition(
  y = cust_churn$churn,
  #the outcome data are needed
  p = .70,
  #the percentage of data in the training set
  list = FALSE,
  times = 1)

#split the data frame into training and testing sets
#based on the above set flag
churn_train <- cust_churn[ inTrain_adj,]
prop.table(table(churn_train$churn))
churn_test <- cust_churn[-inTrain_adj,]
prop.table(table(churn_train$churn))


###~~~
#ROSE oversampling
###~~~

#rose sampled file
churn_train_ROSE <- ROSE::ovun.sample(formula = churn ~ .,
                                      data = churn_train,
                                      N = nrow(churn_train), #needed record count
                                      p = 0.50, #balanced classes
                                      method = 'both', #use under/over and synthetic samples
                                      seed = 12171968)$data #this last part '$data' is needed for a df output


#check initial class mix for training set
prop.table(table(churn_train_ROSE$churn))
```
The output of the oversampling for the training set and confirmation that the test set is at the original mix:

<img width="342" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/4db56a7f-ff7f-4146-ae31-3ec1bc9eedaf">

Step 6 - build an intial tree learning model and review the output

Build the initial tree model using caret and set up a 10 fold cross validation train control. 
```
#set the environmentals
set.seed(12345)


#load the needed libs
library(plyr) #load this before loading tidyverse
library(tidyverse) #data wrangling
library(rattle) #plotting
library(rpart.plot) #plotting
library(RColorBrewer) #custom color pallet
library(party) #plotting
library(partykit) #plotting
library(caret) #train controls

###~~~
#calculate the tree model section
###~~~

##train control params
#set up the cross validation scheme and other needed bits
tm_tctrl <- trainControl(method = 'cv',
                         number = 10,
                         #index = folds,
                         allowParallel = TRUE,
                         summaryFunction = twoClassSummary,
                         classProbs = TRUE,
                         verboseIter = TRUE)

#run the tree model
tree_model <- train(make.names(churn) ~ .,
                    data = churn_train_ROSE,
                    trControl = tm_tctrl,
                    method = 'rpart',
                    metric = 'ROC',
                    na.action = na.omit)


#model diagnostics
tree_model #look at the model
tree_model$finalModel #look at the text version of the final model
```

The initial tree model output shows that the most important predictor is contract. Since this is an EDA model, no tuning was
performed on the model. This model is only trying to see if there is any signal at all coming from the predictor variables.
```
#plot the model
windows()
fancyRpartPlot(tree_model$finalModel)
```
![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/c0d3f88e-6d27-4a32-b8cc-de059e746898)

The top 5 predictor variables VIF plot is below. This plot shows the specific factor levels for each predictor variable that 
has the mode influence on the response variable. 
```
#display the most important variables
windows()
tm_variables <- varImp(tree_model)
plot(tm_variables,
     top = 5,
     main = 'Top 5 Most VIF \n Tree Model')
```
![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/c2b65c4e-5e28-4bc9-82bc-ffc7207c2d71)

To evaluate the tree model, I will use a confusion matrix shown below. Since the objective is to predict churn, the model goal should 
be to get the highest prediction = Yes and Reference = Yes counts. This means that the model correctly predicted customer churn and
that the most important predictor variables are the knobs and levers that are able to be adjusted in order to prevent churn. The model
will also be able to predict customers at risk of churning with the accuracy of the overall model. In this specific case, the tree 
model corectly predicted ~42% (495/(495+683) of the churn = Yes data from the test set - which is not very good.
```
#make predictions
predictions_tm <- predict(tree_model,
                          newdata = churn_test,
                          na.action = na.pass)


##determine the confusion matrix
confusionMatrix(data = predictions_tm,
                reference = as.factor(churn_test$churn))
```

<img width="278" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/c9bde9f4-af30-44a8-8107-4cd142c52720">

Since the response variable is imbalanced, focusing on correct Yes churn predictions alone is not enough to compare models. Since the main concerns
from a business perspective would be false negatives (model predicts churn = No but the customer actually churns), the calculating the F2 score
is a good comparative metric between models. 
```
#calculate f2 score
#create a confusion matrix
confusion_matrix <- as.matrix(table(as.factor(predictions_tm),
                                    as.factor(churn_test$churn)))

#calculate the precision
precision <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])

#calculate the recall
recall <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])

#calculate the F2 score
f2_score <- (1 + (precision * recall)^2) * precision * recall

#print the F2 score
print(f2_score)

```
<img width="437" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/042d1792-6b3b-4d44-a9ce-3a0eb24431a0">

Step 7 - build a distributed random forest model and a gradient boosting model.

These models are more complicated than a tree learning model. Also I am using the H2O platform to develop these models.
The first section of this code is around house keeping and setting up the H2O cluster:

```
#load the needed libs
library(tidyverse) #data wrangling
library(h2o) #develop models
library(caret) #confusion matrix calcs

set.seed(12345) #set the seed value

###~~~
#h2o housekeeping section
###~~~

#make sure the JVM JRE SE environment variable is pointed at the
##correct location ~ use the 64bit version
Sys.setenv(JAVA_HOME = 'C:/Program Files/Java/jdk-20')
print(Sys.getenv('JAVA_HOME')) #check to make sure this worked


h2o::h2o.removeAll() ## clean slate - just in case a cluster was already running

#h2o housekeeping, allow use of all cores, declare the local machine version of h20,
#initialize H2O JVM
localh2o <- h2o::h2o.init(ip = 'localhost',
						              port = 54321,
          			          nthreads = -1,
          			          max_mem_size = '25G')

#basic cluster info
h2o::h2o.clusterInfo() #check the cluster info for correctness
h2o::h2o.clusterStatus() #validate that the cluster status is good
h2o::h2o.shutdown(prompt = FALSE) #shut down the h2o cluster
```
Next I prepare the data for consumption by the H2O models:
```
###~~~
#build the models
###~~~

glimpse(churn_train_ROSE)

#all predictors need to be factors
churn_train_ROSE <- churn_train_ROSE %>% mutate_if(is.character, as.factor)
churn_test <- churn_test %>% mutate_if(is.character, as.factor)
glimpse(churn_test)


#change to h2o tensors
train_h2o <- h2o::as.h2o(churn_train_ROSE)
str(train_h2o) #check out the structure and number of levels per variable
#change to h2o tensors
test_h2o <- h2o::as.h2o(churn_test)
str(test_h2o) #check out the structure and number of levels per variable

##split data into train, test, validate
#set the splitting criteria
split_h2o <- h2o::h2o.splitFrame(train_h2o, c(0.6, 0.39999999), seed = 12171968)
train_h2o <- h2o::h2o.assign(split_h2o[[1]], 'train' ) # 60%
valid_h2o <- h2o::h2o.assign(split_h2o[[2]], 'valid' ) # 39.9%

#set response variable & predictor variable names for h2o
#the 'y' value is the response variable
#the 'x' values are the predictor variables
y <- 'churn'
x <- setdiff(names(train_h2o), y)

#find the order of the levels which is needed to properly
#set the class_sampling_factor
h2o::h2o.levels(train_h2o$churn)
h2o::h2o.levels(test_h2o$churn)
```
Netx, build the DRF - distributed random forest model
```
#compute the DRF model
rf_h2o <- h2o::h2o.randomForest(x = x,
              y = y,
              training_frame = train_h2o,
              validation_frame = valid_h2o,
              balance_classes = TRUE,
              model_id = 'rf_h2o',
              ntrees = 1000,
              max_depth = 50,
              mtries = -1,
              categorical_encoding = 'AUTO',
              stopping_metric = 'AUC',
              auc_type = 'AUTO',
              gainslift_bins = -1,
              build_tree_one_node = FALSE,
              histogram_type = 'Random',
              calibrate_model = FALSE,
              calibration_frame = NULL,
              sample_rate = 0.75,
              col_sample_rate_per_tree = 0.9,
              min_rows = 20,
              min_split_improvement = 0.01,
              nfolds = 10,
              fold_assignment = 'Stratified',
              keep_cross_validation_predictions = TRUE,
              seed = 12171968,
              verbose = TRUE)
```
The top 5 predictor variable VIF plot is below:
```
#vif plot
windows()
h2o::h2o.varimp_plot(rf_h2o,
					           num_of_features = 5)
```
![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/2cd6fb18-133e-440a-aa8b-c19a311d1289)

The resulting confusion matrix for the DRF model is below. The 'Yes' churn accuracy is ~50% (454/(454+448)) which is an imporovement over the tree model.
This model also correctly predicted Yes for churn more that it got wrong. 

<img width="281" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/fd7cd077-e64c-4395-9230-36ff027adc1c">

Next, calcualte the F2 score for the DRF model

```
#calculate f2 score
#create a confusion matrix
confusion_matrix <- as.matrix(table(as.factor(pred_h2o_rf_df$predict),
                                    as.factor(churn_test$churn)))

#calculate the precision
precision <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])

#calculate the recall
recall <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])

#calculate the F2 score
f2_score <- (1 + (precision * recall)^2) * precision * recall

#print the F2 score
print(f2_score)
```
The DRF model has a much better F2 score than the tree learning model:
<img width="476" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/1eb9fb2f-c506-46bd-afdc-36271a1dab40">

Next, develop a gradient boosting machine model:
```
#compute the gbm model
h2o_gbm <- h2o::h2o.gbm(x = x,
            y = y,
            training_frame = train_h2o,
            validation_frame = valid_h2o,
            distribution = 'bernoulli',
            model_id = 'h2o_gbm',
            ntrees = 1000,
            max_depth = 5,
            min_rows = 25,
            min_split_improvement = 0.1,
            sample_rate = 0.9,
            learn_rate = 0.05,
            balance_classes = TRUE,
            class_sampling_factors = c(0.50, 0.50),
            nfolds = 10,
            fold_assignment = 'Stratified',
            score_tree_interval = 5,      #used for early stopping
            stopping_rounds = 3,          #used for early stopping
            stopping_metric = 'AUC',      #used for early stopping
            stopping_tolerance = 0.0005,  #used for early stopping
            seed = 12171968)
```
The top 5 predictor VIF plot is below:

![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/44d8de11-5124-4d31-930d-cba8636f9de7)

The confusion matrix for the GBM model is below. This model had ~42% (495/(495+683)) accuracy with predicting Yes for customer churn which is about the 
same as the tree learning model.

```
#predict on hold-out set, test_h2o predictive viability
pred_h2o_gbm <- h2o::h2o.predict(object = h2o_gbm,
								                 newdata = test_h2o)

#determine the confusion matrix
pred_h2o_gbm <- as.data.frame(pred_h2o_gbm[,1])
glimpse(pred_h2o_gbm)

#display the confusion matrix
confusion_matrix_h2o_gbm <- caret::confusionMatrix(data = pred_h2o_gbm$predict,
												                           reference = churn_test$churn)

#check out the confusion matrix
print(confusion_matrix_h2o_gbm)
```

<img width="276" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/c7919fe2-cdf0-48cb-b98b-cc3e6cdd683a">

The F2 score for the GBM model is 0.66012 which is about the same as the tree learning model. 

<img width="456" alt="image" src="https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/6ea2a804-dcbd-4cfb-bb3e-39a1e7dcd231">

```
#calculate f2 score
#create a confusion matrix
confusion_matrix <- as.matrix(table(as.factor(pred_h2o_gbm$predict),
                                    as.factor(churn_test$churn)))

#calculate the precision
precision <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])

#calculate the recall
recall <- confusion_matrix[1, 1] / sum(confusion_matrix[1, ])

#calculate the F2 score
f2_score <- (1 + (precision * recall)^2) * precision * recall

#print the F2 score
print(f2_score)
```

Step 8 - Evaluate the results


