# R exploratory classaification modeling

The objective is to explore the customer churn data set from Kaggle to identify potentially good predictor variables as well as explore relationship between the predictors. 
Since this data was already put together in the same file, my base assumption is that the predictor variables are reasonably valid predictors for the response variable.

The source data description is:
source -> Kaggle.com

file name -> Telco-Customer_Churn.csv

reponse variable -> Churn {binary: Yes/No}

Step 1: import the source data file:

A description of the source data is below and the dimensions 7032X20 match the source data file. Note that seniorcitizan is converted to a factor along with all 
other chr data.
![image](https://github.com/garth-c/r_exploratory_classification_modeling/assets/138831938/21a3a75c-1065-4414-b197-8af40bc5a02a)

The data quality was evaluated with skimr and no data quality issues were noted.
[Uploading data_pre

#data ingest and prep process

###~~~
#set up the computing environment
###~~~

#load the needed libs

library(readxl) 

library(tidyverse) 

library(skimr) 

###~~~
#read in source data set
###~~~

#customer churn data

cust_churn <- read_excel('cust_churn.xlsx', 
                         sheet = 'churn')

#validate the input

View(cust_churn)

glimpse(cust_churn)

dim(cust_churn) 

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


###
p.R…]()

Step 2: Explore the data set

produce a quick summary of the data
