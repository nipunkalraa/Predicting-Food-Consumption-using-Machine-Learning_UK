# set working directory
setwd("M:/Portfolio/My_Predictive/PRED_3")

# Install required packages
install.packages("tidyverse")
install.packages("caret")
install.packages("colorspace")


# load individuals csv and explore using head(), tail(), summary()
ind <- read_csv("individuals.csv")
head(ind)
tail(ind)

# understand the structure using str(), dimensions of the data using dim()
str(ind)
dim(ind)

# PRE-PROCESS THE DATA

# we will predict whether individuals are more likely to consume red meat.
# we may not be interested in groups who are very young or very old
# we will find row using which() function, it returns the value which satisfies the given condition. 
# select a column of data with $ sign. The '-' minus sign in front of 'index' means that the rows are EXCLUDED



index <- which(ind$Age == "a0_02" |
                 ind$Age == "a03_04" |
                 ind$Age == "a05_09" | 
                 ind$Age == "a10_15"|
                 ind$Age == "a75_pl")
ind <- ind[-index,]
# we have removed age 0-15 and age>75 from our data

# Now, we will use ONE-HOT ENCODING for Sex and Work
# and use ORDINAL ENCODING for Age and Health
# Encoding categorical data into a numeric format is a crucial pre processing step for many machine learnign algorithms
# One-Hot Encoding creates binary columns (1 or 0) for each cateogry
# Ordinal Encoding assigs a numerical  ue to each category based on its order or rank.

ind$Sex[ind$Sex=="F"] <- 0
ind$Sex[ind$Sex=="M"] <- 1

ind$Work[ind$Work=="not_working"] <- 0
ind$Work[ind$Work=="employment"] <- 1

ind$Age[ind$Age=="a16_19" | ind$Age=="a20_44"] <- 1
ind$Age[ind$Age=="a45_64"] <- 2
ind$Age[ind$Age=="a65_74"] <- 3

ind$Health[ind$Health=="verybad"] <- 1
ind$Health[ind$Health=="bad"] <- 2
ind$Health[ind$Health=="fair"] <- 3
ind$Health[ind$Health=="good"] <- 4
ind$Health[ind$Health=="verygood"] <- 5

# calculate Red Meat Consumption and create a new column in the table labelled as RedMeat

ind$RedMeat <- ind$Beef + ind$Lamb + ind$Pork + ind$ProcessedRedMeat

# create a data frame that incules only a subset of predictor variable and outcome variable

ind_subset=ind[c(2:6,20)]


# Label red meat consumption into two categories: (1) those consume large amount of red meat and (2) those not
# the threshold here is arbitrarily defined based on the (mean)average value of read meat consumption

threshold <- mean(ind_subset$RedMeat)
ind_subset$RedMeat[ind_subset$RedMeat >= threshold] <- NA
ind_subset$RedMeat[ind_subset$RedMeat < threshold] <- 0
ind_subset$RedMeat[is.na(ind_subset$RedMeat)] <- 1

# Convert predictor and outcome variables to factors using as.factor() function
#factors are a type of data object in R that represent categorical values
# by converting categorical variables into factors, we ensure that the algorithm treats each category as a separate entity, rather than as continuous data
# this improves accuracy of the model as it allows us to capture any non-linear relationship that may exist b/w the categorical variable and the response variable

ind_subset$RedMeat <- as.factor(ind_subset$RedMeat)
ind_subset$Age <- as.factor(ind_subset$Age)
ind_subset$Sex <- as.factor(ind_subset$Sex)
ind_subset$Ethnicity <- as.factor(ind_subset$Ethnicity)
ind_subset$Health <- as.factor(ind_subset$Health)
ind_subset$Work <- as.factor(ind_subset$Work)

# set '1' as the first level (positive class) in the model
ind_subset$RedMeat = relevel(ind_subset$RedMeat,"1")



#A. Partition of the data in TRAINING and TEST datasets
# we will use 80% of the data to train the model algorithm and 20% of data will be used to test the model to evaluate model performance
# data can be split with the createDataPartition() function from the caret package
# 'p' is th e% of data that goes to training. 'list' is set to False to return results in a matrix with number of rows rather a list
# 'times' specifies the number of partitions to create
# before we train the model, a speed is specifies to a number (any number) so the exact results can be produced

set.seed(2024)
index <- createDataPartition(ind_subset$RedMeat, p=0.8, list = FALSE, times = 1)
data_train <- ind_subset[index,]
data_test <- ind_subset[-index,]

#B. K-fold cross validation
# we use 10-fold cross validation to train the model
# it means that the testing data are split into 10 equal folds. Each time, 9 folds are used to train the algorithm and the last fold is used to evaluate the model
# before we train, a seed is specified to a number (any number) so the exact results can be produced
# training control can be defined using trainControl() function. 
# the method specified as 'cv' which means cross validation and the data set is randomly split into '10' groups

set.seed(2024)
ctrlspecs <- trainControl(method = "cv", number = 10)

#C. Logistic Regression Model
# Train a Model
# we use train() function from carat pack to fir predictive models
# different machine learning algorithms can be specified in the 'method'
# the 'glm' refers to generalized linear model and we need to specify 'family' and 'binomial' for logistic regression

model_lr <- train(RedMeat~., data=data_train, method="glm", family="binomial",
                  trControl=ctrlspecs)

#D. Predictive Variables
# print the model to see summary. Cohen's Kappa measures the model performance compared to guessing with the target distribution by considering the marginal distribution of the response variable.

print(model_lr)

# output coefficients of predictive variables. significant levels are marked with the number of *.
summary(model_lr)

# examine the most predictive variable
varImp(model_lr)

# create a plot to visualise the importance of predictive variables
# load ggplot package

library(ggplot2)

# Extract variable importance into a data frame
importance <- varImp(model_lr, useModel = TRUE, scale = TRUE)
importance_df <- as.data.frame(importance$importance)
importance_df$Variable <- rownames(importance_df)

# Plot using ggplot2
ggplot(importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_point(color = "red", size = 3) +
  coord_flip() +
  labs(title = "Red Meat Consumption",
       x = "Predictor Variables",
       y = "Importance (Scaled)") +
  theme_minimal()


#E. Model Validation and Performance
# apply the model to predict outcome with the test data(unseen data)
outcome_pred_lr <- predict(model_lr, newdata = data_test)

# generate confusion matrix to compare the predicted labels against the actucal labels
# this also generates model matrics, the most important of which are accuaracy, kappa, sensitivity, and specificity
con_matrix_lr <- confusionMatrix(table(outcome_pred_lr, data_test$RedMeat))
print(con_matrix_lr)

#Accuracy: The ratio of correctly predicted observations to the total number of observations. It is 
#calculated as (TP + TN) / (TP + TN + FP + FN). It is not a good indicator for imbalanced data

#Sensitivity:  The ability of the model to correctly identify negative cases. It is calculated as TN / (TN + 
#FP). Specificity measures the ratio of actual negative cases that are correctly identified

# Precision: The ratio of correctly predicted positive cases to all predicted positive cases. It is 
#calculated as TP / (TP + FP). Precision calculates the model’s ability to minimize false positive 
#predictions.

precision_lr <- con_matrix_lr$table[1,1]/sum(con_matrix_lr$table[1,1:2])
print(precision_lr)

# F1 score: the precision and recall contribute equally to the score to evaluate the reliabilty of a model
# it is calculated as 2*precision*recall/(precision+recall)
recall_lr <- con_matrix_lr$table[1,1]/sum(con_matrix_lr$table[1:2,1])

F1_lr <- 2*precision_lr*recall_lr/(precision_lr+recall_lr)

#F. random forest model
# same training method we used for logistic regression
# we will also tune hyperparameters 'mtry' which is the number of randomly selected predictor varialbes to considered at each split point
# 'mtry' is set the square root of the no. of columns in the training set and passed to the 'tunegrid' command for use in the training process

mtry <- sqrt(ncol(data_train))
tunegrid <- expand.grid(.mtry=mtry)

# the same train() function is used to train a mode. 
# best model is selected based on Kappa which is a metric that compares an observed accuracy with an expected (random chance)

model_RF <- train(RedMeat ~ ., 
                  data = data_train, method = "rf", metric = "Kappa",
                  trControl = ctrlspecs, tunegrid = tunegrid)
print(model_RF)

# the same process of model validations nd performance  valuation will be followed as the logistic regression



# TEST: create variable importance plot with ggplot
df <- data.frame(matrix(unlist(varImp(model_RF)$importance), ncol = max(lengths(varImp(model_RF)$importance)), byrow = TRUE))
names(df) <- names(varImp(model_RF)$importance[[which(lengths(varImp(model_RF)$importance)>0)[1]]])

varImp_data = varImp(model_RF)$importance %>%
  as.data.frame() %>%
  rownames_to_column() %>%
  arrange(Overall) %>%
  mutate(rowname = forcats::fct_inorder(rowname))




# TEST: build your own ML models to predict product consumption of your choice

varImp_data$rowname = rev(c("Male", "Ethnicity_white", "Employment", "Age 45-64", "Age 65-74", 
                            "Health_very good", "Health_good", "Health_fair", "Ethnicity_black", 
                            "Health_poor", "Ethnicity_other", "Ethnicity_mixed"))

ggplot(varImp_data, aes(x = reorder(rowname, Overall), y = as.numeric(Overall), fill = as.numeric(Overall))) +
  geom_bar(stat = 'identity') + 
  geom_text(aes(label = sprintf("%0.2f", round(as.numeric(Overall), digits = 2))), position = position_dodge(width = 0.9)) +
  coord_flip() +
  labs(x = 'variable', y = 'Importance score', fill = 'Strength') +
  theme(panel.background = element_rect(fill = 'white')) + 
  theme(text = element_text(size=15)) + 
  scale_fill_gradientn(colours = c("gray87", "lightskyblue", "skyblue2", "skyblue3"))









