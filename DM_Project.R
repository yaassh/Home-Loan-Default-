# Setting path
path <- "/Users/yaasshrao/Desktop/home credit risk default"
setwd(path)

train <- read.csv("application_train.csv")
test <- read.csv("application_test.csv")

dim(train) 
dim(test)

# Testing majority class distribution
unique(train$TARGET)
unique(test$TARGET)

round(prop.table(table(train$TARGET))*100)
# Majority class distribution = 92%

#library(dplyr)
# Taking a slice to see what features are present
train %>%
  slice(1:10) %>%
  knitr::kable()

# No 'TARGET' feature in testing dataset. So, partitioning the dataset into training & testing 
x_train <- train %>% select(-TARGET)
y_train <- train %>% select(TARGET)   

# Testing data
x_test  <- test

# Removing the original data to save memory
rm(train)
rm(test)

# library(skimr)
skim_to_list(x_train)
# Results: 3 data types: integer, numeric, factor

# ------------------------------------------------

# DATA TRANSFORMATION

# Collecting names of all character data & changing into factors
string_2_factor_names <- x_train %>%
  select_if(is.character) %>%
  names()

#library(purrr)
# Factoring numnerical data
#unique_numeric_values_tbl <- x_train %>%
#  select_if(is.numeric) %>%
#  map_df(~ unique(.) %>% length()) %>%
#  gather() %>%
#  arrange(value) %>%
#  mutate(key = as_factor(key))

#unique_numeric_values_tbl

#factor_limit <- 7

#num_2_factor_names <- unique_numeric_values_tbl %>%
#  filter(value < factor_limit) %>%
#  arrange(desc(value)) %>%
#  pull(key) %>%
#  as.character()

#num_2_factor_names

# Missing data
missing_tbl <- x_train %>%
  summarize_all(.funs = ~ sum(is.na(.)) / length(.)) %>%
  gather() %>%
  arrange(desc(value)) %>%
  filter(value > 0)

missing_tbl
# A number of features have over 65% missing values
# Strategy : Impute missing values based on mean(for numeric) & mode(for categorical)

# --------------------------------------------

# DATA PREPROCESSING 
# Imputing missing data
rec_obj <- recipe(~ ., data = x_train) %>%
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>%
  prep(stringsAsFactors = FALSE)

rec_obj

# Using bake() on training & testing datasets to implement minimal transformations
x_train_processed <- bake(rec_obj, x_train) 
x_test_processed  <- bake(rec_obj, x_test)

# -------------------------------------------

# RESULTS OF DATA TRANSFORMATION

# Before transformation
x_train %>%
  select(1:30) %>%
  glimpse()

# After transformation
x_train_processed %>%
  select(1:30) %>%
  glimpse()

# We see that the NA values are imputed and we we are left with factor and numeric data.

# COnverting the TARGET to a factor, as it is the format needed to perform binary classification in H2O.
# Numeric --> Character --> Factor
y_train_processed <- y_train %>%
  mutate(TARGET = TARGET %>% as.character() %>% as.factor())

# Removing unprocessed datasets to save memory
rm(rec_obj)
rm(x_train)
rm(x_test)
rm(y_train)

# ---------------------------------

# MODELING
# We implemented AutoML with H2O
# Initializing a local cluster
h2o.init()

# COnverting the dataframe to an H2O frameusing as.h2o, as h2o expects both the target & training
# features to be in the same data frame, so bind_cols is used.
data_h2o <- as.h2o(bind_cols(y_train_processed, x_train_processed))

# Spliting the training data into train, validation & test sets(70/15/15) split
splits_h2o <- h2o.splitFrame(data_h2o, ratios = c(0.7, 0.15), seed = 1234)

train_h2o <- splits_h2o[[1]]
valid_h2o <- splits_h2o[[2]]
test_h2o  <- splits_h2o[[3]]

# Running the AutoML model
y <- "TARGET"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame    = train_h2o,
  validation_frame  = valid_h2o,
  leaderboard_frame = test_h2o,
)

automl_leader <- automl_models_h2o@leader

# Checking performance
performance_h2o <- h2o.performance(automl_leader, newdata = test_h2o)
# Created an h2o performance object . test_h2o is a random sample of 15% of the application training set

#Testing performance at threshold of 0.139
performance_h2o %>%
  h2o.confusionMatrix()
# 1668 people that defaulted were correctly predicted.
# 5828 people were incorrectly predicted to default, but actually did not.
# 2045 people were incorrectly predicted to default, but actually did default.(FALSE NEGATIVES)

# Checking AUC
performance_h2o %>%
  h2o.auc()
# AUC : 0.75009

# ----------------------------------

# PREDICTION

prediction_h2o <- h2o.predict(automl_leader, newdata = as.h2o(x_test_processed))
prediction_h2o
# p0 = SK_ID_CURR
# P1 = TARGET
