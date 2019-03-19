# Home Credit Default Risk

## Introduction
In this project, we’ll be focusing on a specific type of risk called Credit or Default Risk, which has both systemic and unsystemic drivers. The main point is that the drivers of default risk can be measured and analyzed for patterns related to default. As a result, the probability of default for a person or an institution is not random. This is where machine learning can help.

## About the company
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

## Challenge
While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Data : https://www.kaggle.com/c/home-credit-default-risk/data

## Video Explanation : https://youtu.be/BpEuAZCqaRI

### A few points about the data:
● The training data contains 307K observations, which are people applied for and received loans.

● The “TARGET” column identifies whether or not the person defaulted (0 v/s 1).

● The remaining 121 features describe various attributes about the person, loan, and application.

● There are several additional data sets (bureau.csv, previous_application.csv). These auxiliary
files contain data that is one-to-many relationship, which would require aggregation (a feature engineering method) and considerably more work for the unknown benefit. Hence, we have chosen to leave analysis of variables included in these additional datasets.

Our analysis delves into how certain factors influence loan default rates. In the end we build a model that predicts if a user is going to default on their loan.

We were more interested in agile iteration: Building a good model quickly, then going back to try to improve later. We will be evaluating our model accuracy using ROC AUC curves.
   
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and
safe borrowing experience.In order to make sure this underserved population has a positive loan
experience, Home Credit makes use of a variety of alternative data–including telco and transactional
information–to predict their clients’ repayment abilities.
 
The data is provided by Home Credit, a service dedicated to provided lines of credit (loans) to the unbanked population.
We also used a very disciplined Agile process to ensure that we execute critical path tasks in parallel and in small chunks.

The target variable is appropriately titled ‘TARGET’ - 1 (defaulters) v/s 0 (non-defaulters).

The input variables include binary variables for gender, car and realty ownership, the number of children,
income, annuity, occupation type, marital status etc.

The model is a binary classification supervised learning model.
