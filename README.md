# Fannie Mae Loan Prediction

## Problem Satement

Can we use machine learning to successfully predict whether a loan will foreclose before it does?
 * Current model has 80% successs on test data

## Data Preprocessing

Data required extensive preprocessing to get it into a usable form
 * Needed to make each line a single loan
 * Testing set was 8,278,657 rows each representing an individual loan after processing
 * Dataset had a high imbalance, only .27% of loans were foreclosed on

## Data Analysis

I looked at the distrobution of variables and tried to find anomalies
![distrobutions](https://user-images.githubusercontent.com/84877574/126810531-443b55d6-0b02-41d9-8a17-23e524c991b7.png)


This is a look at how imbalanced the data was
![imbalance_countplot](https://user-images.githubusercontent.com/84877574/126810405-bf91bc88-fce5-4753-b93a-92f56302526f.png)

I checked for multicolinearity and correlation with my target varaible using a heatmap
![correlation_heatmap](https://user-images.githubusercontent.com/84877574/126810977-8a56d028-698a-4d3e-a940-e0af0c2fb436.png)

## Sampling

To give any model to a decent chance to predict foreclosures with such a high imbalance, both undersampling and oversampling were reqired
I used SMOTE for oversampling, which is a common technique that creates synthetic data based on existing data
I tested many different ratios for sampling and discovered that higher oversampling gave better results, but I didn't want to train on exclusively synthetic data.
Settled on undersampling so that we only had twice as much non-foreclosures as foreclosures, then oversampling to even amounts
Here is that imbalanced visualized before and after
![before_smote](https://user-images.githubusercontent.com/84877574/126812784-d4d9dd49-1a93-44d7-a9cb-92b3be812d59.png)
![after_smote](https://user-images.githubusercontent.com/84877574/126812796-fe79184c-753a-4e81-b881-090907e12958.png)

