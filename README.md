# Fannie Mae Loan Prediction

## Problem Satement

Can we use machine learning to successfully predict whether a loan will foreclose before it does?
 * Current model has 80% accuracy on test data

## Data Preprocessing

Data required extensive preprocessing to get it into a usable form
 * Originally had 108 columns, narrowed to 12
   * Current Interest Rate
   * Original UPB
   * Original Loan Term
   * Loan Age
   * Original Loan to Value Ratio (LTV)
   * Number of Borrowers
   * Debt-To-Income (DTI)
   * First Time Home Buyer Indicator
   * Modification Flag
   * Home Ready Program Indicator
   * High Balance Loan Indicator
   * Min Borrower Credit Score
 * Needed to make each line a single loan
 * Testing set was 8,278,657 rows each representing an individual loan after processing
 * Dataset had a high imbalance, only .27% of loans were foreclosed on

## Data Analysis

I looked at the distrobution of variables and tried to find anomalies

![distrobutions](https://user-images.githubusercontent.com/84877574/126810531-443b55d6-0b02-41d9-8a17-23e524c991b7.png)


This is a look at how imbalanced the data was

![imbalance_countplot](https://user-images.githubusercontent.com/84877574/126810405-bf91bc88-fce5-4753-b93a-92f56302526f.png)

I checked for multicolinearity and correlation with my target varaible using a heatmap

![correlation_heatmap](https://user-images.githubusercontent.com/84877574/126833374-3da60278-b7b7-4024-8fe9-d943666c5869.png)


## Sampling

To give any model to a decent chance to predict foreclosures with such a high imbalance, both undersampling and oversampling were reqired
I used SMOTE for oversampling, which is a common technique that creates synthetic data based on existing data
I tested many different ratios for sampling and discovered that higher oversampling gave better results, but I didn't want to train on exclusively synthetic data.
Settled on undersampling so that we only had twice as much non-foreclosures as foreclosures, then oversampling to even amounts
Here is that imbalanced visualized before and after

![before_smote](https://user-images.githubusercontent.com/84877574/126812784-d4d9dd49-1a93-44d7-a9cb-92b3be812d59.png)

![after_smote](https://user-images.githubusercontent.com/84877574/126812796-fe79184c-753a-4e81-b881-090907e12958.png)

## Modeling

Tested several models, eventually settling on XGBoost for it's performance on the loans that were foreclosed
These are the results for my model against the testing set

![confusion_matrix](https://user-images.githubusercontent.com/84877574/126816695-dc79601d-4738-4afd-8e42-ec3cc688cb3c.png)

Here is an example of an individual tree from the model

![tree](https://user-images.githubusercontent.com/84877574/126816626-432b495e-5668-4c23-9c35-a96538369e09.png)

Here is the importance the model assigned to each variable

![feature_importance](https://user-images.githubusercontent.com/84877574/127024507-b3e715d2-a729-4490-933a-c8acf9a2319c.png)


This is the ROC curve for the model predictions

![roc_curve](https://user-images.githubusercontent.com/84877574/126817081-15741b8f-2568-4fc8-81df-5cb230872f9e.png)

## Next Steps

I would like to test this model further on new data aquired from the Fannie Mae database

Adding future data to train on would give the model more real foreclosures

Improvements to front end deployment to make it more user friendly

## Acknowledgements

Huge thanks to:
 * Raul Pena for his subject matter expertise and help selecting features
 * Oswald Vinueza for answering my endless questions and providing me with great resources

