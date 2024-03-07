# Arya-Data-Scientist-Assignment
# General Appraoch and Thought Process

## EDA_PP_Train_Val,ipynb

### EDA, Feature Engineering, Preprocessing

1. Given data consisted of 57 features (X1-X57) and column Y.
2. Explored all features to check for NaN values, relevant features etc.
3. Selected features (7) and then removed possible outliers from 4 fetaures.
4. Noramlized the data and performed a 4:1 (Train:Val) split.

### Model Training and Analysis(on val_data)

Selected XGBoostclassifier as it is fast effecient and highly scalable 
While dealing with large datasets with millions of rows and columns,
these models perform the best and are state of the art.

5. Trained XGBoost Model with default fetaures on CPU.
6. Used oputna to tune 3 main HPs learning_rate,max_depth,subsample for XGBoost model.
7. Compared the metrics for val data and selected best model.
8. Plotted the Confuison Matrix and Classification Report for base_model and tuned_model

## Observations and Conclusion

This section includes an exhaustive analysis of 2 models and selction of best_model
using evaluation metrics.

Further , the analysis of performance of best_model on the validation data is discussed in depth
followed by conclusion.

## inference.py

This script takes as input the filepath of test_set and then stores predictions of model.
The predictions are stored in 2 csv : i)  all fetaures (X1-X57) with preds 
                                      ii) Selected 7 fetaures with preds

### Find the versions of libraries used in requirments.txt 
    
