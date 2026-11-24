# House sale price prediction

## Description

This project predicts the sale price of a house from its size, location, and quality. It uses the Ames, Iowa housing data from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Each row is one house. The columns describe the lot, the neighborhood, the year it was built, the living area, and quality ratings for the kitchen, basement, exterior, and the house overall. The value to predict is `SalePrice`.

The training file has 1,460 houses and 79 features. Many fields look empty because the house has no pool, alley, fence, fireplace, basement, or garage, not because the measurement was lost. The notebook fills those blanks, encodes the text columns so scikit-learn can use them, and checks which features are actually tied to price.

Three models are trained on the same 80/20 split of the labeled houses: a Linear Regression baseline, a Random Forest, and a Gradient Boosting model. Each one is scored with RMSE and R², and the notebook plots actual sale prices against the prices the model predicted. Gradient Boosting is the most accurate of the three. Overall quality and living area explain most of the price, and the neighborhood changes the typical price by a wide margin.

All of the steps are in `house_prices.ipynb`. Figures and the metric table are saved in `outputs/`.

## Results

Scores are on a 20% holdout of the labeled training houses (`random_state=42`). Models fit `log(1 + SalePrice)`. RMSE and R² are computed after converting predictions back to dollars.

| Model | RMSE | R² | log RMSE |
|---|---:|---:|---:|
| Gradient Boosting | $19,434 | 0.932 | 0.114 |
| Linear Regression | $21,786 | 0.914 | 0.135 |
| Random Forest | $23,974 | 0.896 | 0.144 |

Gradient Boosting is the best model on this split. Its typical error is about 12% of the median sale price ($163,000).

The strongest price drivers are:

- **Quality.** `OverallQual` (correlation +0.79). Exterior, basement, and kitchen quality also separate cheap homes from expensive ones.
- **Size.** Above-grade living area (`GrLivArea`, +0.71), then basement and garage size.
- **Location.** Neighborhood median prices run from $88,000 in MeadowV to $315,000 in NridgHt.

Plots are saved under `outputs/`, including `actual_vs_predicted.png` and `model_comparison.png`.

## Project layout

```
house_prices.ipynb    Analysis, models, and plots
data/train.csv        1,460 labeled houses
requirements.txt      Python packages used to run the notebook
outputs/              Saved figures and metrics.csv
```

`train.csv` has 79 features plus `SalePrice`. Kaggle's test file has no sale prices, so the notebook scores a holdout of the training set instead of a Kaggle submission.


## What the notebook does

1. **Explore the target and the features.** Sale price is right-skewed. Numeric features are ranked by correlation with price. Categories are ranked by how much of the price variance sits between groups. Quality, living area, and neighborhood are plotted on their own.
2. **Handle missing values.** In this dataset, a blank pool, alley, fence, fireplace, basement, or garage field means the house does not have that feature, so those cells are filled with `None`. `GarageYrBlt` is filled with the year the house was built. Remaining numeric gaps use the median, and remaining categorical gaps use the most frequent value. Those statistics are learned on the training split only.
3. **Encode categories.** `MSSubClass` and `MoSold` are codes, so they are treated as categories. All categories are one-hot encoded. Levels that appear fewer than 5 times are grouped together.
4. **Drop two partial sales before training.** Two homes in Edwards have living area above 4,000 sq ft and sold for far less than that size implies (Ids 524 and 1299).
5. **Fit three models** on the same holdout:
   - Linear Regression, after dropping encoded columns that repeat information already in the matrix. Living area is the sum of the floor areas, basement total is the sum of the basement parts, and one-hot columns add up to 1. Leaving those copies in makes ordinary least squares overflow.
   - Random Forest (`300` trees, `min_samples_leaf=2`)
   - Gradient Boosting (`400` trees, learning rate `0.05`, depth `3`)
6. **Score RMSE and R²**, plot actual against predicted price, and group Gradient Boosting importances back to the original features.

## Data

Training data: Dean De Cock, Ames Housing dataset, as published for the Kaggle competition linked above. The copy used here is `data/train.csv`.
