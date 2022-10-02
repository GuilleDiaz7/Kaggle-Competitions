# Model explanation

## Load libraries and data
In first place we load the libraries, one to clean and preprocess data and the other to create the statistical model.
```R
library(tidymodels)
library(tidyverse)
library(ggplot2)
```

The we load the data provided by [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data). I converted both character and logical data into factor, although it is not strictly neccesary. Here I am using relative paths to load the data. I cloned this GitHub repository in my computer to work more comfortable with RStudio Desktop.
```R
space_train <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
space_test <- space_test_final <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <-  <- read_csv("data/sample_submission.csv")
```


## Preprocess data and create the recipe
In this case, the data is already split, but this is the code used if the dataset comes as a whole, not separated in train and test. The `set.seed()` function makes the splitting randomly.
```R
set.seed(123)
space_split <- initial_split(space_df, strata = Transported)
space_train_split <- training(space_split)
space_test_split <- testing(space_split)
```

Now it is time to create the recipe where every preprocessing step applied to the data is stored. First we indicate the formula, which means that the `Transported` variable will be predicted by the rest of variables and the training data will be `space_train`.
```R
(Transported ~ ., data = space_train)
```

With `update_role` we convert a couple variables into Id's, then `step_rm` deletes any variable selected, `step_impute_median` imputes the median to the missing values of `all_numeric_predictors` and `step_impute_knn` uses nearest neighbourhoods algorithm to impute missing values for `all_nominal_predictors`. Finally, `step_dummy` creates dummy variables for the nominal predictores, `step_zv` drops variables with *zero variance* (basically just 1 category or value) and `step_scale` normalizes numeric variables.
```R
space_recipe <- recipe(Transported ~ ., data = space_train) %>% 
  update_role(PassengerId, Name, new_role = "ID") %>% 
  step_rm(Cabin) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_impute_knn(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_scale(all_numeric_predictors())
```

## Create and tune the model
We use a technique called random forest, which combines the results of a number of decision trees. In this case I choose 2, which is a very low number. The default value is 500. Random forest is incredibly slow and this is just an example. Normally, with a high enough number of trees, let's say 1000, we wouldn't need hyperparameter tuning. In this example we tune `mtry` (number of randomly selected predictors) and `min_n` (minimal node size of the trees).
```r
tune_spec <- rand_forest(
  mtry = tune(),
  trees = 2,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")
```

Then, we create the workflow.
```r
tune_wf <- workflow() %>%
  add_recipe(space_recipe) %>%
  add_model(tune_spec)
```

It is recommended to train the model with different samples of the dataset (cross validation) so we create ten folds of the same size, approximately.
```r
set.seed(234)
space_folds <- vfold_cv(space_train)
```

Now we are going to see which parameters work better with a grid search. As I said, random forest takes some time, so parallelization will speed things up. This part may take a while, so grab a cup of coffee and open a book.
```r
doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = space_folds,
  grid = 10
)
```

Once the process is completed we can plot the results.
```R
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")
```
![grid_search_metrics](https://user-images.githubusercontent.com/42537388/193423194-da1ed0ce-b9fe-4da4-9b0a-e67377159f69.png)
