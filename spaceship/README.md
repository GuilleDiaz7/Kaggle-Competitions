# Model explanation

In first place we load the libraries, one to clean and preprocess data and the other to create the statistical model.
```r
library(tidymodels)
library(tidyverse)
```

The we load the data provided by [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data). I converted both character and logical data into factor, although it is not strictly neccesary. Here I am using relative paths to load the data. I cloned this GitHub repository in my computer to work more comfortable with RStudio Desktop.
```
space_train <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
space_test <- space_test_final <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <-  <- read_csv("data/sample_submission.csv")
```

In this case, the data is already split, but I prefer to use this code because normally the dataset comes as a whole.
```
set.seed(123)
space_split <- initial_split(space_df, strata = Transported)
space_train <- training(space_split)
space_test <- testing(space_split)
```

Now it is time to create the recipe where every preprocessing step applied to the data is stored.
```
space_recipe <- recipe(Transported ~ ., data = space_train) %>% 
  update_role(PassengerId, Name, new_role = "ID") %>% 
  step_rm(Cabin) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_impute_knn(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_scale(all_numeric_predictors())
```