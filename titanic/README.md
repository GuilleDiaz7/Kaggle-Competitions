# Basic model explanation

This is the explanation of a really basic workflow of a logistic model which will give bad results. On this [file]() there is a better model.

## Load libraries

Firstly, we load the libraries: (1) `tidyverse` to clean and preprocess data, (2) `tidymodels` to create the statistical or ML model and (3) `ggplot2` to visualiza the data or the model results.
```R
library(tidymodels)
library(tidyverse)
library(ggplot2)
```

## Load data

Then we load the data provided by [Kaggle](https://www.kaggle.com/competitions/titanic/data). I converted both character and logical data into factor, although it is not strictly neccesary. Here I am using relative paths to load the data. I cloned this GitHub repository in my computer to work more comfortable with RStudio Desktop.
```R
titanic_train <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
titanic_test <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <-  <- read_csv("data/gender_submission.csv")
```

If you want to download data directly from this repository without cloning it, you can use the following code.
```R
titanic_train <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/train.csv")
titanic_test <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/test.csv") 
submission <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/gender_submission.csv")
``` 

## Preprocess the data

We convert the `Survived` variable into a factor and change the labels to be more readable.
```R
titanic_train <- titanic_train %>% 
  mutate(
    Survived = factor(
      Survived,
      levels = c(0, 1),
      labels = c("Deceased", "Survived")
    )
  )
```

## Create the model

Ecribir

 ```R
 glm_recipe <- recipe(Survived ~., data = titanic_train) %>% 
  update_role(PassengerId, new_role = "Id") %>% 
  step_mutate(Survived = factor(
    Survived,
    levels = c(0, 1),
    labels = c("Deceased", "Survived")
  )) %>% 
  step_string2factor(all_nominal_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

glm_spec <- logistic_reg() %>% 
  set_engine("glm")

glm_wf <- workflow() %>% 
  add_recipe(glm_recipe) %>% 
  add_model(glm_spec)
 ```
 
 ## Fit the model
 
 Escribir
 
 ```R
 set.seed(123)
titanic_folds <- vfold_cv(titanic_train, strata = Survived)
list_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity)

doParallel::registerDoParallel()

glm_results <- glm_wf %>% 
  fit_resamples(
    resamples = titanic_folds,
    metrics = list_metrics,
    control = control_resamples(save_pred = TRUE)
  )
 ```
 
 ## Check the model performance
 
 Escribir
 
 ```R
 glm_results %>% 
  collect_metrics()

glm_results %>% 
  conf_mat_resampled()

glm_results %>% 
  collect_predictions()

glm_results %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Survived, .pred_Survived) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal()
 ```
