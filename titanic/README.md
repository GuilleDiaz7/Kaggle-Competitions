# Basic model explanation

This is the explanation of a quite basic workflow for a logistic model which gave a 0.77511 score in Kaggle.

## Load libraries

Firstly, we load the libraries: (1) `tidyverse` to clean and preprocess data, (2) `tidymodels` to create the statistical or ML model and (3) `ggplot2` to visualiza the data or the model results.
```R
library(tidymodels)
library(tidyverse)
library(ggplot2)
```

## Load data

Then we load the data provided by [Kaggle](https://www.kaggle.com/competitions/titanic/data). I converted both character and logical data into factor, although it is not strictly neccesary. Here I am using relative paths to load the data. I cloned this GitHub repository in my computer to work more comfortable with RStudio Desktop.
The `gender_submission` file contains my last submission, not the original submission file from Kaggle.

```R
titanic_train <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
titanic_test <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <- read_csv("data/gender_submission.csv")
```

If you want to download data directly from this repository without cloning it, you can use the following code.

```R
titanic_train <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/train.csv")
titanic_test <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/test.csv") 
submission <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/gender_submission.csv")
``` 

## Preprocess the data

We convert the `Survived` variable into a factor and change the labels to be more readable. We also extract title from the `Name` variable such as Mr or Master or Dr.

```R
titanic_train <- titanic_train %>% 
  mutate(Survived = factor(
  Survived,
  levels = c(0, 1),
  labels = c("Deceased", "Survived")))
  
titanic_train <- titanic_train %>% 
  mutate(
    title = str_match(Name, ", ([:alpha:]+)\\."),
    title = if_else(is.na(title[, 2]), "NA", title[, 2])
    ) 

titanic_test <- titanic_test %>% 
  mutate(
    title = str_match(Name, ", ([:alpha:]+)\\."),
    title = if_else(is.na(title[, 2]), "NA", title[, 2])
  ) 
```

## Create the model

The formula `Survived ~.` means that the variable Survived will be predicted by every other variable in the dataset.

With `update_role` we convert a variable into and Id, so it will not be used for prediction. `Step_rm` drops a variable (in this case, Ticket, cause it would require quite a lot of preprocessing). `Step_other` musters every category under a threshold into a Other category. `Step_zv` removes any variable with zero variance and `step_normalize` standardizes any numeric variable.

 ```R
glm_recipe <- recipe(Survived ~., data = titanic_train) %>% 
  update_role(PassengerId, new_role = "Id") %>%   
  step_rm(Ticket) %>% 
  step_string2factor(all_nominal_predictors()) %>% 
  step_num2factor(Pclass, levels = c("First", "Second", "Third")) %>% 
  step_other(title, threshold = 0.02, other = "Other") %>% 
  update_role(Name, new_role = "Id") %>% 
  step_mutate(Cabin = if_else(is.na(Cabin), "Missing", "Available")) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())
```

I recommend to run the following code to check if there is any problem with the recipe.
```R
prep(glm_recipe)
```

Then we choose the engine for the logistic regression...

```R
glm_spec <- logistic_reg() %>% 
  set_engine("glm")
```

...and create the workflow.

```R
glm_wf <- workflow() %>% 
  add_recipe(glm_recipe) %>% 
  add_model(glm_spec)
 ```
 
 ## Fit the model
 
 We create folds, namely same size splits of the original data to train the model in each of them. The list of metrics will be used to check how well the model performed.
 
 ```R
 set.seed(123)
titanic_folds <- vfold_cv(titanic_train, strata = Survived)
list_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity)
```

Parallelization will help speed things up.

```R
doParallel::registerDoParallel()

glm_results <- glm_wf %>% 
  fit_resamples(
    resamples = titanic_folds,
    metrics = list_metrics,
    control = control_resamples(save_pred = TRUE)
  )
 ```
 
 ## Check the model performance
 
Here we see every metric we chose...

```R
glm_results %>% 
 collect_metrics()
```
...and the confusion matrix, which basically tells us how well each category was predicted. In this case, with two classes (Deceased and Survived) it will provide four combinations of predicted-observed results: Deceased-Deceases, Deceased-Survived, Survived-Deceased and Survived-Survived.

```R
glm_results %>% 
  conf_mat_resampled()
```

With the `collect_predictions` funtion we plot the `roc_curve`.

```R
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

## Optional - Submit results to Kaggle

If you want to submit your prediction on test data to Kaggle just run the following code.

```R
fitted <- fit(glm_wf, titanic_train)
predictions <- predict(fitted, new_data = titanic_test)
submission$Survived <- predictions
submission <- submission %>% 
  mutate(
    Survived = if_else(Survived == "Deceased", 0, 1)
  )
write.csv(submission, "titanic/data/gender_submission.csv", row.names = FALSE)
```
