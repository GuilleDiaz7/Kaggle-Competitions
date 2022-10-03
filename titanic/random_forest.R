library(tidymodels)
library(tidyverse)
library(ggplot2)
library(vip)
library(themis)

rm(list = ls())

titanic_train <- read_csv("titanic/data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
titanic_test <- read_csv("titanic/data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <- read_csv("titanic/data/gender_submission.csv")

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

rf_recipe <- recipe(Survived ~., data = titanic_train) %>% 
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
  step_normalize(all_numeric_predictors()) %>% 
  step_smote()

titanic_prep <- prep(rf_recipe)
bake(prep(rf_recipe), new_data = NULL)

rf_spec <- rand_forest(trees = 2500) %>% 
  set_engine("ranger", importance = "permutation") %>% 
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(rf_recipe) %>% 
  add_model(rf_spec)

set.seed(323)
titanic_folds <- vfold_cv(titanic_train, strata = Survived)
list_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity)

doParallel::registerDoParallel()

rf_results <- rf_wf %>% 
  fit_resamples(
    resamples = titanic_folds,
    metrics = list_metrics,
    control = control_resamples(save_pred = TRUE)
  )

rf_results %>% 
  collect_metrics()

rf_results %>% 
  conf_mat_resampled()

rf_results %>% 
  collect_predictions()

rf_results %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Survived, .pred_Deceased) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal()

fitted <- fit(rf_wf, titanic_train)

fitted %>% 
  extract_fit_parsnip() %>% 
  vip(geom = "point")

predictions <- predict(fitted, new_data = titanic_test)
rf_submission <- submission
rf_submission$Survived <- predictions
rf_submission <- rf_submission %>% 
  mutate(
    Survived = if_else(Survived == "Deceased", 0, 1)
  )
write.csv(rf_submission, "titanic/data/rf_submission.csv", row.names = FALSE)

# Kaggle accuracy score: 0.77511