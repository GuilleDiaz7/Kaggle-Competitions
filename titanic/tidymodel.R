library(tidymodels)
library(tidyverse)
library(ggplot2)

rm(list = ls())

titanic_train <- read_csv("titanic/data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
titanic_test <- read_csv("titanic/data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <- read_csv("titanic/data/gender_submission.csv")

titanic_train <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/train.csv")
titanic_test <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/test.csv") 
submission <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/gender_submission.csv")

titanic_train <- titanic_train %>% 
  mutate(
    Survived = factor(
      Survived,
      levels = c(0, 1),
      labels = c("Deceased", "Survived")
    )
  )

set.seed(123)
titanic_folds <- vfold_cv(titanic_train, strata = Survived)

glm_recipe <- recipe(Survived ~., data = titanic_train) %>% 
  update_role(PassengerId, new_role = "Id") %>%   
  step_rm(Ticket) %>% 
  step_string2factor(all_nominal_predictors()) %>% 
  step_num2factor(Pclass, levels = c("First", "Second", "Third")) %>% 
  step_mutate(Survived = factor(
    Survived,
    levels = c(0, 1),
    labels = c("Deceased", "Survived")
  )
  ) %>% 
  step_mutate(
    title = str_match(Name, ", ([:alpha:]+)\\."),
    title = if_else(is.na(title[, 2]), "NA", title[, 2])
  ) %>% 
  step_other(title, threshold = 0.02, other = "Other") %>% 
  update_role(Name, new_role = "Id") %>% 
  step_mutate(Cabin = if_else(is.na(Cabin), "Missing", "Available")) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())
prep(glm_recipe)

glm_baked <- glm_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)
names(glm_baked)

glm_spec <- logistic_reg() %>% 
  set_engine("glm")

glm_wf <- workflow() %>% 
  add_recipe(glm_recipe) %>% 
  add_model(glm_spec)

list_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity)

doParallel::registerDoParallel()

glm_results <- glm_wf %>% 
  fit_resamples(
    resamples = titanic_folds,
    metrics = list_metrics,
    control = control_resamples(save_pred = TRUE)
  )

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

fitted <- fit(glm_wf, titanic_train)
predict(fitted, new_data = titanic_test)