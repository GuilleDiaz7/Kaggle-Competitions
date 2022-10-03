library(tidymodels)
library(tidyverse)
library(ggplot2)
library(vip) 
library(xgboost)

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

xgb_recipe <- recipe(Survived ~., data = titanic_train) %>% 
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

xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), titanic_train),
  learn_rate(),
  size = 30
)

xgb_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

set.seed(323)
titanic_folds <- vfold_cv(titanic_train, strata = Survived)

doParallel::registerDoParallel()

set.seed(234)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = titanic_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

xgb_res %>% 
  collect_metrics()
xgb_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

show_best(xgb_res, "roc_auc")

best_auc <- select_best(xgb_res, "roc_auc")
best_auc

final_xgb <- finalize_workflow(
  xgb_wf,
  best_auc
)

final_xgb %>%
  fit(data = titanic_train) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point")

collect_metrics(final_res)

final_res %>%
  collect_predictions() %>%
  roc_curve(Survived, .pred_Deceased) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )

fitted <- fit(final_xgb, titanic_train)

predictions <- predict(fitted, new_data = titanic_test)
xgb_submission <- submission
xgb_submission$Survived <- predictions
xgb_submission <- xgb_submission %>% 
  mutate(
    Survived = if_else(Survived == "Deceased", 0, 1)
  )
write.csv(xgb_submission, "titanic/data/xgb_submission.csv", row.names = FALSE)

# Kaggle accuracy score: 0.79904