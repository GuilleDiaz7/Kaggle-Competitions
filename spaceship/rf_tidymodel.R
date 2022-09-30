#### EXPLORATORY DATA ANALYSIS OF THE SCRAPED DATA ####
#### BASIC CODE ####

## Clean data memory
rm(list = ls())
rm("model_one_recipe") # Clean a particular R object

## Remove plots
dev.off(dev.list()["RStudioGD"]) # Apply dev.off() & dev.list()


#### LOAD LIBRARIES ####
library(tidymodels)
library(tidyverse)

#### LOAD DATASET, TRAIN AND TEST ####
space_df <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
space_test_final <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <- read_csv("data/sample_submission.csv")

set.seed(123)
space_split <- initial_split(space_df, strata = Transported)
space_train <- training(space_split)
space_test <- testing(space_split)

space_recipe <- recipe(Transported ~ ., data = space_train) %>% 
  update_role(PassengerId, Name, new_role = "ID") %>% 
  step_rm(Cabin) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_impute_knn(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_scale(all_numeric_predictors())
model_prep <- prep(space_recipe)
juiced <- juice(model_prep)

tune_spec <- rand_forest(
  mtry = tune(),
  trees = 2,
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

tune_wf <- workflow() %>%
  add_recipe(space_recipe) %>%
  add_model(tune_spec)

set.seed(234)
space_folds <- vfold_cv(space_train)

doParallel::registerDoParallel()

set.seed(345)
tune_res <- tune_grid(
  tune_wf,
  resamples = space_folds,
  grid = 10
)

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

doParallel::registerDoParallel()
rf_grid <- grid_regular(
  mtry(range = c(0, 1500)),
  min_n(range = c(30, 40)),
  levels = 5
)

set.seed(456)
regular_res <- tune_grid(
  tune_wf,
  resamples = space_folds,
  grid = rf_grid
)

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC")

best_auc <- select_best(regular_res, "roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_auc
)

final_rf

library(vip)

final_rf %>%
  set_engine("ranger", importance = "permutation") %>%
  fit(Transported ~ .,
      data = juice(model_prep) %>% select(-c(PassengerId, Name))
  ) %>%
  vip(geom = "point")

final_wf <- workflow() %>%
  add_recipe(space_recipe) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(space_split)

final_res %>%
  collect_metrics()

final_res %>% 
  collect_predictions()

conf_mat_resampled(final_res, tidy = FALSE) %>% 
  autoplot()

## TO submit to kaggle ##
test_pred <- final_res %>% 
  extract_workflow() %>% 
  predict(space_test_final)


options(warn = getOption("warn"))
test_pred_new <- test_pred %>% 
  mutate(.pred_class = str_to_title(.pred_class))

submission$Transported <- test_pred_new$.pred_class
write_csv(submission, "data/submission.csv")
