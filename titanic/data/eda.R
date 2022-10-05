
#### LOAD LIBRARIES ####
library(readr)
library(tidyverse)
library(gpglot2)
library(correlationfunnel)
library(forcats)

rm(list = ls())


#### LOAD AND CLEAN DATA ####
df <- read_csv("https://raw.githubusercontent.com/GuilleDiaz7/Kaggle-Competitions/main/titanic/data/train.csv")
titanic_test <- read_csv("titanic/data/test.csv")
submission <- read_csv("titanic/data/gender_submission.csv")


df
sapply(df, class)

df <- df %>% 
  mutate(
    Survived = factor(Survived,
                      levels = c(0, 1),
                      labels = c("Deceased", "Survived")),
    Pclass = factor(Pclass,
                    levels = c(1, 2, 3),
                    labels = c("First", "Second", "Third")),
    Title = str_match(Name, ", ([:alpha:]+)\\."),
    Title = Title[, 2], 
    Family_Size = SibSp + Parch
  )

titanic_test <- titanic_test %>% 
  mutate(
    Pclass = factor(Pclass,
                    levels = c(1, 2, 3),
                    labels = c("First", "Second", "Third")),
    Title = str_match(Name, ", ([:alpha:]+)\\."),
    Title = Title[, 2], 
    Family_Size = SibSp + Parch
  )


#### SOME BASIC COUNTS ####
## TARGET VARIABLE ###
#We can see that there is a certain class imbalance between Deceased (0) and Survived (1)
# A 61.6% of the people died
df %>% 
  count(Survived) %>% 
  mutate(Perc = n / 891)

### TARGET AND SEX ###
# Of those who survived, most of them were female
# A 74.2% of female survived whereas a 81.1% of male didn't
df %>%
  group_by(Sex) %>% 
  count(Survived)

df %>%
  group_by(Sex) %>% 
  count(Survived) %>% 
  ggplot(aes(x = n, y = Survived, fill = Sex)) +
  geom_col(position = "fill")



### FAMILY SIZE ###
df %>% 
  group_by(Sex, Family_Size) %>% 
  count(Survived) %>% 
  mutate(Perc =  n / sum(n)) %>% 
  ggplot(aes(x = Sex, y = Perc,  fill = Survived)) +
  geom_col(position = "dodge") +
  facet_wrap(~Family_Size) +
  theme_classic() +
  theme(legend.position = "bottom")

# Lets refactorize family size
df <- df %>% 
  mutate(Family_Size = case_when(
    Family_Size == 0 | Family_Size == 1 ~ "Uno o dos",
    Family_Size == 2 | Family_Size == 3 ~ "Dos o tres",
    TRUE ~ "Más de 3"
  ))
titanic_test <- titanic_test %>% 
  mutate(Family_Size = case_when(
    Family_Size == 0 | Family_Size == 1 ~ "Uno o dos",
    Family_Size == 2 | Family_Size == 3 ~ "Dos o tres",
    TRUE ~ "Más de 3"
  ))



df %>% 
  group_by(Sex, Family_Size) %>% 
  count(Survived) %>% 
  mutate(Perc =  n / sum(n)) %>% 
  ggplot(aes(x = Sex, y = Perc,  fill = Survived)) +
  geom_col(position = "dodge") +
  facet_wrap(~fct_reorder(Family_Size, desc(n))) +
  theme_classic() +
  theme(legend.position = "bottom")

#### TITLE VARIABLE ####
df %>% 
  group_by(Title) %>% 
  count() %>% 
  arrange(desc(n))
# Lets collapse some title into an Other category
df <- df %>% mutate(
  Title = case_when(
    Title == "Mr" ~ "Mr",
    Title == "Miss"  ~ "Miss",
    Title == "Mrs" ~ "Mrs", 
    Title == "Master" ~ "Master",
    TRUE ~ "Other"
  )
)
titanic_test <- titanic_test %>% mutate(
  Title = case_when(
    Title == "Mr" ~ "Mr",
    Title == "Miss"  ~ "Miss",
    Title == "Mrs" ~ "Mrs", 
    Title == "Master" ~ "Master",
    TRUE ~ "Other"
  )
) 

 
df %>% 
  group_by(Title) %>% 
  count(Survived) %>% 
  filter(n > 10) %>% 
  ggplot(aes(x = n, y = Survived, fill = Title)) +
  geom_col(position = "fill")

# There seem to be a strong relation between age and title
df %>% 
  ggplot(aes(x = Age, y = fct_reorder(Title, Age, median))) +
  geom_boxplot() +
  theme_classic()

df %>% 
  drop_na(Age) %>% 
  group_by(Title) %>% 
  summarize(size = n(), age = median(Age), fare = median(Fare)) %>% 
  arrange(desc(age)) %>% 
  filter(size > 6)
# How is the Master median age so little
df %>% 
  filter(Title == "Master") %>% 
  print(n = 50)


#### NUMERIC VARIABLES ####
# Some numeric variableS
# This two graphics are produced with the following warning:
# Removed 177 rows containing non-finite values (NAs basically)
df %>% 
  ggplot(aes(x = Age, y = reorder(Survived, Age, median), color = Survived, fill = Survived)) +
  geom_boxplot(size = 0.9, alpha = 0.2, show.legend = F) +
  theme_classic()
df %>% 
  ggplot(aes(x = Age, y = reorder(Sex, Age, median), color = Sex, fill = Sex)) +
  geom_boxplot(size = 0.9, alpha = 0.2, show.legend = F) +
  theme_classic()

df %>% 
  ggplot(aes(x = Fare, y = reorder(Survived, Fare, median), color = Survived, fill = Survived)) +
  geom_boxplot(size = 0.9, alpha = 0.2, show.legend = F) +
df %>% 
  ggplot(aes(x = Fare, y = reorder(Sex, Fare, median), color = Sex, fill = Sex)) +
  geom_boxplot(size = 0.9, alpha = 0.2, show.legend = F) +
  theme_classic()  theme_classic()







#### Correlation funnel ####

df %>% 
  drop_na() %>% 
  select(-PassengerId, -Name) %>% 
  binarize(n_bins = 4, thresh_infreq = 0.01) %>% 
  correlate(Survived__Survived) %>% 
  plot_correlation_funnel(interactive = FALSE)


## Visualization of missing data
library(naniar)

# There is a lot of NAs in Cabin and some in Age
df %>% 
  summarise(across(everything(), ~sum(is.na(.))))
miss_var_summary(df)
## 77.1 of Cabin cells are NA's. Before removing it, let's see if there is any correlation with surviving
df %>% 
  mutate(Cabin = if_else(is.na(Cabin), "Missing", "Available")) %>% 
  ggplot(aes(y = Cabin, fill = factor(Survived))) +
  geom_bar(position = "fill")
# It clearly has some information
df <- df %>% 
  mutate(
    Cabin = case_when(
      str_match(df$Cabin, "C") == "C" ~ "C",
      str_match(df$Cabin, "A") == "A" ~ "A",
      str_match(df$Cabin, "E") == "E" ~ "E",
      str_match(df$Cabin, "B") == "B" ~ "B",
      str_match(df$Cabin, "D") == "D" ~ "D",
      str_match(df$Cabin, "F") == "F" ~ "F",
      str_match(df$Cabin, "F") == "F" ~ "F",
    )
  )

df %>% 
  group_by(Cabin) %>% 
  summarise(median(Fare))


gg_miss_var(df)
miss_case_summary(df) %>% 
  count(as.character(n_miss))
NA_2 <- miss_case_summary(df) %>% 
  filter(n_miss == 2) %>% 
  select(case) %>% 
  unlist()
df %>% 
  filter(PassengerId %in% NA_2) %>% 
  summarise(mean(Fare), median(Fare))
df %>% 
  summarise(mean(Fare), median(Fare))


df %>% 
  ggplot(aes(x = Age, y = Fare)) +
  geom_point()

df %>% 
  ggplot(aes(x = Age, y = Fare)) +
  geom_miss_point() +
  facet_wrap(~Survived) +
  theme(legend.position = "bottom")

df %>% 
  bind_shadow() %>% 
  ggplot(aes(x = Fare, 
             fill = Age_NA)) +
  geom_density(alpha = 0.5)

df %>% 
  ggplot(aes(x = Age, y = Fare)) +
  geom_miss_point() +
  facet_wrap(~Sex) +
  theme(legend.position = "bottom")










#### XGB BOOST CLEAN DATA ####

xgb_recipe <- recipe(Survived ~., data = df) %>% 
  update_role(PassengerId, new_role = "Id") %>%   
  step_rm(Ticket) %>% 
  step_string2factor(all_nominal_predictors()) %>% 
  update_role(Name, new_role = "Id") %>% 
  step_mutate(Cabin = if_else(is.na(Cabin), "Missing", "Available")) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_corr(all_numeric(), threshold = 0.9) %>% 
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
  finalize(mtry(), df),
  learn_rate(),
  size = 30
)

xgb_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

set.seed(323)
titanic_folds <- vfold_cv(df, strata = Survived)

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
  fit(data = df) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point")

fitted <- fit(final_xgb, df)

predictions <- predict(fitted, new_data = titanic_test)
xgb_submission_prueba <- submission
xgb_submission_prueba$Survived <- predictions
xgb_submission_prueba <- xgb_submission_prueba %>% 
  mutate(
    Survived = if_else(Survived == "Deceased", 0, 1)
  )
write.csv(xgb_submission_prueba, "titanic/data/xgb_submission_prueba.csv", row.names = FALSE)
