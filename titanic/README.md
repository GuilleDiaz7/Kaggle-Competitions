# Basic model explanation

## Load libraries and data

Firstly, we load the libraries: (1) `tidyverse` to clean and preprocess data, (2) `tidymodels` to create the statistical or ML model and (3) `ggplot2` to visualiza the data or the model results.
```R
library(tidymodels)
library(tidyverse)
library(ggplot2)
```

Then we load the data provided by [Kaggle](https://www.kaggle.com/competitions/titanic/data). I converted both character and logical data into factor, although it is not strictly neccesary. Here I am using relative paths to load the data. I cloned this GitHub repository in my computer to work more comfortable with RStudio Desktop.
```R
titanic_train <- read_csv("data/train.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
titanic_test <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character) | where(is.logical), as.factor))
submission <-  <- read_csv("data/gender_submission.csv")
```

Split the training data into train and validation sets.
```R
set.seed(212)
train_split <- initial_split(
```
