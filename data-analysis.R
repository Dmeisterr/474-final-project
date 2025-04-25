############################################################
# Wildfire Cause Classification Pipeline (SQLite Version)
# --------------------------------------------------------
# Connects to a local SQLite database containing the 1.88 Million US Wildfires
# dataset, cleans/engineers features, trains and evaluates multiple
# classification models, and stores the best workflow. This version fixes the
# `as_date()` error by explicitly namespacing it as `lubridate::as_date()`.
############################################################

############################################################
# 1. Setup -------------------------------------------------
############################################################
# Ensure a default CRAN mirror for non‑interactive installs
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::activate()

required_pkgs <- c(
  "tidyverse", "data.table", "janitor", "lubridate", "here",
  "tidymodels", "themis", "ranger", "xgboost", "vip", "SHAPforxgboost",
  "sf", "DBI", "RSQLite"
)
missing_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if (length(missing_pkgs)) install.packages(missing_pkgs, dependencies = TRUE)

lapply(required_pkgs, library, character.only = TRUE)
set.seed(42)

############################################################
# 2. Paths & Data (SQLite) --------------------------------
############################################################
proj_dir <- here::here()
raw_dir  <- file.path(proj_dir, "data")
if (!dir.exists(raw_dir)) dir.create(raw_dir, recursive = TRUE)

db_path <- file.path(raw_dir, "wildfires.sqlite")
if (!file.exists(db_path)) {
  stop("SQLite database not found. Place 'wildfires.sqlite' inside the data/ folder or adjust `db_path` in the script.")
}

# Establish connection
con <- DBI::dbConnect(RSQLite::SQLite(), db_path)

table_name <- "Fires"          # <-- change if your table name differs
if (!table_name %in% DBI::dbListTables(con)) {
  stop(paste("Table", table_name, "not found in database."))
}

fires_raw <- tbl(con, table_name) %>% collect()
DBI::dbDisconnect(con)

############################################################
# 3. Initial Exploration ----------------------------------
############################################################
print(glimpse(fires_raw))

fires_raw %>%
  count(STAT_CAUSE_DESCR, sort = TRUE) %>%
  print(n = 50)

############################################################
# 4. Cleaning & Feature Engineering -----------------------
############################################################
fires_tbl <- fires_raw %>%
  janitor::clean_names() %>%
  mutate(
    # DISCOVERY_DATE is Julian-day style numeric from 1970‑01‑01 origin
    discovery_date = lubridate::as_date(discovery_date, origin = "1970-01-01"),
    year  = lubridate::year(discovery_date),
    month = lubridate::month(discovery_date, label = TRUE, abbr = TRUE),
    doy   = lubridate::yday(discovery_date),
    fire_size_log = log1p(fire_size)
  ) %>%
  filter(!is.na(latitude), !is.na(longitude), latitude != 0, longitude != 0) %>%
  group_by(stat_cause_descr) %>%
  mutate(cause_n = n()) %>%
  ungroup() %>%
  mutate(stat_cause_descr = forcats::fct_lump_min(stat_cause_descr, min = 5000, other_level = "Other")) %>%
  select(-cause_n)

############################################################
# 5. Partitioning -----------------------------------------
############################################################
split      <- rsample::initial_split(fires_tbl, strata = stat_cause_descr, prop = 0.7)
train_data <- rsample::training(split)
 test_data <- rsample::testing(split)

folds <- rsample::vfold_cv(train_data, v = 5, strata = stat_cause_descr)

############################################################
# 6. Preprocessing Recipe ---------------------------------
############################################################
rec <- recipes::recipe(stat_cause_descr ~ ., data = train_data) %>%
  recipes::update_role(fire_year, new_role = "ID") %>%
  recipes::step_rm(fire_year) %>%
  recipes::step_mutate_at(recipes::all_numeric(), -recipes::all_outcomes(), fn = list(zero2na = ~ dplyr::na_if(., 0))) %>%
  recipes::step_impute_median(recipes::all_numeric(), -recipes::all_outcomes()) %>%
  recipes::step_impute_mode(recipes::all_nominal(), -recipes::all_outcomes()) %>%
  recipes::step_other(recipes::all_nominal(), threshold = 0.01) %>%
  recipes::step_dummy(recipes::all_nominal_predictors()) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  themis::step_smote(stat_cause_descr)

############################################################
# 7. Model Specifications ---------------------------------
############################################################
rf_spec <- parsnip::rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  parsnip::set_mode("classification") %>%
  parsnip::set_engine("ranger", importance = "impurity")

xgb_spec <- parsnip::boost_tree(
  trees = tune(), mtry = tune(), learn_rate = tune(), min_n = tune(), loss_reduction = tune()
) %>%
  parsnip::set_mode("classification") %>%
  parsnip::set_engine("xgboost")

log_spec <- parsnip::multinom_reg(penalty = tune(), mixture = tune()) %>%
  parsnip::set_engine("glmnet")

############################################################
# 8. Tuning -----------------------------------------------
############################################################
ctrl <- tune::control_grid(save_pred = TRUE, verbose = TRUE)

run_tuning <- function(model_spec, grid = 10) {
  workflows::workflow() %>%
    workflows::add_model(model_spec) %>%
    workflows::add_recipe(rec) %>%
    tune::tune_grid(resamples = folds, grid = grid,
                    metrics = yardstick::metric_set(accuracy, bal_accuracy, kap, f_meas),
                    control = ctrl)
}

rf_res  <- run_tuning(rf_spec, grid = 20)
xgb_res <- run_tuning(xgb_spec, grid = 30)
log_res <- run_tuning(log_spec, grid = 15)

############################################################
# 9. Select Best Model ------------------------------------
############################################################
best_rf  <- tune::select_best(rf_res,  metric = "bal_accuracy")
best_xgb <- tune::select_best(xgb_res, metric = "bal_accuracy")
best_glm <- tune::select_best(log_res, metric = "bal_accuracy")

model_metrics <- tibble::tibble(
  model = c("Random Forest", "XGBoost", "GLMnet"),
  bal_accuracy = c(best_rf$bal_accuracy, best_xgb$bal_accuracy, best_glm$bal_accuracy)
)
print(model_metrics)

best_model <- dplyr::arrange(model_metrics, dplyr::desc(bal_accuracy)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(model)

if (best_model == "Random Forest") {
  final_wf <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(parsnip::finalize_model(rf_spec, best_rf))
} else if (best_model == "XGBoost") {
  final_wf <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(parsnip::finalize_model(xgb_spec, best_xgb))
} else {
  final_wf <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(parsnip::finalize_model(log_spec, best_glm))
}

############################################################
# 10. Fit Full Model --------------------------------------
############################################################
final_fit <- workflows::fit(final_wf, data = train_data)

############################################################
# 11. Test Evaluation -------------------------------------
############################################################
test_predictions <- predict(final_fit, test_data, type = "prob") %>%
  dplyr::bind_cols(predict(final_fit, test_data)) %>%
  dplyr::bind_cols(test_data %>% dplyr::select(stat_cause_descr))

test_metrics <- yardstick::metrics(test_predictions, truth = stat_cause_descr, estimate = .pred_class, .pred_lightning:.pred_other)
print(test_metrics)

print(yardstick::conf_mat(test_predictions, truth = stat_cause_descr, estimate = .pred_class))

############################################################
# 12. Explainability --------------------------------------
############################################################
if (best_model == "Random Forest") {
  vip::vip(final_fit$fit$fit, num_features = 20)
} else if (best_model == "XGBoost") {
  shap_values <- SHAPforxgboost::shap.values(xgb_model = final_fit$fit$fit,
                                             X_train = recipes::bake(recipes::prep(rec), new_data = NULL, recipes::all_predictors()))
  SHAPforxgboost::shap.plot.summary(shap_values)
}

############################################################
# 13. Save Artifacts --------------------------------------
############################################################
output_dir <- file.path(proj_dir, "output")
if (!dir.exists(output_dir)) dir.create(output_dir)
model_path <- file.path(output_dir, "wildfire_cause_model.rds")

saveRDS(final_fit, model_path)

predict_cause <- function(new_data) {
  mdl <- readR
