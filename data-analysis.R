############################################################
# Wildfire Cause Classification Pipeline (SQLite Version)
# --------------------------------------------------------
# Connects to a local SQLite database containing the 1.88 Million US Wildfires
# dataset, cleans/engineers features, trains and evaluates multiple
# classification models, and stores the best workflow.
#
############################################################

############################################################
# 1. Setup -------------------------------------------------
############################################################
cat("\n[STEP 1] Setting up environment and packages...\n")
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")
renv::activate()

required_pkgs <- c(
  "tidyverse", "data.table", "janitor", "lubridate", "here",
  "tidymodels", "themis", "ranger", "xgboost", "vip", "SHAPforxgboost",
  "sf", "DBI", "RSQLite", "glmnet" 
)
missing_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if (length(missing_pkgs)) {
  cat("Installing missing packages:", paste(missing_pkgs, collapse=", "), "\n")
  install.packages(missing_pkgs, dependencies = TRUE)
} else {
  cat("All required packages are already installed.\n")
}

lapply(required_pkgs, library, character.only = TRUE)
set.seed(42)

############################################################
# 2. Paths & Data (SQLite) --------------------------------
############################################################
cat("\n[STEP 2] Loading data from SQLite database...\n")
proj_dir <- here::here()
raw_dir  <- file.path(proj_dir, "data")
if (!dir.exists(raw_dir)) {
  cat("Creating data directory at", raw_dir, "\n")
  dir.create(raw_dir, recursive = TRUE)
}

db_path <- file.path(raw_dir, "wildfires.sqlite")
if (!file.exists(db_path)) {
  stop("SQLite database not found. Place 'wildfires.sqlite' inside the data/ folder or adjust `db_path` in the script.")
}
cat("Database found at", db_path, "\n")

cat("Connecting to SQLite database...\n")
con <- DBI::dbConnect(RSQLite::SQLite(), db_path)

table_name <- "Fires"          
if (!table_name %in% DBI::dbListTables(con)) {
  stop(paste("Table", table_name, "not found in database."))
}
cat("Found table:", table_name, "\n")

cat("Fetching wildfire data...\n")
fires_raw <- tbl(con, table_name) %>% collect()
cat("Loaded", nrow(fires_raw), "rows and", ncol(fires_raw), "columns\n")
DBI::dbDisconnect(con)

############################################################
# 3. Cleaning & Feature Engineering -----------------------
############################################################
cat("\n[STEP 3] Cleaning data and engineering features...\n")
fires_tbl <- fires_raw %>%
  janitor::clean_names() %>%
  # ---- Remove columns that can't be processed by recipes ------------------
  # Filter out columns that are list‑columns, raw, or blob (e.g., `shape`)
  dplyr::select(where(~ !(is.list(.x) || inherits(.x, "raw") || inherits(.x, "blob")))) %>%
  # ----------------------------------------------------------------------
  mutate(
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

cat("Cleaned data has", nrow(fires_tbl), "rows and", ncol(fires_tbl), "columns\n")
cat("Cause distribution:\n")
print(table(fires_tbl$stat_cause_descr))

############################################################
# 4. Partitioning -----------------------------------------
############################################################
cat("\n[STEP 4] Partitioning data into training and testing sets...\n")
set.seed(42)
fires_tbl <- fires_tbl %>%
  group_by(stat_cause_descr) %>%
  slice_sample(prop = 0.01) %>%  # Use just 1% of data
  ungroup()

split      <- rsample::initial_split(fires_tbl, strata = stat_cause_descr, prop = 0.7)
train_data <- rsample::training(split)
test_data  <- rsample::testing(split)

cat("Training set:", nrow(train_data), "rows\n")
cat("Testing set:", nrow(test_data), "rows\n")

folds <- rsample::vfold_cv(train_data, v = 3, strata = stat_cause_descr)
cat("Created 3-fold cross-validation splits\n")

############################################################
# 4b. Feature Selection (Manual) --------------------------
############################################################
cat("\n[STEP 4b] Performing manual feature selection...\n")

# Create a smaller dataset for quick feature importance calculation
set.seed(42)
feature_sel_data <- train_data %>% 
  group_by(stat_cause_descr) %>%
  slice_sample(prop = 0.1) %>%  # Use 10% of training data
  ungroup()
cat("Using", nrow(feature_sel_data), "rows for feature selection\n")

# Fit a quick Random Forest on the data
quick_rf <- ranger::ranger(
  stat_cause_descr ~ ., 
  data = feature_sel_data,
  importance = "impurity",
  num.trees = 100,
  seed = 42
)

# Get variable importance
var_imp <- tibble::tibble(
  variable = names(quick_rf$variable.importance),
  importance = quick_rf$variable.importance
) %>%
  arrange(desc(importance))

# Select top features (keep the top 15)
top_vars <- var_imp %>%
  slice_head(n = 15) %>%
  pull(variable)

# Print top features
cat("Selected top 15 features:\n")
print(var_imp %>% slice_head(n = 15))

# Now adjust your recipe to use only these variables
rec <- recipes::recipe(stat_cause_descr ~ ., data = train_data) %>%
  # Only keep important features
  recipes::step_rm(-all_of(c("stat_cause_descr", top_vars))) %>%
  # exclude variables that should not enter modeling ------------------
  recipes::step_rm(stat_cause_code) %>%            # drop stat_cause_code
  #-------------------------------------------------------------------
  # drop stat_cause_code feature
  recipes::step_rm(stat_cause_code) %>%
  # Continue with normal preprocessing
  recipes::step_mutate_at(recipes::all_numeric(), -recipes::all_outcomes(),
                        fn = list(zero2na = ~ dplyr::na_if(., 0))) %>%
  recipes::step_impute_median(recipes::all_numeric(), -recipes::all_outcomes()) %>%
  recipes::step_impute_mode(recipes::all_nominal(), -recipes::all_outcomes()) %>%
  recipes::step_other(recipes::all_nominal(), threshold = 0.01) %>%
  recipes::step_dummy(recipes::all_nominal_predictors()) %>%
  recipes::step_zv(recipes::all_predictors()) %>%
  # Balance classes
  themis::step_smote(stat_cause_descr)

cat("Recipe created with feature selection and preprocessing steps\n")
print(rec)

############################################################
# 5. Preprocessing Recipe ---------------------------------
############################################################
cat("\n[STEP 5] Creating preprocessing recipe with feature selection...\n")
rec <- recipes::recipe(stat_cause_descr ~ ., data = train_data) %>%
  # Exclude variables that should not enter modeling ------------------
  recipes::step_rm(discovery_date, fire_year) %>%            # drop date & ID-like columns
  #-------------------------------------------------------------------
  recipes::step_mutate_at(recipes::all_numeric(), -recipes::all_outcomes(),
                          fn = list(zero2na = ~ dplyr::na_if(., 0))) %>%
  recipes::step_impute_median(recipes::all_numeric(), -recipes::all_outcomes()) %>%
  recipes::step_impute_mode(recipes::all_nominal(), -recipes::all_outcomes()) %>%
  recipes::step_other(recipes::all_nominal(), threshold = 0.01) %>%
  recipes::step_dummy(recipes::all_nominal_predictors()) %>%
  # Add feature selection steps
  recipes::step_corr(recipes::all_numeric_predictors(), threshold = 0.8) %>% # Remove highly correlated features
  recipes::step_zv(recipes::all_predictors()) %>%
  # Feature selection with filter methods instead of RFE
  recipes::step_nzv(all_predictors(), freq_cut = 95/5) %>%  # Remove near-zero variance predictors
  # Balance classes (after feature selection for speed)
  themis::step_smote(stat_cause_descr)

cat("Recipe created with the following steps:\n")
print(rec)

############################################################
# 6. Model Specifications ---------------------------------
############################################################
cat("\n[STEP 6] Creating model specifications...\n")

rf_spec <- parsnip::rand_forest(
  mtry = parsnip::tune(), 
  min_n = parsnip::tune(), 
  trees = 500  
) %>%
  parsnip::set_mode("classification") %>%
  parsnip::set_engine("ranger", importance = "impurity")
cat("- Random Forest model specified\n")

xgb_spec <- parsnip::boost_tree(
  trees = 100,  
  mtry = parsnip::tune(), 
  learn_rate = parsnip::tune(), 
  min_n = parsnip::tune(),
  loss_reduction = 0  
) %>%
  parsnip::set_mode("classification") %>%
  parsnip::set_engine("xgboost")
cat("- XGBoost model specified\n")

log_spec <- parsnip::multinom_reg(
  penalty = parsnip::tune(), 
  mixture = parsnip::tune()
) %>%
  parsnip::set_engine("glmnet")
cat("- Multinomial logistic regression (glmnet) specified\n")

############################################################
# 7. Tuning (Optimized for Speed) ------------------------
############################################################
cat("\n[STEP 7] Tuning hyperparameters (optimized for speed)...\n")

# 1. Use parallel processing 
library(doParallel)
n_cores <- min(parallel::detectCores() - 1, 4)  # Reduced to 2 cores to avoid memory issues
cat("Setting up parallel processing with", n_cores, "cores\n")
cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)

# 2. Use standard grid search with error catching
ctrl <- tune::control_grid(
  save_pred = TRUE, 
  verbose = TRUE,
  parallel_over = "resamples",
  allow_par = TRUE,
  extract = function(x) extract_model_info(x)
)

# Helper function to safely extract model info
extract_model_info <- function(x) {
  if (inherits(x, "try-error")) {
    return(NULL)
  } else {
    return(x)
  }
}

# 3. Define smaller, simpler grids
rf_grid <- tibble::tibble(
  mtry = c(3, 5),          # Reduced options
  min_n = c(10, 15)        # Reduced options
)
cat("Created Random Forest grid with", nrow(rf_grid), "combinations\n")

xgb_grid <- tibble::tibble(
  mtry = c(3, 5),          # Reduced options
  min_n = c(10, 15),       # Reduced options
  learn_rate = c(0.01, 0.1) # Direct values instead of transformations
)
cat("Created XGBoost grid with", nrow(xgb_grid), "combinations\n")

log_grid <- tibble::tibble(
  penalty = c(0.01, 0.1),  # Direct values
  mixture = c(0.5, 1)      # Reduced options
)
cat("Created Logistic Regression grid with", nrow(log_grid), "combinations\n")

# 4. Modified tuning function with error handling
run_tuning <- function(model_spec, grid, name = "Model") {
  cat(paste0("Starting hyperparameter tuning for ", name, "...\n"))
  start_time <- Sys.time()
  
  # Try to run the tuning with error handling
  result <- tryCatch({
    workflows::workflow() %>%
      workflows::add_model(model_spec) %>%
      workflows::add_recipe(rec) %>%
      tune::tune_grid(
        resamples = folds,
        grid = grid,
        metrics = yardstick::metric_set(accuracy, bal_accuracy),
        control = ctrl
      )
  }, error = function(e) {
    cat("Error in tuning:", e$message, "\n")
    return(NULL)
  })
  
  end_time <- Sys.time()
  time_taken <- round(as.numeric(difftime(end_time, start_time, units = "mins")), 2)
  cat(paste0("Completed ", name, " tuning in ", time_taken, " minutes\n"))
  return(result)
}

# 5. Run tuning one at a time
cat("Starting Random Forest tuning...\n")
rf_res <- run_tuning(rf_spec, grid = rf_grid, name = "Random Forest")

# Check if RF worked before continuing
if (!is.null(rf_res)) {
  cat("Random Forest tuning successful. Moving to XGBoost...\n")
  xgb_res <- run_tuning(xgb_spec, grid = xgb_grid, name = "XGBoost")
} else {
  cat("Skipping XGBoost due to Random Forest failure\n")
  xgb_res <- NULL
}

# Only try logistic if previous models worked
if (!is.null(rf_res) || !is.null(xgb_res)) {
  cat("Starting Logistic Regression tuning...\n")
  log_res <- run_tuning(log_spec, grid = log_grid, name = "Multinomial Regression")
} else {
  cat("Skipping Logistic Regression due to previous failures\n")
  log_res <- NULL
}

# Clean up parallel cluster
stopCluster(cl)
registerDoSEQ()

# Modified selection of best model to handle NULL results
cat("\n[STEP 8] Selecting best model...\n")

# Helper function to safely get metrics
get_best_metrics <- function(res, model_name) {
  if (is.null(res)) {
    cat(model_name, "failed to tune, skipping metrics\n")
    return(NULL)
  }
  
  best <- try(tune::select_best(res, metric = "bal_accuracy"), silent = TRUE)
  if (inherits(best, "try-error")) {
    cat("Could not select best parameters for", model_name, "\n")
    return(NULL)
  }
  
  cat("Best", model_name, "parameters:\n")
  print(best)
  
  metrics <- tune::collect_metrics(res) %>% 
    filter(.metric == "bal_accuracy") %>% 
    slice_max(mean, n = 1)
  
  return(list(best = best, metrics = metrics))
}

# Get metrics for each model
rf_results <- get_best_metrics(rf_res, "Random Forest")
xgb_results <- get_best_metrics(xgb_res, "XGBoost")
log_results <- get_best_metrics(log_res, "Multinomial Regression")

# Build model comparison table from available results
model_metrics <- tibble::tibble(
  model = character(),
  bal_accuracy = numeric()
)

if (!is.null(rf_results)) {
  model_metrics <- model_metrics %>%
    add_row(model = "Random Forest", bal_accuracy = rf_results$metrics$mean)
}

if (!is.null(xgb_results)) {
  model_metrics <- model_metrics %>%
    add_row(model = "XGBoost", bal_accuracy = xgb_results$metrics$mean)
}

if (!is.null(log_results)) {
  model_metrics <- model_metrics %>%
    add_row(model = "GLMnet", bal_accuracy = log_results$metrics$mean)
}

# Only proceed if we have at least one successful model
if (nrow(model_metrics) > 0) {
  cat("Model comparison metrics:\n")
  print(model_metrics)
  
  best_model <- dplyr::arrange(model_metrics, dplyr::desc(bal_accuracy)) %>%
    dplyr::slice(1) %>%
    dplyr::pull(model)
  
  cat("\nBest model:", best_model, "\n")
} else {
  cat("No models were successfully tuned. Need to troubleshoot data or simplify further.\n")
}

############################################################
# 8. Select Best Model ------------------------------------
############################################################
cat("\n[STEP 8] Selecting best model...\n")
best_rf  <- tune::select_best(rf_res,  metric = "bal_accuracy")
cat("Best Random Forest parameters:\n")
print(best_rf)

best_xgb <- tune::select_best(xgb_res, metric = "bal_accuracy")
cat("Best XGBoost parameters:\n")
print(best_xgb)

best_glm <- tune::select_best(log_res, metric = "bal_accuracy")
cat("Best Multinomial Regression parameters:\n")
print(best_glm)

rf_metrics <- tune::collect_metrics(rf_res) %>% 
  filter(.metric == "bal_accuracy") %>% 
  slice_max(mean)
xgb_metrics <- tune::collect_metrics(xgb_res) %>% 
  filter(.metric == "bal_accuracy") %>% 
  slice_max(mean)
glm_metrics <- tune::collect_metrics(log_res) %>% 
  filter(.metric == "bal_accuracy") %>% 
  slice_max(mean)

model_metrics <- tibble::tibble(
  model = c("Random Forest", "XGBoost", "GLMnet"),
  bal_accuracy = c(rf_metrics$mean, xgb_metrics$mean, glm_metrics$mean)
)
cat("Model comparison metrics:\n")
print(model_metrics)

best_model <- dplyr::arrange(model_metrics, dplyr::desc(bal_accuracy)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(model)

cat("\nBest model:", best_model, "\n")

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
# 9. Fit Full Model ---------------------------------------
############################################################
cat("\n[STEP 9] Fitting final model on full training data...\n")
start_time <- Sys.time()
final_fit <- workflows::fit(final_wf, data = train_data)
end_time <- Sys.time()
time_taken <- round(as.numeric(difftime(end_time, start_time, units = "mins")), 2)
cat("Final model fitting completed in", time_taken, "minutes\n")

############################################################
# 10. Test Evaluation -------------------------------------
############################################################
cat("\n[STEP 10] Evaluating model on test data...\n")
preds <- predict(final_fit, test_data)
preds_with_truth <- bind_cols(
  prediction = preds$.pred_class,
  truth = test_data$stat_cause_descr
)

# Calculate metrics
accuracy <- yardstick::accuracy(preds_with_truth, truth, prediction)
bal_accuracy <- yardstick::bal_accuracy(preds_with_truth, truth, prediction)
kappa <- yardstick::kap(preds_with_truth, truth, prediction)
f1 <- yardstick::f_meas(preds_with_truth, truth, prediction)

cat("\nTest set metrics:\n")
cat("Accuracy:", round(accuracy$.estimate, 4), "\n")
cat("Balanced Accuracy:", round(bal_accuracy$.estimate, 4), "\n")
cat("Kappa:", round(kappa$.estimate, 4), "\n")
cat("F1 Score:", round(f1$.estimate, 4), "\n")

# Confusion Matrix
conf_mat <- yardstick::conf_mat(preds_with_truth, truth, prediction)
cat("\nConfusion Matrix:\n")
print(conf_mat)

cat("\nAnalysis complete!\n")