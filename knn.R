####################################################################################################
#  Multi‑class KNN with parameter tuning and parallel processing for wildfire cause prediction
####################################################################################################

suppressPackageStartupMessages({
  library(dplyr);   library(caret);  library(Matrix);  library(class)
  library(doParallel);  library(MLmetrics)   # optional for extra metrics
})

set.seed(1)

# ── Parallel set‑up ────────────────────────────────────────────────────────────────────────────────
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(sprintf("Using %d cores for parallel processing\n", num_cores))

# ── Data load ─────────────────────────────────────────────────────────────────────────────────────
trainData <- read.csv("data/train_fires10.csv")
testData  <- read.csv("data/test_fires10.csv")

# ── Factor casting ────────────────────────────────────────────────────────────────────────────────
fact_cols <- c("stat_cause_descr","fire_size_class","owner_code",
               "state","discovery_hour")

trainData[fact_cols] <- lapply(trainData[fact_cols], as.factor)
testData[fact_cols]  <- lapply(testData[fact_cols],  as.factor)

trainData$discovery_month <- factor(trainData$discovery_month, levels = month.abb)
testData$discovery_month  <- factor(testData$discovery_month,  levels = month.abb)

# ── Remove predictor we're not using ──────────────────────────────────────────────────────────────
trainData <- trainData %>% select(-fire_size_class)
testData  <- testData  %>% select(-fire_size_class)

# ── DROP rows whose target is NA ──────────────────────────────────────────────────────────────────
trainData <- trainData %>% filter(!is.na(stat_cause_descr))
testData  <- testData  %>% filter(!is.na(stat_cause_descr))   # usually none, but be safe

# ── Harmonise factor levels across train & test (predictors only) ─────────────────────────────────
for (v in names(trainData)) {
  if (is.factor(trainData[[v]]) && v != "stat_cause_descr") {
    lvls <- union(levels(trainData[[v]]), levels(testData[[v]]))
    trainData[[v]] <- factor(trainData[[v]], levels = lvls)
    testData[[v]]  <- factor(testData[[v]],  levels = lvls)
  }
}

# ── Report class imbalance ────────────────────────────────────────────────────────────────────────
cat("Class distribution (train):\n")
print(table(trainData$stat_cause_descr))

# ── One‑hot encode categorical predictors ───────────────────────────────────────────────────────────
dv <- dummyVars(~ ., data = trainData %>% select(-stat_cause_descr), fullRank = TRUE)

train_x <- predict(dv, trainData %>% select(-stat_cause_descr))
test_x <- predict(dv, testData %>% select(-stat_cause_descr))

# ── Handle missing columns in test data ────────────────────────────────────────────────────────────
missing_cols <- setdiff(colnames(train_x), colnames(test_x))
for (col in missing_cols) {
  test_x <- cbind(test_x, rep(0, nrow(test_x)))
  colnames(test_x)[ncol(test_x)] <- col
}
test_x <- test_x[, colnames(train_x)]  # Ensure column order matches

# ── Scale features (critical for KNN) ────────────────────────────────────────────────────────────
train_center <- apply(train_x, 2, mean)
train_scale <- apply(train_x, 2, sd)
train_scale[train_scale == 0] <- 1  # Prevent division by zero

train_scaled <- scale(train_x, center = train_center, scale = train_scale)
test_scaled <- scale(test_x, center = train_center, scale = train_scale)

# ── Prepare response variables ───────────────────────────────────────────────────────────────────
train_y <- trainData$stat_cause_descr
test_y <- testData$stat_cause_descr

# ── Setup cross-validation for parameter tuning ─────────────────────────────────────────────────
# We'll try different k values and different distance weightings
cat("Setting up KNN parameter tuning...\n")

# Sample smaller dataset for tuning (KNN can be slow with large datasets)
set.seed(1)
tune_indices <- createDataPartition(train_y, p = 0.2, list = FALSE)  # Use 20% for tuning
tune_x <- train_scaled[tune_indices, ]
tune_y <- train_y[tune_indices]

# Create training control
ctrl <- trainControl(
  method = "cv",
  number = 5,  # 5-fold CV
  allowParallel = TRUE
)

# Define tuning grid
knn_grid <- expand.grid(
  kmax = c(3, 5, 7, 9, 11, 15, 21, 25),  # Number of neighbors (k)
  distance = 2,                         # Euclidean distance (fixed)
  kernel = c("rectangular", "triangular", "epanechnikov", "gaussian")
)

# ── Tune KNN model ─────────────────────────────────────────────────────────────────────────────────
cat("Tuning KNN model...\n")
start_time <- Sys.time()

# Train with caret
knn_tuned <- train(
  x = tune_x, 
  y = tune_y, 
  method = "kknn",  # Using kknn for weighted KNN
  tuneGrid = knn_grid,
  trControl = ctrl,
  preProcess = NULL,  
  metric = "Accuracy"
)

cat("KNN tuning completed in:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "minutes\n")
cat("Best parameters:\n")
print(knn_tuned$bestTune)

# ── Apply best model to full dataset ─────────────────────────────────────────────────────────────
cat("Training final KNN model with best parameters...\n")
start_time <- Sys.time()

# Extract best parameters
best_k <- knn_tuned$bestTune$kmax       # changed from 'k' to 'kmax'
best_kernel <- as.character(knn_tuned$bestTune$kernel)  # changed from 'weight' to 'kernel'

# Re-train using best parameters on full data
library(kknn)
final_knn <- kknn(
  formula = stat_cause_descr ~ ., 
  train = data.frame(train_scaled, stat_cause_descr = train_y),
  test = data.frame(test_scaled),
  k = best_k,
  kernel = best_kernel,        # changed from 'weight' to 'kernel'
  distance = 2  # Euclidean distance
)

cat("Final model training completed in:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "minutes\n")

# ── Evaluation ────────────────────────────────────────────────────────────────────────────────────
knn_pred <- final_knn$fitted.values
knn_prob <- final_knn$prob

# Convert predictions to factor with the same levels as the original data
knn_pred <- factor(knn_pred, levels = levels(testData$stat_cause_descr))

# Calculate accuracy
acc <- mean(knn_pred == test_y)
cat(sprintf("Test-set accuracy: %.4f\n", acc))

# Generate confusion matrix
conf_mat <- confusionMatrix(knn_pred, test_y)
print(conf_mat)

# ── Calculate balanced accuracy for handling class imbalance ──────────────────────────────────────
balanced_acc <- mean(conf_mat$byClass[, "Balanced Accuracy"], na.rm = TRUE)
cat(sprintf("Balanced accuracy: %.4f\n", balanced_acc))

# ── Save artifacts ────────────────────────────────────────────────────────────────────────────────
saveRDS(knn_tuned, "knn_tuned_model.rds")
saveRDS(list(center = train_center, scale = train_scale), "knn_scaling_params.rds")

# ── Clean-up ───────────────────────────────────────────────────────────────────────────────────────
stopCluster(cl)
registerDoSEQ()
gc()
cat("All done!\n")

comment("
Using 10 cores for parallel processing
Class distribution (train):

            Arson          Campfire          Children    Debris Burning 
             9699              3653              1869             12092 
    Equipment Use         Fireworks         Lightning     Miscellaneous 
             3398               732             15284             10341 
Missing/Undefined         Powerline          Railroad           Smoking 
             2480               535               589              1570 
        Structure 
              197 
Setting up KNN parameter tuning...
Tuning KNN model...
KNN tuning completed in: 5.36 minutes
Best parameters:
   kmax distance       kernel
31   25        2 epanechnikov
Training final KNN model with best parameters...

Attaching package: ‘kknn’

The following object is masked from ‘package:caret’:

    contr.dummy

Warning message:
In model.matrix.default(mt2, test, contrasts.arg = contrasts.arg) :
  variable 'stat_cause_descr' is absent, its contrast will be ignored
Final model training completed in: 1.24 minutes
Test-set accuracy: 0.5155
Confusion Matrix and Statistics

                   Reference
Prediction          Arson Campfire Children Debris Burning Equipment Use
  Arson              1879       85      123            833           198
  Campfire             44      281       19             58            47
  Children             63        7      126             97            35
  Debris Burning     1111      270      282           3249           388
  Equipment Use        99       27       26            100           152
  Fireworks            22       11       25             28             8
  Lightning           279      737       95            341           342
  Miscellaneous       567      152      137            417           263
  Missing/Undefined    65        8       12             76            49
  Powerline             1        1        1              6             6
  Railroad              7        5        1             10             1
  Smoking               8        6        3             10             6
  Structure             1        1        0              0             0
                   Reference
Prediction          Fireworks Lightning Miscellaneous Missing/Undefined
  Arson                    65       147           515                49
  Campfire                  5       122           127                16
  Children                 18        18            62                 0
  Debris Burning           30       216           700                83
  Equipment Use             9        52           147                15
  Fireworks                73        10            22                 0
  Lightning                81      5470           822               186
  Miscellaneous            35       339          1990               134
  Missing/Undefined         0        19           106               560
  Powerline                 0         6             4                 0
  Railroad                  1         1             4                 0
  Smoking                   1         4            15                 2
  Structure                 0         0             0                 0
                   Reference
Prediction          Powerline Railroad Smoking Structure
  Arson                    30       28      66        14
  Campfire                  2        8      35         1
  Children                  2        2      14        11
  Debris Burning           56       94     166        18
  Equipment Use            21        6      26         5
  Fireworks                 1        1       0         2
  Lightning                34       48     153        13
  Miscellaneous            43       17     201        11
  Missing/Undefined         3        9      12         0
  Powerline                 2        0       0         0
  Railroad                  0       11       2         1
  Smoking                   1        1       2         0
  Structure                 0        0       0         0

Overall Statistics
                                          
               Accuracy : 0.5155          
                 95% CI : (0.5095, 0.5215)
    No Information Rate : 0.2393          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4083          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: Arson Class: Campfire Class: Children
Sensitivity               0.45321         0.17662        0.148235
Specificity               0.90480         0.98077        0.987303
Pos Pred Value            0.46602         0.36732        0.276923
Neg Pred Value            0.90026         0.94961        0.972478
Prevalence                0.15493         0.05945        0.031763
Detection Rate            0.07021         0.01050        0.004708
Detection Prevalence      0.15067         0.02859        0.017002
Balanced Accuracy         0.67900         0.57869        0.567769
                     Class: Debris Burning Class: Equipment Use
Sensitivity                         0.6218              0.10167
Specificity                         0.8415              0.97890
Pos Pred Value                      0.4876              0.22190
Neg Pred Value                      0.9017              0.94850
Prevalence                          0.1952              0.05586
Detection Rate                      0.1214              0.00568
Detection Prevalence                0.2490              0.02560
Balanced Accuracy                   0.7316              0.54029
                     Class: Fireworks Class: Lightning Class: Miscellaneous
Sensitivity                  0.229560           0.8542              0.44085
Specificity                  0.995084           0.8462              0.89590
Pos Pred Value               0.359606           0.6360              0.46215
Neg Pred Value               0.990775           0.9486              0.88760
Prevalence                   0.011883           0.2393              0.16868
Detection Rate               0.002728           0.2044              0.07436
Detection Prevalence         0.007586           0.3214              0.16091
Balanced Accuracy            0.612322           0.8502              0.66837
                     Class: Missing/Undefined Class: Powerline Class: Railroad
Sensitivity                           0.53589        1.026e-02        0.048889
Specificity                           0.98604        9.991e-01        0.998756
Pos Pred Value                        0.60936        7.407e-02        0.250000
Neg Pred Value                        0.98123        9.928e-01        0.991990
Prevalence                            0.03905        7.287e-03        0.008408
Detection Rate                        0.02093        7.474e-05        0.000411
Detection Prevalence                  0.03434        1.009e-03        0.001644
Balanced Accuracy                     0.76096        5.047e-01        0.523823
                     Class: Smoking Class: Structure
Sensitivity               2.954e-03        0.000e+00
Specificity               9.978e-01        9.999e-01
Pos Pred Value            3.390e-02        0.000e+00
Neg Pred Value            9.747e-01        9.972e-01
Prevalence                2.530e-02        2.840e-03
Detection Rate            7.474e-05        0.000e+00
Detection Prevalence      2.205e-03        7.474e-05
Balanced Accuracy         5.004e-01        5.000e-01
Balanced accuracy: 0.6168
           used  (Mb) gc trigger  (Mb) limit (Mb) max used  (Mb)
Ncells  2341758 125.1    4168430 222.7         NA  4168430 222.7
Vcells 32682794 249.4  107695268 821.7      36864 90251580 688.6
All done!
")