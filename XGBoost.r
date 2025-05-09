####################################################################################################
#  Multi‑class XGBoost with class weights – fixed for NA labels & factor warnings
####################################################################################################

suppressPackageStartupMessages({
  library(dplyr);   library(caret);  library(Matrix);  library(xgboost)
  library(doParallel);  library(MLmetrics)   # optional for extra metrics
})

set.seed(1)

# ── Parallel set‑up ────────────────────────────────────────────────────────────────────────────────
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(sprintf("Using %d cores for parallel processing\n", num_cores))

Sys.setenv(OMP_NUM_THREADS = num_cores)

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

# ── Remove predictor we’re not using ──────────────────────────────────────────────────────────────
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

# ── One‑hot encode predictors (target excluded) ───────────────────────────────────────────────────
dv <- dummyVars(~ ., data = trainData %>% select(-stat_cause_descr), fullRank = TRUE)

train_x <- predict(dv, trainData %>% select(-stat_cause_descr)) %>% Matrix(sparse = TRUE)
test_x  <- predict(dv, testData  %>% select(-stat_cause_descr)) %>% Matrix(sparse = TRUE)

train_y <- as.integer(trainData$stat_cause_descr) - 1    # 0‑based integer labels
test_y  <- as.integer(testData$stat_cause_descr)  - 1

# ── Row‑level class weights (inverse frequency) ───────────────────────────────────────────────────
cls_counts  <- table(train_y)
cls_weights <- nrow(train_x) / (length(cls_counts) * cls_counts)
row_weights <- cls_weights[train_y + 1]

# ── DMatrix objects ───────────────────────────────────────────────────────────────────────────────
dtrain <- xgb.DMatrix(train_x, label = train_y, weight = row_weights)
dtest  <- xgb.DMatrix(test_x,  label = test_y)

# ── XGBoost parameters ────────────────────────────────────────────────────────────────────────────
params <- list(
  objective        = "multi:softprob",
  num_class        = length(cls_counts),
  eval_metric      = "mlogloss",
  eta              = 0.1,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  nthread          = num_cores
)

# ── Train with early stopping ─────────────────────────────────────────────────────────────────────
cat("Training XGBoost …\n")
watchlist <- list(train = dtrain, eval = dtest)
start_time <- Sys.time()

xgb_model <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 2000,
  watchlist             = watchlist,
  early_stopping_rounds = 50,
  print_every_n         = 25
)

cat("Best iteration:", xgb_model$best_iteration, "\n")
cat("Training time:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "minutes\n")

# ── Evaluation ────────────────────────────────────────────────────────────────────────────────────
pred_prob <- predict(xgb_model, test_x) |>
            matrix(ncol = params$num_class, byrow = TRUE)

pred_int  <- max.col(pred_prob, ties.method = "first") - 1
pred_fact <- factor(pred_int,
                    levels = 0:(params$num_class - 1),
                    labels = levels(trainData$stat_cause_descr))

acc <- mean(pred_fact == testData$stat_cause_descr)
cat(sprintf("Test‑set accuracy: %.4f\n", acc))

conf_mat <- caret::confusionMatrix(pred_fact, testData$stat_cause_descr)
print(conf_mat)

# ── Save artefacts ────────────────────────────────────────────────────────────────────────────────
xgb.save(xgb_model, "xgb_multiclass.model")
saveRDS(dv,         "xgb_dummyVars.rds")

# ── Clean‑up ───────────────────────────────────────────────────────────────────────────────────────
stopCluster(cl);  registerDoSEQ();  gc()
cat("All done!\n")

comment("
Best iteration: 1236 
Training time: 7.58 minutes
Test‑set accuracy: 0.5717
Confusion Matrix and Statistics

                   Reference
Prediction          Arson Campfire Children Debris Burning Equipment Use
  Arson              2170       71       69            762           127
  Campfire            107      864       28            169            74
  Children            160       16      290            234            58
  Debris Burning      778      158      205           3011           240
  Equipment Use       196       41       69            255           513
  Fireworks            39        8       32             55            20
  Lightning           120      254       22            114           121
  Miscellaneous       431      110       85            370           193
  Missing/Undefined    34       10        8             68            58
  Powerline            27        7        5             53            38
  Railroad             17        7        8             32            12
  Smoking              62       45       25             96            33
  Structure             5        0        4              6             8
                   Reference
Prediction          Fireworks Lightning Miscellaneous Missing/Undefined
  Arson                    21        88           379                10
  Campfire                  6       320           341                 9
  Children                 28        35           150                 5
  Debris Burning           24       128           515                17
  Equipment Use            27       141           413                36
  Fireworks               141        22            51                 2
  Lightning                17      5309           336                34
  Miscellaneous            39       251          2000                66
  Missing/Undefined         2        52           126               856
  Powerline                 4        17            44                 4
  Railroad                  1         6            26                 0
  Smoking                   8        33           130                 4
  Structure                 0         2             3                 2
                   Reference
Prediction          Powerline Railroad Smoking Structure
  Arson                     9       23      44        10
  Campfire                 11        9      90         0
  Children                  6        7      33        15
  Debris Burning           42       56     135        19
  Equipment Use            41       23      83         8
  Fireworks                 6        3       6         2
  Lightning                11       33      53         3
  Miscellaneous            34       14     147         4
  Missing/Undefined         4        1      16         0
  Powerline                26        3       4         1
  Railroad                  0       46       3         1
  Smoking                   5        6      63         2
  Structure                 0        1       0        11

Overall Statistics
                                          
               Accuracy : 0.5717          
                 95% CI : (0.5658, 0.5777)
    No Information Rate : 0.2393          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4938          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: Arson Class: Campfire Class: Children
Sensitivity               0.52340         0.54305         0.34118
Specificity               0.92868         0.95375         0.97117
Pos Pred Value            0.57362         0.42604         0.27965
Neg Pred Value            0.91400         0.97061         0.97823
Prevalence                0.15493         0.05945         0.03176
Detection Rate            0.08109         0.03229         0.01084
Detection Prevalence      0.14136         0.07578         0.03875
Balanced Accuracy         0.72604         0.74840         0.65617
                     Class: Debris Burning Class: Equipment Use
Sensitivity                         0.5763              0.34314
Specificity                         0.8924              0.94724
Pos Pred Value                      0.5651              0.27790
Neg Pred Value                      0.8967              0.96059
Prevalence                          0.1952              0.05586
Detection Rate                      0.1125              0.01917
Detection Prevalence                0.1991              0.06898
Balanced Accuracy                   0.7343              0.64519
                     Class: Fireworks Class: Lightning Class: Miscellaneous
Sensitivity                  0.443396           0.8290              0.44307
Specificity                  0.990697           0.9451              0.92161
Pos Pred Value               0.364341           0.8260              0.53419
Neg Pred Value               0.993289           0.9461              0.89078
Prevalence                   0.011883           0.2393              0.16868
Detection Rate               0.005269           0.1984              0.07474
Detection Prevalence         0.014461           0.2402              0.13991
Balanced Accuracy            0.717047           0.8870              0.68234
                     Class: Missing/Undefined Class: Powerline Class: Railroad
Sensitivity                           0.81914        0.1333333        0.204444
Specificity                           0.98526        0.9922081        0.995742
Pos Pred Value                        0.69312        0.1115880        0.289308
Neg Pred Value                        0.99260        0.9936294        0.993271
Prevalence                            0.03905        0.0072867        0.008408
Detection Rate                        0.03199        0.0009716        0.001719
Detection Prevalence                  0.04615        0.0087067        0.005941
Balanced Accuracy                     0.90220        0.5627707        0.600093
                     Class: Smoking Class: Structure
Sensitivity                0.093058         0.144737
Specificity                0.982786         0.998838
Pos Pred Value             0.123047         0.261905
Neg Pred Value             0.976609         0.997567
Prevalence                 0.025298         0.002840
Detection Rate             0.002354         0.000411
Detection Prevalence       0.019132         0.001569
Balanced Accuracy          0.537922         0.571788
[1] TRUE
           used  (Mb) gc trigger  (Mb) limit (Mb) max used  (Mb)
Ncells  2206645 117.9    4168365 222.7         NA  4168365 222.7
Vcells 12798298  97.7   28663127 218.7      36864 28590019 218.2
All done!
")