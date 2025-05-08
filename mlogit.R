# Load required libraries
library(nnet)      # For multinomial logistic regression
library(caret)     # For model training and evaluation
library(dplyr)     # For data manipulation
library(ggplot2)   # For visualization
library(pROC)      # For ROC curves
library(tidyr)     # For data reshaping
library(MLmetrics) # For performance metrics
library(doParallel) # For parallel processing

# Set seed for reproducibility
set.seed(1)

# Setup parallel processing - use more cores for full dataset
num_cores <- parallel::detectCores() - 1  # Leave one core free
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat(paste("Using", num_cores, "cores for parallel processing\n"))

# Load data
trainData <- read.csv("data/train_fires10.csv")
testData <- read.csv("data/test_fires10.csv")

# Convert categorical variables to factors
trainData$stat_cause_descr <- as.factor(trainData$stat_cause_descr)
trainData$fire_size_class <- as.factor(trainData$fire_size_class)
trainData$owner_code <- as.factor(trainData$owner_code)
trainData$state <- as.factor(trainData$state)
trainData$discovery_hour <- as.factor(trainData$discovery_hour)
trainData$discovery_month <- factor(trainData$discovery_month, levels = month.abb)

testData$stat_cause_descr <- as.factor(testData$stat_cause_descr)
testData$fire_size_class <- as.factor(testData$fire_size_class)
testData$owner_code <- as.factor(testData$owner_code)
testData$state <- as.factor(testData$state)
testData$discovery_hour <- as.factor(testData$discovery_hour)
testData$discovery_month <- factor(testData$discovery_month, levels = month.abb)

# Prepare data by excluding fire_size_class
trainDataVarSelect <- trainData %>% select(-fire_size_class)
testDataVarSelect <- testData %>% select(-fire_size_class)

# USE FULL DATASETS - Remove sampling
# trainDataSample <- trainDataVarSelect %>% sample_n(5000)
# testDataSample <- testDataVarSelect %>% sample_n(1000)

# Using full datasets instead
trainDataFull <- trainDataVarSelect
testDataFull <- testDataVarSelect

# Fix factor levels to ensure valid R variable names
trainDataFull$stat_cause_descr <- factor(trainDataFull$stat_cause_descr)
levels(trainDataFull$stat_cause_descr) <- make.names(levels(trainDataFull$stat_cause_descr))
testDataFull$stat_cause_descr <- factor(testDataFull$stat_cause_descr)
levels(testDataFull$stat_cause_descr) <- make.names(levels(testDataFull$stat_cause_descr))

# Check class distribution of full dataset
cause_distribution <- table(trainDataFull$stat_cause_descr)
print("Class distribution (full dataset):")
print(cause_distribution)

# Set up cross-validation with parallel processing enabled
# Reduce number of folds for computational efficiency with large dataset
ctrl <- trainControl(
  method = "cv",
  number = 5,       # Reduced from 5 to 3 for full dataset
  classProbs = FALSE,
  summaryFunction = defaultSummary,
  allowParallel = TRUE
)

# Check for rare classes in full dataset
rare_classes <- names(which(table(trainDataFull$stat_cause_descr) < 100))
if (length(rare_classes) > 0) {
  cat("Warning: Relatively rare classes detected:", paste(rare_classes, collapse=", "), "\n")
}

# Memory management for large dataset
gc()  # Force garbage collection

# Train with multithreading on full dataset
cat("Training multinomial logistic regression model on FULL dataset...\n")
start_time <- Sys.time()

# Use tryCatch to handle potential errors
tryCatch({
  # Full model with all variables - will potentially be very time-consuming!
  mlr_model <- train(
    stat_cause_descr ~ ., 
    data = trainDataFull,
    method = "multinom",
    trControl = ctrl,
    trace = FALSE,
    maxit = 300,      # Increase for convergence on large dataset
    MaxNWts = 10000   # Increase for complex model with full data
  )
  
  print(mlr_model)
  
}, error = function(e) {
  cat("Error in full model training:", e$message, "\n")
  cat("Trying with reduced feature set...\n")
  
  # Fallback with important predictors only
  mlr_model <- train(
    stat_cause_descr ~ fire_year + latitude + longitude + state + discovery_month + discovery_hour, 
    data = trainDataFull,
    method = "multinom",
    trControl = ctrl,
    trace = FALSE,
    maxit = 200
  )
  
  print(mlr_model)
})

# Print execution time
end_time <- Sys.time()
cat("Full model training time:", difftime(end_time, start_time, units = "mins"), "minutes\n")

# Continue with predictions on full test dataset
if(exists("mlr_model")) {
  # Predict on full test set
  cat("Making predictions on full test dataset...\n")
  mlr_preds <- predict(mlr_model, newdata = testDataFull)
  
  # Calculate accuracy
  accuracy <- sum(mlr_preds == testDataFull$stat_cause_descr) / length(mlr_preds)
  print(paste("Full Test Set Accuracy:", round(accuracy, 4)))
  
  # Generate confusion matrix
  tryCatch({
    conf_matrix <- confusionMatrix(mlr_preds, testDataFull$stat_cause_descr)
    print(conf_matrix)
    
    # Save confusion matrix to a file for later reference
    capture.output(conf_matrix, file = "confusion_matrix_full.txt")
    
  }, error = function(e) {
    cat("Error generating confusion matrix:", e$message, "\n")
    table_results <- table(Actual = testDataFull$stat_cause_descr, Predicted = mlr_preds)
    print(table_results)
    write.csv(table_results, "confusion_table_full.csv")
  })
  
  # Save the model for future use
  saveRDS(mlr_model, "mlr_full_model.rds")
}

# Clean up parallel processing cluster
stopCluster(cl)
registerDoSEQ()  # Switch back to sequential processing

# Final garbage collection
gc()

cat("Analysis complete!\n")

comment("
[1] "Class distribution (full dataset):"

            Arson          Campfire          Children    Debris.Burning 
             9699              3653              1869             12092 
    Equipment.Use         Fireworks         Lightning     Miscellaneous 
             3398               732             15284             10341 
Missing.Undefined         Powerline          Railroad           Smoking 
             2480               535               589              1570 
        Structure 
              197 
          used  (Mb) gc trigger  (Mb) limit (Mb) max used  (Mb)
Ncells 2138997 114.3    4272944 228.2         NA  2820907 150.7
Vcells 4535242  34.7   10146329  77.5      36864  8388425  64.0
Training multinomial logistic regression model on FULL dataset...
Penalized Multinomial Regression 

62439 samples
   11 predictor
   13 classes: 'Arson', 'Campfire', 'Children', 'Debris.Burning', 'Equipment.Use', 'Fireworks', 'Lightning', 'Miscellaneous', 'Missing.Undefined', 'Powerline', 'Railroad', 'Smoking', 'Structure' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 49952, 49953, 49951, 49950, 49950 
Resampling results across tuning parameters:

  decay  Accuracy   Kappa    
  0e+00  0.5357547  0.4334937
  1e-04  0.5362032  0.4340233
  1e-01  0.5368437  0.4346678

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was decay = 0.1.
Full model training time: 9.624981 minutes
Making predictions on full test dataset...
[1] "Full Test Set Accuracy: 0.5368"
Confusion Matrix and Statistics

                   Reference
Prediction          Arson Campfire Children Debris.Burning Equipment.Use
  Arson              1939       84       91            796           137
  Campfire             86      378        9             40            40
  Children             31        3      124             83            29
  Debris.Burning     1172      265      357           3456           485
  Equipment.Use        33       11        6             39           103
  Fireworks            27       10       26             29            14
  Lightning           265      657       91            246           303
  Miscellaneous       499      158      121            379           271
  Missing.Undefined    94       22       24            152           111
  Powerline             0        1        1              0             0
  Railroad              0        1        0              5             0
  Smoking               0        1        0              0             2
  Structure             0        0        0              0             0
                   Reference
Prediction          Fireworks Lightning Miscellaneous Missing.Undefined
  Arson                    48       107           530                20
  Campfire                  3        95           161                 7
  Children                 26        12            51                 0
  Debris.Burning           41       209           726                73
  Equipment.Use             6        18            66                15
  Fireworks               105        20            29                 4
  Lightning                62      5618           748               130
  Miscellaneous            22       254          1954               115
  Missing.Undefined         4        71           242               681
  Powerline                 0         0             0                 0
  Railroad                  1         0             3                 0
  Smoking                   0         0             3                 0
  Structure                 0         0             1                 0
                   Reference
Prediction          Powerline Railroad Smoking Structure
  Arson                    20       21      41         9
  Campfire                  3        3      54         0
  Children                  1        0       6        16
  Debris.Burning           76      109     216        30
  Equipment.Use            14        5      11         2
  Fireworks                 2        4       2         0
  Lightning                35       46     133        13
  Miscellaneous            38       18     185         4
  Missing.Undefined         5       14      28         1
  Powerline                 1        0       0         0
  Railroad                  0        5       1         0
  Smoking                   0        0       0         0
  Structure                 0        0       0         1

Overall Statistics
                                          
               Accuracy : 0.5368          
                 95% CI : (0.5308, 0.5428)
    No Information Rate : 0.2393          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.435           
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: Arson Class: Campfire Class: Children
Sensitivity               0.46768         0.23759        0.145882
Specificity               0.91581         0.98010        0.990043
Pos Pred Value            0.50455         0.43003        0.324607
Neg Pred Value            0.90370         0.95313        0.972478
Prevalence                0.15493         0.05945        0.031763
Detection Rate            0.07246         0.01413        0.004634
Detection Prevalence      0.14360         0.03285        0.014275
Balanced Accuracy         0.69174         0.60884        0.567963
                     Class: Debris.Burning Class: Equipment.Use
Sensitivity                         0.6614             0.068896
Specificity                         0.8255             0.991055
Pos Pred Value                      0.4790             0.313070
Neg Pred Value                      0.9095             0.947337
Prevalence                          0.1952             0.055865
Detection Rate                      0.1291             0.003849
Detection Prevalence                0.2696             0.012294
Balanced Accuracy                   0.7434             0.529976
                     Class: Fireworks Class: Lightning Class: Miscellaneous
Sensitivity                  0.330189           0.8773              0.43288
Specificity                  0.993685           0.8659              0.90722
Pos Pred Value               0.386029           0.6731              0.48631
Neg Pred Value               0.991959           0.9573              0.88744
Prevalence                   0.011883           0.2393              0.16868
Detection Rate               0.003924           0.2099              0.07302
Detection Prevalence         0.010164           0.3119              0.15014
Balanced Accuracy            0.661937           0.8716              0.67005
                     Class: Missing.Undefined Class: Powerline Class: Railroad
Sensitivity                           0.65167        5.128e-03       0.0222222
Specificity                           0.97014        9.999e-01       0.9995855
Pos Pred Value                        0.46998        3.333e-01       0.3125000
Neg Pred Value                        0.98562        9.927e-01       0.9917742
Prevalence                            0.03905        7.287e-03       0.0084078
Detection Rate                        0.02545        3.737e-05       0.0001868
Detection Prevalence                  0.05415        1.121e-04       0.0005979
Balanced Accuracy                     0.81090        5.025e-01       0.5109038
                     Class: Smoking Class: Structure
Sensitivity               0.0000000        1.316e-02
Specificity               0.9997700        1.000e+00
Pos Pred Value            0.0000000        5.000e-01
Neg Pred Value            0.9746963        9.972e-01
Prevalence                0.0252980        2.840e-03
Detection Rate            0.0000000        3.737e-05
Detection Prevalence      0.0002242        7.474e-05
Balanced Accuracy         0.4998850        5.066e-01
           used  (Mb) gc trigger  (Mb) limit (Mb) max used  (Mb)
Ncells  2344842 125.3    4272944 228.2         NA  3453570 184.5
Vcells 34015323 259.6   78054185 595.6      36864 78048519 595.5
Analysis complete!
")