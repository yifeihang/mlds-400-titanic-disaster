library(readr)
library(dplyr)
library(tibble)
library(stringr)
library(tidyr)

# --- Config / paths ---
TRAIN_CSV <- "src/data/train.csv"
TEST_CSV  <- "src/data/test.csv"
OUT_PRED  <- "src/data/results/test_predictions_r.csv"

DROP_COLS <- c("PassengerId", "Name", "Ticket", "Cabin")
DUMMY_COLS <- c("Sex", "Embarked", "Pclass")   # to be one-hot encoded
NUM_COLS <- c("Age", "SibSp", "Parch", "Fare") # numeric columns to scale

# Helper: make dummy matrix from dataframe for given formula columns,
# returning data.frame (no intercept) with safe column names.
make_dummies <- function(df, cols) {
  if (length(cols) == 0) return(data.frame())
  formula_str <- paste0("~ -1 + ", paste(cols, collapse = " + "))
  mm <- stats::model.matrix(as.formula(formula_str), data = df)
  dummies_df <- as.data.frame(mm)
  colnames(dummies_df) <- make.names(colnames(dummies_df))
  return(dummies_df)
}

# --- Load train
train_raw <- read_csv(TRAIN_CSV, show_col_types = FALSE)

# -------------- PREPROCESS TRAIN ----------------
# 1) Drop columns
train_proc <- train_raw %>% select(-any_of(DROP_COLS))

# 2) Ensure categorical NAs become an explicit "Missing" value so rows are preserved
train_proc <- train_proc %>%
  mutate(across(all_of(DUMMY_COLS), ~ replace_na(as.character(.), "Missing")))

# 3) One-hot encode specified categorical columns
train_dummies <- make_dummies(train_proc, DUMMY_COLS)

# 4) Numeric and other columns
train_numeric <- train_proc %>% select(all_of(NUM_COLS))
train_other   <- train_proc %>% select(-all_of(NUM_COLS), -all_of(DUMMY_COLS))

# 5) Combine (now all parts should have the same row count)
train_combined <- bind_cols(train_numeric, train_other, train_dummies)

# 6) Impute Age and Fare with TRAIN medians
age_median  <- median(train_combined$Age, na.rm = TRUE)
fare_median <- median(train_combined$Fare, na.rm = TRUE)
train_combined$Age  <- ifelse(is.na(train_combined$Age),  age_median, train_combined$Age)
train_combined$Fare <- ifelse(is.na(train_combined$Fare), fare_median, train_combined$Fare)

# 7) Standardize numeric columns using TRAIN stats (mean/sd)
num_means <- sapply(train_combined[NUM_COLS], mean)
num_sds   <- sapply(train_combined[NUM_COLS], sd)
num_sds[num_sds == 0] <- 1
train_scaled_nums <- as.data.frame(scale(train_combined[NUM_COLS], center = num_means, scale = num_sds))
train_final <- bind_cols(train_scaled_nums, train_combined %>% select(-all_of(NUM_COLS)))

# Ensure Survived exists
if (!"Survived" %in% colnames(train_final)) stop("Survived column not found in training data.")

# -------------- MODEL FITTING ----------------
y_train <- train_final$Survived
X_train <- train_final %>% select(-Survived)

# Build formula using available features (may be many)
glm_formula <- as.formula(paste("Survived ~", paste(colnames(X_train), collapse = " + ")))
model <- stats::glm(glm_formula, data = train_final, family = stats::binomial(link = "logit"))

# Training predictions and accuracy
train_probs <- predict(model, newdata = train_final, type = "response")
train_pred  <- ifelse(train_probs >= 0.5, 1, 0)
train_acc   <- mean(train_pred == y_train)

# -------------- PREPROCESS TEST ----------------
test_raw <- read_csv(TEST_CSV, show_col_types = FALSE)

# Preserve PassengerId for output if present
pid <- if ("PassengerId" %in% names(test_raw)) test_raw$PassengerId else seq_len(nrow(test_raw))

# Drop same cols, convert categorical NA -> "Missing"
test_proc <- test_raw %>% select(-any_of(DROP_COLS))
test_proc <- test_proc %>%
  mutate(across(all_of(DUMMY_COLS), ~ replace_na(as.character(.), "Missing")))

# Create test dummies and numeric/other parts
test_dummies <- make_dummies(test_proc, DUMMY_COLS)
test_numeric <- test_proc %>% select(all_of(NUM_COLS))
test_other   <- test_proc %>% select(-all_of(NUM_COLS), -all_of(DUMMY_COLS))

# Impute with TRAIN medians (no leakage)
test_numeric$Age  <- ifelse(is.na(test_numeric$Age),  age_median, test_numeric$Age)
test_numeric$Fare <- ifelse(is.na(test_numeric$Fare), fare_median, test_numeric$Fare)

# Combine
test_combined <- bind_cols(test_numeric, test_other, test_dummies)

# Align test to training features (X_train columns). Create missing columns with 0
train_feature_cols <- colnames(X_train)
for (col in train_feature_cols) {
  if (!col %in% colnames(test_combined)) {
    test_combined[[col]] <- 0
  }
}
# Drop extra columns in test not present in train
test_aligned <- test_combined %>% select(all_of(train_feature_cols))

# Standardize numeric cols on test using TRAIN mean/sd
test_aligned[NUM_COLS] <- sweep(test_aligned[NUM_COLS], 2, num_means, "-")
test_aligned[NUM_COLS] <- sweep(test_aligned[NUM_COLS], 2, num_sds, "/")

# -------------- PREDICT TEST ----------------
test_probs <- predict(model, newdata = test_aligned, type = "response")
test_pred  <- ifelse(test_probs >= 0.5, 1, 0)

# Save predictions to CSV with PassengerId if available
out_df <- tibble::tibble(PassengerId = pid, Predicted = test_pred)
dir.create(dirname(OUT_PRED), showWarnings = FALSE, recursive = TRUE)
readr::write_csv(out_df, OUT_PRED)

# -------------- Minimal PRINTS: CHANGES & RESULTS ----------------
cat("\nCHANGES\n")
cat("- Dropped columns:", paste(DROP_COLS, collapse = ", "), "\n")
cat("- Replaced NA in categorical columns with 'Missing' to preserve rows\n")
cat("- One-hot encoded columns:", paste(DUMMY_COLS, collapse = ", "), "\n")
cat("- Imputed Age and Fare using TRAIN medians (no test leakage)\n")
cat("- Standardized numeric columns using TRAIN mean/sd:", paste(NUM_COLS, collapse = ", "), "\n")
cat("- Aligned test features to TRAIN features before predicting\n")

cat("\nRESULTS\n")
cat(sprintf("- Training accuracy (on TRAIN set): %.4f\n", train_acc))
cat(sprintf("- Predictions saved to: %s\n", OUT_PRED))
cat(sprintf("- Total predictions: %d (Predicted=1 count: %d)\n", nrow(out_df), sum(out_df$Predicted == 1)))