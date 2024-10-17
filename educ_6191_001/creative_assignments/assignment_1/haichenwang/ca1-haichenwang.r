library(tidyverse)

df <- read_csv("/Users/john/Library/CloudStorage/Box-Box/Website/educ_6191_001/creative_assignments/assignment_1/data/ca1-dataset.csv")
df$`Unique-id` <- NULL
df$namea <- NULL

library(janitor)
df <- clean_names(df)

library(ROSE)
databalanced <- ovun.sample(
  off_task ~ ., 
  data = df, 
  method = "both", 
  p = 0.5,
  N = 1000, 
  seed = 1
)$data

table(databalanced$off_task)
table(df$off_task)
set.seed(502)

ind <- sample(2, nrow(databalanced), replace = TRUE, prob = c(0.7, 0.3))
databalanced$off_task <- ifelse(databalanced$off_task == "Y", 1, 0)
databalanced$off_task <- as.factor(databalanced$off_task)

Train <- databalanced[ind == 1, ]
Test <- databalanced[ind == 2, ]

add_cv_cohorts <- function(dat, cv_K) {
  if (nrow(dat) %% cv_K == 0) {
    # If perfectly divisible
    set.seed(123)
    dat$cv_cohort <- sample(rep(1:cv_K, each = (nrow(dat) %/% cv_K)))
  } else {
    # If not perfectly divisible
    dat$cv_cohort <- sample(
      c(
        rep(1:(nrow(dat) %% cv_K), each = (nrow(dat) %/% cv_K + 1)),
        rep((nrow(dat) %% cv_K + 1):cv_K, each = (nrow(dat) %/% cv_K))
      )
    )
  }
  return(dat)
}

# Add 10-fold CV labels to real estate data
train_cv <- add_cv_cohorts(Train, 10)
train_cv$off_task <- as.factor(train_cv$off_task)

rf.grid <- expand.grid(
  nt = seq(100, 300, by = 100),
  mrty = c(1, 3, 5, 7, 10)
)
rf.grid$acc <- rf.grid$f1 <- NA
rf.f1 <- rf.acc <- c()

library(randomForest)
library(MetricsWeighted)

for (k in 1:nrow(rf.grid)) {
  for (i in 1:10) {
    # Segment data by fold using the which() function
    indexes <- which(train_cv$cv_cohort == i)
    train <- train_cv[-indexes, ]
    val <- train_cv[indexes, ]

    # Model
    set.seed(123)
    rf.model <- randomForest(
      off_task ~ ., 
      data = train,
      ntrees = rf.grid$nt[k], 
      mtry = rf.grid$mtry[k]
    )
    rf.pred <- predict(rf.model, val)

    # Evaluate
    rf.f1[i] <- f1_score(
      as.numeric(val$off_task) - 1,
      as.numeric(rf.pred) - 1
    )
    rf.acc[i] <- sum(rf.pred == val$off_task) / nrow(val)
  }
  rf.grid$f1[k] <- mean(rf.f1)
  rf.grid$acc[k] <- mean(rf.acc)
  print(paste("finished with:", k))
}

rf.grid[which.max(rf.grid$f1), ]

# Best model
bestrfmodel <- randomForest(
  off_task ~ ., 
  data = Train,
  n.trees = 100,
  mtry = 5
)
pre <- predict(bestrfmodel, Test)

library(caret)
confusionMatrix(Test$off_task, pre)

plot_table <- function(x, xlab = 'Predicted label', ylab = 'True label', normalize = FALSE) {
  library(ggplot2)
  
  if (!is.table(x)) {
    warning('Input should be a table, not a ', class(x))
    x <- as.table(x)
  }
  if (!is.numeric(x)) {
    stop('Input should be numeric, not ', mode(x), call. = FALSE)
  }

  if (normalize) {
    x <- round(prop.table(x, 1), 2)
    mar <- as.data.frame(x)
  } else {
    mar <- as.data.frame(x)
  }

  ggplot(mar, aes(mar[, 2], mar[, 1])) +
    geom_tile(aes(fill = Freq), color = 'black') +
    scale_fill_gradientn(colours = c('gray98', 'steelblue1', 'midnightblue')) +
    geom_label(aes(label = Freq), size = 10) +
    labs(fill = '', x = xlab, y = ylab, title = "Confusion Matrix") +
    ylim(rev(levels(mar[, 2]))) +
    scale_y_discrete(expand = c(0, 0)) +
    scale_x_discrete(expand = c(0, 0)) +
    theme(
      plot.title = element_text(hjust = 0.5, family = "serif", size = 20),
      legend.position = "none",
      axis.text = element_text(family = "serif", size = 20),
      axis.title = element_text(family = "serif", size = 20)
    )
}

con_matrix <- table(Test$off_task, pre)
colnames(con_matrix) <- c("N", "Y")
rownames(con_matrix) <- c("N", "Y")
names(dimnames(con_matrix)) <- c("actual", "predicted")
con_matrix <- as.table(con_matrix)

plot_table(con_matrix, "predicted", "actual")

performance <- function(table, n = 2) {
  if (!all(dim(table) == c(2, 2)))
    stop("MUST be a 2*2 table")

  tn <- table[2, 2]
  fp <- table[2, 1]
  fn <- table[1, 2]
  tp <- table[1, 1]

  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  positive <- tp / (tp + fp)
  negative <- tn / (tn + fn)
  hitrate <- (tp + tn) / (tp + tn + fp + fn)

  result <- paste(
    "Sensitivity = ", round(sensitivity, n),
    "\nSpecificity = ", round(specificity, n),
    "\nPositive Predictive Value = ", round(positive, n),
    "\nNegative Predictive Value = ", round(negative, n),
    "\nAccuracy = ", round(hitrate, n),
    "\n", sep = ""
  )
  cat(result)
}
performance(con_matrix)

library(pROC)
testlabel <- as.numeric(Test$off_task) - 1
pred <- as.numeric(pre) - 1
rocobj <- roc(testlabel, pred)

# Get the value of AUC
auc <- round(auc(testlabel, pred), 4)
auc

ggroc(
  rocobj,
  color = "red",
  linetype = 1,
  size = 1,
  alpha = 1,
  legacy.axes = TRUE
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    color = "grey",
    size = 1,
    linetype = 1
  ) +
  labs(
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity or Recall)"
  ) +
  annotate(
    "text",
    x = 0.70,
    y = 0.30,
    label = paste("AUC =", auc),
    size = 5,
    family = "serif"
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_bw() +
  theme(
    panel.background = element_rect(fill = "transparent"),
    axis.ticks.length = unit(0.4, "lines"),
    axis.ticks = element_line(color = "black"),
    axis.line = element_line(size = 0.5, colour = "black"),
    axis.title = element_text(colour = "black", size = 10, face = "bold"),
    axis.text = element_text(colour = "black", size = 10, face = "bold"),
    text = element_text(size = 8, color = "black", family = "serif")
  )

library(vip)
vip(bestrfmodel) +
  theme(
    axis.text = element_text(family = "serif", size = 17),
    axis.title = element_text(family = "serif", size = 17)
  )