install.packages('rstatix')
library(car)             # Load the car package for leveneTest
library(lmerTest)
library(haven)
library(tidyverse)
library(RColorBrewer)
library(lme4)
library(lmtest)
library(flexplot)
library(DHARMa)
library(car)
# Load required library
library(dplyr)
library(DHARMa)
library(glmmTMB)
library(rstatix)

# Load data
data_movies_p1 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_movies_p1.csv")
data_movies_p2 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_movies_p2.csv")
data_movies_p3 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_movies_p3.csv")
data_movies_p4 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_movies_p4.csv")


data_birdstrikes_p1 <- read.csv('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_birdstrikes_p1.csv')
data_birdstrikes_p2 <- read.csv('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_birdstrikes_p2.csv')
data_birdstrikes_p3 <- read.csv('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_birdstrikes_p3.csv')
data_birdstrikes_p4 <- read.csv('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_birdstrikes_p4.csv')

data_both_pall <- read.csv('/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/data/zheng/mixed-model-data/zheng_both_datasets.csv', stringsAsFactors = TRUE)
data_both_pall$u <- as.factor(data_both_pall$u)
data_both_pall$Task <- as.factor(data_both_pall$Task)
data_both_pall$Trial <- as.factor(data_both_pall$Trial)
data_both_pall$Openend <- as.factor(data_both_pall$Openend)

# #remove all the rows with 0 probability
# data_both_pall <- data_both_pall[!data_both_pall$probabsame == 0,]
# data_both_pall <- data_both_pall[!data_both_pall$probabmodify1 == 0,]
# data_both_pall <- data_both_pall[!data_both_pall$probabmodify2 == 0,]
# data_both_pall <- data_both_pall[!data_both_pall$probabmodify3 == 0,]
#
# data_both_pall$probabsame <- log(data_both_pall$probabsame)
# data_both_pall$probabmodify1 <- log(data_both_pall$probabmodify1)
# data_both_pall$probabmodify2 <- log(data_both_pall$probabmodify2)
# data_both_pall$probabmodify3 <- log(data_both_pall$probabmodify3)

res.aov <- anova_test(
  data =data_both_pall  , dv = probabmodify3, wid = u,
  within = c(Trial, Task)
  )
get_anova_table(res.aov)


action_probabs <- c('probabsame', 'probabmodify1', 'probabmodify2', 'probabmodify3')
combined_model_full <- lmer(formula = probabmodify3~ Trial + (1|Task) + (1|u), data = data_both_pall, REML = FALSE)
  # Reduced model: include random effect for u only
reducedmodel_same <- lmer(formula =  probabmodify3 ~ 1 +  (1|Task)  + (1|u), data = data_both_pall, REML = FALSE)
summary(reducedmodel_same)
lrtest(reducedmodel_same, combined_model_full)

combined_model_reporting <- lmer(formula = probabmodify3 ~ Trial + (1|Openendedness) + (1|Dataset)+ (1|u), data = data_both_pall)
summary(combined_model_reporting)
  # Reduced model: include random effect for u only

# anova_test(probabsame~ Trial + Error(u), data = data_both_pall, type = 3)
#
# combined_model_full_ML <- lmer(formula = probabsame~ Trial  + Openend   + (1|u), data = data_both_pall)
shapiro.test(resid(combined_model_full))
qqnorm(resid(combined_model_full))
qqline(resid(combined_model_full))
coef(combined_model_full)
#check with shapiro test
ks.test(resid(combined_model_full_ML), "pnorm")

help('isSingular')
simulationOutput <- simulateResiduals(fittedModel = fullmodel_same, plot = FALSE)
plot(simulationOutput)
#check if resiudals are norma

#check if levene test is significant
leveneTest(resid(combined_model_full) ~ combined_data$Trial)
summary(combined_model_full)

# Results
# The number of unique 'u' values is 36
# same: 0.01741
# modify1: 1.853e-05
# modify2: 0.9041
# modify3: 0.6341


# Define the columns for the new dataset
new_columns <- c('u', 'Trial', 'action', 'probab', 'Task')
new_data <- list()

# Iterate through each row of the dataset
for (i in 1:nrow(combined_data)) {
  # Iterate through each action type: same, modify1, modify2, modify3
  for (action in c('same', 'modify1', 'modify2', 'modify3')) {
    # Extract the action and corresponding probability
    probab_col <- paste0("probab", action)
    probab <- combined_data[i, probab_col]
    # If probability is not 0, add to new dataset
    if (probab != 0) {
      new_data[[length(new_data) + 1]] <- c(combined_data[i, 'u'], combined_data[i, 'Trial'], action, probab, combined_data[i, 'Task'])
    }
  }
}

# Convert the new data to a data frame
new_df <- as.data.frame(do.call(rbind, new_data), stringsAsFactors = TRUE)
colnames(new_df) <- new_columns

# Print the new data frame
print(new_df)
#check is there is null
new_df$action <- as.factor(new_df$action)

new_df$u <- as.factor(new_df$u)
new_df$Trial <- as.factor(new_df$Trial)
new_df$probab <- as.numeric(new_df$probab)

combined_model_full <-lmer(formula = probab ~  Trial  + Task + (1|u), data = new_df, REML = FALSE)
combined_model_reduced <- lmer(formula = probab ~  Task + (1|u), data = new_df, REML = FALSE)
lrtest(combined_model_reduced, combined_model_full)
summary(combined_model_full)

##############################################################################################################################
##############################################################################################################################
# MAJOR TEST: LMER model for p4 data, where the dependent variable is probability of each action. Doing for each action separately
# When NUMBER OF TRIALS is 2
##############################################################################################################################

##############################################################################################################################
# Load the data
# This block is for GLMER model for p4 data, where the dependent variable is COUNT, so we use the poisson family
data_p4 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng_logistic_p4.csv")
# Convert Trial to a factor with ordered levels


# Convert u to a factor
data_p4$u <- as.factor(data_p4$u)
# Full model: include Trial as a fixed effect and random effect for u
fullmodel_same <- glmer(formula = action ~   + (1|u), data = data_p4, family = binomial)
summary(fullmodel_same)
# reducedmodel_same <- lmer(formula = probabmodify3 ~ 1 + (1|u), data = data_p4)
# lrtest_result <- anova(reducedmodel_same, fullmodel_same)
# print(lrtest_result)

# LR Test Results:  Number of Trials is 2
# same: 0.9755
# modify1: 0.1534
# modify2: 0.3455
# modify3: 0.9106


# LR Test Results:  Number of Trials is 3
# same: 0.9755
# modify1: 0.1534
# modify2: 0.3455
# modify3: 0.9106

# Results: 2 Trials on Probability of each action


##############################################################################################################################

#manova : multivariate analysis of variance to use when there are multiple dependent variables and a single independent variable
#(or factor), or when the independent variable has more than two levels.
# dep_vars <- cbind( data_p4$probabmodify1, data_p4$probabmodify2, data_p4$probabsame,data_p4$probabmodify3)
# fit <- manova(dep_vars ~ Trial , data = data_p4)
# summary.aov(fit)


##############################################################################################################################

# T-test- paired sample t-test
# Assumptions: The dependent variable is continuous, the independent variable (or factor) is categorical, and the observations are independent.
# The dependent variable should be approximately normally distributed for each category of the independent variable.
# The dependent variable should have no significant outliers for each category of the independent variable.
# The dependent variable should have homogeneity of variances for each category of the independent variable.
# The observations are independent.
# The dependent variable should be approximately normally distributed for each category of the independent variable.

# Load the data
# This block is for GLMER model for p4 data, where the dependent variable is COUNT, so we use the poisson family
original_data <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng_p4.csv")
# Convert Trial to a factor with ordered levels

data_p4$Trial <- as.factor(data_p4$Trial)
# Convert u to a factor
data_p4$u <- as.factor(data_p4$u)
# Filter the dataset for Trial 0 and Trial 1 separately
trial_0_data <- subset(original_data, Trial == 0)
trial_1_data <- subset(original_data, Trial == 1)

# Select only the columns probabsame and Trial for the new datasets
trial_0_data <- trial_0_data[, c("u", "probabsame")]
trial_1_data <- trial_1_data[, c("u", "probabsame")]
print(trial_0_data)
# Rename the probabsame column to differentiate between Trial 0 and Trial 1
names(trial_0_data)[2] <- "probabsame_trial0"
names(trial_1_data)[2] <- "probabsame_trial1"

# Combine the two datasets based on the common 'u' column
new_dataset <- merge(trial_0_data, trial_1_data, by = "u")
print(new_dataset)
hist(new_dataset$probabsame_trial1 - new_dataset$probabsame_trial0)

# Or use a QQ plot
qqnorm(new_dataset$probabsame_trial1 - new_dataset$probabsame_trial0)
qqline(new_dataset$probabsame_trial1 - new_dataset$probabsame_trial0)

# Perform Shapiro-Wilk test for normality
shapiro.test(new_dataset$probabsame_trial1 - new_dataset$probabsame_trial0)
#p-value = 0.2037 > 0.05, so we fail to reject the null hypothesis that the data is normally distributed. Therefore we can use a paired t-test.

t.test(new_dataset$probabsame_trial1, new_dataset$probabsame_trial0, paired = TRUE, alternative = "two.sided")

# data:  new_dataset$probabsame_trial1 and new_dataset$probabsame_trial0
# t = 0.38712, df = 35, p-value = 0.701
# alternative hypothesis: true mean difference is not equal to 0
# 95 percent confidence interval:
#  -7.663039 11.274150
# sample estimates:
# mean difference
#        1.805556



##############################################################################################################################
#The paired t-test’s analog in the linear modeling framework is the linear mixed model with varying intercepts.
# TEST: GLMER model for p4 data, where the dependent variable is COUNT, so we use the poisson family
# Assumptions: The dependent variable is continuous, the independent variable (or factor) is categorical, and the observations are independent.
# The dependent variable should be approximately normally distributed for each category of the independent variable.
# The dependent variable should have no significant outliers for each category of the independent variable.
# The dependent variable should have homogeneity of variances for each category of the independent variable.
# The observations are independent.
# The dependent variable should be approximately normally distributed for each category of the independent variable.

##############################################################################################################################
#This is True we get the same t-value for paired-sample T-test and the t-value for Trial in the LMER model: 0.387
##############################################################################################################################

##############################################################################################################################

# Define actions
actions_all <- c('same', 'modify1', 'modify2', 'modify3')
require(flexplot)
# Load required libraries
library(stats)
# Load the data
# This block is for GLMER model for p4 data, where the dependent variable is COUNT, so we use the poisson family
data_p4 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng_p4.csv")
# Convert Trial to a factor with ordered levels

data_p4$Trial <- as.factor(data_p4$Trial)
# Convert u to a factor
data_p4$u <- as.factor(data_p4$u)
# Full model: include Trial as a fixed effect and random effect for u
fullmodel_same <- lme4::glmer(formula = probabsame ~ Trial + (Trial|u), data = data_p4, family = poisson)
reducedmodel_same <- lme4::lmer(formula = probabsame ~ 1 + (1|u), data = data_p4, REML = FALSE)
# Reduced model: include random effect for u only
reducedmodel_same <- lme4::glmer(formula = probabsame ~ 1 + (1|u), data = data_p4, family = poisson)
# Summary of the reduced model
summary(fullmodel_same)
# Visualize the full model
visualize(fullmodel_same)
# Likelihood ratio test between the full and reduced models
lrtest_result <- anova(reducedmodel_same, fullmodel_same)
print(lrtest_result)
# Perform simulation-based residuals analysis
simulationOutput <- simulateResiduals(fittedModel = fullmodel_same, plot = FALSE)
plot(simulationOutput)

# Results: 2 Trials on Count of each action
#same: 0.236
#modify1:0.0006006
#modify2: 0.3852
#modify3: 0.6419, -0.05401

##############################################################################################################################

# TEST: GLMER model for p2 data, where the dependent variable is COUNT, so we use the poisson family

##############################################################################################################################

# Load the data
# This block is for GLMER model for p4 data, where the dependent variable is COUNT, so we use the poisson family
data_p2 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng_p2.csv")
# Convert Trial to a factor with ordered levels

data_p2$Trial <- as.factor(data_p2$Trial)
# Convert u to a factor
data_p2$u <- as.factor(data_p2$u)
# Full model: include Trial as a fixed effect and random effect for u
fullmodel_same_p2 <- lme4::glmer(formula = probabmodify1 ~ 1 + Trial + (1|u), data = data_p2, family = poisson)
# Reduced model: include random effect for u only
reducedmodel_same_p2 <- lme4::glmer(formula = probabsame ~ 1 + (1|u), data = data_p2, family = poisson)
# Summary of the reduced model
summary(fullmodel_same_p2)
# Visualize the full model
visualize(fullmodel_same_p2)
# Likelihood ratio test between the full and reduced models
lrtest_result <- lrtest(reducedmodel_same_p2, fullmodel_same_p2)
print(lrtest_result)
# Perform simulation-based residuals analysis
simulationOutput <- simulateResiduals(fittedModel = fullmodel_same_p2, plot = FALSE)
plot(simulationOutput)


#same:0.0262
#modify1: 6.617e-05
#modify2: 0.8153
#modify3: 0.004462



##############################################################################################################################



# data_p2 <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng_p2.csv")
# fullmodel_same <- lme4::lmer(formula = probabsame ~ 1 + (Trial|u) , data = data_p2, family = poisson)
# # Reduced model for 'same' action
# reducedmodel_same <- lme4::lmer(formula = probabsame ~  1 +  (1|u), data = data_p2, family=poisson)
# print(lrtest(reducedmodel_same, fullmodel_same))
# visualize(fullmodel_same)
# coef(fullmodel_same)
# # qqnorm(resid(fullmodel_same))
# # qqline(resid(fullmodel_same))
#
# #same: 0.02033 ,
# #modify1: 0.
# #modify2: 0.0000
# #modify3: 0.0000
#
# cat("\nFull Model Summary for same:\n")
# print(summary(fullmodel_same))
#
# visualize(reducedmodel_same, plot="model",sample=3)
# visualize(reducedmodel_same, plot="residuals")
#
# #Residual plot
# plot(resid(fullmodel_same) ~ fitted(fullmodel_same), main = "Residual Plot", xlab = "Fitted Values", ylab = "Residuals")
# abline(h = 0, col = "red")
# # Create a QQ plot
# qqnorm(resid(fullmodel_same))
# qqline(resid(fullmodel_same))
#
#
# # Reduced model for 'same' action
# reducedmodel_same <- lmer(formula = probabsame ~  0 + (1 | u), data = data, REML = FALSE)
# cat("\nReduced Model Summary for same:\n")
# print(summary(reducedmodel_same))
#
# # ANOVA for 'same' action
# cat("\nANOVA for same:\n")
# print(anova(fullmodel_same, reducedmodel_same))
#
#
# # Plot the data points
# plot(data$Trial, data$probabsame, xlab = "Trial", ylab = "probabsame", main = "Linear Regression Line")
#
# # Add the linear regression line
# abline(coef(fullmodel_same)[1], coef(fullmodel_same)[2], col = "red")
#
# # Add a legend
# legend("topleft", legend = "Linear Regression Line", col = "red", lty = 1)
#
# # Full model for 'modify1' action
# fullmodel_modify1 <- lmer(formula = probabmodify1 ~ Trial + (1 | u), data = data, REML = FALSE)
# cat("\nFull Model Summary for modify1:\n")
# print(summary(fullmodel_modify1))
#
# # Reduced model for 'modify1' action
# reducedmodel_modify1 <- lmer(formula = probabmodify1 ~ 1 + (1 | u), data = data, REML = FALSE)
# cat("\nReduced Model Summary for modify1:\n")
# print(summary(reducedmodel_modify1))
#
# # ANOVA for 'modify1' action
# cat("\nANOVA for modify1:\n")
# print(anova(fullmodel_modify1, reducedmodel_modify1))
#
# # Full model for 'modify2' action
# fullmodel_modify2 <- lmer(formula = probabmodify2 ~ Trial + (1 | u), data = data, REML = FALSE)
# cat("\nFull Model Summary for modify2:\n")
# print(summary(fullmodel_modify2))
#
# # Reduced model for 'modify2' action
# reducedmodel_modify2 <- lmer(formula = probabmodify2 ~ 1 + (1 | u), data = data, REML = FALSE)
# cat("\nReduced Model Summary for modify2:\n")
# print(summary(reducedmodel_modify2))
#
# # ANOVA for 'modify2' action
# cat("\nANOVA for modify2:\n")
# print(anova(fullmodel_modify2, reducedmodel_modify2))
#
# # Full model for 'modify3' action
# fullmodel_modify3 <- lmer(formula = probabmodify3 ~ Trial + (1 | u), data = data, REML = FALSE)
# cat("\nFull Model Summary for modify3:\n")
# print(summary(fullmodel_modify3))
#
# # Reduced model for 'modify3' action
# reducedmodel_modify3 <- lmer(formula = probabmodify3 ~ 1 + (1 | u), data = data, REML = FALSE)
# cat("\nReduced Model Summary for modify3:\n")
# print(summary(reducedmodel_modify3))
#
# # ANOVA for 'modify3' action
# cat("\nANOVA for modify3:\n")
# print(anova(fullmodel_modify3, reducedmodel_modify3))




