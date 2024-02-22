install.packages(c("lmerTest", "haven", "tidyverse", "RColorBrewer", "lme4", "lmtest"))
library(lmerTest)
library(haven)
library(tidyverse)
library(RColorBrewer)
library(lme4)
library(lmtest)
install.packages("sjPlot")
library(sjPlot)
install.packages("glmmTMB")
library(glmmTMB)
install.packages("dotwhisker")
install.packages("flexplot")
install.packages("devtools") # Devtools is a package which allows to do this.
devtools::install_github("dustinfife/flexplot")
# Load necessary libraries
library(lme4)
library(ggplot2)
library(flexplot)

# Load the data


require(flexplot)
# Load the data
data <- read.csv("/Users/aryal/Desktop/ForeCache/foreCache-interaction/ForeCache_Models/zheng.csv")
# data standarize probabsame

testdata <- math
print(testdata)
# Define actions
actions_all <- c('same', 'modify1', 'modify2', 'modify3')

# Full model for 'same' action
fullmodel_same <- lme4::lmer(formula = probabsame ~ Trial + (1|u) , data = data, REML = FALSE)


# Reduced model for 'same' action
reducedmodel_same <- lme4::lmer(formula = probabsame  ~  1 +  (1|u), data = data, REML = FALSE)
icc(fullmodel_same)
visualize(fullmodel_same)
# ANOVA for 'same' action
cat("\nANOVA for same:\n")
print(anova(reducedmodel_same, fullmodel_same))


cat("\nFull Model Summary for same:\n")
print(summary(fullmodel_same))

visualize(reducedmodel_same, plot="model",sample=3)
visualize(reducedmodel_same, plot="residuals")

#Residual plot
plot(resid(fullmodel_same) ~ fitted(fullmodel_same), main = "Residual Plot", xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red")
# Create a QQ plot
qqnorm(resid(fullmodel_same))
qqline(resid(fullmodel_same))


# Reduced model for 'same' action
reducedmodel_same <- lmer(formula = probabsame ~  0 + (1 | u), data = data, REML = FALSE)
cat("\nReduced Model Summary for same:\n")
print(summary(reducedmodel_same))

# ANOVA for 'same' action
cat("\nANOVA for same:\n")
print(anova(fullmodel_same, reducedmodel_same))


# Plot the data points
plot(data$Trial, data$probabsame, xlab = "Trial", ylab = "probabsame", main = "Linear Regression Line")

# Add the linear regression line
abline(coef(fullmodel_same)[1], coef(fullmodel_same)[2], col = "red")

# Add a legend
legend("topleft", legend = "Linear Regression Line", col = "red", lty = 1)

# Full model for 'modify1' action
fullmodel_modify1 <- lmer(formula = probabmodify1 ~ Trial + (1 | u), data = data, REML = FALSE)
cat("\nFull Model Summary for modify1:\n")
print(summary(fullmodel_modify1))

# Reduced model for 'modify1' action
reducedmodel_modify1 <- lmer(formula = probabmodify1 ~ 1 + (1 | u), data = data, REML = FALSE)
cat("\nReduced Model Summary for modify1:\n")
print(summary(reducedmodel_modify1))

# ANOVA for 'modify1' action
cat("\nANOVA for modify1:\n")
print(anova(fullmodel_modify1, reducedmodel_modify1))

# Full model for 'modify2' action
fullmodel_modify2 <- lmer(formula = probabmodify2 ~ Trial + (1 | u), data = data, REML = FALSE)
cat("\nFull Model Summary for modify2:\n")
print(summary(fullmodel_modify2))

# Reduced model for 'modify2' action
reducedmodel_modify2 <- lmer(formula = probabmodify2 ~ 1 + (1 | u), data = data, REML = FALSE)
cat("\nReduced Model Summary for modify2:\n")
print(summary(reducedmodel_modify2))

# ANOVA for 'modify2' action
cat("\nANOVA for modify2:\n")
print(anova(fullmodel_modify2, reducedmodel_modify2))

# Full model for 'modify3' action
fullmodel_modify3 <- lmer(formula = probabmodify3 ~ Trial + (1 | u), data = data, REML = FALSE)
cat("\nFull Model Summary for modify3:\n")
print(summary(fullmodel_modify3))

# Reduced model for 'modify3' action
reducedmodel_modify3 <- lmer(formula = probabmodify3 ~ 1 + (1 | u), data = data, REML = FALSE)
cat("\nReduced Model Summary for modify3:\n")
print(summary(reducedmodel_modify3))

# ANOVA for 'modify3' action
cat("\nANOVA for modify3:\n")
print(anova(fullmodel_modify3, reducedmodel_modify3))




