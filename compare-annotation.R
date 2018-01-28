annotation_src = '/compare-predictions.csv'
annotations = read.csv(annotation_src)

method1 <- annotations[['truth']]
method2 <- annotations[['predictions']]

library(epiR)

tmp <- data.frame(method1, method2)
tmp.ccc <- epi.ccc(method1, method2, ci = "z-transform", conf.level = 0.95)

tmp.lab <- data.frame(lab = paste("CCC: ", 
                                  round(tmp.ccc$rho.c[,1], digits = 2), " (95% CI ", 
                                  round(tmp.ccc$rho.c[,2], digits = 2), " - ",
                                  round(tmp.ccc$rho.c[,3], digits = 2), ")", sep = ""))

z <- lm(method2 ~ method1)
alpha <- summary(z)$coefficients[1,1]
beta <-  summary(z)$coefficients[2,1]
tmp.lm <- data.frame(alpha, beta)

## Concordance correlation plot:
library(ggplot2)

ggplot(tmp, aes(x = method1, y = method2)) + 
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  geom_abline(data = tmp.lm, aes(intercept = alpha, slope = beta), 
              linetype = "dashed") +
  xlim(0, 1.1) +
  ylim(0, 1.1) +
  xlab("Annotator 1") +
  ylab("Annotator 2") +
  geom_text(data = tmp.lab, x = 0.5, y = 2.95, label = tmp.lab$lab) + 
  coord_fixed(ratio = 1 / 1)

## In this plot the dashed line represents the line of perfect concordance. 
## The solid line represents the reduced major axis.  


## Bland and Altman plot (Figure 2 from Bland and Altman 1986):
tmp.ccc <- epi.ccc(method1, method2, ci = "z-transform", conf.level = 0.95, 
                   rep.measure = FALSE)
tmp <- data.frame(mean = tmp.ccc$blalt[,1], delta = tmp.ccc$blalt[,2])


library(ggplot2)

ggplot(tmp.ccc$blalt, aes(x = mean, y = delta)) + 
  geom_point() +
  geom_hline(data = tmp.ccc$sblalt, aes(yintercept = lower), linetype = 2) +
  geom_hline(data = tmp.ccc$sblalt, aes(yintercept = upper), linetype = 2) +
  geom_hline(data = tmp.ccc$sblalt, aes(yintercept = est), linetype = 1) +
  xlab("Average PEFR by two meters (L/min)") +
  ylab("Difference in PEFR (L/min)") +
  xlim(0, 1) +
  ylim(-1,1)


## Interclass Correlation
library(psych)

sf <- data.matrix(annotations)
ICC(sf, missing=FALSE, alpha = 0.05)

