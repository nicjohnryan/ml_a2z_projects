

install.packages("Hmisc")
library(Hmisc)

numeric_data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric")

# as we have many categoricals converted to dummies, let's use spearman correlation
pairs(numeric_data)

predictor_clusters = varclus(as.matrix(numeric_data))

summary(predictor_clusters)

# plot the hierarchical clusters as a tree
plot(predictor_clusters)



