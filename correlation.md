# Correlation (Pearson)

## NAN handle

Replace NaN in Correlation Matrix: If after calculating the correlation matrix, you have NaN values, it is generally safe to replace those NaN values with zero for the purpose of finding the least correlated features. This is because a NaN correlation indicates that there is no linear relationship between the features, which is akin to having zero correlation.