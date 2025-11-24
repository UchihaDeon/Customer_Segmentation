import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/transactions.csv")
df.head()

# feature distribution
features = ["PurchaseFrequency", "AmountSpent", "ProductTypes"]
df[features].hist(bins=15, figsize=(10,6))
plt.suptitle("Feature Distributions")
plt.show()

# correlation heatmap
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# pairplot of for relationships
sns.pairplot(df[features])
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

