import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
print(os.listdir("../data"))
df = pd.read_excel("../data/Online_Retail.xlsx")

# Data cleaning
df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Feature engineering
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# RFM calculation
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Final KMeans (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Segment naming
rfm['Customer_Segment'] = rfm['Cluster'].map({
    0: 'Regular Customers',
    1: 'Loyal Customers',
    2: 'VIP Customers'
})

# Visualization
plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'])
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segmentation using RFM & K-Means")
plt.show()

# Export for Power BI
rfm.reset_index(inplace=True)
rfm.to_csv("../data/customer_segments.csv", index=False)

print("customer_segments.csv file created successfully")