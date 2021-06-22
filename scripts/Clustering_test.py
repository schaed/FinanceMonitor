import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm1
import pytz,os,sys
import base as b
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

if __name__ == "__main__":
    # execute only if run as a script
    sec_map = {'TECHNOLOGY': 0, 'TRADE & SERVICES': 1, 'FINANCE': 2, 'Financial Services': 3, 'Technology': 4, 'Consumer Cyclical': 5, 'MANUFACTURING': 6, 'Healthcare': 7, 'LIFE SCIENCES': 8, 'Basic Materials': 9, 'REAL ESTATE & CONSTRUCTION': 10, 'Industrials': 11, 'Consumer Defensive': 12, 'ENERGY & TRANSPORTATION': 13, 'Energy': 14, 'Utilities': 15, 'Communication Services': 16, 'Real Estate': 17, 'Other': 18, '': 19}
    stock_returns = pd.read_csv('stock_returns_kmeansv2.csv')
    
    # A list holds the SSE values for each k
    sse = []
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []
    maxTestCluster = 31
    data_points_kmeans = ['day_return', 'day2_return', 'day3_return', 'day4_return',
       'day5_return', 'day15_return', '30d_return', '60d_return',
       '180d_return', 'volatitilty', '5d_vol', '30d_vol', '180d_vol',
       'ShortPercentFloat', 'PercentInsiders', 'PercentInstitutions',
       'PERatio', 'ForwardPE', 'MarketCapitalization', 'AnalystTargetPrice',
       'Industry', 'Sector']
    data_points_kmeans = ['day_return', 'day2_return', 'day3_return', 'day4_return','day5_return', 'day15_return', '30d_return', '60d_return','180d_return', 'volatitilty', '5d_vol', '30d_vol', '180d_vol','ShortPercentFloat', 'PercentInsiders', 'PercentInstitutions', 'PERatio', 'ForwardPE','MarketCapitalization']
    for h in ['5d_vol','30d_vol','180d_vol','PERatio', 'ForwardPE']:
        stock_returns[h]/=stock_returns[h].max()
    for h in ['ShortPercentFloat','PercentInsiders','PercentInstitutions']:
        stock_returns[h]/=100.0
    print(data_points_kmeans)
    for k in range(2, maxTestCluster):
        kmeans = KMeans(n_clusters=k)
        #kmeans = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',assign_labels='kmeans')
        kmeans.fit(stock_returns[data_points_kmeans].dropna())
        sse.append(kmeans.inertia_)

        #kmeans.fit(scaled_features)
        #score = silhouette_score(scaled_features, kmeans.labels_)
        #silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, maxTestCluster), sse)
    plt.xticks(range(2, maxTestCluster))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    stock_returns_dropna = stock_returns[data_points_kmeans+['ticker']].dropna()    
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(stock_returns[data_points_kmeans].dropna())
    stock_returns_dropna['kmeans'] =kmeans.predict(stock_returns[data_points_kmeans].dropna())
    
    #kmeans = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',assign_labels='kmeans')
    #stock_returns_dropna['kmeans'] =kmeans.fit_predict(stock_returns[data_points_kmeans].dropna())    

    print(stock_returns_dropna)
    stock_returns_kmeans = stock_returns_dropna.groupby('kmeans')
    print(stock_returns_dropna.groupby('kmeans').describe())
    for c in stock_returns_dropna.columns:
        print(c)
        print(stock_returns_kmeans[c].describe())
    for k in stock_returns_dropna['kmeans'].unique():
        print(k)
        print(stock_returns_dropna[stock_returns_dropna['kmeans']==k]['ticker'].values)
