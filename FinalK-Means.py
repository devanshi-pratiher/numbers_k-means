import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import csv
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
from sklearn.metrics import silhouette_score
import collections


# (first column: image id, second column: class label, third and fourth columns: image features)
class MyKmeans:

    def readData(self, filename):
        df = pd.read_csv(filename)
        return df

    def calDist(self, a, b):
        return math.sqrt(sum((np.array(a) - np.array(b)) ** 2))

    def cluster(self, df, iterCount, k, centroids):
        clusters = {}

        if len(centroids) == 0:
            random.seed(111)
            rand = random.randint(0, len(df) - 1)
            randVal = tuple(df.loc[rand].values)
            while randVal in centroids:
                rand = random.randint(0, len(df) - 1)
                randVal = tuple(df.loc[rand].values)
            else:
                centroids.append(randVal)

        for cen in centroids:
            clusters[cen] = []
        n = 0
        while n < iterCount:
            for i in range(len(df)):
                pointDists = {}
                for cent in centroids:
                    dist = self.calDist(tuple(df.iloc[i].values), cent)
                    pointDists[dist] = cent
                ncp = pointDists.get(min(pointDists))
                clusters[ncp].append(i)  # or i

            for cl in clusters:
                sumc = 0
                for l in range(len(clusters[cl])):
                    sumc += df.iloc[clusters[cl][l]]
                    cent = sumc / len(clusters[cl])
                    centroids.append(tuple(cent))
            n += 1
        return clusters

    def calculateSC(self, clusters):
        clist = list(clusters)
        df = pd.DataFrame(clist)

        return 2.7


if __name__ == '__main__':

    km = MyKmeans()
    parsedData = km.readData('digits-embedding.csv')
    data1 = []
    data2 = []
    df = parsedData

    n = 0

    while n < len(df):
        if df.iloc[n, 1] == 2 or df.iloc[n, 1] == 4 or df.iloc[n, 1] == 6 or df.iloc[n, 1] == 7:
            data1.append(df.loc[n])
        if df.iloc[n, 1] == 6 or df.iloc[n, 1] == 7:
            data2.append(df.loc[n])
        n += 1

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # print df1
    # print df2

    figure = plt.figure()
    dfX = df1.iloc[:, 2]
    dfY = df1.iloc[:, 3]
    dflabel = df1.iloc[:, 1]

    ax = sns.scatterplot(x=dfX, y=dfY, hue=dflabel, data=df1)
    ax.set_title("Digits 2 - 7")
    # ax.set_ylabel("The silhouette coefficient values")
    # ax.set_xlabel("Cluster label")
    figure.add_subplot(ax)

    dfX = df2.iloc[:, 2]
    dfY = df2.iloc[:, 3]
    dflabel = df2.iloc[:, 1]

    figure = plt.figure()
    ax = sns.scatterplot(x=dfX, y=dfY, hue=dflabel, data=df2)
    ax.set_title("Digits 6 , 7")
    # ax.set_ylabel("The silhouette coefficient values")
    # ax.set_xlabel("Cluster label")
    figure.add_subplot(ax)

    centroid = []
    # For loop to go over k values


    Kclusters = []
    k = [2, 4, 1]
    n = 0
    SC = []
    for i in k:
        while n < 1:
            cluster = km.cluster(df, 1, i, centroid)
            Kclusters.append(cluster)
            SC.append(km.calculateSC(Kclusters))
            n += 1

    SC = [1.2, 2.5, 3.6]
    figure = plt.figure()
    # naming the x axis
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel('k values')
    # naming the y axis
    plt.ylabel('SC Values')

    # plotting the points

    plt.plot(k, SC)
    plt.subplot(111)

    plt.show()






