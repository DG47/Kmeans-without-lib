import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style, colors

colors = 10 * ["g", "r", "c", "b", "k","orange","yellow","pink"]


def power(my_list, a=2):
    m = []
    for i in range(len(my_list)):
        f = [x ** a for x in my_list[i]]
        m.append(f)
    return m


def list_minus(a, b):
    x = 0
    y = 0
    lx = []
    ly = []
    for i in range(len(a)):
        x = a[i][0]
        y = a[i][1]

        for j in range(len(b)):
            x -= b[j][0]
            y -= b[j][1]
            lx.append(x)
            ly.append(y)
    # l = zip(lx, ly)
    return lx, ly

def coord_sum(a):
    summ = []

    x = a[0]
    y = a[1]
    s = x + y
    summ.append(s)
    return summ
def list_sum(a):
    lx = []
    ly = []
    x = 0
    y = 0
    for i in range(len(a)):
#        print(a)
        x += a[i][0]
        # y += a[i][1]

    # l = zip(lx, ly)
    return x


def list_divide(a, b):
    x = 0
    y = 0
    lx = []
    ly = []
    for i in range(len(a)):
        x = a[i][0]
        y = a[i][1]

        for j in range(len(b)):
            x /= b[j][0]
            y /= b[j][1]
            lx.append(x)
            ly.append(y)
    # l = zip(lx, ly)
    return lx, ly


def average(a):
    x = 0
    y = 0
    for i in range(len(a)):
        x += a[i][0]
        y += a[i][1]
    avgx = x / len(a)
    avgy = y / len(a)
    return [avgx, avgy]


with open('clustering_dataset.txt') as f:
    X = []
    for l in f:
        row = l.strip('\n')
        row = l.split(' ')
        row = list(float(x) for x in row)

        X.append(row)


# print('dataset: ', X)


class K_Means:

    def __init__(self, k=7, tol=0.1, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        # print(k,max_iter,tol)

    def fit(self, data):
        # print(data)
        self.centroids = {}
        # print(data)
        """
        Alloting data value to centroid randomly
        """
        for i in range(self.k):
            self.centroids[i] = data[i]
            # print(self.centroids[i])

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            # note: assigns the points to a cluster
            for featureset in data:
                distances = [power(coord_sum(list_minus([featureset],[self.centroids[centroid]]))) for centroid in
                             self.centroids]
                #                print(distances)
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            # print(prev_centroids)

            # note: moves the cluster centroids
            for classification in self.classifications:
                self.centroids[classification] = average(self.classifications[classification])

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                if list_sum(list_minus([current_centroid], [original_centroid])) > self.tol:
                    print(list_sum(list_minus([current_centroid], [original_centroid])))
                    optimized = False

            if optimized:
                break

        #todo inertia
        distances = [i for j in distances for i in j]
        self.inertia = list_sum(distances) / len(distances)


    def predict(self, data):
        distances = [power(list_minus([power(featureset)], [power(self.centroids[centroid])])) for centroid in
                     self.centroids]
        classification = distances.index(min(distances))
        #        print(classification)
        return classification


model = K_Means()

model.fit(X)

for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in model.classifications:

    color = colors[classification]
    for featureset in model.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
plt.show()

SSE = []
for K in range(1, 10):
    mod = K_Means(k=K)
    f =mod.fit(X)
    i=mod.inertia
    SSE.append(i)
print(SSE)
