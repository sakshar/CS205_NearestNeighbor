# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def load_data(filename):
    data = np.loadtxt(filename)
    return data[:, 1:], data[:, 0]


def normalize_data(dataX):
    minX, maxX = np.min(dataX, axis=0), np.max(dataX, axis=0)
    normX = (dataX - minX)/(maxX - minX)
    return normX


def distance(a, b):
    return np.sum((a-b)**2)


def leave_one_out_cross_validation(X, Y, currentSet, feature, add_remove):
    correctPredCount = 0
    currentFeatures = currentSet.copy()
    if add_remove == 1:
        currentFeatures += [feature]
    elif add_remove == 0:
        currentFeatures.remove(feature)
    #print(currentFeatures)
    for i in range(X.shape[0]):
        nearestNeighborPoint = -1
        nearestNeighborDist = np.inf
        for j in range(X.shape[1]):
            if i == j:
                continue
            dist = distance(X[i][currentFeatures], X[j][currentFeatures])
            if dist < nearestNeighborDist:
                nearestNeighborDist = dist
                nearestNeighborPoint = j
        if Y[i] == Y[nearestNeighborPoint]:
            correctPredCount += 1
    return correctPredCount/X.shape[0]


def forward_selection(X, Y):
    print("-------Forward Selection-------")
    features = []
    best = []
    defaultClassCount = np.max([np.where(Y == 1)[0].shape[0], np.where(Y == 2)[0].shape[0]])
    best.append((-1, defaultClassCount/X.shape[0]))
    for i in range(X.shape[1]):
        print("At level "+str(i+1))
        currentBestFeature = -1
        currentBestAccuracy = 0
        for j in range(X.shape[1]):
            if j not in features:
                acc = leave_one_out_cross_validation(X, Y, features, j, 1)
                print("--considering feature " + str(j) + " with accuracy " + str(acc))
                if acc > currentBestAccuracy:
                    currentBestAccuracy = acc
                    currentBestFeature = j
        features.append(currentBestFeature)
        best.append((features.copy(), currentBestAccuracy))
        print("Added feature " + str(currentBestFeature) + " at level " + str(i+1))
    return best


def backward_selection(X, Y):
    print("-------Backward Selection-------")
    features = list(np.arange(X.shape[1]))
    best = []
    acc = leave_one_out_cross_validation(X, Y, features, -1, -1)
    best.append((features.copy(), acc))
    for i in range(X.shape[1] - 1):
        print("At level "+str(i+1))
        currentBestFeature = -1
        currentBestAccuracy = 0
        for j in features:
            acc = leave_one_out_cross_validation(X, Y, features, j, 0)
            print("--considering feature " + str(j) + " with accuracy " + str(acc))
            if acc > currentBestAccuracy:
                currentBestAccuracy = acc
                currentBestFeature = j
        features.remove(currentBestFeature)
        best.append((features.copy(), currentBestAccuracy))
        print("Removed feature " + str(currentBestFeature) + " at level " + str(i + 1))
    defaultClassCount = np.max([np.where(Y == 1)[0].shape[0], np.where(Y == 2)[0].shape[0]])
    best.append((-1, defaultClassCount / X.shape[0]))
    return best


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    size = "large"
    num = "2"
    X, Y = load_data("CS205_"+size+"_testdata__"+num+".txt")
    normX = normalize_data(X)
    bestFeaturesForward = forward_selection(normX, Y)
    bestFeaturesBackward = backward_selection(normX, Y)
    #print(leave_one_out_cross_validation(normX, Y, [4, 1], 9))
    print(bestFeaturesForward)
    print(bestFeaturesBackward)
    f = open("Result_"+size+"_testdata__"+num+".txt", 'w')
    f.write("-----Forward Selection-----\n")
    for (i, j) in bestFeaturesForward:
        if i == -1:
            feat_str = "{}"
        else:
            feat_str = "{"+(",").join([str(k) for k in i])+"}"
        f.write(feat_str+": "+str(j))
        f.write("\n")
    f.write("-----Backward Selection-----\n")
    for (i, j) in bestFeaturesBackward:
        if i == -1:
            feat_str = "{}"
        else:
            feat_str = "{"+(",").join([str(k) for k in i])+"}"
        f.write(feat_str + ": " + str(j))
        f.write("\n")
    f.close()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
