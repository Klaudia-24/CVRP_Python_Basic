import numpy as np
from Data.Extractor import Extract
from Plots import Route


def parseRouteArrays(ar, fullRoute):
    result = []
    routeArray = np.empty(shape=[0, 2])
    routeArray = np.append(routeArray, [ar[0]], axis=0)
    for node in fullRoute.replace("]", "").replace("[", "").split(",")[1:]:
        if int(node) == 1:
            routeArray = np.append(routeArray, [ar[int(node) - 1]], axis=0)
            result.append(routeArray)
            routeArray = np.empty(shape=[0, 2])
            routeArray = np.append(routeArray, [ar[0]], axis=0)
        else:
            routeArray = np.append(routeArray, [ar[int(node) - 1]], axis=0)
    return result


def main():
    cmtFile = "CMT12"
    c = 1
    m = 1

    with open(f"../cmtResultGen/result_{c}_{m}/bestResult{cmtFile}.txt", "r+") as f:
        dic = f.read().split("[")[2]
        data = Extract(f"../Data/CMT/{cmtFile}.vrp")
        ar = np.empty(shape=[0, 2])
        for elem in data['nodes'].keys():
            ar = np.append(ar, [[data['nodes'][elem]['x'], data['nodes'][elem]['y']]], axis=0)
    result = Route(0, 0, 0, parseRouteArrays(ar, dic), dic)
    result.plotRoute(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/bestRoutePlot.jpg")


if __name__ == "__main__":
    main()
