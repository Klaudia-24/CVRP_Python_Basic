import matplotlib.pyplot as plt


class ParsedData:
    carCapacity: int
    nAnt: int
    nIter: int
    avgScore: float
    minScore: float
    maxScore: float
    lostCap: float
    avgTime: float

    def __init__(self, line):
        data = line.split(",")
        self.nAnt = int(data[1])
        self.nIter = int(data[2])
        self.avgScore = float(data[3])
        self.minScore = float(data[4])
        self.maxScore = float(data[5])
        self.lostCap = float(data[6])
        self.avgTime = float(data[7])


def main():
    cmtFile = "CMT1"

    with open(f"../cmtResultAnt/results{cmtFile}.txt", "r") as f:
        data = f.read().split("\n")
        elements = [ParsedData(x) for x in data if x != ""]
    criteria = (10, 20, 40, 60)
    selCrit = criteria[3]
    selected = [a for a in elements if a.nAnt == selCrit]

    plt.plot([x.nIter for x in selected], [x.minScore for x in selected], label="Min")
    plt.plot([x.nIter for x in selected], [x.avgScore for x in selected], label="Avg")
    plt.plot([x.nIter for x in selected], [x.maxScore for x in selected], label="Max")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Iteration count")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.xticks(ticks=[x.nIter for x in selected])
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultMinAvgMax_{selCrit}.jpg")
    plt.close()

    # plt.plot([x.nIter for x in selected], [x.avgTime for x in selected], label="Execution time")
    # plt.legend(loc='upper right')
    # plt.xticks(ticks=[x.nIter for x in selected])
    # plt.pause(0.1)
    # plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultTime.jpg")
    # plt.close()

    for i in criteria:
        selected = [a for a in elements if a.nAnt == i]
        plt.plot([x.nIter for x in selected], [x.avgTime for x in selected], label=f"{i} ants")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.ylabel("time [s]")
    plt.xlabel("Iteration count")
    plt.tight_layout()
    plt.xticks(ticks=[x.nIter for x in selected])
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultTime.jpg")
    plt.close()

    z = [-4, -2, 0, 2]
    for i, j in zip(criteria, z):
        selected = [a for a in elements if a.nAnt == i]
        plt.bar([x.nIter + j for x in selected], [(1-x.lostCap)*100 for x in selected], label=f"{i} ants", width=2)
    plt.xlabel("Iteration count")
    plt.ylabel("%")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(ticks=[x.nIter for x in selected])
    # plt.ylim((9.25, 10))
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultLostCap.jpg")
    plt.close()

    z = [-4, -2, 0, 2]
    for i, j in zip(criteria, z):
        selected = [a for a in elements if a.nAnt == i]
        plt.bar([x.nIter + j for x in selected], [x.avgScore for x in selected], label=f"{i} ants", width=2)
    plt.xlabel("Iteration count")
    plt.ylabel("??????")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(ticks=[x.nIter for x in selected])
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultAvgCost.jpg")
    plt.close()

    z = [-4, -2, 0, 2]
    for i, j in zip(criteria, z):
        selected = [a for a in elements if a.nAnt == i]
        plt.bar([x.nIter + j for x in selected], [x.avgScore for x in selected], label=f"{i} ants", width=2)
    plt.xlabel("Iteration count")
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(ticks=[x.nIter for x in selected])
    plt.ylim((500, 700))
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsAnt/{cmtFile}/resultAvgCost_2.jpg")
    plt.close()


if __name__ == "__main__":
    main()
