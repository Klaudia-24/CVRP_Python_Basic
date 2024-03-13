import matplotlib.pyplot as plt


class ParsedData:
    carCapacity: int
    cross: float
    mut: float
    popC: int
    genC: int
    avgScore: float
    minScore: float
    maxScore: float
    lostCap: float
    avgTime: float

    def __init__(self, line):
        data = line.split(",")
        self.cross = float(data[1])
        self.mut = float(data[2])
        self.popC = int(data[3])
        self.genC = int(data[4])
        self.avgScore = float(data[5])
        self.minScore = float(data[6])
        self.maxScore = float(data[7])
        self.lostCap = float(data[8])
        self.avgTime = float(data[9])


def main():
    cmtFile = "CMT1"
    c = 2
    m = 1

    with open(f"../cmtResultGen/result_{c}_{m}/results{cmtFile}.txt", "r") as f:
        data = f.read().split("\n")
        elements = [ParsedData(x) for x in data if x != ""]
    criteria = (0.05, 0.2, 20)
    selected = [a for a in elements if a.cross == criteria[0] and a.mut == criteria[1] and a.popC == criteria[2]]
    plt.plot([x.genC for x in selected], [x.minScore for x in selected], label="Min")
    plt.plot([x.genC for x in selected], [x.avgScore for x in selected], label="Avg")
    plt.plot([x.genC for x in selected], [x.maxScore for x in selected], label="Max")
    plt.xticks(ticks=[x.genC for x in selected])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Generation count")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultMinAvgMax.jpg")
    plt.close()

    crit = (0.05, 0.01)# cross, mutation
    z = [-3, -1.5, 0, 1.5]
    crit_2 = [5, 10, 20, 40] # populationOptions
    for i, j in zip(crit_2, z):
        selected = [a for a in elements if a.cross == crit[0] and a.mut == crit[1] and a.popC == i]
        plt.plot([x.genC for x in selected], [x.avgTime for x in selected], label=f"Execution time {i}")
    plt.xticks(ticks=[x.genC for x in selected])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.ylabel("time [s]")
    plt.xlabel("Generation count")
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultTime.jpg")
    plt.close()


    for i, j in zip(crit_2, z):
        selected = [a for a in elements if a.cross == crit[0] and a.mut == crit[1] and a.popC == i]
        plt.bar([12*x +j for x in range(1,7)], [(1-x.lostCap)*100 for x in selected], label=f"Unused capacity {i}", width=1.5)
    plt.xticks(ticks=[12*x for x in range(1,7)], labels=[str(x.genC) for x in selected])

    # ticks = [x.genC for x in selected]
    plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center')
    plt.ylabel("%")
    plt.xlabel("Generation count")
    # plt.ylim((8, 10))
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultLostCap.jpg")
    plt.close()


if __name__ == "__main__":
    main()
