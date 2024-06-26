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
    cmtFile = "CMT12"
    c = 2
    m = 1

    with open(f"../cmtResultGen/result_{c}_{m}/results{cmtFile}.txt", "r") as f:
        data = f.read().split("\n")
        elements = [ParsedData(x) for x in data if x != ""]
    # crossOptions = (0.05, 0.25)
    # mutationOptions = (0.01, 0.2)
    # populationOptions = (5, 10, 20, 40)
    # generationOptions = (10, 20, 40, 80)
    criteria = (0.05, 0.2, 5)
    selected = [a for a in elements if a.cross == criteria[0] and a.mut == criteria[1] and a.popC == criteria[2]]
    plt.plot([x.genC for x in selected], [x.minScore for x in selected], label="Minimum")
    plt.plot([x.genC for x in selected], [x.avgScore for x in selected], label="Średnia")
    plt.plot([x.genC for x in selected], [x.maxScore for x in selected], label="Maksimum")
    plt.xticks(ticks=[x.genC for x in selected])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Liczba generacji")
    plt.ylabel("Dystans")
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultMinAvgMax{criteria[2]}.jpg")
    plt.close()

    criteria = (0.05, 0.2, 40)
    selected = [a for a in elements if a.cross == criteria[0] and a.mut == criteria[1] and a.popC == criteria[2]]
    plt.plot([x.genC for x in selected], [x.minScore for x in selected], label="Minimum")
    plt.plot([x.genC for x in selected], [x.avgScore for x in selected], label="Średnia")
    plt.plot([x.genC for x in selected], [x.maxScore for x in selected], label="Maksimum")
    plt.xticks(ticks=[x.genC for x in selected])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.xlabel("Liczba generacji")
    plt.ylabel("Dystans")
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultMinAvgMax{criteria[2]}.jpg")
    plt.close()


    crit = (0.05, 0.01)# cross, mutation
    z = [-4.5, -3, -1.5, 0, 1.5, 3]
    crit_2 = [5, 10, 20, 40, 80, 160] # populationOptions
    for i, j in zip(crit_2, z):
        selected = [a for a in elements if a.cross == crit[0] and a.mut == crit[1] and a.popC == i]
        plt.plot([x.genC for x in selected], [x.avgTime for x in selected], label=f"{i} osobników")
    plt.xticks(ticks=[x.genC for x in selected])
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.ylabel("Czas [s]")
    plt.xlabel("Liczba generacji")
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultTime.jpg")
    plt.close()


    for i, j in zip(crit_2, z):
        selected = [a for a in elements if a.cross == crit[0] and a.mut == crit[1] and a.popC == i]
        plt.bar([12*x +j for x in range(1, len(selected)+1)], [(1-x.lostCap)*100 for x in selected], label=f"{i} osobników", width=1.5)
    plt.xticks(ticks=[12*x for x in range(1, len(selected)+1)], labels=[str(x.genC) for x in selected])

    # ticks = [x.genC for x in selected]
    plt.legend(bbox_to_anchor=(0.5, 1.3), loc='upper center')
    plt.ylabel("%")
    plt.xlabel("Liczba generacji")
    # plt.ylim((8, 10))
    plt.tight_layout()
    plt.pause(0.1)
    plt.savefig(f"../cmtPlotsGen/result_{c}_{m}/{cmtFile}/resultLostCap.jpg")
    plt.close()


if __name__ == "__main__":
    main()
