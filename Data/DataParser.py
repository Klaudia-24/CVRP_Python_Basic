import numpy as np


def parseRawData(file):
    """Parse data from vrp file to metadata, nodes, demand"""
    with open(file, "r") as f:
        raw_string = [x.split("DEMAND_SECTION") for x in f.read().split("NODE_COORD_SECTION")]
    metadata = [x.split(":") for x in raw_string[0][0].split("\n") if x!=""]
    metadata = {x[0].strip(" "): convert(x[1]) for x in metadata}

    demand = [int(x.strip(" ").split(" ")[1]) for x in raw_string[1][1].split("DEPOT_SECTION")[0].split("\n") if x!="" and x!=" "]
    nodes = [x.strip(" ").split(" ") for x in raw_string[1][0].split("\n") if x!="" and x!=" "]
    collection = [list(z) for z in list(dict.fromkeys([(float(x[1]),float(x[2]),y) for x,y in zip(nodes,demand)]))]
    demand = [x[2] for x in collection]
    nodes = np.array([x[:2] for x in collection])

    metadata["DIMENSION"]=len(demand)

    return metadata, nodes, demand


def convert(ob):
    """Try to convert ob to float. Return ob if fail."""
    try:
        return float(ob)
    except Exception:
        return ob


if __name__ == "__main__":
    metadata, nodes, demands = parseRawData("RawData/christofides/CMT1_snip.vrp")
