
def Extract(file, mode : int = 1):
    with open(file, "r") as f:
        raw_string =f.read().replace("DEPOT_SECTION","BREAK").replace("CAPACITY :","BREAK").replace("NODE_COORD_SECTION","BREAK").replace("DEMAND_SECTION","BREAK").split("BREAK")
        data=dict()
        data['nodes']= dict()
        for line in raw_string[2].split("\n"):
            if len(line) != 0:
               temp=line.split(" ")
               data['nodes'][int(temp[0])]={'x': float(temp[1]),'y': float(temp[2])}
        for line in raw_string[3].split("\n"):
            if len(line) != 0:
               temp=line.split(" ")
               data['nodes'][int(temp[0])]['demand']=int(temp[1])
        data['capacity']=int(raw_string[1])
    return data

if __name__=="__main__":
    print(Extract("christofides and elion/E-n22-k4.vrp",1))
