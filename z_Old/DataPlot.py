import matplotlib.pyplot as plt
import xmltodict

file=open("augerat-1995-set-a/A-n80-k10.xml","r")
xmlfile=file.read()
dictionary=xmltodict.parse(xmlfile)
x= []
y= []
for element in dictionary["instance"]["network"]["nodes"]["node"]:
    x.append(float(element["cx"]))
    y.append(float(element["cy"]))

plt.plot(x,y,'bo')
plt.show()