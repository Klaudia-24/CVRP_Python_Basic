import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

#url = "http://web.cba.neu.edu/~msolomon/c101.htm"
#url = "http://web.cba.neu.edu/~msolomon/r101.htm"
url = "http://web.cba.neu.edu/~msolomon/rc101.htm"

page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")
result = soup.find_all("font", color="#3366FF")
capacity = result[0].text
#result.pop(0)
result.pop(0)
i = 1
while result:
    root = ET.Element('instance')
    info = ET.SubElement(root, 'info')
    dataset = ET.SubElement(info, 'dataset')
    dataset.text = "Salomon R1-Type"
    name = ET.SubElement(info, 'name')
    name.text = f"R{i+100}"
    network = ET.SubElement(root, 'network')
    nodes = ET.SubElement(network, 'nodes')
    fleet = ET.SubElement(root, 'fleet')
    vehicle_profile = ET.SubElement(fleet, 'vehicle_profile')
    departure_node = ET.SubElement(vehicle_profile, 'departure_node')
    departure_node.text = "101"
    arrival_node = ET.SubElement(vehicle_profile, 'arrival_node')
    arrival_node.text = "101"
    capacity = ET.SubElement(vehicle_profile, 'capacity')
    capacity.text = capacity
    requests = ET.SubElement(root, 'requests')

    temp = result[0].text.replace("\xa0", "").lstrip().replace("\r\n", " ").replace("  ", " ")
    lista = temp.split(" ")
    node = ET.SubElement(nodes, 'node', id=lista[0], type="0")
    cx = ET.SubElement(node, 'cx')
    cx.text = lista[1]
    cy = ET.SubElement(node, 'cy')
    cy.text = lista[2]
    result.pop(0)

    for _ in range(100):
        print(result[0].text)
        temp = result[0].text.replace("\xa0", "").lstrip().replace("\r\n", " ").replace("  ", " ")
        lista = temp.split(" ")
        node = ET.SubElement(nodes, 'node', id=lista[0], type="1")
        cx = ET.SubElement(node, 'cx')
        cx.text = lista[1]
        cy = ET.SubElement(node, 'cy')
        cy.text = lista[2]
        request = ET.SubElement(requests, 'request', node=lista[0])
        quantity = ET.SubElement(request, 'quantity')
        quantity.text = lista[3]
        result.pop(0)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(f"Data_XML_R1/R{i+100}.xml", encoding="UTF-8", xml_declaration=True)
    i += 1


