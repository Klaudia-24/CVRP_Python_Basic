import xml.etree.ElementTree as ET

root = ET.Element('instance')

info = ET.SubElement(root, 'info')
dataset = ET.SubElement(info, 'dataset')
name = ET.SubElement(info, 'name')

network = ET.SubElement(root, 'network')
nodes = ET.SubElement(network, 'nodes')
node = ET.SubElement(nodes, 'node')
cx = ET.SubElement(node, 'cx')
cy = ET.SubElement(node, 'cy')
euclidean = ET.SubElement(network, 'euclidean')
decimals = ET.SubElement(network, 'decimals')

fleet = ET.SubElement(root, 'fleet')
vehicle_profile = ET.SubElement(fleet, 'vehicle_profile')
departure_node = ET.SubElement(vehicle_profile, 'departure_node')
arrival_node = ET.SubElement(vehicle_profile, 'arrival_node')
capacity = ET.SubElement(vehicle_profile, 'capacity')

requests = ET.SubElement(root, 'requests')
request = ET.SubElement(requests, 'request')
quantity = ET.SubElement(request, 'quantity')


tree = ET.ElementTree(root)
ET.indent(tree, space="\t", level=0)
tree.write('basic.xml', encoding="UTF-8", xml_declaration=True)


