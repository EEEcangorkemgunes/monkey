import json
import os
from xml.etree import ElementTree as ET
from xml.dom import minidom
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


read_file = "test.json"
output_path = "bounding_boxes"
output_image_path = "bounding_boxes_images"
with open(read_file, "r") as f:
    data = json.load(f)
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_image_path, exist_ok=True)
i = 0
for patch in data:
    asap_annotation = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(asap_annotation, "Annotations")
    image = Image.open(patch["file_name"])
    draw = ImageDraw.Draw(image)
    for ant in patch["annotations"]:
        outline = "red" if ant["category_id"] == 0 else "yellow"
        draw.rectangle([ant["bbox"][0], ant["bbox"][1], ant["bbox"][0] + ant["bbox"][2], ant["bbox"][1] + ant["bbox"][3]], outline=outline, width=2)
        annotation = ET.SubElement(annotations, "Annotation")
        annotation.set("Name", f"BB_{i}")
        i += 1
        annotation.set("Type", "Rectangle")
        annotation.set("PartOfGroup", "None")
        annotation.set("Color", "255, 0, 0")
        coordinates = ET.SubElement(annotation, "Coordinates")
        j = 0
        for coord in [[ant["bbox"][0], ant["bbox"][1]], [ant["bbox"][0] + ant["bbox"][2], ant["bbox"][1]], [ant["bbox"][0] + ant["bbox"][2], ant["bbox"][1] + ant["bbox"][3]], [ant["bbox"][0], ant["bbox"][1] + ant["bbox"][3]]]:
            coordinate = ET.SubElement(coordinates, "Coordinate")
            coordinate.set("Order", str(j))
            coordinate.set("X", str(coord[0]))
            coordinate.set("Y", str(coord[1]))
            j += 1
    image.save(os.path.join(output_image_path, patch["file_name"].split("/")[-1]))

    with open(os.path.join(output_path, patch["file_name"].split("/")[-1].replace(".png", ".xml")), "w") as f:
        f.write(minidom.parseString(ET.tostring(asap_annotation)).toprettyxml())
    
