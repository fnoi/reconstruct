import bpy
import ifcopenshell
import bonsai.tool as tool
import sys
import re


class OutputRedirector:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = OutputRedirector("/Users/fnoic/Downloads/sandbox44_output.txt")


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def search_value(value, target_numbers, target_strings):
    str_value = str(value)
    for num in target_numbers:
        if str(num) in str_value:
            if not is_float(str_value) and not re.search(r'\d*\.\d*' + str(num), str_value):
                return str(num)

    for target in target_strings:
        if target in str_value:
            return target

    return None


def print_finding(element, attribute_name, value, access_path, found_target):
    print(f"Found target '{found_target}' in: {value}")
    print(f"In element: {element.is_a()} (ID: {element.id()})")
    print(f"Attribute: {attribute_name}")
    print(f"Access path: {access_path}")
    print("Element details:")
    print_element_details(element)
    print()


def print_element_details(element):
    attributes = element.get_info()
    for attr, value in attributes.items():
        if attr not in ['id', 'type']:
            if isinstance(value, ifcopenshell.entity_instance):
                print(f"  {attr}: {value.is_a()}(ID: {value.id()})")
            elif isinstance(value, (list, tuple)):
                print(f"  {attr}: [List with {len(value)} items]")
            else:
                print(f"  {attr}: {value}")


def parse_attributes(element, attribute_name, value, target_numbers, target_strings, access_path):
    found_target = search_value(value, target_numbers, target_strings)
    if found_target:
        print_finding(element, attribute_name, value, access_path, found_target)

    if isinstance(value, ifcopenshell.entity_instance):
        parse_element(value, target_numbers, target_strings, f"{access_path}.{attribute_name}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            if isinstance(item, ifcopenshell.entity_instance):
                parse_element(item, target_numbers, target_strings, f"{access_path}.{attribute_name}[{idx}]")


def parse_element(element, target_numbers, target_strings, access_path):
    for attribute_name, value in element.get_info().items():
        parse_attributes(element, attribute_name, value, target_numbers, target_strings, access_path)


def parse_elements():
    model = tool.Ifc.get()
    target_numbers = {73}#71, 66, 69, 76, 72, 74, 81, 77, 79}
    target_strings = {}#"W8X67", "HP12X53"}

    for ifc_class in model.types():
        for element in model.by_type(ifc_class):
            parse_element(element, target_numbers, target_strings, f"model.by_id({element.id()})")

    for obj in bpy.data.objects:
        found_target = search_value(obj.BIMObjectProperties.ifc_definition_id, target_numbers, target_strings)
        if found_target:
            print(f"Found target '{found_target}' in: {obj.BIMObjectProperties.ifc_definition_id}")
            print(f"In Blender object: {obj.name}")
            print(f"Access path: bpy.data.objects['{obj.name}'].BIMObjectProperties.ifc_definition_id")
            print()

        for prop in obj.keys():
            value = obj[prop]
            found_target = search_value(value, target_numbers, target_strings)
            if found_target:
                print(f"Found target '{found_target}' in Blender object property: {value}")
                print(f"Object: {obj.name}, Property: {prop}")
                print(f"Access path: bpy.data.objects['{obj.name}']['{prop}']")
                print()


print("\n>>>>>>>\nParsing elements...\n")
parse_elements()

sys.stdout = sys.stdout.terminal