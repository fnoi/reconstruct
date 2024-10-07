import bpy


def blender_beams():
    # bpy.ops.bim.create_project()
    # bpy.ops.bim.new_project() # causes crash for some reason, for now just set up ifc project manually

    bpy.context.scene.BIMProjectProperties.library_file = 'IFC4 US Steel.ifc'

    bpy.ops.bim.select_library_file(filepath="/Users/fnoic/Library/Application Support/Blender/4.2/extensions/.local/lib/python3.11/site-packages/bonsai/bim/data/libraries/IFC4 US Steel.ifc")


    bpy.ops.bim.change_library_element(element_name="IfcProfileDef")
    bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")

    # Element 332:
    # Name: W6X15
    # ifc_definition_id: 3587
    # is_appended: False
    # is_declared: False
    # name: W6X15

    needed_names = ['W8X67', 'W6X12']

    print("Inspecting Library Elements:")
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in needed_names:
            ifc_definition_id = getattr(element, 'ifc_definition_id')
            print(f'index: {index}, name: {element.name}, ifc_definition_id: {ifc_definition_id}')


        # print(f"\nElement {index}:")
        # print(f"  Name: {element.name}")
        # for attr in dir(element):
        #     if not attr.startswith("__") and attr not in ['bl_rna', 'rna_type']:
        #         value = getattr(element, attr)
        #         print(f"  {attr}: {value}")


    # somhehow parse the existing profile library to identify the profiles
    # try append to parse, append only if matching, if matching store id in a dictionary

    # here starts the loop later
    # add a beam

    # assign the correct profile

    # transformation



if __name__ == "__main__":
    blender_beams()