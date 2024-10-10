import bpy
import bonsai.tool as tool


def get_attribute_value(attr_value):
    if isinstance(attr_value, ifcopenshell.entity_instance):
        return f"{attr_value.is_a()}:{attr_value.id()}"
    elif isinstance(attr_value, (list, tuple)):
        return [get_attribute_value(v) for v in attr_value]
    else:
        return attr_value


def assign_material_and_create_instance(profile_type_name, element_name):
    objectname = f'IfcBeamType/{profile_type_name}'

    if objectname not in bpy.data.objects:
        print(f"Error: Object {objectname} not found. Skipping material assignment and instance creation.")
        return

    obj = bpy.data.objects[objectname]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Enable editing of assigned material
    bpy.ops.bim.enable_editing_assigned_material()

    # Attempt to set the profile using the element name
    bpy.context.scene.BIMMaterialProperties.active_profile_set = element_name

    # Edit the material set
    bpy.ops.bim.edit_material_set()

    # Find the correct type ID
    relating_type = obj.BIMObjectProperties.relating_type
    if relating_type:
        bpy.context.scene.BIMModelProperties.relating_type_id = str(relating_type)
        bpy.ops.bim.add_constr_type_instance()
    else:
        print(f"Error: No relating type found for {objectname}")

    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    bpy.ops.bim.disable_editing_assigned_material(obj=objectname)


def blender_beams(query_profile_dict):
    model = tool.Ifc.get()

    # load standard library
    bpy.context.scene.BIMProjectProperties.library_file = 'IFC4 US Steel.ifc'
    bpy.ops.bim.select_library_file(filepath="/Users/fnoic/Library/Application Support/Blender/4.2/extensions/.local/lib/python3.11/site-packages/bonsai/bim/data/libraries/IFC4 US Steel.ifc")

    # change into the correct library element type
    bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")

    query_profile_names = query_profile_dict.keys()
    ref_dict = {}

    print("inspecting library elements and loading necessary types:")
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in query_profile_names:
            ifc_definition_id = getattr(element, 'ifc_definition_id')
            print(f'index: {index}, name: {element.name}, ifc_definition_id: {ifc_definition_id}')
            ref_dict[element.name] = {
                'index': index,
                'ifc_definition_id': ifc_definition_id
            }

            # Load the profile to the project
            bpy.ops.bim.append_library_element(definition=ifc_definition_id, prop_index=index)

            # Set up for beam type creation
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            bpy.ops.bim.enable_add_type()
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
            bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'

            profile_type_name = f'beamprofiletype_{element.name}'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name

            # Create the beam type
            bpy.ops.bim.add_type()
            bpy.ops.bim.disable_add_type()
            print(f'Added beam profile type: {profile_type_name}')

            # Assign material and create instance
            assign_material_and_create_instance(profile_type_name, element.name)


if __name__ == "__main__":
    profiles = {
        'W8X67': {'length': 1.5},
        'W6X12': {'length': 1.0},
        'HP12X53': {'length': 0.2}
    }

    blender_beams(profiles)