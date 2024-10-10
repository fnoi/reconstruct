import bpy
import bonsai.tool as tool


def get_attribute_value(attr_value):
    if isinstance(attr_value, ifcopenshell.entity_instance):
        return f"{attr_value.is_a()}:{attr_value.id()}"
    elif isinstance(attr_value, (list, tuple)):
        return [get_attribute_value(v) for v in attr_value]
    else:
        return attr_value

def blender_beams(query_profile_dict):
    # bpy.ops.bim.create_project()
    # bpy.ops.bim.new_project() # causes crash for some reason, for now just set up ifc project manually
    model = tool.Ifc.get()

    # load standard library
    bpy.context.scene.BIMProjectProperties.library_file = 'IFC4 US Steel.ifc'
    bpy.ops.bim.select_library_file(filepath="/Users/fnoic/Library/Application Support/Blender/4.2/extensions/.local/lib/python3.11/site-packages/bonsai/bim/data/libraries/IFC4 US Steel.ifc")

    # change into the correct library element type
    bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")

    query_profile_names = query_profile_dict.keys()
    ref_dict = {}
    beam_counter = 0

    print("inspecting library elements and loading necessary types:")
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in query_profile_names:
            # ... (keep this part as is)

            profile_type_name = f'beamprofiletype_{element.name}'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name

            # Create the beam type
            bpy.ops.bim.add_type()
            bpy.ops.bim.disable_add_type()
            print(f'Added beam profile type: {profile_type_name}')

            # Assign material and create instance
            assign_material_and_create_instance(profile_type_name, element.name)

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

        # Find the correct profile
        profile_id = None
        for profile in bpy.context.scene.BIMModelProperties.profiles:
            if profile.name == element_name:
                profile_id = profile.id
                break

        if profile_id is not None:
            # Set the profile directly
            bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)

            # Edit the material set (this might create a material_set_item if it doesn't exist)
            bpy.ops.bim.edit_material_set()
        else:
            print(f"Error: Profile {element_name} not found.")

        # Find the correct type ID
        relating_type = obj.BIMObjectProperties.relating_type
        if relating_type:
            bpy.context.scene.BIMModelProperties.relating_type_id = str(relating_type)
            bpy.ops.bim.add_constr_type_instance()
        else:
            print(f"Error: No relating type found for {objectname}")

        bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
        bpy.ops.bim.disable_editing_assigned_material(obj=objectname)


if __name__ == "__main__":
    profiles = {
        'W8X67': {'length': 1.5},
        'W6X12': {'length': 1.0},
        'HP12X53': {'length': 0.2}
    }

    blender_beams(profiles)