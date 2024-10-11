import os
import tempfile

import bpy
import ifcopenshell

import bonsai.tool as tool



def get_attribute_value(attr_value):
    if isinstance(attr_value, ifcopenshell.entity_instance):
        return f"{attr_value.is_a()}:{attr_value.id()}"
    elif isinstance(attr_value, (list, tuple)):
        return [get_attribute_value(v) for v in attr_value]
    else:
        return attr_value


def retrieve_profiles_info(model):
    profiles = {}
    elements = model.by_type("IfcProfileDef")

    for element in elements:
        profile_info = {
            "id": element.id(),
            "type": element.is_a(),
            "name": getattr(element, "ProfileName", "Unnamed"),
            "attributes": {}
        }

        # Retrieve all attributes
        for attribute, value in element.__dict__.items():
            if not attribute.startswith('_'):
                profile_info["attributes"][attribute] = get_attribute_value(value)

        profiles[element.id()] = profile_info

    return profiles


def print_profile_info(profiles):
    for profile_id, info in profiles.items():
        print(f"\nProfile ID: {profile_id}")
        print(f"Type: {info['type']}")
        print(f"Name: {info['name']}")
        print("Attributes:")
        for attr, value in info['attributes'].items():
            print(f"  {attr}: {value}")


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
    sonder_dict = {}

    print("inspecting library elements and load necessary types:")
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in query_profile_names:
            ifc_definition_id = getattr(element, 'ifc_definition_id')
            print(f'\nstandard steel attributes:\n'
                  f'index: {index}, name: {element.name}, ifc_definition_id: {ifc_definition_id}')
            ref_dict[element.name] = {
                'index': index,
                'ifc_definition_id': ifc_definition_id
            }

            # load the profile to the project
            bpy.ops.bim.append_library_element(definition=ifc_definition_id, prop_index=index)

            # add the profile to the project and add a beam object
            # add_beam(element.name, query_profile_dict[element.name])

            # add directly in the loop to avoid issues (lol does not actually avoid issues)
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            bpy.ops.bim.enable_add_type()
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
            bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
            # bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_one_for_all'
            profile_type_name = f'beamprofiletype_{element.name}'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")

            # add type and specify the profile
            print('try to clear the cache')
            cache_dir = os.path.join(tempfile.gettempdir(), "bonsai")
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    os.remove(os.path.join(cache_dir, file))
            print('cache cleared (?)')
            print(f'adding profile type {profile_type_name}')
            bpy.ops.bim.add_type()
            print(f'added profile type {profile_type_name }')
            bpy.ops.bim.disable_add_type()
            print(f'added profile type {profile_type_name }')

            profiles = retrieve_profiles_info(model)
            print_profile_info(profiles)

            # store all relevant info in sonder_dict
            sonder_dict[element.name] = {
                'profile_type_name': profile_type_name,
                'element': element
            }

            # # create beam instance
            # bpy.ops.bim.add_constr_type_instance()

    print('types setup complete')





if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }

    blender_beams(profiles)