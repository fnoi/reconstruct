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


##### SPICY PART #####
def retrieve_material_profile_id(beam_type_name, model):
    for rel in model.by_type('IfcRelAssociatesMaterial'):
        for related_obj in rel.RelatedObjects:
            if related_obj.is_a('IfcBeamType') and related_obj.Name == beam_type_name:
                if rel.RelatingMaterial.is_a('IfcMaterialProfileSet'):
                    return rel.RelatingMaterial.MaterialProfiles[0].id()
    return None


def find_material_profile_id_initial(model, profile_name):
    # Step 1: Find the IfcBeamType by its Name (profile-related name)
    for beam_type in model.by_type("IfcBeamType"):
        if profile_name.strip() in beam_type.Name.strip():
            # Ensure the exact length of the beam name matches the expected pattern
            if len(beam_type.Name.strip()) == len(f"beamprofiletype_{profile_name.strip()}"):
                beam_type_id = beam_type.id()

                # Step 2: Find the IfcRelAssociatesMaterial related to the found IfcBeamType
                for assoc_material in model.by_type("IfcRelAssociatesMaterial"):
                    if beam_type in assoc_material.RelatedObjects:
                        material_profile_set = assoc_material.RelatingMaterial

                        # Step 3: Access the IfcMaterialProfileSet and then the IfcMaterialProfile
                        if material_profile_set:
                            material_profiles = material_profile_set.MaterialProfiles
                            if material_profiles:
                                for material_profile in material_profiles:
                                    profile_def = material_profile.Profile

                                    # Step 4: Match IfcIShapeProfileDef ProfileName with the input profile name
                                    if profile_def and profile_def.ProfileName.strip() == profile_name.strip():
                                        return material_profile.id()
    # Return None if no match is found
    return None


def find_material_profile_id_alternative_1(model, profile_name):
    # Step 1: Iterate over all IfcRelAssociatesMaterial elements
    for assoc_material in model.by_type("IfcRelAssociatesMaterial"):
        # Step 2: Access IfcMaterialProfileSet from RelatingMaterial
        material_profile_set = assoc_material.RelatingMaterial
        if material_profile_set:
            material_profiles = material_profile_set.MaterialProfiles
            if material_profiles:
                for material_profile in material_profiles:
                    # Step 3: Check if the profile name in IfcIShapeProfileDef matches the target profile
                    profile_def = material_profile.Profile
                    if profile_def and profile_def.ProfileName.strip() == profile_name.strip():
                        # Step 4: Ensure exact matching of IfcBeamType Name attribute
                        for obj in assoc_material.RelatedObjects:
                            if isinstance(obj, model.get_class("IfcBeamType")) and profile_name.strip() in obj.Name.strip():
                                # Additional check: Ensure profile name lengths match exactly
                                if len(obj.Name.strip()) == len(f"beamprofiletype_{profile_name.strip()}"):
                                    # Return the ID of the IfcMaterialProfile
                                    return material_profile.id()
    # Return None if no match is found
    return None

def get_ids_for_profile(model, profile_name):
    material_set_item_id = None
    material_set_item_material_id = None
    profile_id = None
    material_set_id = None

    for beam_type in model.by_type("IfcBeamType"):
        if f"beamprofiletype_{profile_name}" == beam_type.Name:
            for rel in model.by_type("IfcRelAssociatesMaterial"):
                if beam_type in rel.RelatedObjects:
                    relating_material = rel.RelatingMaterial
                    if relating_material.is_a("IfcMaterialProfileSet"):
                        material_set_id = relating_material.id()
                        for material_profile in relating_material.MaterialProfiles:
                            if material_profile.Profile.ProfileName == profile_name:
                                material_set_item_id = material_profile.id()
                                material_set_item_material_id = material_profile.Material.id()
                                profile_id = material_profile.Profile.id()
                                return material_set_item_id, material_set_item_material_id, profile_id, material_set_id
                    elif relating_material.is_a("IfcMaterialProfileSetUsage"):
                        material_set_id = relating_material.ForProfileSet.id()
                        for material_profile in relating_material.ForProfileSet.MaterialProfiles:
                            if material_profile.Profile.ProfileName == profile_name:
                                material_set_item_id = material_profile.id()
                                material_set_item_material_id = material_profile.Material.id()
                                profile_id = material_profile.Profile.id()
                                return material_set_item_id, material_set_item_material_id, profile_id, material_set_id

    return material_set_item_id, material_set_item_material_id, profile_id, material_set_id

def find_material_profile_id_alternative_2(model, profile_name):
    # Step 1: Iterate over all IfcIShapeProfileDef elements
    for profile_def in model.by_type("IfcIShapeProfileDef"):
        if profile_def.ProfileName.strip() == profile_name.strip():
            # Step 2: Backtrack to find the IfcMaterialProfile that references this IfcIShapeProfileDef
            for material_profile in model.by_type("IfcMaterialProfile"):
                if material_profile.Profile == profile_def:
                    # Step 3: Find IfcRelAssociatesMaterial linking to IfcBeamType with exact Name matching
                    for assoc_material in model.by_type("IfcRelAssociatesMaterial"):
                        if material_profile in assoc_material.RelatingMaterial.MaterialProfiles:
                            for obj in assoc_material.RelatedObjects:
                                if isinstance(obj, model.get_class("IfcBeamType")) and profile_name.strip() in obj.Name.strip():
                                    # Ensure the exact length of name
                                    if len(obj.Name.strip()) == len(f"beamprofiletype_{profile_name.strip()}"):
                                        # Return the ID of the IfcMaterialProfile
                                        return material_profile.id()
    # Return None if no match is found
    return None






def retrieve_profile_id(profile_name):
    model = tool.Ifc.get()
    for profile in model.by_type('IfcProfileDef'):
        if hasattr(profile, 'ProfileName') and profile.ProfileName == profile_name:
            return profile.id()
    return None

def retrieve_relating_type_id(type_name):
    model = tool.Ifc.get()
    for beam_type in model.by_type('IfcBeamType'):
        if beam_type.Name == type_name:
            return beam_type.id()
    return None

def print_debug_info(profile_name):
    model = tool.Ifc.get()
    print(f"Debugging info for profile: {profile_name}")
    for rel in model.by_type('IfcRelAssociatesMaterial'):
        if rel.RelatingMaterial.is_a('IfcMaterialProfileSet'):
            print(f"Found IfcMaterialProfileSet: {rel.RelatingMaterial.id()}")
            for material_profile in rel.RelatingMaterial.MaterialProfiles:
                print(f"  MaterialProfile: {material_profile.id()}")
                print(f"    Profile: {material_profile.Profile.is_a()}")
                print(f"    ProfileName: {material_profile.Profile.ProfileName}")
    print("End of debug info")

##### SPICY PART #####


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

            material_name = f'M_{element.name}'
            bpy.ops.bim.add_material(name=material_name)

            # profiles = retrieve_profiles_info(model)
            # print_profile_info(profiles)

            # store all relevant info in sonder_dict
            sonder_dict[element.name] = {
                'profile_type_name': profile_type_name,
                'element': element
            }

            # # create beam instance
            # bpy.ops.bim.add_constr_type_instance()

    print('types setup complete')

    for profile_name, sonder in sonder_dict.items():
        # retrieve ids with new fct
        material_set_item_id, material_set_item_material_id, profile_id, material_set_id = get_ids_for_profile(model, element.name)
        print(f"\nProfile: {profile_name},\n"
              f"material_set_item_id: {material_set_item_id},\n"
              f"material_set_item_material_id: {material_set_item_material_id},\n"
              f"profile_id: {profile_id},\n"
              f"material_set_id: {material_set_id}")

        bpy.ops.outliner.item_activate(deselect_all=True)
        bpy.ops.bim.enable_edititng_assigned_material()
        bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
        bpy.data.objects[f"IfcBeamType/{sonder['profile_type_name']}"].BIMObjectMaterialProperties.material = str(material_set_item_material_id)
        bpy.ops.bim.enable_editing_assigned_material(material_set_item=material_set_item_id)
        bpy.data.objects[f"IfcBeamType/{sonder['profile_type_name']}"].BIMObjectMaterialProperties.material_set_item_material = str(material_set_item_material_id)
        bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
        bpy.ops.bim.disable_editing_material_set_item(obj=f"IfcBeamType/{sonder['profile_type_name']}")
        bpy.ops.bim.edit_material_set_item(material_set_item=material_set_item_id)
        bpy.ops.bim.disable_editing_material_set_item()
        bpy.ops.bim.disable_editing_assigned_material(obj=f"IfcBeamType/{sonder['profile_type_name']}")
        bpy.ops.bim.enable_editing_assigned_material(material_set=material_set_id)


    raise Exception('stop here')



    # iter through sonder_dict and assign profiles, create instances
    for profile_name, sonder in sonder_dict.items():
        profile_type_name = sonder['profile_type_name']
        element = sonder['element']
        print(f"\nProcessing element: {element.name}")

        # this is where the fun starts
        objectname = f"IfcBeamType/{profile_type_name}"
        obj = bpy.data.objects[objectname]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.bim.enable_editing_assigned_material()
        # retrieve IDx
        id_material = retrieve_material_profile_id(element.name, model)
        if id_material is None:
            id_material = find_material_profile_id_initial(model, element.name)
        if id_material is None:
            id_material = find_material_profile_id_alternative_1(model, element.name)
        if id_material is None:
            id_material = find_material_profile_id_alternative_2(model, element.name)
            if id_material is None:
                raise Exception(f"Could not find material profile ID for {element.name}")

        id_profile = retrieve_profile_id(element.name)
        id_relating_type = retrieve_relating_type_id(profile_type_name)
        print(f'obj_name: {objectname}, material_set_item_id: {id_material}, profile_id: {id_profile}, relating_type_id: {id_relating_type}')

        # enable editing of material set item
        # bpy.ops.bim.enable_editing_material_set_item(material_set_item=id_material)
        # bpy.context.scene.BIMMaterialProperties.profiles = str(id_profile)
        # bpy.ops.bim.disable_editing_material_set_item(obj=objectname)
        # bpy.ops.bim.edit_material_set_item(material_set_item=id_material)
        # bpy.context.scene.BIMModelProperties.relating_type_id = str(id_relating_type)
        # bpy.ops.bim.add_constr_type_instance()
        # bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
        # bpy.ops.bim.disable_editing_assigned_material(obj=objectname)

    #
    #         # this is where the fun starts
    #         objectname = f"IfcBeamType/{profile_type_name}"
    #         obj = bpy.data.objects[objectname]
    #         obj.select_set(True)
    #         bpy.context.view_layer.objects.active = obj
    #
    #         bpy.ops.bim.enable_editing_assigned_material()
    #         # retrieve IDx
    #         id_material = retrieve_material_set_item(element.name)
    #         id_profile = retrieve_profile_id(element.name)
    #         id_relating_type = retrieve_relating_type_id(profile_type_name)
    #
    #         # enable editing of material set item
    #         bpy.ops.bim.enable_editing_material_set_item(material_set_item=id_material)
    #         bpy.context.scene.BIMMaterialProperties.profiles = str(id_profile)
    #         bpy.ops.bim.disable_editing_material_set_item(obj=objectname)
    #         bpy.ops.bim.edit_material_set_item(material_set_item=id_material)
    #         bpy.context.scene.BIMModelProperties.relating_type_id = str(id_relating_type)
    #         bpy.ops.bim.add_constr_type_instance()
    #         bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    #         bpy.ops.bim.disable_editing_assigned_material(obj=objectname)
    #
    # raise Exception('stop here')




    #
    # print('hardcode')
    # objectname = 'IfcBeamType/beamprofiletype_W6X12'
    #
    # obj = bpy.data.objects[objectname]
    # obj.select_set(True)
    # bpy.context.view_layer.objects.active = obj
    #
    # bpy.ops.bim.enable_editing_assigned_material()
    # bpy.ops.bim.enable_editing_material_set_item(material_set_item=76)
    # bpy.context.scene.BIMMaterialProperties.profiles = '72'
    # bpy.ops.bim.disable_editing_material_set_item(obj="IfcBeamType/beamprofiletype_W6X12")
    # bpy.ops.bim.edit_material_set_item(material_set_item=76)
    # bpy.context.scene.BIMModelProperties.relating_type_id = '73'
    # bpy.ops.bim.add_constr_type_instance()
    # bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    # bpy.ops.bim.disable_editing_assigned_material(obj="IfcBeamType/beamprofiletype_W6X12")
    #
    # raise Exception('stop here')




if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                # 'W6X12':
                #     {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }

    blender_beams(profiles)