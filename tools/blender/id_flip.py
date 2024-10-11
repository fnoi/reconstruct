import bonsai.tool as tool
import bpy

def print_out_material_infos(model):
    for rel in model.by_type('IfcRelAssociatesMaterial'):
        if rel.RelatingMaterial.is_a('IfcMaterialProfileSet'):
            print(f"Found IfcMaterialProfileSet: {rel.RelatingMaterial.id()}")
            for material_profile in rel.RelatingMaterial.MaterialProfiles:
                print(f"  MaterialProfile: {material_profile.id()}")
                print(f"    Profile: {material_profile.Profile.is_a()}")
                print(f"    ProfileName: {material_profile.Profile.ProfileName}")
    print("End of debug info")


def retrieve_material_id(model):
    materials = model.by_type("IfcMaterial")
    material_id = None
    for material in materials:
        material_id = material.id()

    return material_id



def print_materials_info(model):

    materials = model.by_type("IfcMaterial")

    for material in materials:
        print(f"Material: {material.Name}")
        print(f"  ID: {material.id()}")
        print(f"  Attributes:")
        for attr, value in material.get_info().items():
            if attr not in ['id', 'type']:
                print(f"    {attr}: {value}")
        print()

def retrieve_id_magic(model):
    id_0 = None
    id_1 = None
    for rel in model.by_type('IfcRelAssociatesMaterial'):
        if rel.RelatingMaterial.is_a('IfcMaterialProfileSet'):
            id_0 = rel.RelatingMaterial.id()
            for material_profile in rel.RelatingMaterial.MaterialProfiles:
                id_1 = material_profile.id()

    return id_0, id_1



#######################
def retrieve_profile_id(profile_name, model):
    for profile in model.by_type('IfcProfileDef'):
        if hasattr(profile, 'ProfileName') and profile.ProfileName == profile_name:
            return profile.id()
    raise ValueError(f'Profile {profile_name} not found in model')


def blender_beams(query_profile_dict):
    # bpy.ops.bim.create_project()
    # bpy.ops.bim.new_project() # causes crash for some reason, for now just set up ifc project manually
    model = tool.Ifc.get()

    # load standard library for IfcIShapeProfileDef import
    bpy.context.scene.BIMProjectProperties.library_file = 'IFC4 US Steel.ifc'
    bpy.ops.bim.select_library_file(filepath="/Users/fnoic/Library/Application Support/Blender/4.2/extensions/.local/lib/python3.11/site-packages/bonsai/bim/data/libraries/IFC4 US Steel.ifc")
    bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")

    query_profile_names = query_profile_dict.keys()
    print(f'inspecting types: {query_profile_names}')

    ref_dict = {} # dict to store reference data between loops

    # import the necessary profiles from library to project
    print('\n--\n1.')
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in query_profile_names:
            print(f':::looking for {element.name}')
            ifc_definition_id = getattr(element, 'ifc_definition_id')
            bpy.ops.bim.append_library_element(definition=ifc_definition_id, prop_index=index)
            # retrieve profile id
            profile_id = retrieve_profile_id(element.name, model)
            ref_dict[element.name] = {
                '_index': index,
                '_ifc_definition_id': ifc_definition_id,
                'profile_id': profile_id
            }
            print(f':::loaded profile: {element.name} to project with id {profile_id}')

    # create beam type, assign profile (no psets)
    print('\n--\n2.')
    for profile_name, profile_data in ref_dict.items():
        print(f':::creating beam type for {profile_name}')
        profile_type_name = f'beamprofiletype_{profile_name}'
        bpy.ops.bim.enable_add_type()
        bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
        bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
        bpy.context.scene.BIMModelProperties.type_name = profile_type_name
        bpy.ops.bim.add_type()
        bpy.ops.bim.disable_add_type()
        ref_dict[profile_name]['profile_type_name'] = profile_type_name
        id_0, id_1 = retrieve_id_magic(model)
        print(f':::created beam type: {profile_type_name}')

        material_name = f'M_{profile_name}'
        bpy.ops.bim.add_material(name=material_name)
        material_id = retrieve_material_id(model)
        print(f':::created material: {material_name}, id: {material_id}')

        print_out_material_infos(model)
        print_materials_info(model)
        print(f':::material set id: {id_0}, material profile id: {id_1}')


        print(f'id_0={id_0}, id_1={id_1}, material_id={material_id}, profile_id={profile_data["profile_id"]}')
    #     raise ValueError('End of script')
    #
    # for _ in False:
        obj_name = f"IfcBeamType/{profile_type_name}"
        obj = bpy.data.objects.get(obj_name)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.bim.enable_editing_assigned_material()
        bpy.ops.bim.enable_editing_material_set_item(material_set_item=id_1)
        bpy.data.objects[obj_name].BIMObjectMaterialProperties.material_set_item_material = str(material_id)
        bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
        bpy.ops.bim.disable_editing_material_set_item(obj=obj_name)
        bpy.ops.bim.edit_material_set_item(material_set_item=id_1)
        bpy.ops.bim.disable_editing_material_set_item()
        bpy.ops.bim.disable_editing_assigned_material(obj=obj_name)

        # bpy.ops.bim.edit_material_set_item(material_set_item=id_0) # this call kills blender






if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }

    blender_beams(profiles)