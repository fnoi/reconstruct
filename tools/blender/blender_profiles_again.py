import bpy

import bonsai.tool as tool


def export_scene_to_fbx(filepath):
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')

    # Set the export path
    filepath = bpy.path.ensure_ext(filepath, ".fbx")

    # Export to FBX
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_NONE',
        bake_space_transform=False,
        object_types={'EMPTY', 'CAMERA', 'LIGHT', 'ARMATURE', 'MESH', 'OTHER'},
        use_mesh_modifiers=True,
        mesh_smooth_type='OFF',
        use_mesh_edges=False,
        use_tspace=False,
        use_custom_props=False,
        add_leaf_bones=True,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=True,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=1.0,
        path_mode='AUTO',
        embed_textures=False,
        batch_mode='OFF',
        use_batch_own_dir=True,
        use_metadata=True,
    )

    print(f"FBX export completed: {filepath}")



def add_beam(profile_name, profile_dict):
    # create a new profile type
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    bpy.ops.bim.enable_add_type()
    bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
    bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
    bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_{profile_name}'
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")

    bpy.ops.bim.add_type()

    raise NotImplementedError

    bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
    bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_{profile_name}'
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    bpy.ops.bim.add_type()
    print('from nothing')
    bpy.ops.bim.add_constr_type_instance(relating_type_id=67, from_invoke=True)
    print('to something')
    raise NotImplementedError

    # bpy.context.scene.BIMModelProperties.type_name = f'beamtype_{profile_name}'



    # bpy.ops.bim.add_type()
    # bpy.ops.bim.add_default_type(ifc_element_type="IfcBeamType")

    # # Assign profile to beam type
    # beam_type = bpy.context.active_object
    #
    # # Create beam instance
    # bpy.context.scene.BIMModelProperties.extrusion_depth = profile_dict['length']
    # bpy.ops.bim.add_constr_type_instance()





def retrieve_relating_type_id(typename):
    for obj in bpy.context.scene.objects:
        if obj.BIMObjectProperties.ifc_definition_id:
            # print(f"Object: {obj.name}")
            # print(f"IFC Definition ID: {obj.BIMObjectProperties.ifc_definition_id}")
            # print(f"Relating Type ID: {bpy.context.scene.BIMModelProperties.relating_type_id}")
            if obj.name == typename:
                return obj.BIMObjectProperties.ifc_definition_id

    raise ValueError(f"type with name {typename} not found")



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

    print("inspecting library elements and load necessary types:")
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        if element.name in query_profile_names:
            ifc_definition_id = getattr(element, 'ifc_definition_id')
            print(f'index: {index}, name: {element.name}, ifc_definition_id: {ifc_definition_id}')
            ref_dict[element.name] = {
                'index': index,
                'ifc_definition_id': ifc_definition_id
            }

            # load the profile to the project
            bpy.ops.bim.append_library_element(definition=ifc_definition_id, prop_index=index)

            # add the profile to the project and add a beam object
            # add_beam(element.name, query_profile_dict[element.name])

            # add directly in the loop to avoid issues
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            bpy.ops.bim.enable_add_type()
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
            bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
            # bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_one_for_all'
            bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_{element.name}'
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")

            bpy.ops.bim.add_type()
            bpy.ops.bim.disable_add_type()
            # print(f'newly added object has id: {bpy.context.active_object.name}')

            # rel_id = bpy.context.scene.BIMProperties.last_added_type_id

            bpy.context.scene.BIMModelProperties.extrusion_depth = query_profile_dict[element.name]['length']
            target_type = retrieve_relating_type_id(f'IfcBeamType/beamprofiletype_{element.name}')
            print(f'target_type: {target_type}')
            bpy.ops.bim.add_constr_type_instance(relating_type_id=target_type, from_invoke=True)
            new_object = bpy.context.active_object
            new_object.name = f'beam_{beam_counter}_{element.name}'

            # select the new object by name through bpy
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = new_object
            new_object.select_set(True)

            # then change the profile
            # bpy.ops.bim.enable_editing_assigned_material()
            # bpy.context.scene.BIMMaterialProperties.profiles = target_type
            # bpy.ops.bim.disable_editing_material_set_item(obj=


            # print(bpy.context.scene.BIMModelProperties.relating_type_id)

            # another one




            beam_counter += 1
    raise NotImplementedError



    # bpy.ops.bim.change_library_element(element_name="IfcProfileDef")
    # bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")
    #
    # for


    # somhehow parse the existing profile library to identify the profiles
    # try append to parse, append only if matching, if matching store id in a dictionary

    # here starts the loop later
    # add a beam

    # assign the correct profile

    # transformation



if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0}
                }

    blender_beams(profiles)
    # export_scene_to_fbx("C:/Users/fnoic/Download/exported_scene.fbx")




## for reference:
# print(f"\nElement {index}:")
# print(f"  Name: {element.name}")
# for attr in dir(element):
#     if not attr.startswith("__") and attr not in ['bl_rna', 'rna_type']:
#         value = getattr(element, attr)
#         print(f"  {attr}: {value}")
## returns:
# Element 332:
# Name: W6X15
# ifc_definition_id: 3587
# is_appended: False
# is_declared: False
# name: W6X15