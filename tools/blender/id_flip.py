import math
import pandas as pd
import numpy as np
import bmesh
import mathutils
# from omegaconf import OmegaConf

try:
    import bonsai.tool as tool
    import bpy
    import numpy as np
    import pandas as pd
except ImportError:
    pass


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


def retrieve_relating_type_id(profile_type_name, model):
    for beam_type in model.by_type('IfcBeamType'):
        if beam_type.Name == profile_type_name:
            return beam_type.id()
    raise ValueError(f'Beam type {profile_type_name} not found in model')


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
    # config = OmegaConf.load('config_0.yaml')
    # cs_path = config.cs_fit.ifc_cs_path
    ### hard fix
    cs_path = '/Users/fnoic/PycharmProjects/IfcOpenShell/src/bonsai/bonsai/bim/data/libraries/IFC4 EU Steel.ifc'
    cs_library = cs_path.split('/')[-1]
    bpy.context.scene.BIMProjectProperties.library_file = cs_library
    bpy.ops.bim.select_library_file(filepath=cs_path)
    bpy.ops.bim.change_library_element(element_name="IfcIShapeProfileDef")

    query_profile_names = []
    for _key, _value in query_profile_dict.items():
        query_profile_names.append(_value['cstype'])
    query_profile_names = list(set(query_profile_names))
    # query_profile_names = query_profile_dict.keys()
    print(f'inspecting types: {query_profile_names}')

    ref_dict = {} # dict to store reference data between loops

    # import the necessary profiles from library to project
    print('\n--\n1.')
    all_cs = []
    for index, element in enumerate(bpy.context.scene.BIMProjectProperties.library_elements):
        all_cs.append(element.name)
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
            # create beam type
            profile_type_name = f'T_{element.name}'
            # replace . with _ in profile type name to avoid blender errors
            profile_type_name = profile_type_name.replace('.', '_')
            ref_dict[element.name]['profile_type_name'] = profile_type_name
            bpy.ops.bim.enable_add_type()
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
            bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name

            material_name = f'M_{element.name}'

            print(f':::creating beam type: {profile_type_name} and material: {material_name}')
            bpy.ops.bim.add_type()
            bpy.ops.bim.disable_add_type()
            ref_dict[element.name]['profile_type_name'] = profile_type_name
            id_0, id_1 = retrieve_id_magic(model)
            ref_dict[element.name]['id_0'] = id_0
            ref_dict[element.name]['id_1'] = id_1
            ref_dict[element.name]['relating_type_id'] = retrieve_relating_type_id(profile_type_name, model)

            bpy.ops.bim.add_material(name=material_name)
            material_id = retrieve_material_id(model)
            ref_dict[element.name]['material_id'] = material_id
            print(f':::created material: {material_name}, id: {material_id}')

            # link beam type to material and profile
            bpy.ops.object.select_all(action='DESELECT')
            obj = bpy.data.objects.get(f"IfcBeamType/{profile_type_name}")

            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.bim.enable_editing_assigned_material()
            bpy.ops.bim.enable_editing_material_set_item(material_set_item=id_1)
            bpy.data.objects[f"IfcBeamType/{profile_type_name}"].BIMObjectMaterialProperties.material_set_item_material = str(material_id)
            bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
            bpy.ops.bim.disable_editing_material_set_item(obj=f"IfcBeamType/{profile_type_name}")
            bpy.ops.bim.edit_material_set_item(material_set_item=id_1)
            bpy.ops.bim.disable_editing_material_set_item()
            bpy.ops.bim.disable_editing_assigned_material(obj=f"IfcBeamType/{profile_type_name}")

            # print_out_material_infos(model)
            # print_materials_info(model)
            print(f':::material set id: {id_0}, material profile id: {id_1}')

    print('\n--\n2.')
    beam_iter = 0
    for beam_name, beam_data in query_profile_dict.items():
        profile_type_data = ref_dict[beam_data['cstype']]
        type_obj_name = f"IfcBeamType/{profile_type_data['profile_type_name']}"
        beam_obj_name = f"Beam_{beam_iter}_{beam_data['cstype']}"
        print(f'\n:::creating beam {beam_name} -> {beam_obj_name}')

        bpy.ops.object.select_all(action='DESELECT')
        obj = bpy.data.objects.get(type_obj_name)
        obj.select_set(True)

        bpy.context.scene.BIMModelProperties.extrusion_depth = beam_data['length']
        bpy.context.scene.BIMMaterialProperties.profiles = str(profile_type_data['profile_id'])
        bpy.context.scene.BIMModelProperties.relating_type_id = str(profile_type_data['relating_type_id'])

        bpy.ops.bim.add_constr_type_instance()

        obj = bpy.context.active_object
        obj.name = beam_obj_name

        # Set the beam object's origin to its geometric center
        # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # Set the beam's position to the world origin
        # obj.location = (0, 0, 0)

        # Apply rotation
        rot_mat = beam_data['rot_mat']
        obj.rotation_euler = (
            math.atan2(rot_mat[2][1], rot_mat[2][2]),
            math.atan2(-rot_mat[2][0], math.sqrt(rot_mat[2][1] ** 2 + rot_mat[2][2] ** 2)),
            math.atan2(rot_mat[1][0], rot_mat[0][0])
        )

        # Set the final position
        # center = [(a + b) / 2 for a, b in zip(beam_data['start'], beam_data['end'])]
        obj.location = beam_data['start']

        beam_iter += 1

        # export single object to fbx
        bpy.ops.object.select_all(action='DESELECT')
        obj = bpy.data.objects.get(beam_obj_name)
        obj.select_set(True)
        bpy.ops.export_scene.fbx(filepath=f'/Users/fnoic/PycharmProjects/reconstruct/data/parking/{beam_obj_name}.fbx', use_selection=True)

        bpy.ops.export_scene.fbx(
            filepath=f'/Users/fnoic/PycharmProjects/reconstruct/data/parking/beamies/{beam_iter}_{beam_obj_name}.fbx',
            use_selection=True,
            global_scale=1.0,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_ALL',
            use_space_transform=True,
            bake_space_transform=True,
            axis_forward='Y',  # -Y Forward
            axis_up='Z',  # Z Up
            use_mesh_modifiers=True,
            mesh_smooth_type='OFF',
            use_mesh_edges=False,
            use_tspace=False,
            use_custom_props=False,
            path_mode='AUTO',
            embed_textures=False,
            batch_mode='OFF',
            use_batch_own_dir=True,
            use_metadata=True
        )
        bpy.ops.wm.obj_export(filepath=f'/Users/fnoic/PycharmProjects/reconstruct/data/parking/beamies/{beam_iter}_{beam_obj_name}.obj', export_selected_objects=True)


    # export all objects to fbx
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(filepath='/Users/fnoic/PycharmProjects/reconstruct/data/parking/all_beams.fbx', use_selection=True)
    bpy.ops.wm.obj_export(filepath='/Users/fnoic/PycharmProjects/reconstruct/data/parking/all_beams.obj', export_selected_objects=True)


    # save ifc project
    bpy.ops.bim.save_project(filepath='/Users/fnoic/PycharmProjects/reconstruct/data/parking/all_beams.ifc', should_save_as=True, save_as_invoked=True)

    print(':::done')

    # # save all_cs to txt file  ## legacy, activate to save all_cs to txt file: use this to reduce profile list for MOO / CS fitting step
    # with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/all_cs.txt', 'w') as f:
    #     for cs in all_cs:
    #         f.write(f'{cs}\n')


if __name__ == "__main__":
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
        skeleton = pd.read_json(f)
    # print(skeleton)

    profiles = {}
    # iterate over columns in skeleton dataframe
    for bone, data in skeleton.items():
        start = data['start']
        end = data['end']
        length = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))

        rot_mat = np.asarray(data['rot_mat']).T
        # z_angle_add = data['angle_xy']
        # # define rotation matrix for z axis rotation
        # rot_mat_z = np.asarray([[np.cos(z_angle_add), -np.sin(z_angle_add), 0],
        #                         [np.sin(z_angle_add), np.cos(z_angle_add), 0],
        #                         [0, 0, 1]])
        # # multiply rotation matrices
        # rot_mat = np.dot(rot_mat.T, rot_mat_z)

        # apply reverse rotation to start and end
        # _start = np.dot(rot_mat.T, start)
        # __start = _start + start
        # ___start = _start + end
        #
        # _end = np.dot(rot_mat.T, end)
        # __end = _end + start
        # ___end = _end + end
        #
        # print(f'{start}, start')
        # print(f'{__start}, __start')
        # print(f'{___start}, ___start')
        #
        # print(f'{end}, end')
        # print(f'{__end}, __end')
        # print(f'{___end}, ___end')
        #
        #
        # #
        #
        # thresh = 0.01
        # print('bone name:', bone)
        # if np.linalg.norm(target - end) < thresh:
        #     print(f'A', np.linalg.norm(target - end))
        # if np.linalg.norm(target - start) < thresh:
        #     print('B', np.linalg.norm(target - start))
        #
        # # continue


        profiles[data.name] = {
            'cstype': data.cstype,
            'length': length,
            'rot_mat': rot_mat,
            'start': start,
            'end': end
        }
    print(f'working with {len(profiles)} profiles')

    # print(profiles)
    # raise ValueError('End of script')
    #
    #
    # profiles = {'W8X67':
    #                 {'length':1.5},
    #             'W6X12':
    #                 {'length':1.0},
    #             'HP12X53':
    #                 {'length':0.2}
    #             }
    # raise ValueError('End of script')
    blender_beams(profiles)