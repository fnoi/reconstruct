import bpy
import json


def data_loader():
    path_local = '/Users/fnoic/PycharmProjects/reconstruct/'
    path_config = 'config_reconstruct_a.json'
    config = json.load(open(path_local + path_config, 'r'))

    with open(path_local + config['path_skeleton'], 'rb') as f:
        skeleton_dict = json.load(f)

    print(f'loaded skeleton data and reconstruct config')

    return skeleton_dict, config


def blender_beams():
    skeleton_dict, config = data_loader()
    print(skeleton_dict['bone_0']['beam_params'])
    raise ValueError("This is a test exception")



    bpy.ops.bim.create_project()
    bpy.ops.bim.set_tab(tab="OBJECT")




    bpy.ops.bim.load_profiles()
    bpy.context.scene.BIMProfileProperties.profile_classes = 'IfcIShapeProfileDef'
    bpy.ops.bim.enable_editing_profile(profile=66)
    bpy.ops.bim.enable_editing_profile()
    bpy.context.scene.BIMProfileProperties.profile_attributes[1].string_value = "dummy_profile"
    bpy.ops.bim.enable_editing_profile(profile=0)
    bpy.ops.bim.enable_editing_profile(profile=1)
    bpy.context.scene.BIMProfileProperties.profile_attributes[1].string_value = "dummy_profile"
    bpy.context.scene.BIMProfileProperties.profile_attributes[2].float_value = 0.2
    bpy.context.scene.BIMProfileProperties.profile_attributes[3].float_value = 0.02
    bpy.context.scene.BIMProfileProperties.profile_attributes[4].float_value = 0.02
    bpy.ops.bim.disable_editing_arbitrary_profile()
    bpy.ops.bim.disable_editing_profile()
    bpy.context.scene.BIMProfileProperties.active_profile_index = 0
    bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
    bpy.ops.bim.enable_add_type()

    # store as ifc
    export_name = "/Users/fnoic/Downloads/scripting_ifc_test_0.ifc"
    bpy.ops.bim.save_project(filepath=export_name, should_save_as=False, save_as_invoked=True)




    bpy.context.scene.BIMProjectProperties.template_file = 'IFC4 Demo Template.ifc'
    bpy.ops.bim.select_library_file(
        filepath="/Users/fnoic/Library/Application Support/Blender/4.2/extensions/.local/lib/python3.11/site-packages/bonsai/bim/data/templates/projects/IFC4 Demo Template.ifc")
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType", limit=9, offset=0)
    bpy.ops.bim.create_project()
    bpy.ops.outliner.item_activate(deselect_all=True)
    # bpy.context. = False
    # bpy.context. = True
    bpy.ops.bim.load_profiles()
    bpy.ops.bim.edit_profiles()
    bpy.context.scene.BIMProfileProperties.profile_attributes[1].string_value = "meins"
    bpy.context.scene.BIMProfileProperties.profile_attributes[2].float_value = 0.5
    bpy.context.scene.BIMProfileProperties.profile_attributes[3].float_value = 0.2
    bpy.context.scene.BIMProfileProperties.profile_attributes[4].float_value = 0.01
    bpy.context.scene.BIMProfileProperties.profile_attributes[5].float_value = 0.005
    bpy.context.scene.BIMProfileProperties.profile_attributes[6].float_value = 0.01
    bpy.context.scene.BIMProfileProperties.profile_attributes[7].float_value = 0.01
    bpy.context.scene.BIMProfileProperties.profile_attributes[8].float_value = 0.01
    bpy.ops.bim.disable_editing_arbitrary_profile()
    bpy.ops.bim.disable_editing_profile()
    bpy.ops.bim.load_profiles()

    raise ValueError("This is a test exception")



if __name__ == "__main__":
    blender_beams()