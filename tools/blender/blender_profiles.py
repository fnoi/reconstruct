import bpy
import bonsai.tool as tool
import ifcopenshell
import json


def data_loader():
    path_local = '/Users/fnoic/PycharmProjects/reconstruct/'
    path_config = 'config_reconstruct_a.json'
    config = json.load(open(path_local + path_config, 'r'))

    with open(path_local + config['path_skeleton'], 'rb') as f:
        skeleton_dict = json.load(f)

    print(f'loaded skeleton data and reconstruct config')

    return skeleton_dict, config

def project_setup():
    bpy.ops.bim.create_project()

    model = tool.Ifc.get()
    # ifcopenshell.api.root.create_entity(model, ifc_class="IfcProject")
    # ifcopenshell.api.unit.assign_unit(model)

    return model

def profile_def(skeleton_dict, model, config):
    for bone_id, bone in skeleton_dict.items():
        params = bone['beam_params']
        print(bone_id)

        profile = model.create_entity(
            "IfcIShapeProfileDef",
            ProfileName=params['label'], ProfileType="AREA",
            OverallWidth=params['bf'], OverallDepth=params['d'],
            WebThickness=params['tw'], FlangeThickness=params['tf'],
            FilletRadius=0.005  # default value
        )  # can we track naming?

        profile_id = profile.id()
        bone['profile_id'] = profile_id


def beam_placement(skeleton_dict, model, config):
    for bone_id, bone in skeleton_dict.items():
        profile_name = bone['beam_params']['label']
        profile_id = bone['profile_id']

        bpy.ops.bim.add_type()
        bpy.ops.bim.add_default_type(ifc_element_type="IfcBeamType")
        bpy.ops.bim.add_constr_type_instance()

        # Get the newly created object and set its name
        new_obj = bpy.context.active_object
        # split the int from the string
        bone_id_int = int(bone_id.split('_')[1])
        new_obj.name = f"Beam_{bone_id_int}"

        # Update the profile
        bpy.ops.bim.enable_editing_assigned_material()
        bpy.ops.bim.enable_editing_material_set_item(material_set_item=100)
        bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
        bpy.ops.bim.edit_material_set_item(material_set_item=100)
        bpy.ops.bim.disable_editing_material_set_item()
        bpy.ops.bim.disable_editing_assigned_material(obj=new_obj.name)

        print(f'fyi: this should be cross section profile {profile_name} with id {profile_id}')


        # perform transformation, if necessary by translating the rotation matrix to displacement values and euler angles


        # raise ValueError("This is a test exception")

def blender_beams():
    skeleton, config = data_loader()
    # set up ifc project
    model = project_setup()

    # define required profiles
    profile_def(skeleton, model, config)
    # create and place beams
    beam_placement(skeleton, model, config)


    raise ValueError("This is a test exception")



if __name__ == "__main__":
    blender_beams()