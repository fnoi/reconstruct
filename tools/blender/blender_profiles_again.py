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

            # add directly in the loop to avoid issues (lol does not actually avoid issues)
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            bpy.ops.bim.enable_add_type()
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'
            bpy.context.scene.BIMModelProperties.type_template = 'PROFILESET'
            # bpy.context.scene.BIMModelProperties.type_name = f'beamprofiletype_one_for_all'
            profile_type_name = f'beamprofiletype_{element.name}'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")

            bpy.ops.bim.add_type()
            bpy.ops.bim.disable_add_type()
            print(f'added profile type {profile_type_name }')

    print('rawww')
    objectname = 'IfcBeamType/beamprofiletype_W6X12'

    obj = bpy.data.objects[objectname]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.bim.enable_editing_assigned_material()
    bpy.ops.bim.enable_editing_material_set_item(material_set_item=76)
    bpy.context.scene.BIMMaterialProperties.profiles = '72'
    bpy.ops.bim.disable_editing_material_set_item(obj="IfcBeamType/beamprofiletype_W6X12")
    bpy.ops.bim.edit_material_set_item(material_set_item=76)
    bpy.context.scene.BIMModelProperties.relating_type_id = '73'
    bpy.ops.bim.add_constr_type_instance()
    bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
    bpy.ops.bim.disable_editing_assigned_material(obj="IfcBeamType/beamprofiletype_W6X12")




if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }

    blender_beams(profiles)
