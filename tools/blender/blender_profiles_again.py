import bpy

import bonsai.tool as tool



def get_attribute_value(attr_value):
    if isinstance(attr_value, ifcopenshell.entity_instance):
        return f"{attr_value.is_a()}:{attr_value.id()}"
    elif isinstance(attr_value, (list, tuple)):
        return [get_attribute_value(v) for v in attr_value]
    else:
        return attr_value


def inspect_object(obj_name):
    obj = None
    for o in bpy.data.objects:
        if o.name.endswith(obj_name.split('/')[-1]):
            obj = o
            break

    if obj is None:
        print(f"Object ending with '{obj_name.split('/')[-1]}' not found in the scene.")
        print("Available objects:")
        for o in bpy.data.objects:
            print(f"  - {o.name} (Type: {o.type})")
        return None

    print(f"Inspecting object: {obj.name}")
    print(f"  Type: {obj.type}")
    print(f"  Data: {'Present' if obj.data else 'None'}")
    print(f"  Material slots: {len(obj.material_slots)}")
    if hasattr(obj, 'BIMObjectProperties'):
        print(f"  IFC class: {obj.BIMObjectProperties.ifc_definition_id}")
    else:
        print("  No BIMObjectProperties found")
    return obj


def ensure_material(obj):
    if obj.type != 'MESH':
        print(f"Warning: Object '{obj.name}' is not a mesh object. Attempting to convert...")
        try:
            bpy.ops.bim.add_representation(obj=obj.name)
            print(f"Conversion successful. New type: {obj.type}")
        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

    if not obj.data:
        print(f"Warning: Object '{obj.name}' has no mesh data after conversion attempt.")
        return False

    if not obj.material_slots:
        mat = bpy.data.materials.new(name=f"Material_{obj.name}")
        obj.data.materials.append(mat)
        print(f"Added new material to {obj.name}")
    return True


def assign_profile(obj, profile_name):
    if not ensure_material(obj):
        return

    try:
        bpy.ops.bim.add_material_set_item(obj=obj.name, set_type="PROFILE")
        print(f"Added material set item to {obj.name}")
    except Exception as e:
        print(f"Error adding material set item: {e}")
        return

    try:
        bpy.ops.bim.enable_editing_assigned_material(obj=obj.name)
        bpy.ops.bim.enable_editing_material_set_item(obj=obj.name, material_set_item=0)
        bpy.context.scene.BIMMaterialProperties.profiles = profile_name
        bpy.ops.bim.edit_material_set_item(obj=obj.name, material_set_item=0)
        bpy.ops.bim.disable_editing_material_set_item(obj=obj.name)
        bpy.ops.bim.disable_editing_assigned_material(obj=obj.name)
        print(f"Successfully assigned profile {profile_name} to {obj.name}")
    except Exception as e:
        print(f"Error assigning profile: {e}")


def get_beam_ids(obj):
    if not obj.material_slots:
        print(f"Warning: Object '{obj.name}' has no material slots.")
        return None, None, None

    if not obj.material_slots[0].material:
        print(f"Warning: Object '{obj.name}' has an empty first material slot.")
        return None, None, None

    material = obj.material_slots[0].material

    if not hasattr(material, 'BIMMaterialProperties'):
        print(f"Warning: Material '{material.name}' has no BIMMaterialProperties.")
        return None, None, None

    bim_props = material.BIMMaterialProperties

    if not bim_props.material_set_items:
        print(f"Warning: BIMMaterialProperties has no material_set_items.")
        return None, None, None

    material_set_item_id = bim_props.material_set_items[0].id

    if not bim_props.profile_set_items:
        print(f"Warning: BIMMaterialProperties has no profile_set_items.")
        profile_id = None
    else:
        profile_id = bim_props.profile_set_items[0].id

    relating_type_id = obj.BIMObjectProperties.relating_type_id if hasattr(obj, 'BIMObjectProperties') else None

    return material_set_item_id, profile_id, relating_type_id



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
        if element.name in query_profile_dict:
            print(f"\nProcessing element: {element.name}")

            profile_type_name = f'beamprofiletype_{element.name}'
            bpy.context.scene.BIMModelProperties.type_name = profile_type_name
            bpy.context.scene.BIMModelProperties.type_class = 'IfcBeamType'  # Ensure we're creating a beam type
            bpy.ops.bim.add_type()
            print(f'Added profile type {profile_type_name}')

            obj_name = f"IfcBeamType/{profile_type_name}"
            obj = inspect_object(obj_name)

            if obj:
                assign_profile(obj, element.name)
                try:
                    bpy.ops.bim.add_constr_type_instance()
                    print(f"Added construction type instance for {obj.name}")
                except Exception as e:
                    print(f"Error adding construction type instance: {e}")
                bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            else:
                print(f"Skipping profile assignment for {obj_name} due to missing object.")

            print('Profile assignment completed')
            material_set_item_id, profile_id, relating_type_id = get_beam_ids(obj)

            if material_set_item_id is None:
                print(f"Skipping profile assignment for {obj_name} due to missing data.")
                continue

            print(f'obj_name: {obj_name}, material_set_item_id: {material_set_item_id}, profile_id: {profile_id}, relating_type_id: {relating_type_id}')

            bpy.ops.bim.enable_editing_assigned_material()
            bpy.ops.bim.enable_editing_material_set_item(material_set_item=material_set_item_id)
            bpy.context.scene.BIMMaterialProperties.profiles = str(profile_id)
            bpy.ops.bim.disable_editing_material_set_item(obj=obj_name)
            bpy.ops.bim.edit_material_set_item(material_set_item=material_set_item_id)
            bpy.context.scene.BIMModelProperties.relating_type_id = str(relating_type_id)
            bpy.ops.bim.add_constr_type_instance()
            bpy.ops.bim.load_type_thumbnails(ifc_class="IfcBeamType")
            bpy.ops.bim.disable_editing_assigned_material(obj=obj_name)


if __name__ == "__main__":
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }

    blender_beams(profiles)
