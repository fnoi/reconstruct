import os
from cProfile import Profile
from datetime import datetime

import bpy
import bmesh
import bonsai.tool as tool
import ifcopenshell
import json
import mathutils

from mathutils import Vector
import bonsai.bim.import_ifc
from bonsai.bim.ifc import IfcStore


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

def profile_def(bone, model):
    params = bone['beam_params']
    profile_name = f"{params['label']}"

    profile = model.create_entity(
        "IfcIShapeProfileDef",
        ProfileName=profile_name,
        ProfileType="AREA",
        OverallWidth=params['bf'],
        OverallDepth=params['d'],
        WebThickness=params['tw'],
        FlangeThickness=params['tf'],
        FilletRadius=0.005  # default value
    )

    profile_id = profile.id()
    bone['profile_id'] = profile_id
    bone['profile_name'] = profile_name

    return profile


def create_profile_definition(model, bone):
    params = bone['beam_params']
    profile_name = f"{params['label']}"
    profile = model.create_entity(
        "IfcIShapeProfileDef",
        ProfileName=profile_name,
        ProfileType="AREA",
        OverallWidth=float(params['bf']),
        OverallDepth=float(params['d']),
        WebThickness=float(params['tw']),
        FlangeThickness=float(params['tf']),
        FilletRadius=0.005  # default value, adjust if needed
    )
    return profile, profile_name


def create_custom_profile(model, bone):
    beam_verts = bone['beam_verts']
    profile_name = f"CustomProfile_{bone['beam_params']['label']}"

    # Create IfcCartesianPoints for the profile
    ifc_points = [model.create_entity("IfcCartesianPoint", Coordinates=(float(v[0]), float(v[1]))) for v in beam_verts]

    # Ensure the profile is closed by adding the first point at the end if necessary
    if ifc_points[0] != ifc_points[-1]:
        ifc_points.append(ifc_points[0])

    # Create IfcPolyline from the points
    poly_curve = model.create_entity("IfcPolyline", Points=ifc_points)

    # Create IfcArbitraryClosedProfileDef
    profile = model.create_entity(
        "IfcArbitraryClosedProfileDef",
        ProfileType="AREA",
        ProfileName=profile_name,
        OuterCurve=poly_curve
    )

    return profile, profile_name


def beam_placement(bone, bone_id, model):
    # Create custom profile definition
    custom_profile, custom_profile_name = create_custom_profile(model, bone)

    # Create IShapeProfileDef
    params = bone['beam_params']
    ishape_profile_name = f"IProfile_{params['label']}"
    ishape_profile = model.create_entity(
        "IfcIShapeProfileDef",
        ProfileName=ishape_profile_name,
        ProfileType="AREA",
        OverallWidth=float(params['bf']),
        OverallDepth=float(params['d']),
        WebThickness=float(params['tw']),
        FlangeThickness=float(params['tf']),
        FilletRadius=0.005  # default value, adjust if needed
    )

    # Create beam type and instance in IFC model
    beam_type = model.create_entity(
        "IfcBeamType",
        Name=f"BeamType_{bone_id}",
        PredefinedType="BEAM"
    )

    beam_instance = model.create_entity(
        "IfcBeam",
        GlobalId=ifcopenshell.guid.new(),
        Name=f"Beam_{bone_id}",
        PredefinedType="BEAM"
    )

    # Assign both profiles to beam type
    model.create_entity(
        "IfcRelDefinesByProperties",
        RelatedObjects=[beam_type],
        RelatingPropertyDefinition=custom_profile
    )
    model.create_entity(
        "IfcRelDefinesByProperties",
        RelatedObjects=[beam_type],
        RelatingPropertyDefinition=ishape_profile
    )

    # Assign beam type to beam instance
    model.create_entity(
        "IfcRelDefinesByType",
        RelatedObjects=[beam_instance],
        RelatingType=beam_type
    )

    # Calculate beam length and direction
    start = mathutils.Vector(bone['start'])
    end = mathutils.Vector(bone['end'])
    length = (end - start).length
    direction = (end - start).normalized()

    # Create extrusion
    extrusion_direction = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))

    # Create placement for the extrusion
    extrusion_placement = model.create_entity(
        "IfcAxis2Placement3D",
        Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
        Axis=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    )

    extruded_area_solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=custom_profile,  # Use custom_profile here
        Position=extrusion_placement,
        ExtrudedDirection=extrusion_direction,
        Depth=float(length)
    )

    # Create shape representation
    shape_representation = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=model.by_type("IfcGeometricRepresentationContext")[0],
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[extruded_area_solid]
    )

    # Assign geometry to beam instance
    product_definition_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_representation]
    )
    beam_instance.Representation = product_definition_shape

    # Create and assign IfcLocalPlacement
    beam_axis2placement3d = model.create_entity(
        "IfcAxis2Placement3D",
        Location=model.create_entity("IfcCartesianPoint", Coordinates=tuple(map(float, start))),
        Axis=model.create_entity("IfcDirection", DirectionRatios=tuple(direction)),
        RefDirection=model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    )
    local_placement = model.create_entity(
        "IfcLocalPlacement",
        RelativePlacement=beam_axis2placement3d
    )
    beam_instance.ObjectPlacement = local_placement

    # Create Blender object for visualization
    mesh = bpy.data.meshes.new(name=f"Mesh_Beam_{bone_id}")
    obj = bpy.data.objects.new(f"Beam_{bone_id}", mesh)
    bpy.context.collection.objects.link(obj)

    # Create beam geometry in Blender
    bm = bmesh.new()
    for vert in bone['beam_verts']:
        bm.verts.new((float(vert[0]), float(vert[1]), 0))
    bm.faces.new(bm.verts)
    bmesh.ops.extrude_face_region(bm, geom=bm.faces[:] + bm.edges[:] + bm.verts[:])
    bmesh.ops.translate(bm, vec=(0, 0, length), verts=[v for v in bm.verts if v.co.z > 0])
    bm.to_mesh(mesh)
    bm.free()

    # Set object location and rotation
    obj.location = start
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z', 'Y')

    # Assign IFC properties to Blender object
    obj.BIMObjectProperties.ifc_definition_id = beam_instance.id()

    # Store beam_verts, beam_params, and profile IDs as custom properties
    obj['beam_verts'] = bone['beam_verts']
    obj['beam_params'] = bone['beam_params']
    obj['custom_profile_id'] = custom_profile.id()
    obj['ishape_profile_id'] = ishape_profile.id()

    print(f'Created beam {bone_id} with custom profile {custom_profile_name} and I-shape profile {ishape_profile_name}')

    return obj


def triangulate_object(obj):
    # Triangulate the mesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    bm.to_mesh(me)
    me.update()
    bm.free()


def export_beam_to_obj(obj, export_folder):
    # Triangulate the mesh
    triangulate_object(obj)

    # Export the object as OBJ
    obj_file_path = os.path.join(export_folder, f"{obj.name}.obj")

    # Create a temporary mesh for exporting
    temp_mesh = obj.data.copy()
    temp_mesh.transform(obj.matrix_world)

    # Write OBJ file
    with open(obj_file_path, 'w') as f:
        f.write(f"# OBJ file: {obj.name}\n")
        for v in temp_mesh.vertices:
            f.write(f"v {v.co.x:.6f} {v.co.y:.6f} {v.co.z:.6f}\n")
        for p in temp_mesh.polygons:
            f.write(f"f")
            for idx in p.vertices:
                f.write(f" {idx + 1}")
            f.write("\n")

    # Remove the temporary mesh
    bpy.data.meshes.remove(temp_mesh)

    print(f"Exported {obj.name} to {obj_file_path}")


def export_to_ifc(model):
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"beam_project_{timestamp}"
    filepath = os.path.join("/Users/fnoic/Downloads", filename + ".ifc")

    # Create folder for OBJ files
    obj_folder = os.path.join("/Users/fnoic/Downloads", filename)
    os.makedirs(obj_folder, exist_ok=True)

    # Create a new IFC file
    ifc_file = ifcopenshell.file(schema="IFC4")

    # Create IfcProject
    project = ifc_file.create_entity("IfcProject", Name="BeamProject")

    # Set up default units
    unit_assignment = ifc_file.create_entity("IfcUnitAssignment")
    units = []
    for unit_type, unit_name in [("LENGTHUNIT", "METRE"), ("AREAUNIT", "SQUARE_METRE"), ("VOLUMEUNIT", "CUBIC_METRE")]:
        unit = ifc_file.create_entity("IfcSIUnit", UnitType=unit_type, Name=unit_name)
        units.append(unit)
    unit_assignment.Units = units
    project.UnitsInContext = unit_assignment

    # Create IfcSite and IfcBuilding (required for a valid IFC structure)
    site = ifc_file.create_entity("IfcSite", Name="Site")
    building = ifc_file.create_entity("IfcBuilding", Name="Building")

    # Set up spatial structure
    ifc_file.create_entity("IfcRelAggregates", RelatingObject=project, RelatedObjects=[site])
    ifc_file.create_entity("IfcRelAggregates", RelatingObject=site, RelatedObjects=[building])

    # Dictionary to store profile definitions
    custom_profile_defs = {}
    ishape_profile_defs = {}

    # Add all objects from the original model to the new IFC file
    for element in model.by_type("IfcBeam"):
        blender_obj = bpy.data.objects.get(element.Name)
        if not blender_obj:
            print(f"Warning: Blender object not found for beam {element.Name}")
            continue

        beam_verts = blender_obj.get('beam_verts', [])
        beam_params = blender_obj.get('beam_params', {})
        custom_profile_id = blender_obj.get('custom_profile_id')
        ishape_profile_id = blender_obj.get('ishape_profile_id')

        if not beam_verts or not beam_params or not custom_profile_id or not ishape_profile_id:
            print(f"Warning: Required data not found for beam {element.Name}")
            continue

        # Create IfcArbitraryClosedProfileDef if it doesn't exist
        custom_profile_name = f"CustomProfile_{beam_params['label']}"
        if custom_profile_name not in custom_profile_defs:
            ifc_points = [ifc_file.create_entity("IfcCartesianPoint", Coordinates=(float(v[0]), float(v[1]))) for v in beam_verts]
            if ifc_points[0] != ifc_points[-1]:
                ifc_points.append(ifc_points[0])
            poly_curve = ifc_file.create_entity("IfcPolyline", Points=ifc_points)
            custom_profile = ifc_file.create_entity(
                "IfcArbitraryClosedProfileDef",
                ProfileType="AREA",
                ProfileName=custom_profile_name,
                OuterCurve=poly_curve
            )
            custom_profile_defs[custom_profile_name] = custom_profile
        else:
            custom_profile = custom_profile_defs[custom_profile_name]

        # Create IShapeProfileDef if it doesn't exist
        ishape_profile_name = f"IProfile_{beam_params['label']}"
        if ishape_profile_name not in ishape_profile_defs:
            ishape_profile = ifc_file.create_entity(
                "IfcIShapeProfileDef",
                ProfileName=ishape_profile_name,
                ProfileType="AREA",
                OverallWidth=float(beam_params['bf']),
                OverallDepth=float(beam_params['d']),
                WebThickness=float(beam_params['tw']),
                FlangeThickness=float(beam_params['tf']),
                FilletRadius=0.0  # Adjust if needed
            )
            ishape_profile_defs[ishape_profile_name] = ishape_profile
        else:
            ishape_profile = ishape_profile_defs[ishape_profile_name]

        # Create new beam element
        new_element = ifc_file.create_entity(
            "IfcBeam",
            GlobalId=ifcopenshell.guid.new(),
            Name=element.Name,
            ObjectType="BEAM",
            ObjectPlacement=ifc_file.add(element.ObjectPlacement),
            Representation=ifc_file.add(element.Representation)
        )

        # Assign both profiles to beam
        ifc_file.create_entity(
            "IfcRelDefinesByProperties",
            RelatedObjects=[new_element],
            RelatingPropertyDefinition=custom_profile
        )
        ifc_file.create_entity(
            "IfcRelDefinesByProperties",
            RelatedObjects=[new_element],
            RelatingPropertyDefinition=ishape_profile
        )

        # Add beam to building
        ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            RelatingStructure=building,
            RelatedElements=[new_element]
        )

    # Write the IFC file
    ifc_file.write(filepath)
    print(f"IFC file exported to: {filepath}")

    # Export individual OBJ files for each beam
    for obj in bpy.data.objects:
        if obj.name.startswith("Beam_"):
            export_beam_to_obj(obj, obj_folder)

    return filepath

def blender_beams():
    # load data
    skeleton, config = data_loader()
    # set up ifc project
    model = project_setup()

    # bone by bone
    for bone_id, bone in skeleton.items():
        beam_obj = beam_placement(bone, bone_id, model)

    print("All beams created successfully.")

    filepath = export_to_ifc(model)

    # write filepath to a txt file
    with open("/Users/fnoic/Downloads/exported_ifc.txt", "w") as f:
        f.write(filepath)


if __name__ == "__main__":
    blender_beams()