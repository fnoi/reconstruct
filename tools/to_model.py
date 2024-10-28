import math
import ifcopenshell
import ifcopenshell.api.root
import ifcopenshell.geom as geom
import pandas as pd
import uuid
import string
import numpy as np
from omegaconf import OmegaConf
from tools.IO import data_from_IFC, config_io
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IMeshTools import IMeshTools_Parameters


def newGUID():
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase + "_$"
    g = uuid.uuid4().hex
    bs = [int(g[i: i + 2], 16) for i in range(0, len(g), 2)]

    def b64(v, l=4):
        return "".join([chars[(v // (64 ** i)) % 64] for i in range(l)][::-1])

    return "".join([b64(bs[0], 2)] + [b64((bs[i] << 16) + (bs[i + 1] << 8) + bs[i + 2]) for i in range(1, 16, 3)])


def data_preprocessor(skeleton):
    profiles = {}
    for bone, data in skeleton.items():
        start = data['start']
        end = data['end']
        length = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))

        profiles[data.name] = {
            'cstype': data.cstype,
            'length': length,
            'rot_mat': np.array(data.rot_mat),
            'start': start,
            'end': end
        }
    return profiles


def get_profile(profile_name, profiles):
    iprofile = [i for i in profiles if i.ProfileName in [profile_name]]
    if not iprofile:
        raise ValueError(f"Profile {profile_name} not found")
    return iprofile[0]


def create_mesh_params():
    params = IMeshTools_Parameters()
    params.Deflection = 0.01
    params.Angle = 0.1
    params.Relative = False
    params.InParallel = True
    params.MinSize = 0.01
    params.InternalVerticesMode = True
    params.ControlSurfaceDeflection = True
    return params


def model_builder(skeleton, config, output_dir, scale_factor=1e3):
    settings = geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)

    profile_dict = data_preprocessor(skeleton)
    model = ifcopenshell.file(schema="IFC4")

    # Setup IFC context
    units = model.createIfcUnitAssignment([model.createIfcSIUnit(None, "LENGTHUNIT", "MILLI", "METRE")])
    origin = model.createIfcAxis2Placement3D(
        model.createIfcCartesianPoint((0.0, 0.0, 0.0)),
        model.createIfcDirection((0.0, 0.0, 1.0)),
        model.createIfcDirection((1.0, 0.0, 0.0))
    )
    context = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1.0e-05, origin)
    ctx = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, origin)

    # Load profiles
    profiles = data_from_IFC(config.cs_fit.ifc_cs_path, direct=True)

    # Global placement
    z_axis_global = model.createIfcDirection([0.0, 0.0, 1.0])
    x_axis_global = model.createIfcDirection([1.0, 0.0, 0.0])
    origin_global = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
    placement_global = model.createIfcLocalPlacement(
        None,
        model.createIfcAxis2Placement3D(origin_global, z_axis_global, x_axis_global)
    )

    mesh_params = create_mesh_params()

    for bone_name, profile_properties in profile_dict.items():
        # Transform coordinates
        rot = profile_properties["rot_mat"]
        length = profile_properties["length"] * scale_factor
        start = np.array(profile_properties["start"]) * scale_factor
        end = np.array(profile_properties["end"]) * scale_factor

        # Setup transformation matrix
        rotation = rot[:3, :3]
        translation = rot[:3, 3] * scale_factor
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -translation + start

        # Validate transformation
        test_point = np.array([0, 0, length, 1])
        test_transformed = np.dot(transform, test_point)[:3]
        assert np.allclose(end, test_transformed, rtol=1e-5), f"Transform validation failed for {bone_name}"

        # Create beam
        profile_name = profile_properties['cstype']
        ifc_profile = model.add(get_profile(profile_name, profiles))
        body = model.createIfcExtrudedAreaSolid(ifc_profile, None, z_axis_global, length)
        bodyRep = model.createIfcShapeRepresentation(ctx, 'Body', 'SweptSolid', [body])
        prdDefShape = model.createIfcProductDefinitionShape(None, None, (bodyRep,))

        beam = model.createIfcBeam(newGUID(), None, bone_name, None, None,
                                   placement_global, prdDefShape, None, None)
        ifcopenshell.api.geometry.edit_object_placement(model, product=beam, matrix=transform)

        # Generate mesh and export STL
        shape = geom.create_shape(settings, beam).geometry
        mesh = BRepMesh_IncrementalMesh(shape, mesh_params)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.Write(shape, f"{output_dir}/{bone_name}.stl")

    # Export IFC model
    model.write(f"{output_dir}/model.ifc")


if __name__ == '__main__':
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
        skeleton = pd.read_json(f)
    config = OmegaConf.load('/Users/fnoic/PycharmProjects/reconstruct/config_0.yaml')
    config = config_io(config)

    output_dir = '/Users/fnoic//Downloads/'

    model_builder(skeleton, config, output_dir)