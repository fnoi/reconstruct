import math

import ifcopenshell
import ifcopenshell.api.root
import ifcopenshell.geom as geom
import pandas as pd

import uuid
import string

from omegaconf import OmegaConf

from tools.IO import data_from_IFC, config_io

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
import open3d as o3d
import numpy as np

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IMeshTools import IMeshTools_Parameters




def newGUID():
    chars = string.digits + string.ascii_uppercase + string.ascii_lowercase + "_$"

    g = uuid.uuid4().hex
    bs = [int(g[i : i + 2], 16) for i in range(0, len(g), 2)]

    def b64(v, l=4):
        return "".join([chars[(v // (64**i)) % 64] for i in range(l)][::-1])

    return "".join([b64(bs[0], 2)] + [b64((bs[i] << 16) + (bs[i + 1] << 8) + bs[i + 2]) for i in range(1, 16, 3)])


# print(newGUID())


def data_preprocessor(skeleton):
    profiles = {}
    # iterate over columns in skeleton dataframe
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
    print(f'working with {len(profiles)} profiles')
    return profiles

def get_profile(profile_name, profiles):
    iprofile = [i for i in profiles if i.ProfileName in [profile_name]]
    if len(iprofile) == 0:
        raise ValueError (f"Profile {profile_name} not found")
    return iprofile[0]

def model_builder(skeleton, config):
    settings = geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    # settings.set(settings.SEW_SHELLS, True)
    profile_dict = data_preprocessor(skeleton)
    model = ifcopenshell.file(schema="IFC4")

    # Create units and context
    units = model.createIfcUnitAssignment([model.createIfcSIUnit(None, "LENGTHUNIT", "MILLI", "METRE")])
    origin = model.createIfcAxis2Placement3D(model.createIfcCartesianPoint((0.0, 0.0, 0.0)), model.createIfcDirection((0.0, 0.0, 1.0)), model.createIfcDirection((1.0, 0.0, 0.0)))
    context = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1.0e-05, origin)

    origin_local = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
    gcf = model.createIfcAxis2Placement3D(origin_local, None, None)
    ctx = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, gcf, None)

    profiles = data_from_IFC(config.cs_fit.ifc_cs_path, direct=True)

    zDir = model.createIfcDirection((0.0, 0.0, 1.0))
    xDir = model.createIfcDirection((1.0, 0.0, 0.0))
    frame = model.createIfcAxis2Placement3D(origin_local, zDir, xDir)
    placement_local = model.createIfcLocalPlacement(None, frame)

    scale_fac = 1e3

    # for key value pair in profile_dict
    for bone_name, profile_properties in profile_dict.items():
        rot = profile_properties["rot_mat"]
        # fourth column * scale_fac
        rot = rot.T
        rot[:, 3] = rot[:, 3] * scale_fac
        # Convert numpy values to float
        origin_global = model.createIfcCartesianPoint([0.0, 0.0, 0.0])
        z_axis_global = model.createIfcDirection([0.0, 0.0, 1.0])
        x_axis_global = model.createIfcDirection([1.0, 0.0, 0.0])
        placement_global = model.createIfcLocalPlacement(
            None, model.createIfcAxis2Placement3D(origin_global, z_axis_global, x_axis_global))

        # origin_local = model.createIfcCartesianPoint([float(x) for x in rot[:3, 3]])
        # z_axis_local = model.createIfcDirection([float(x) for x in rot[:3, 2]])
        # x_axis_local = model.createIfcDirection([float(x) for x in rot[:3, 0]])
        # placement_local = model.createIfcLocalPlacement(
        #     None, model.createIfcAxis2Placement3D(origin_local, z_axis_local, x_axis_local))

        profile_name = profile_properties['cstype']
        ifc_profile = model.add(get_profile(profile_name, profiles))
        body = model.createIfcExtrudedAreaSolid(ifc_profile, None, z_axis_global, profile_properties['length'] * scale_fac)
        bodyRep = model.createIfcShapeRepresentation(ctx, 'Body', 'SweptSolid', [body])
        prdDefShape = model.createIfcProductDefinitionShape(None, None, (bodyRep,))

        guid_ = newGUID()
        print(f'guid: {guid_}')

        beam = model.createIfcBeam(guid_, None, bone_name, None, None, placement_global, prdDefShape, None, None)
        ifcopenshell.api.geometry.edit_object_placement(model, product=beam, matrix=rot)

        shape = geom.create_shape(settings, beam).geometry

        params = IMeshTools_Parameters()
        params.Deflection = 0.01
        params.Angle = 0.1
        params.Relative = False
        params.InParallel = True
        params.MinSize = 0.01
        params.InternalVerticesMode = True
        params.ControlSurfaceDeflection = True

        mesh = BRepMesh_IncrementalMesh(shape, params)
        mesh.Perform()

        # Export directly to STL
        writer = StlAPI_Writer()
        writer.Write(shape, f"/Users/fnoic/Downloads/{bone_name}.stl")


    model.write("/Users/fnoic/Downloads/model.ifc")



if __name__ == '__main__':
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
        skeleton = pd.read_json(f)
    config = OmegaConf.load('/Users/fnoic/PycharmProjects/reconstruct/config_0.yaml')
    config = config_io(config)

    model_builder(skeleton, config)
