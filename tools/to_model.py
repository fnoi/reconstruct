import math

import ifcopenshell
import ifcopenshell.geom as geom
import numpy as np
import pandas as pd

import uuid
import string

from omegaconf import OmegaConf

from tools.IO import data_from_IFC, config_io


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
    origin = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
    gcf = model.createIfcAxis2Placement3D(origin, None, None)
    ctx = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, gcf, None)

    profiles = data_from_IFC(config.cs_fit.ifc_cs_path, direct=True)

    zDir = model.createIfcDirection((0.0, 0.0, 1.0))
    xDir = model.createIfcDirection((1.0, 0.0, 0.0))
    frame = model.createIfcAxis2Placement3D(origin, zDir, xDir)
    placement = model.createIfcLocalPlacement(None, frame)

    scale_fac = 1e3

    # for key value pair in profile_dict
    for bone_name, profile_properties in profile_dict.items():
        rot = profile_properties["rot_mat"]
        # Convert numpy values to float
        origin = model.createIfcCartesianPoint([float(x) for x in rot[:3, 3]])
        z_axis = model.createIfcDirection([float(x) * scale_fac for x in rot[:3, 2]])
        x_axis = model.createIfcDirection([float(x) * scale_fac for x in rot[:3, 0]])
        placement = model.createIfcLocalPlacement(
            None, model.createIfcAxis2Placement3D(origin, z_axis, x_axis))

        profile_name = profile_properties['cstype']
        ifc_profile = model.add(get_profile(profile_name, profiles))
        body = model.createIfcExtrudedAreaSolid(ifc_profile, None, z_axis, profile_properties['length'])
        bodyRep = model.createIfcShapeRepresentation(ctx, 'Body', 'SweptSolid', [body])
        prdDefShape = model.createIfcProductDefinitionShape(None, None, (bodyRep,))

        guid_ = newGUID()
        print(f'guid: {guid_}')
        beam = model.createIfcBeam(guid_, None, bone_name, None, None, placement, prdDefShape, None, None)
        
        shape = geom.create_shape(settings, beam).geometry








    model.write("/Users/fnoic/Downloads/model.ifc")








if __name__ == '__main__':
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
        skeleton = pd.read_json(f)
    config = OmegaConf.load('/Users/fnoic/PycharmProjects/reconstruct/config_0.yaml')
    config = config_io(config)

    model_builder(skeleton, config)
