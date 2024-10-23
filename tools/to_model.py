import math

import ifcopenshell
import numpy as np
import pandas as pd


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

def get_profile(profile_name):
    profiles = ifcopenshell.open(r"/Users/fnoic/PycharmProjects/IfcOpenShell/src/bonsai/bonsai/bim/data/libraries/IFC4 US Steel.ifc")
    ishapes = profiles.by_type("IfcIshapeProfileDef")
    iprofile = [i for i in ishapes if i.ProfileName in [profile_name]]
    if len(iprofile) == 0:
        raise ValueError (f"Profile {profile_name} not found")
    return iprofile[0]

def model_builder(skeleton):
    # settings = ifcopenshell.geom.settings()
    # settings.set(settings.USE_PYTHON_OPENCASCADE, True)
    # settings.set(settings.SEW_SHELLS, True)

    profile_dict = data_preprocessor(skeleton)
    model = ifcopenshell.file(schema="IFC4")
    origin = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
    gcf = model.createIfcAxis2Placement3D(origin, None, None)
    ctx = model.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, gcf, None)

    zDir = model.createIfcDirection((0.0, 0.0, 1.0))
    xDir = model.createIfcDirection((1.0, 0.0, 0.0))
    frame = model.createIfcAxis2Placement3D(origin, zDir, xDir)
    placement = model.createIfcLocalPlacement(None, frame)

    # for key value pair in profile_dict
    for bone_name, profile_properties in profile_dict.items():
        profile_name = profile_properties['cstype']
        ifc_profile = model.add(get_profile(profile_name))
        body = model.createIfcExtrudedAreaSolid(ifc_profile, None, zDir, profile_properties['length'])
        bodyRep = model.createIfcShapeRepresentation(ctx, "Body", "SweptSolid", [body])
        prdDefShape = model.createIfcProductDefinitionShape(None, None, (bodyRep,))

        print(f'rot mat {profile_properties["rot_mat"]}')

        beam = model.createIfcBeam(ifcopenshell.guid.new(), None, bone_name, None, None, placement, prdDefShape, None, None)


    model.write("/Users/fnoic/Downloads/model.ifc")








if __name__ == '__main__':
    with open('/Users/fnoic/PycharmProjects/reconstruct/data/parking/skeleton_cache.json', 'r') as f:
        skeleton = pd.read_json(f)
    # print(skeleton)

    model_builder(skeleton)
