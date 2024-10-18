import math

import ifcopenshell
import numpy as np


def data_preprocessor(skeleton):
    profiles = {}
    # iterate over columns in skeleton dataframe
    for bone, data in skeleton.items():
        start = data['start']
        end = data['end']

        length = math.sqrt(sum((a - b) ** 2 for a, b in zip(start, end)))

        rot_mat = np.asarray(data['rot_mat'])
        z_angle_add = data['angle_xy']
        # define rotation matrix for z axis rotation
        rot_mat_z = np.asarray([[np.cos(z_angle_add), -np.sin(z_angle_add), 0],
                                [np.sin(z_angle_add), np.cos(z_angle_add), 0],
                                [0, 0, 1]])
        # multiply rotation matrices
        rot_mat = np.dot(rot_mat.T, rot_mat_z)

        profiles[data.name] = {
            'cstype': data.cstype,
            'length': length,
            'rot_mat': rot_mat,
            'start': start,
            'end': end,
            'offset_2D': data['offset']
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
        beam = model.createIfcBeam(ifcopenshell.guid.new(), None, bone_name, None, None, placement, prdDefShape, None, None)


    model.write("model.ifc")








if __name__ == '__main__':
    profiles = {'W8X67':
                    {'length':1.5},
                'W6X12':
                    {'length':1.0},
                'HP12X53':
                    {'length':0.2}
                }
    model_builder(profiles)

