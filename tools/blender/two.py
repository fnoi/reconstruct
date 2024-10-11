import bpy
import ifcopenshell
import bonsai.tool as tool

def find_beam_type(model, beam_type_name):
    for beam_type in model.by_type("IfcBeamType"):
        print(f"Found beam type: {beam_type.Name}")
        if beam_type.Name == beam_type_name:
            return beam_type
    return None

def create_beam(profile_name, length):
    model = tool.Ifc.get()
    beam_type_name = f"beamprofiletype_{profile_name}"
    beam_type = find_beam_type(model, beam_type_name)

    if not beam_type:
        print(f"Beam type {beam_type_name} not found")
        return

    print(f"Creating beam of type {beam_type.Name} with ID {beam_type.id()}")

    bpy.ops.bim.add_constr_type_instance()
    beam = bpy.context.active_object

    # Set beam length
    beam.scale[2] = length

    # Assign the beam type
    beam.BIMObjectProperties.relating_type = str(beam_type.id())
    print(f"Assigned type {beam_type.Name} to beam {beam.name}")

    # Verify assignment
    assigned_type = model.by_id(int(beam.BIMObjectProperties.relating_type))
    print(f"Beam {beam.name} is assigned to type: {assigned_type.Name}")

def blender_beams(query_profile_dict):
    for profile_name, data in query_profile_dict.items():
        create_beam(profile_name, data['length'])

if __name__ == "__main__":
    profiles = {
        'W8X67': {'length': 1.5},
        'W6X12': {'length': 1.0},
        'HP12X53': {'length': 0.2}
    }
    blender_beams(profiles)