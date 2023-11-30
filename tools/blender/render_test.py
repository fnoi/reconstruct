import bpy

# Clear all mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Create a new cube
bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))

# Set render engine to CYCLES
bpy.context.scene.render.engine = 'CYCLES'

# Set the output path
bpy.context.scene.render.filepath = "/path/to/output.png"

# Render the scene
bpy.ops.render.render(write_still=True)