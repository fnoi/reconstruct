import os
import pickle
import Part
import Mesh

projectpath = '/Users/fnoic/PycharmProjects/freecad_base/'
file = 'rick__1'
file_path = projectpath + file
with open(file_path, 'rb') as pkl:
    try:
        while True:
            pklin = pickle.load(pkl)
    except EOFError:
        pass

print(pklin)
beta = float(pklin[1] * 1.5)
App = FreeCAD
filename = str('freePIPEpkl_' + str(file[-1]))
App.newDocument(filename)
fcdoc = App.getDocument(filename)
fcdoc.addObject('PartDesign::Body', 'Body')
fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'line_path')
line_1 = fcdoc.getObject('line_path')
line_1.Support = (fcdoc.getObject('XY_Plane'), [''])
line_1.MapMode = 'FlatFace'

line_1.addGeometry(Part.LineSegment(
    App.Vector(float(pklin[3]), float(pklin[4]), float(pklin[0])),
    App.Vector(float(pklin[5]), float(pklin[6]), float(pklin[0]))),
    False
)

line_1.addGeometry(Part.LineSegment(
    App.Vector(float(pklin[7]), float(pklin[8]), float(pklin[0])),
    App.Vector(float(pklin[9]), float(pklin[10]), float(pklin[0]))),
    False
)
fcdoc.recompute()
line_1.fillet(
    0, 1,
    App.Vector(float(pklin[11]), float(pklin[12]), float(pklin[0])),
    App.Vector(float(pklin[13]), float(pklin[14]), float(pklin[0])),
    beta,
    True, False
)

print(line_1.Geometry)
fcdoc.recompute()

fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'cross_sec')
cross = fcdoc.getObject('cross_sec')
cross.addGeometry(Part.Circle(
    App.Vector(0, 0, 0),
    App.Vector(0, 1, 0),
    float(pklin[1])),
    False)
rot = App.Rotation(0, 90, float(pklin[2]))
ctr = App.Vector(0, 0, 0)
pos = App.Vector(float(pklin[3]), float(pklin[4]), float(pklin[0]))
plc = App.Placement(pos, rot, ctr)
cross.Placement = plc
fcdoc.recompute()

fcdoc.addObject('Part::Sweep', 'pipe')
fcdoc.ActiveObject.Sections = [cross, ]
fcdoc.ActiveObject.Spine = (line_1, ['Edge1', 'Edge2', 'Edge3', ])
fcdoc.ActiveObject.Solid = False
fcdoc.ActiveObject.Frenet = False
fcdoc.recompute()
