import os
import pickle
import Part
import Mesh

DO_THIS = 2

projectpath = '/Users/fnoic/PycharmProjects/freecad_base/'
for file in os.listdir(projectpath):
    if file.startswith('rick__'):
        file_path = projectpath + file
        with open(file_path, 'rb') as pkl:
            try:
                while True:
                    pklin = pickle.load(pkl)
            except EOFError:
                pass

        if pklin[-1] == None: # and file.endswith(str(DO_THIS)):
            print('solo')
            print(pklin)
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
                False)
            fcdoc.recompute()

            fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'cross_sec')
            cross = fcdoc.getObject('cross_sec')
            cross.addGeometry(Part.Circle(
                App.Vector(0, 0, 0),
                App.Vector(0, 1, 0),
                pklin[1]),
                False)
            rot = App.Rotation(0, 90, pklin[2])
            ctr = App.Vector(0, 0, 0)
            pos = App.Vector(pklin[3], pklin[4], pklin[0])
            plc = App.Placement(pos, rot, ctr)
            cross.Placement = plc

            fcdoc.recompute()

            fcdoc.addObject('Part::Sweep', 'pipe')
            fcdoc.ActiveObject.Sections = [cross, ]
            fcdoc.ActiveObject.Spine = (line_1, ['Edge1', ])
            fcdoc.ActiveObject.Solid = False
            fcdoc.ActiveObject.Frenet = False

            fcdoc.recompute()

            #__objs__ = []
            # macos
            #savename = projectpath + 'stl/' + filename + '.stl'
            #__objs__.append(fcdoc.getObject('pipe'))
            #Mesh.export(__objs__, savename)
            #del __objs__

        elif pklin[-1] != None:
            print('double')