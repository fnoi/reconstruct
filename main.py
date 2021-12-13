import math
#import subprocess

#subprocess.call(['pip', 'install', 'shapely'])
#import shapely
#from shapely.geometry import LineString, Point
import numpy as np
import statistics
# import sweeps

import sys

# pt = '/usr/lib/freecad/lib/'
#pt = '/Applications/FreeCAD.app/Contents/Resources/lib/'
#sys.path.append(pt)

#import FreeCAD

App = FreeCAD
import Part
#import Sketcher
#import _PartDesign
import Mesh
#import os


def init_FC(pipe):
    App = FreeCAD
    filename = str('freePIPE_' + str(pipe.num))
    App.newDocument(filename)
    fcdoc = App.getDocument(filename)
    return fcdoc, filename


def one_line(pipe, fcdoc):
    print('\none-part pipe')
    pta = pipe.parts[0].A
    ptb = pipe.parts[0].B
    hgh = pipe.z
    rad = pipe.dia

    x = ptb[0] - pta[0]
    y = ptb[1] - pta[1]
    frac = x / y
    alpha = math.atan(frac) * 180 / math.pi

    fcdoc.addObject('PartDesign::Body', 'Body')
    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'line_path')
    line_1 = fcdoc.getObject('line_path')
    line_1.Support = (fcdoc.getObject('XY_Plane'), [''])
    line_1.MapMode = 'FlatFace'
    line_1.addGeometry(Part.LineSegment(App.Vector(pta[0], pta[1], hgh), App.Vector(ptb[0], ptb[1], hgh)), False)
    fcdoc.recompute()

    # line_1_oriplac = line_1.Placement
    # print(line_1_oriplac)

    # rot = App.Rotation(App.Vector(0, 0, 1), alpha)
    # pos = line_1.Placement.Base
    # ctr = App.Vector(pta[0], pta[1], hgh)
    # plc = App.Placement(pos, rot, ctr)
    # print(plc)

    # line_1.Placement = plc
    # fcdoc.recompute()

    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'cross_sec')
    cross = fcdoc.getObject('cross_sec')
    cross.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 1, 0), rad), False)
    rot = App.Rotation(0, 90, alpha)
    ctr = App.Vector(0, 0, 0)
    pos = App.Vector(pta[0], pta[1], hgh)
    plc = App.Placement(pos, rot, ctr)
    cross.Placement = plc

    fcdoc.recompute()

    fcdoc.addObject('Part::Sweep', 'pipe')
    fcdoc.ActiveObject.Sections = [cross, ]
    fcdoc.ActiveObject.Spine = (line_1, ['Edge1', ])
    fcdoc.ActiveObject.Solid = False
    fcdoc.ActiveObject.Frenet = False

    # fcdoc.ActiveObject.Placement = line_1_oriplac
    # print(fcdoc.ActiveObject.Placement)

    fcdoc.recompute()
    return fcdoc
    STOP = 0


def two_lines(pipe, fcdoc):
    print('\ntwo-part pipe')

    pta0 = pipe.parts[0].A
    ptb0 = pipe.parts[0].B
    pta1 = pipe.parts[1].A
    ptb1 = pipe.parts[1].B

    hgh = pipe.z
    rad = pipe.dia

    L1 = line(pta0, ptb0)
    L2 = line(ptb1, pta1)

    SP = intersection(L1, L2)
    if SP:
        print("Intersection detected:", SP)
    else:
        print("No single intersection point detected")

    print('\n0:', pta0, ptb0, '\n1:', pta1, ptb1, '\nhgh, rad:', hgh, rad, '\nL1L2SP', L1, L2, SP)

    ### shapely
    # import shapely

    #A0 = (pta0[0], pta0[1])
    #B0 = (ptb0[0], ptb0[1])
    #A1 = (pta1[0], pta1[1])
    #B1 = (ptb1[0], ptb1[1])
    #line1 = LineString([A0, B0])
    #line2 = LineString([A1, B1])

    #int_pt = line1.intersection(line2)
    #poi = int_pt.x, int_pt.y
    #print('sheplly', poi)

    ptb0[0] = SP[0]
    ptb0[1] = SP[1]
    # ptb1[0] = SP[0]
    # ptb1[1] = SP[1]

    x0 = ptb0[0] - pta0[0]
    y0 = ptb0[1] - pta0[1]
    frac = x0 / y0
    alpha = math.atan(frac) * 180 / math.pi
    beta = 1.5 * rad
    vec0 = [x0, y0]
    vec0_l = math.sqrt(vec0[0] ** 2 + vec0[1] ** 2)
    vec0_n = [veci / vec0_l for veci in vec0]
    print('vec0n:', vec0_n)
    vec1 = [
        ptb1[0] - pta1[0],
        ptb1[1] - pta1[1]
    ]
    vec1_l = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    # vec1_n = vec1 / vec1_l
    vec1_n = [veci / vec1_l for veci in vec1]
    print('vec1n', vec1_n)

    fillet_0 = []
    fillet_0.append(float(SP[0] - beta * vec0_n[0]))
    fillet_0.append(float(SP[1] - beta * vec0_n[1]))
    fillet_1 = []
    fillet_1.append(float(SP[0] + beta * vec1_n[0]))
    fillet_1.append(float(SP[1] + beta * vec1_n[1]))
    print('\nfillet in:', fillet_0, '\nfillet_out:', fillet_1)

    fcdoc.addObject('PartDesign::Body', 'Body')
    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'line_path')
    line_1 = fcdoc.getObject('line_path')
    line_1.Support = (fcdoc.getObject('XY_Plane'), [''])
    line_1.MapMode = 'FlatFace'

    trial = 'fillet'  # 'triple' or 'fillet'
    if trial == 'triple':
        line_1.addGeometry(Part.LineSegment(
            App.Vector(pta0[0], pta0[1], hgh),
            App.Vector(fillet_0[0], fillet_0[1], hgh)),
            False
        )

        line_1.addGeometry(Part.LineSegment(
            App.Vector(fillet_0[0], fillet_0[1], hgh),
            App.Vector(fillet_1[0], fillet_1[1], hgh)),
            False
        )

        line_1.addGeometry(Part.LineSegment(
            App.Vector(fillet_1[0], fillet_1[1], hgh),
            App.Vector(ptb1[0], ptb1[1], hgh)),
            False
        )
    elif trial == 'fillet':
        line_1.addGeometry(Part.LineSegment(
            App.Vector(pta0[0], pta0[1], hgh),
            App.Vector(ptb0[0], ptb0[1], hgh)),
            False
        )

        line_1.addGeometry(Part.LineSegment(
            App.Vector(pta1[0], pta1[1], hgh),
            App.Vector(ptb1[0], ptb1[1], hgh)),
            False
        )

        line_1.fillet(
            0, 1,
            App.Vector(fillet_0[0], fillet_0[1], hgh),
            App.Vector(fillet_1[0], fillet_1[1], hgh),
            beta,
            True, False
        )

    print(line_1.Geometry)
    fcdoc.recompute()

    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'cross_sec')
    cross = fcdoc.getObject('cross_sec')
    cross.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 1, 0), 0.05), False)
    rot = App.Rotation(0, 90, alpha)
    print(alpha)
    ctr = App.Vector(0, 0, 0)
    pos = App.Vector(pta0[0], pta0[1], hgh)
    plc = App.Placement(pos, rot, ctr)
    cross.Placement = plc

    fcdoc.recompute()

    fcdoc.addObject('Part::Sweep', 'pipe')
    fcdoc.ActiveObject.Sections = [cross, ]
    fcdoc.ActiveObject.Spine = (line_1, ['Edge1', 'Edge2', 'Edge3', ])
    fcdoc.ActiveObject.Solid = False
    fcdoc.ActiveObject.Frenet = False

    # fcdoc.ActiveObject.Placement = line_1_oriplac
    # print(fcdoc.ActiveObject.Placement)

    fcdoc.recompute()
    return fcdoc


def multi_lines(pipe):
    print('himany')


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def export(fcdoc, filename):
    __objs__ = []
    # macos
    savename = '/Users/fnoic/DropBox/DropDrive/freecad_base/results_stl/' + filename + '.stl'
    # savename = '/home/fnoic/Desktop/' + filename + '.stl'
    __objs__.append(fcdoc.getObject('pipe'))
    Mesh.export(__objs__, savename)
    del __objs__


# strong limitation: pipe run in (elevated) XY-plane
class pipe_run:
    def __init__(self, num):
        self.num = int(num)
        self.parts = []
        self.dia = None
        self.size = None
        self.z = None

    def calc_run(self):
        dia = []
        z = []
        for part in self.parts:
            dia.append(part.dia)
            z.append(part.Z)
        self.dia = statistics.mean(dia)
        self.z = statistics.mean(z)
        self.size = len(self.parts)
        if self.size > 1:
            switch_2(self.parts[0], self.parts[1])


class pipe_straight:
    def __init__(self, line):
        self.dia = line[-1]
        self.A = [line[1], line[2], line[3]]
        self.B = [line[4], line[5], line[6]]
        # self.B, self.A = list(line[1:3]), list(line[3:5])
        self.Z = statistics.mean([line[3], line[6]])


def switch_2(straight0, straight1):
    start_0 = straight0.A
    end_0 = straight0.B
    start_1 = straight1.A
    end_1 = straight1.B
    dist_is = \
        math.sqrt((start_1[0] - end_0[0]) ** 2 + (start_1[1] - end_0[1]) ** 2)
    dist_cross = min(
        math.sqrt((start_1[0] - start_0[0]) ** 2 + (start_1[1] - start_0[1]) ** 2),
        math.sqrt((end_1[0] - end_0[0]) ** 2 + (end_1[1] - end_0[1]) ** 2)
    )
    dist_against = \
        math.sqrt((start_0[0] - end_1[0]) ** 2 + (start_0[1] - end_1[1]) ** 2)
    if dist_is < dist_cross and dist_is < dist_against:
        print('no switch required, orientation ok')
    elif dist_cross < dist_is and dist_cross < dist_against:
        start_0, end_0 = end_0, start_0
        print('had to switch one')
    elif dist_against < dist_cross and dist_against < dist_is:
        start_0, end_0, start_1, end_1 = end_0, start_0, end_1, start_1
        print('had to switch both')
    else:
        print('you weird, I\'m outta here')
        sys.exit()

    straight0.A, straight0.B = start_0, end_0
    straight1.A, straight1.B = start_1, end_1
    return straight0, straight1


def collector():
    pipes = []
    src = '/Users/fnoic/PycharmProjects/freecad_base/axis.txt'
    # src = '/home/fnoic/PycharmProjects/freecad_base/axis.txt'
    src_array = np.loadtxt(src)

    for inst in np.unique(src_array[:, 0]):
        sub = src_array[src_array[:, 0] == inst]
        this_pipe = pipe_run(inst)

        for line in sub:
            this_pipe.parts.append(pipe_straight(line))

        this_pipe.calc_run()
        pipes.append(this_pipe)

        print('pipe', int(inst), '- size', this_pipe.size)
    return pipes


# TODO pipe to model
# TODO orientation init for sweep and final re-rotate


if __name__ == '__main__':
    pipes = collector()
    for pipe in pipes:
        fcdoc, filename = init_FC(pipe)

        if pipe.size == 1:
            print('nah')
            # fcdoc = one_line(pipe, fcdoc)
        elif pipe.size == 2:
            fcdoc = two_lines(pipe, fcdoc)
        elif pipe.size > 2:
            print('not solved, more than 2 parts')
        export(fcdoc, filename)

    a = 0
