import math
import sys
import numpy as np
import statistics
#import sweeps

import sys

#pt = '/usr/lib/freecad/lib/'
pt = '/Applications/FreeCAD.app/Contents/Resources/lib/'
sys.path.append(pt)

import FreeCAD
App = FreeCAD
import Part
import Sketcher
import _PartDesign
import Mesh
import os


def init_FC(pipe):
    App = FreeCAD
    filename = str('freePIPE_' + str(pipe.num))
    fcdoc = App.newDocument(filename)
    return fcdoc, filename


def one_line(pipe, fcdoc):
    print('hi1')
    pta = pipe.parts[0].A
    ptb = pipe.parts[0].B
    hgh = pipe.z
    rad = pipe.dia

    x = ptb[0] - pta[0]
    y = (ptb[1] - pta[1])
    frac = x / y
    alpha = math.atan(frac) * 180 / math.pi

    fcdoc.addObject('PartDesign::Body', 'Body')
    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'line_path')
    line_1 = fcdoc.getObject('line_path')
    line_1.Support = (fcdoc.getObject('XY_Plane'), [''])
    line_1.MapMode = 'FlatFace'
    line_1.addGeometry(Part.LineSegment(App.Vector(pta[0], pta[1], hgh), App.Vector(ptb[0], ptb[1], hgh)), False)
    fcdoc.recompute()
    line_1_oriplac = line_1.Placement
    print(line_1_oriplac)

    rot = App.Rotation(App.Vector(0, 0, 1), alpha)
    pos = 7
    ctr = App.Vector(pta[0], pta[1], hgh)
    plc = App.Placement(pos, rot, ctr)
    print(plc)

    line_1.Placement = plc
    fcdoc.recompute()

    fcdoc.getObject('Body').newObject('Sketcher::SketchObject', 'cross_sec')
    cross = fcdoc.getObject('cross_sec')
    cross.addGeometry(Part.Circle(App.Vector(0, 0, 0), App.Vector(0, 1, 0), rad), False)
    fcdoc.recompute()

    fcdoc.addObject('Part::Sweep', 'pipe')
    fcdoc.ActiveObject.Sections = [cross, ]
    fcdoc.ActiveObject.Spine = (line_1, ['Edge1', ])
    fcdoc.ActiveObject.Solid = False
    fcdoc.ActiveObject.Frenet = False

    fcdoc.ActiveObject.Placement = line_1_oriplac
    print(fcdoc.ActiveObject.Placement)

    fcdoc.recompute()

    return fcdoc
    STOP = 0


def two_lines(pipe, fcdoc):
    print('hi2')
    return fcdoc

def multi_lines(pipe):
    print('himany')


def export(fcdoc, filename):

    __objs__ = []
    savename = '/home/fnoic/Desktop/' + filename + '.stl'
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
        self.B, self.A = list(line[1:3]), list(line[3:5])
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
    if dist_is > dist_cross and dist_is > dist_against:
        print('no switch required, orientation ok')
    elif dist_cross > dist_is and dist_cross > dist_against:
        start_0, end_0 = end_0, start_0
        print('had to switch one')
    elif dist_against > dist_cross and dist_against > dist_is:
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
    src = '/Users/fnoic/Dropbox/DropDrive/freecad_base/axis.txt'
    #src = '/home/fnoic/PycharmProjects/freecad_base/axis.txt'
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
            fcdoc = one_line(pipe, fcdoc)
        elif pipe.size == 2:
            fcdoc = one_line(pipe, fcdoc)
        elif pipe.size > 2:
            print('not solved, more than 2 parts')

        export(fcdoc, filename)

    #    freecad_pipetomodel(pipe)
    a = 0
