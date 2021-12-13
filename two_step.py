import math
import numpy as np
import statistics
import sys
import pickle


def one_line(pipe):
    print('\none-part pipe')
    pta = pipe.parts[0].A
    ptb = pipe.parts[0].B
    hgh = pipe.z
    rad = pipe.dia

    x = ptb[0] - pta[0]
    y = ptb[1] - pta[1]
    frac = x / y
    alpha = math.atan(frac) * 180 / math.pi

    pkl_cont = [hgh, rad,
                pta[0], pta[1], ptb[0], ptb[1],
                None, None, None, None,
                None, None, None, None]
    dump2pickle(pkl_cont, pipe.num)

    a = 0


def dump2pickle(pkl_cont, num):
    pkl_name = 'rick__' + str(num)
    with open(pkl_name, 'wb') as pkl:
        pickle.dump(pkl_cont, pkl)

    a = 0


def two_lines(pipe):
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

    ptb0[0] = SP[0]
    ptb0[1] = SP[1]

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

    vec1_n = [veci / vec1_l for veci in vec1]
    print('vec1n', vec1_n)

    fillet_0 = []
    fillet_0.append(float(SP[0] - beta * vec0_n[0]))
    fillet_0.append(float(SP[1] - beta * vec0_n[1]))
    fillet_1 = []
    fillet_1.append(float(SP[0] + beta * vec1_n[0]))
    fillet_1.append(float(SP[1] + beta * vec1_n[1]))
    print('\nfillet in:', fillet_0, '\nfillet_out:', fillet_1)

    pkl_cont = [hgh, rad,
                pta0[0], pta0[1], ptb0[0], ptb0[1],
                pta1[0], pta1[1], ptb1[0], ptb1[1],
                fillet_0[0], fillet_0[1], fillet_1[0], fillet_1[1]]
    dump2pickle(pkl_cont, pipe.num)


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

        if pipe.size == 1:
            one_line(pipe)
        elif pipe.size == 2:
            two_lines(pipe)
        elif pipe.size > 2:
            print('not solved, more than 2 parts')

    a = 0
