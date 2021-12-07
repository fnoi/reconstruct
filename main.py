import numpy as np
import statistics


class pipe_run:
    def __init__(self, num):
        self.num = int(num)
        self.parts = []
        self.dia = None
        self.size = None

    def calc_run(self):
        dia = []
        for part in self.parts:
            dia.append(part.dia)
        self.dia = statistics.mean(dia)
        self.size = len(self.parts)

    #def points_dist(self):


class pipe_straight:
    def __init__(self, line):
        self.dia = line[-1]
        self.A = list(line[1:3])
        self.B = list(line[3:5])


def collector():
    pipes = []
    src = './axis.txt'
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



if __name__ == '__main__':
    pipes = collector()
    a = 0
