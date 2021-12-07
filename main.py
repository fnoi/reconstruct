import os
import numpy as np


class pipe_run:
    def __init__(self, num):
        self.num = int(num)
        self.parts = []
        self.dia = None

    def props(self, array):
        self.dia = array[-1]


class pipe_straight:
    def __init__(self, line):
        self.dia = line[-1]
        self.A = lineone
        self.B = None


def collector():
    pipes = []

    src = './axis.txt'
    src_array = np.loadtxt(src)
    for inst in np.unique(src_array[:, 0]):
        sub = src_array[src_array[:, 0] == inst]
        this_pipe = pipe_run(inst)

        for line in sub:
            pipe_run.props(pipe_run, line)
            this_pipe.dia = line[-1]

        pipes.append(this_pipe)
        print(inst)

    a = 0


if __name__ == '__main__':
    collector()
