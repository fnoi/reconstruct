# load txt line by line
from matplotlib import pyplot as plt


def cornerpts(x0, y0, tf, tw, lf, lw):
    p0 = (x0, y0)
    p1 = (p0[0], p0[1] + tf)
    p2 = (p1[0] + lf/2 - tw/2, p1[1])
    p3 = (p2[0], p2[1] + lw)
    p4 = (p1[0], p3[1])
    p5 = (p4[0], p4[1] + tf)
    p6 = (p5[0] + lf, p5[1])
    p7 = (p6[0], p4[1])
    p8 = (p7[0] - lf/2 + tw/2, p7[1])
    p9 = (p8[0], p2[1])
    p10 = (p7[0], p9[1])
    p11 = (p10[0], p0[1])

    plot_polygon(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)

    # return p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11

def plot_polygon(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    fig, ax = plt.subplots()
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]])
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]])
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]])
    ax.plot([p4[0], p5[0]], [p4[1], p5[1]])
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]])
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]])
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]])
    ax.plot([p8[0], p9[0]], [p8[1], p9[1]])
    ax.plot([p9[0], p10[0]], [p9[1], p10[1]])
    ax.plot([p10[0], p11[0]], [p10[1], p11[1]])
    ax.plot([p11[0], p0[0]], [p11[1], p0[1]])
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':

    filename = '../data/first_beam.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        _ = line.split(' ')
        x0 = float(_[0])
        y0 = float(_[1])
        tf = float(_[2])
        tw = float(_[3])
        lf = float(_[4])
        lw = float(_[5])

        cornerpts(x0, y0, tf, tw, lf, lw)




    a = 0
