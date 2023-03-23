import sys

MODULE_PATH = "C:/Program Files/FreeCAD 0.19/bin/"
#MODULE_PATH = 'C:\\Users\\sauce\\AppData\\Local\\FreeCAD 0.19\\bin' # correct location?

DBG_LOAD = True

if not MODULE_PATH in sys.path:
    if DBG_LOAD is True:
        print("no module path")
    sys.path.insert(-1,MODULE_PATH)
else:
    if DBG_LOAD is True:
        print("module path is present")

if DBG_LOAD is True:
    print(sys.path)

import FreeCAD
#import FreeCAD
import Part
import importOBJ
import os


def init_FC(pipe):
    filename = str(pipe.num)
    App.newDocument("FreePIPE")


def one_line(pipe):
    print('hi1')
    pta = pipe.skeleton[0].A
    ptb = pipe.skeleton[0].B
    hgh = pipe.z


def two_lines(pipe):
    print('hi2')
    pt0a = pipe.skeleton[0].A
    pt0b = pipe.skeleton[0].B
    pt1a = pipe.skeleton[1].A
    pt1b = pipe.skeleton[1].B
    hgh = pipe.z


def multi_lines(pipe):
    print('himany')


def export():
    #exportpath = os.getcwd() + "pipey.obj"
    __objs__=[]
    __objs__.append(FreeCAD.getDocument("current").getObject("piperun_sweep"))
    importOBJ.export(__objs__, u"C:/Users/ga25mal/Desktop/pipesweep.obj")
    del __objs__
