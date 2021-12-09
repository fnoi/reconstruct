import FreeCAD
import Part
import importOBJ
import os


def init_FC():
    App.newDocument("FreePIPE")


def one_line():
    print('hi1')


def two_lines():
    print('hi2')


def multi_lines():
    print('himany')


def export():
    #exportpath = os.getcwd() + "pipey.obj"
    __objs__=[]
    __objs__.append(FreeCAD.getDocument("current").getObject("piperun_sweep"))
    importOBJ.export(__objs__, u"C:/Users/ga25mal/Desktop/pipesweep.obj")
    del __objs__
