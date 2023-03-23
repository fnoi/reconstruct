







Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type 'help', 'copyright', 'credits' or 'license' for more information.
>>> Gui.runCommand('Std_Workbench',18)
>>> Gui.runCommand('Std_ViewStatusBar',1)
>>> with open('C:/Program Files/FreeCAD 0.20/data/Mod/Start/StartPage/LoadNew.py') as file:
>>> 	exec(file.read())
>>> # App.setActiveDocument("Unnamed")
>>> # App.ActiveDocument=App.getDocument("Unnamed")
>>> # Gui.ActiveDocument=Gui.getDocument("Unnamed")
>>> Gui.runCommand('Std_OrthographicCamera',1)
>>> Gui.runCommand('Std_DlgMacroExecute',0)
>>> ### Begin command Std_Workbench
>>> Gui.activateWorkbench("PartWorkbench")
>>> ### End command Std_Workbench
>>> ### Begin command Std_Workbench
>>> Gui.activateWorkbench("DraftWorkbench")
>>> ### End command Std_Workbench
>>> Gui.runCommand('Draft_Wire',0)
>>> import Draft
>>> pl = FreeCAD.Placement()
>>> pl.Rotation.Q = (0.44601487980014143, 0.1195089482778592, 0.22957493990078448, 0.8567868376953142)
>>> pl.Base = FreeCAD.Vector(863693.517330978, -26584.184453127615, 339498.93049081304)
>>> points = [FreeCAD.Vector(863693.517330978, -26584.184453127615, 339498.93049081304), FreeCAD.Vector(6.568852902483181e+16, 3.407222200481687e+16, 5166222420728493.0)]
>>> line = Draft.make_wire(points, placement=pl, closed=False, face=True, support=None)
>>> # Gui.Selection.addSelection('Unnamed','Line')
>>> Draft.autogroup(line)
>>> FreeCAD.ActiveDocument.recompute()
>>> ### Begin command Std_Delete
>>> App.getDocument('Unnamed').removeObject('Line')
>>> App.getDocument('Unnamed').recompute()
>>> ### End command Std_Delete
>>> # Gui.Selection.clearSelection()
>>> Gui.runCommand('Draft_Wire',0)
>>> pl = FreeCAD.Placement()
>>> pl.Rotation.Q = (0.44601487980014143, 0.1195089482778592, 0.22957493990078448, 0.8567868376953142)
>>> pl.Base = FreeCAD.Vector(3.7548269071772524, 8.196449647934667, 17.341594161610672)
>>> points = [FreeCAD.Vector(3.7548269071772524, 8.196449647934667, 17.341594161610672), FreeCAD.Vector(5.841589652857254, 19.362075299856876, 39.59892385419461), FreeCAD.Vector(8.183151713743337, 31.48637729128389, 61.412951183775405), FreeCAD.Vector(11.303668341640176, 44.041490355590206, 83.23845030561691)]
>>> line = Draft.make_wire(points, placement=pl, closed=False, face=True, support=None)
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> Draft.autogroup(line)
>>> FreeCAD.ActiveDocument.recompute()
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> ### Begin command Std_Delete
>>> App.getDocument('Unnamed').removeObject('Wire')
>>> App.getDocument('Unnamed').recompute()
>>> ### End command Std_Delete
>>> # Gui.Selection.clearSelection()
>>> Gui.runCommand('Draft_Wire',0)
>>> pl = FreeCAD.Placement()
>>> pl.Rotation.Q = (0.39793688208965394, 0.5370472083547874, 0.727576855915042, 0.15446181601896586)
>>> pl.Base = FreeCAD.Vector(7.35, 17.2, 5.67)
>>> points = [FreeCAD.Vector(7.35, 17.2, 5.67), FreeCAD.Vector(7.39, 23.18, 5.7), FreeCAD.Vector(8.09, 23.22, 4.87), FreeCAD.Vector(8.98, 23.22, 4.89)]
>>> line = Draft.make_wire(points, placement=pl, closed=False, face=True, support=None)
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> Draft.autogroup(line)
>>> FreeCAD.ActiveDocument.recompute()
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> Gui.runCommand('Std_Workbench',1)
>>> Gui.runCommand('Draft_Circle',0)
>>> Gui.runCommand('Draft_Circle',0)
>>> pl=FreeCAD.Placement()
>>> pl.Rotation.Q=(0.6633437055805083, -0.2541508990526751, 0.2471967207107813, 0.6589963809060508)
>>> pl.Base=FreeCAD.Vector(7.35, 17.2, 5.67)
>>> circle = Draft.make_circle(radius=0.25568871039632307, placement=pl, face=True, support=None)
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> Draft.autogroup(circle)
>>> FreeCAD.ActiveDocument.recompute()
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Circle','Edge1',7.1111,17.2025,5.57892)
>>> ### Begin command Std_Workbench
>>> Gui.activateWorkbench("PartWorkbench")
>>> ### End command Std_Workbench
>>> ### Begin command Part_Loft
>>> from FreeCAD import Base
>>> import Part
>>> ### End command Part_Loft
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> # Gui.Selection.removeSelection('Unnamed','Circle','Edge1')
>>> # Gui.Selection.removeSelection('Unnamed','Circle')
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> # Gui.Selection.removeSelection('Unnamed','Wire')
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> ### Begin command Part_Sweep
>>> from FreeCAD import Base
>>> import Part
>>> ### End command Part_Sweep
>>> # Gui.Selection.removeSelection('Unnamed','Circle')
>>> # Gui.Selection.removeSelection('Unnamed','Wire')
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> # Gui.Selection.removeSelection('Unnamed','Circle')
>>> # Gui.Selection.addSelection('Unnamed','Wire')
>>> # Gui.Selection.removeSelection('Unnamed','Wire')
>>> # Gui.Selection.addSelection('Unnamed','Circle')
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Wire','Edge1',7.3523,17.5438,5.67172)
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Wire','Edge2',7.53296,23.1882,5.53049)
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Wire','Edge1',7.38688,22.7128,5.69766)
>>> # Gui.Selection.addSelection('Unnamed','Wire','Edge2',7.59997,23.192,5.45104)
>>> # Gui.Selection.addSelection('Unnamed','Wire','Edge3',8.5276,23.22,4.87983)
>>> App.getDocument('Unnamed').addObject('Part::Sweep','Sweep')
>>> App.getDocument('Unnamed').ActiveObject.Sections=[App.getDocument('Unnamed').Circle, ]
>>> App.getDocument('Unnamed').ActiveObject.Spine=(App.getDocument('Unnamed').getObject('Wire'),['Edge1','Edge2','Edge3',])
>>> App.getDocument('Unnamed').ActiveObject.Solid=False
>>> App.getDocument('Unnamed').ActiveObject.Frenet=False
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Transformed"
>>>
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Round corner"
>>>
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Right corner"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Transformed"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Frenet = True
>>>
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Round corner"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Frenet = False
>>>
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Transformed"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Right corner"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep')
>>> FreeCAD.getDocument('Unnamed').getObject('Sweep').Transition = u"Round corner"
>>>
>>> # Gui.Selection.clearSelection()
>>> # Gui.Selection.addSelection('Unnamed','Sweep','Face1',7.43211,18.1563,5.91756)
>>> ### Begin command Std_Export
>>> __objs__=[]
>>> __objs__.append(FreeCAD.getDocument("Unnamed").getObject("Sweep"))
>>> import importOBJ
>>> importOBJ.export(__objs__,u"C:/Users/ga25mal/PycharmProjects/reconstruct/data/out/0_skeleton/Unnamed-Sweep.obj")
>>>
>>> del __objs__
>>> ### End command Std_Export
>>> ### Begin command Std_Export
>>> __objs__=[]
>>> __objs__.append(FreeCAD.getDocument("Unnamed").getObject("Sweep"))
>>> import Mesh
>>> Mesh.export(__objs__,u"C:/Users/ga25mal/PycharmProjects/reconstruct/data/out/0_skeleton/Unnamed-Sweep.stl")
>>>
>>> del __objs__
>>> ### End command Std_Export
>>> 