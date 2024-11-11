# reconstruct

cleanup goal
* small dummy data
* segmentation (lean)
* cs fit from catalog (store catalog with project)
* visualization and ifc export


goal:
* point cluster enrichment (pca, ransac, metaheur.)
* cluster to system graph
* system graph to model (freecad)
* texturing model


(! current implementation needs overlapping (actual intersection!))

Output: Each Beam / Pipe in an individual obj and an FreeCadfile that containts each Beam / Pipe as structur to
export it into an IFC File. The export is not automated yet. 


Documentation:
Data for FreeCad is exportet into json files.
Beam, Beam_intersections, Pipe, Pipe_intersections

In Beam the information of the boneaxis is containt in an 3D vector and the crosssection.
The Crosssection is given such as that the center of mass is placed at the origion in the XY-Plane.
The Crosssection is given in 2D in the XY-Plane.
Each beam is identified by an ID.

In Beam_intersections each intersection of beams is listed.
The order of ids indicates the dominates of the beam.

In Pipe the information of the boneaxis is containt in an 3D vector and the radius.
Each Pipe is identified by an ID.

In Pipe_intersections each intersection of pipe is listed.
The trensition lenght is an additional information if the Pipes have different radius.



