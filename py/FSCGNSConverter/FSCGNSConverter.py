# Import modules
import sys
import time
from functools import wraps
import numpy

import Converter.PyTree as C
import Converter.Internal as Internal
import Converter.Converter as Converter
import Converter.Mpi as Cmpi
import Generator.PyTree as G
import Transform.PyTree as T
import Post.PyTree as P

from FSDataManager import (
    FSClac,
    FSMesh,
    FSError,
    FSFloatArray,
    FSIntArray,
    FSStringArray,
    FSDataName,
    FSDataSpecArray,
    FSDatasetInfo,
    FSMeshEnums,
    FSUnstructVolumeCellTypes,
    FSUnstructSurfaceCellTypes,
    FS_AT_CADGroupID,
    FSCellInfo,
)

# Import optional packages
try:
    from FSDMPyUtils import ArrayOps
except ImportError as exc:
    raise ImportError("FSCGNSConverter: FSDMPyUtils not found") from exc


__all__ = [
    "FSCGNSConverter",
    "ENABLE_PROFILING",
    "BuildMeshOps",
    "buildMeshOps",
    "GenerateBCDictFromFSMesh",
    "generateBCDictFromFSMesh"
]


# ---------------------------------------------------------------------------- #
# Global variables
# ---------------------------------------------------------------------------- #


# ENABLE_PROFILING: bool; Switch to toggle profiling on or off
ENABLE_PROFILING = False

# CGNS_CONTAINER_NAMES: list; Names of the CGNS containers
CGNS_CONTAINER_NAMES = [
    Internal.__FlowSolutionNodes__,
    Internal.__FlowSolutionCenters__,
]


# ---------------------------------------------------------------------------- #
# Decorator
# ---------------------------------------------------------------------------- #


def ProfileTime(func):
    @wraps(func)
    def Wrapper(*args, **kwargs):
        if ENABLE_PROFILING:
            t0 = time.perf_counter()
            res = func(*args, **kwargs)
            t1 = time.perf_counter()
            elapsed = max(Cmpi.allgather(t1 - t0))
            if Cmpi.master:
                print(f"  > {func.__name__}: executed in {elapsed:.3f} sec.")
            return res
        return func(*args, **kwargs)
    return Wrapper


# ---------------------------------------------------------------------------- #
# Functions
# ---------------------------------------------------------------------------- #


def CreateQuad2Quad(coords, ncFaces, ncFacesCentroids, plane="xy", tol=1e-6):
    if plane == "xy":
        ndir = 2
    elif plane == "xz":
        ndir = 1
    else:
        raise ValueError(
            "CreateQuad2Quad: Plane in CreateQuad2Quad must be "
            f"'xy' or 'xz' instead of {plane}."
        )

    nNCFaces = len(ncFaces)
    print(
        f"Quad2Quad: {nNCFaces} original non conformal faces (incl. "
        f"{nNCFaces // 3} potential Quad2Quad)\nLooking for non conformal "
        f"faces in 2D mesh, {plane} plane."
    )

    listQuad2Quad = []
    lenNCF = len(ncFaces)

    node2CellList = ComputeNode2CellList(
        ncFaces, lenNCF, len(coords[:, 0])
    )
    lengths = numpy.array([len(x) for x in node2CellList])
    points45_init = numpy.where(lengths == 3)[0]

    node2CellList = numpy.array(node2CellList, dtype=object)
    node2CellListShr = numpy.vstack(node2CellList[points45_init])

    indexes = numpy.lexsort(
        numpy.vstack([node2CellListShr[:, 1], node2CellListShr[:, 2]])
    )

    node2cellListShrSorted = node2CellListShr[indexes]

    node2Remove = []
    for i in range(0, len(node2cellListShrSorted[:, 0]), 2):
        idx1, iface1, iface2 = node2cellListShrSorted[i]
        idx2 = node2cellListShrSorted[i + 1][0]

        x1Ctr, y1Ctr, z1Ctr = ncFacesCentroids[iface1][:]
        x2Ctr, y2Ctr, z2Ctr = ncFacesCentroids[iface2][:]
        x1, y1, z1 = coords[idx1][:]
        x2, y2, z2 = coords[idx2][:]

        vector1 = numpy.array(
            [
                (y1 - y1Ctr) * (z2 - z1) - (y2 - y1) * (z1 - z1Ctr),
                (x1 - x1Ctr) * (z2 - z1) - (z1 - z1Ctr) * (x2 - x1),
                (x1 - x1Ctr) * (y2 - y1) - (y1 - y1Ctr) * (x2 - x1),
            ]
        )
        vector1Norm = numpy.linalg.norm(vector1)
        vector1 = vector1 / vector1Norm
        vector2 = numpy.array(
            [
                (y1 - y2Ctr) * (z2 - z1) - (y2 - y1) * (z1 - z2Ctr),
                (x1 - x2Ctr) * (z2 - z1) - (z1 - z2Ctr) * (x2 - x1),
                (x1 - x2Ctr) * (y2 - y1) - (y1 - y2Ctr) * (x2 - x1),
            ]
        )
        vector2Norm = numpy.linalg.norm(vector2)
        vector2 = vector2 / vector2Norm
        scalarProduct = numpy.dot(vector1, vector2)
        if scalarProduct > 0.0:
            node2Remove.append(node2cellListShrSorted[i][0])
            node2Remove.append(node2cellListShrSorted[i + 1][0])

    for i in node2Remove:
        points45_init = points45_init[points45_init != i]
        node2CellListShr = node2CellListShr[node2CellListShr[:, 0] != i]
        node2cellListShrSorted = node2cellListShrSorted[
            node2cellListShrSorted[:, 0] != i
        ]

    if len(node2cellListShrSorted) % 2 != 0:
        raise ValueError(
            "CreateQuad2Quad: Something is off, some small faces are missing."
        )

    bigFace = []
    hns = []
    for conn in range(0, len(node2cellListShrSorted[:, 0]), 2):
        nd1 = node2cellListShrSorted[conn][0]
        nd2 = node2cellListShrSorted[conn + 1][0]
        el1 = node2cellListShrSorted[conn][1]
        el2 = node2cellListShrSorted[conn][2]

        bigFaceConcatenated = numpy.concatenate([ncFaces[el1], ncFaces[el2]])
        bigFaceConcatenated = bigFaceConcatenated[bigFaceConcatenated != nd1]
        bigFaceConcatenated = bigFaceConcatenated[bigFaceConcatenated != nd2]

        if ncFaces[el1][1] == nd1:
            hns.append([nd2, nd1])
        else:
            hns.append([nd1, nd2])
        bigFace.append(bigFaceConcatenated)

    for idx, nodes in enumerate(bigFace):
        point4 = hns[idx][0]
        point5 = hns[idx][1]

        i01 = abs(coords[nodes, ndir] - coords[point4, ndir]) < tol

        point01 = nodes[i01]
        point0 = point01[0]
        point1 = point01[1]
        point23 = nodes[~i01]
        point2 = point23[0]
        point3 = point23[1]

        r02 = coords[point2] - coords[point0]
        r03 = coords[point3] - coords[point0]
        r04 = coords[point4] - coords[point0]
        r05 = coords[point5] - coords[point0]

        if numpy.cross(r02, r03).dot(numpy.cross(r04, r05)) < 0:
            point2 = point23[1]
            point3 = point23[0]
        thisQuad2Quad = [point0, point1, point2, point3, point4, point5]
        listQuad2Quad.append(thisQuad2Quad)
    np_Quad2Quad = numpy.array(listQuad2Quad)
    return np_Quad2Quad


def CreateQuad4Quad(coords, ncFaces, ncFacesCentroids):
    nvertices = coords.shape[0]
    lenNCF = len(ncFaces)

    z = Internal.newZone(
        name="Zone", zsize=[[nvertices, lenNCF]], ztype="Unstructured"
    )
    n_gc = Internal.newGridCoordinates(parent=z)
    Internal.newDataArray("CoordinateX", value=coords[:, 0], parent=n_gc)
    Internal.newDataArray("CoordinateY", value=coords[:, 1], parent=n_gc)
    Internal.newDataArray("CoordinateZ", value=coords[:, 2], parent=n_gc)

    zc = Internal.newZone(
        name="ZoneCenters", zsize=[[lenNCF, lenNCF]], ztype="Unstructured"
    )
    n_gcc = Internal.newGridCoordinates(parent=zc)
    Internal.newDataArray("CoordinateX", ncFacesCentroids[:, 0], parent=n_gcc)
    Internal.newDataArray("CoordinateY", ncFacesCentroids[:, 1], parent=n_gcc)
    Internal.newDataArray("CoordinateZ", ncFacesCentroids[:, 2], parent=n_gcc)

    hook = C.createHook(z, "nodes")
    ids = C.identifyNodes(hook, zc)
    idsP8 = ids[ids[:] > -1] - 1

    offsets, cells = ComputeNode2Cell(
        ncFaces, nelts=lenNCF, nvertices=nvertices
    )

    uniqueIdsP8 = numpy.unique(idsP8)
    nq = uniqueIdsP8.size
    rowIds = numpy.arange(nq)

    # Get face ids for all point8
    beg = offsets[uniqueIdsP8]
    end = offsets[uniqueIdsP8 + 1]
    counts = end - beg

    if not numpy.all(counts == 4):
        raise ValueError(
            "Each hanging point must be listed by 4 non-conformal faces."
        )

    # Build flattened face-id array -> shape (4*nq)
    allFaceIds = numpy.concatenate([cells[beg[i]:end[i]] for i in rowIds])
    allFaceIds = allFaceIds.reshape(nq, 4)
    # Fetch all faces at once -> shape (nq, 4, 4)
    faces = ncFaces[allFaceIds]
    # Find column position of each point8 in its 4 faces
    # Expand point8 for broadcasting
    p8 = uniqueIdsP8[:, None, None]
    mask = faces == p8  # shape (nq, 4, 4)
    # Column index (0..3) where match occurs
    pos = numpy.argmax(mask, axis=2)  # shape (nq, 4)
    # Build array np_Quad4Quad
    np_Quad4Quad = numpy.empty((nq, 9), dtype=Internal.E_NpyInt)
    np_Quad4Quad[:, 8] = uniqueIdsP8  # last column is point8
    for col in range(4):
        sel = pos[:, col]
        rows = faces[rowIds, col]
        mask = sel == 0
        np_Quad4Quad[mask, 7] = rows[mask, 1]
        np_Quad4Quad[mask, 3] = rows[mask, 2]
        np_Quad4Quad[mask, 6] = rows[mask, 3]
        mask = sel == 1
        np_Quad4Quad[mask, 5] = rows[mask, 0]
        np_Quad4Quad[mask, 6] = rows[mask, 2]
        np_Quad4Quad[mask, 2] = rows[mask, 3]
        mask = sel == 2
        np_Quad4Quad[mask, 1] = rows[mask, 0]
        np_Quad4Quad[mask, 4] = rows[mask, 1]
        np_Quad4Quad[mask, 5] = rows[mask, 3]
        mask = sel == 3
        np_Quad4Quad[mask, 4] = rows[mask, 0]
        np_Quad4Quad[mask, 0] = rows[mask, 1]
        np_Quad4Quad[mask, 7] = rows[mask, 2]
    return np_Quad4Quad


def _AddBC2ZoneLoc(z, bndName, bndType, zbc):
    s = bndType.split(":")
    bndType1 = s[0]
    if len(s) > 1:
        bndType2 = s[1]
    else:
        bndType2 = ""

    # Analyse zone zbc
    dims = Internal.getZoneDim(zbc)
    neb = dims[2]  # nbre d'elts de zbc

    eltType, _ = Internal.eltName2EltNo(dims[3])  # type d'elements de zbc
    # On cherche l'element max dans les connectivites de z
    maxElt = 0
    connects = Internal.getNodesFromType(z, "Elements_t")
    for cn in connects:
        r = Internal.getNodeFromName1(cn, "ElementRange")
        m = r[1][1]
        maxElt = max(maxElt, m)
    # on cree un nouveau noeud connectivite dans z1 (avec le nom de la zone z2)
    nebb = neb
    node = Internal.createUniqueChild(
        z, zbc[0], "Elements_t", value=[eltType, nebb]
    )
    Internal.createUniqueChild(
        node, "ElementRange", "IndexRange_t", value=[maxElt + 1, maxElt + neb]
    )
    oldc = Internal.getNodeFromName2(zbc, "ElementConnectivity")[1]
    newc = numpy.copy(oldc)
    hook = C.createHook(z, "nodes")
    ids = C.identifyNodes(hook, zbc)
    newc[:] = ids[oldc[:] - 1]
    Internal.createUniqueChild(
        node, "ElementConnectivity", "DataArray_t", value=newc
    )

    zoneBC = Internal.createUniqueChild(z, "ZoneBC", "ZoneBC_t")
    if len(s) == 1:
        info = Internal.createChild(zoneBC, bndName, "BC_t", value=bndType)
    else:  # familyspecified
        info = Internal.createChild(zoneBC, bndName, "BC_t", value=bndType1)
        Internal.createUniqueChild(
            info, "FamilyName", "FamilyName_t", value=bndType2
        )

    Internal.createUniqueChild(
        info, "GridLocation", "GridLocation_t", value="FaceCenter"
    )
    Internal.createUniqueChild(
        info,
        "ElementRange",
        "IndexRange_t",
        value=numpy.array([[maxElt + 1, maxElt + neb]]),
    )
    return None


def ComputeNode2CellList(np_cell2NodeUnravelled, nelts, nvertices, nvpe=4):
    node2CellList = [[i] for i in range(nvertices)]
    for i in range(nelts):
        for j in range(nvpe):
            node2CellList[np_cell2NodeUnravelled[i][j]].append(i)
    return node2CellList


def ComputeNode2Cell(np_cell2NodeUnravelled, nelts, nvertices, nvpe=4):
    nodes = np_cell2NodeUnravelled.ravel()
    cells = numpy.repeat(numpy.arange(nelts), nvpe)
    # Sort by node index
    order = numpy.argsort(nodes)
    nodes = nodes[order]
    cells = cells[order]
    # Count cells per node
    counts = numpy.bincount(nodes, minlength=nvertices)
    offsets = numpy.empty(nvertices + 1, dtype=Internal.E_NpyInt)
    offsets[0] = 0
    numpy.cumsum(counts, out=offsets[1:])
    return offsets, cells


def ComputeQuadCentroids(xNP, yNP, zNP, eltConn):
    quadIdc = eltConn.reshape(-1, 4)
    centroids = numpy.column_stack(
        (
            xNP[quadIdc].mean(axis=1),
            yNP[quadIdc].mean(axis=1),
            zNP[quadIdc].mean(axis=1),
        )
    )
    return centroids


def BuildMeshOps(
    meshName,
    partitioningLibrary="PARMETIS",
    preserveCellStacks=False,
    verbose=True,
):
    if partitioningLibrary.upper() not in ["PARMETIS", "FSZOLTAN"]:
        raise ValueError(f"WARNING: Partitioning library {partitioningLibrary} "
                         "not found. Options are: FSZoltan, ParMETIS.")
    if partitioningLibrary.upper() == "FSZOLTAN":
        from importlib.util import find_spec
        if find_spec("FSZoltan"):
            import FSZoltan
            partitioningCmd = "RepartitionMeshZOLTAN"
        else:
            partitioningCmd = "RepartitionMeshPARMETIS"
            if Cmpi.master:
                print("WARNING: FSZoltan not found, ParMETIS used instead.")
    else:
        partitioningCmd = "RepartitionMeshPARMETIS"

    meshFormat = meshName.split(".")[-1]
    if meshFormat == "h5":
        meshImportCmd = "ImportMeshHDF5"
    elif meshFormat in ["grid", "cdf"]:
        meshImportCmd = "ImportMeshTAU"
    else:
        raise ValueError(
            f"BuildMeshOps: Input mesh format, {meshFormat}, not "
            "supported. Must be h5, grid or cdf."
        )

    meshOps = [(meshImportCmd, {"MeshFilename": meshName})]
    if verbose:
        meshOps.append("PrintInfo")
    if Cmpi.size > 1:
        meshOps.extend(
            [
                # Create a reasonable partitioning for ZOLTAN
                # (prevents memory bottlenecks)
                "RepartitionMeshRCB",
                # Create local numbering
                "CreateLocalNumbering",
                # Main task
                (
                    partitioningCmd,
                    {
                        "PreserveCellStacks": preserveCellStacks,
                        "LineSectionsExtractionParameters": {
                            "ActiveNodesSelection": {
                                "CellTypes": (
                                    "Prisms",
                                    "Hexahedra",
                                    "Tetrahedra",
                                    "Pyramids",
                                    "Quadrilaterals",
                                    "Triangles",
                                )
                            },
                            "StartNodesSelection": {
                                "CellAttribute": "CADGroupID",
                            },
                        },
                        "GraphExtraction": {"GraphType": "CellBased"},
                        "Approach": "CoordinateGraphMultilevel",
                    },
                ),
                # Create local numbering
                "CreateLocalNumbering",
            ]
        )
        if verbose:
            meshOps.append("PrintInfo")
    return tuple(meshOps)


def GenerateBCDictFromFSMesh(fsmesh=None):
    """Fetch boundary names from an fsmesh and return a partial bcDict."""
    bcDictPartial = {}
    if fsmesh is None:
        return bcDictPartial
    markers = FSIntArray()
    fsmesh.GatherCellAttributeValues(FS_AT_CADGroupID, markers) or FSError.PrintAndExit()
    for marker in markers:
        bcName = str(fsmesh.GetCellAttributeValueName(FS_AT_CADGroupID, marker))
        bcDictPartial[marker] = bcName
    return bcDictPartial


# Lower-case aliases for Cassiopee users
buildMeshOps = BuildMeshOps
generateBCDictFromFSMesh = GenerateBCDictFromFSMesh


# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #


class FSCGNSConverter:
    # Map cell type indices from FS to CGNS and vice versa
    FS2CGNS_CT = {
        FSMeshEnums.CT_Node: 2,
        FSMeshEnums.CT_Edge2: 3,
        FSMeshEnums.CT_Tri3: 5,
        FSMeshEnums.CT_Quad4: 7,
        FSMeshEnums.CT_Tetra4: 10,
        FSMeshEnums.CT_Pyra5: 12,
        FSMeshEnums.CT_Prism6: 14,
        FSMeshEnums.CT_Hexa8: 17,
        FSMeshEnums.CT_Poly3D: 23
    }
    CGNS2FS_CT = {v: k for k, v in FS2CGNS_CT.items()}
    CGNS2FS_CT[22] = FSMeshEnums.CT_Poly3D

    def __init__(
        self,
        meshName=None,
        pyTree=None,
        clac=None,
        fsmesh=None,
        dimPb=2,
        flipYZAxes=False,
        conformal=True,
        IBMParameters=None,
        datasets="all",
        bcDict=None,
        coordsName="Coordinates",
        verbose=True,
        **kwargs,
    ):
        # Sanitize and store input arguments
        if (clac is None) != (fsmesh is None):
            raise ValueError(
                "Must provide both 'clac' and 'fsmesh', or neither of them."
            )
        if ((meshName is None) == (fsmesh is None)) and (pyTree is None):
            raise ValueError(
                "Must either provide 'meshName' or 'fsmesh', "
                "but not both or neither of them."
            )
        if ((meshName is None) == (pyTree is None)) and (fsmesh is None):
            raise ValueError(
                "Must either provide 'meshName' or 'pyTree', "
                "but not both or neither of them."
            )
        if (fsmesh is not None) and (pyTree is not None):
            raise ValueError(
                "Provide either 'fsmesh' or 'pyTree', but not both of them."
            )
        if not (meshName is None or isinstance(meshName, str)):
            raise TypeError("'meshName' must be a string")

        self.meshName = meshName
        if (clac is None) != (fsmesh is None):
            raise ValueError(
                "Must provide both 'clac' and 'fsmesh', or neither of them."
            )
        self.clac = FSClac() if clac is None else clac
        self.fsmesh = fsmesh
        self.pyTree = pyTree
        self.dimPb = dimPb
        self.flipYZAxes = flipYZAxes
        self.conformal = conformal
        self.ibmParameters = (
            {} if IBMParameters is None else IBMParameters.copy()
        )
        self.ibm = len(self.ibmParameters) > 0
        self.bcDict = bcDict
        self.coordsName = coordsName
        self.verbose = verbose

        if datasets is None:
            self.datasets = set()
        elif isinstance(datasets, (list, set)):
            self.datasets = set(datasets)
        elif isinstance(datasets, str):
            if datasets == "all":
                self.datasets = datasets
            else:
                self.datasets = set([datasets])
        else:
            raise TypeError(
                "Invalid datasets argument. Must be None, 'all', a string "
                "(eg. a field name or a CGNS container name), or a list/set of "
                f"dataset names. Got: {datasets}"
            )

        # Determine which convert and export functions to use from the inputs
        isHDF5MeshName = isinstance(
            self.meshName, str
        ) and self.meshName.endswith(".h5")
        if isHDF5MeshName or (self.clac is not None and self.fsmesh is not None):
            self.__Convert = self.Convert2CGNS
            self.__Export2Tecplot = self.ExportCGNS2Tecplot
        else:
            self.__Convert = self.Convert2FSDM
            self.__Export2Tecplot = self.ExportFSMesh2Tecplot

        # Create other class attributes
        self.meshType = None
        self.nvertices = 0
        self.nsurfaceCells = 0
        self.nvolumeCells = 0
        self.fsSurfaceCellTypes = []
        self.fsVolumeCellTypes = []
        self.np_coordinates = None
        self.connectivityDict = {}
        self.cell2ProcDict = {}
        self.bcsNames = []
        self.bcEltsDict = {}
        self.eltRangeMap = {}

    def InitBCDict(self):
        """Check user bcDict and combine with that obtained from the fsmesh"""
        errorMessage = (
            "Invalid bcDict argument. Dictionary values can "
            "either be a BCType (str) or a list containing "
            "(BCName, BCType). When a BCName is provided, it "
            "overrides the name present in the input file."
        )

        bcDictTmp = GenerateBCDictFromFSMesh(self.fsmesh)

        if self.bcDict is None or not isinstance(self.bcDict, dict):
            self.bcDict = {}
        else:
            self.bcDict = {int(k): v for k, v in self.bcDict.items()}
            for marker in self.bcDict:
                if isinstance(self.bcDict[marker], (tuple, list)):
                    if len(self.bcDict[marker]) == 1:
                        self.bcDict[marker] = {
                            "name": None,
                            "type": self.bcDict[marker][0]
                        }
                    elif len(self.bcDict[marker]) == 2:
                        self.bcDict[marker] = {
                            "name": self.bcDict[marker][0],
                            "type": self.bcDict[marker][1]
                        }
                    else:
                        raise ValueError(errorMessage)
                elif isinstance(self.bcDict[marker], str):
                    self.bcDict[marker] = {
                        "name": None,
                        "type": self.bcDict[marker]
                    }
                else:
                    raise ValueError(errorMessage)

        for marker, bcName in bcDictTmp.items():
            if marker not in self.bcDict:
                if "symmetry" in bcName.lower():
                    bcType = "BCSymmetryPlane"
                else:
                    bcType = "UserDefined"
                self.bcDict[marker] = {"name": bcName, "type": bcType}
            elif self.bcDict[marker]["name"] is None:
                self.bcDict[marker]["name"] = bcName

    def Convert(self, **kwargs):
        """Main routine to convert a mesh from FS to CGNS or vice versa"""
        self.__Convert(**kwargs)

    def Convert2CGNS(self, forOverset=False, forFFDX=False, **kwargs):
        """Convert a mesh from FS to CGNS"""
        self.meshType = "Unstructured"
        if forOverset:
            includeSurfaceData = False
            includeGhostCells = True
        else:
            includeSurfaceData = True
            includeGhostCells = False
        self.RecoverFSMeshInfo(includeGhostCells=includeGhostCells)
        self.InitBCDict()

        self.RecoverFSCoordinates()
        self.InitCGNSCoordinates()
        self.RecoverFSConnectivity(
            includeGhostCells=includeGhostCells,
            includeSurfaceData=includeSurfaceData,
        )
        self.InitCGNSConnectivity(includeSurfaceData=includeSurfaceData)

        if forOverset:
            return

        if Cmpi.size > 1:
            Cmpi._setProc(self.pyTree, Cmpi.rank)

        self.RecoverFSFlowSolution()

        if forFFDX:
            self.FreeResources()
            # Convert ME to NGon
            if Cmpi.master and self.verbose:
                print("Converting Multiple-Element mesh to NGon.")
            C._convertArray2NGon(self.pyTree, method="topologic",
                                 recoverBC=True, api=3)

            # Reorient surface normals
            if kwargs.get("reorient", True):
                import Intersector.PyTree as XOR

                if Cmpi.master and self.verbose:
                    print("Reorienting mesh for use in FFD and FFX.")
                XOR._reorient(self.pyTree)

            self.MergeBCsByMarker()
            G._rmOrphans(self.pyTree)
        else:
            status = {}
            indices = []
            BCInfo = C.getBCs(self.pyTree)
            G._close(self.pyTree, indices=indices, status=status)
            if status.get("modified"):
                C._recoverBCs(self.pyTree, BCInfo=BCInfo, removeBC=True) # TODO add indices

    def Convert2FSDM(self):
        """Convert a mesh from CGNS to FS"""
        z_ncFaces = None
        ibmDatasets = None

        self.RecoverCGNSMeshInfo()
        self.InitBCDict()

        if self.nvertices > 0:
            self.PrepareNCFacesDataset()
            z_ncFaces = self.CreateNCFacesZone()
            self.PrepareCGNSConnectivities()
            self.RecoverCGNSCoordinates()
            self.RecoverCGNSConnectivity()

        self.InitFSMesh(z_ncFaces)
        if self.nvertices > 0:
            self.InitIBMDatasets()
            ibmDatasets = self.RecoverPointList2BoundaryMarkers()
        if Cmpi.size > 1:
            self.InitFSBCsMPI(ibmDatasets)
            self.DeduplicateFSNodesMPI()
        else:
            self.InitFSBCs(ibmDatasets)
        self.InitFSFlowSolution()
        self.CheckFSMesh()

    @ProfileTime
    def RecoverFSMeshInfo(self, includeGhostCells=False):
        """Load and/or fetch FS mesh data and initialize the corresponding
        class attributes"""
        if Cmpi.master and self.verbose:
            print("Fetching FS mesh data.")
        if self.fsmesh is None:  # Load FS mesh
            self.fsmesh = FSMesh(self.clac)
            meshOps = BuildMeshOps(self.meshName, verbose=self.verbose)
            if not self.fsmesh.DoOps(meshOps):
                FSError.PrintAndExit()

        # Get mesh info
        self.nsurfaceCells = 0
        self.nvolumeCells = 0
        self.fsVolumeCellTypes = []
        self.fsSurfaceCellTypes = []

        # Get number of vertices
        self.nvertices = self.fsmesh.GetNOwnedCells(FSMeshEnums.CT_Node)

        # Compute number of surface cells
        for cellType in FSUnstructSurfaceCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsSurfaceCellTypes.append(cellType)
                if includeGhostCells:
                    self.nsurfaceCells += self.fsmesh.GetNCells(cellType)
                else:
                    self.nsurfaceCells += self.fsmesh.GetNOwnedCells(cellType)

        # Compute number of volume cells
        for cellType in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsVolumeCellTypes.append(cellType)
                if includeGhostCells:
                    self.nvolumeCells += self.fsmesh.GetNCells(cellType)
                else:
                    self.nvolumeCells += self.fsmesh.GetNOwnedCells(cellType)

    def SetNumpyCoordinates(self, coords):
        if self.flipYZAxes:
            self.np_coordinates = numpy.empty_like(coords)
            self.np_coordinates[:, 0] = coords[:, 0]
            self.np_coordinates[:, 1] = coords[:, 2]
            self.np_coordinates[:, 2] = -coords[:, 1]
        else:
            self.np_coordinates = coords

    @ProfileTime
    def RecoverFSCoordinates(self):
        """
        Fetch FS mesh coordinates and initialize the corresponding numpy class
        attributes
        """
        if Cmpi.master and self.verbose:
            print("Fetching FS mesh coordinates.")
        fs_coordinates = self.fsmesh.GetUnstructDataset(
            self.coordsName
        ).GetValues()
        np_coordinates = numpy.array(fs_coordinates.Buffer(), copy=False)
        self.SetNumpyCoordinates(np_coordinates)

    @ProfileTime
    def MergeUnstructSurfaceConnectivities(self):
        # Find CGNS surface element numbers corresponding to existing
        # FS surface cell types
        cgnsSurfCellNos = []
        for fsCellType in self.fsSurfaceCellTypes:
            cgnsSurfCellNos.append(FSCGNSConverter.FS2CGNS_CT[fsCellType])

        z = Internal.getZones(self.pyTree)[0]
        n_elts = Internal.getNodesFromType1(z, "Elements_t")

        # Group surface element nodes by CGNS element number
        eltType2ElementNodesDict = {}
        for n_elt in n_elts:
            eltName = n_elt[0]
            eltNo = Internal.getValue(n_elt)[0]
            if eltNo in cgnsSurfCellNos and eltName != "NonConformalFaces":
                eltType2ElementNodesDict.setdefault(eltNo, []).append(n_elt)

        ntotFaces = 0
        for eltNo, n_eltOfType in eltType2ElementNodesDict.items():
            if len(n_eltOfType) <= 1:
                continue  # single connectivity at most
            # Concatenate CGNS surface mesh connectivities of this type
            eltConns = []
            for n_elt in n_eltOfType:
                eltName = n_elt[0]
                np_eltConn = Internal.getNodeFromName1(
                    n_elt, "ElementConnectivity"
                )[1]
                eltConns.append(np_eltConn)
            eltConns = numpy.concatenate(eltConns)

            # Update element range and connectivity of the first element node
            # of this type
            np_eltRange = Internal.getNodeFromName1(
                n_eltOfType[0], "ElementRange"
            )[1]
            ec = Internal.getNodeFromName1(
                n_eltOfType[0], "ElementConnectivity"
            )

            nvpe = Internal.eltNo2EltName(eltNo)[1]
            nfaces = int(eltConns.shape[0] // nvpe)
            np_eltRange[0] = self.nvolumeCells + ntotFaces + 1
            np_eltRange[1] = self.nvolumeCells + ntotFaces + nfaces
            ec[1] = eltConns
            n_eltOfType[0][0] = Internal.eltNo2EltName(eltNo)[0]

            ntotFaces += nfaces

            # Delete all but the first element node of this type
            for n_elt in n_eltOfType[1:]:
                Internal._rmNode(z, n_elt)

    def CreateBCZonePerSurfaceElementType(self):
        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
        if len(n_zoneBCs) == 0:
            return None  # No BCs defined

        n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")

        # Loop over all existing conformal BC nodes (IndexArray or IndexRange)
        # and offset vertex indices by the number of volume cells
        np_vertexPL = numpy.array([], dtype=Internal.E_NpyInt)
        ntotBCVertices = 0
        for i, n_bc in enumerate(n_bcs):
            bcName = Internal.getName(n_bc)
            if bcName == "NonConformalFaces":
                continue
            if not any(suffix in bcName for suffix in [".TRI", ".QUAD"]):
                n_bc[0] = self.bcsNames[i]

            offset = self.nvolumeCells + ntotBCVertices
            n_IA = Internal.getNodeFromType1(n_bc, "IndexArray_t")
            if n_IA is not None and n_IA[0] == "PointList":
                np_vertexPL = Internal.getNodeFromName1(n_bc, "PointList")[1][0]
                np_vertexPL += offset
                ntotBCVertices += np_vertexPL.shape[0]
            n_IR = Internal.getNodeFromType1(n_bc, "IndexRange_t")
            if n_IR is not None and n_IR[0] == "ElementRange":
                np_eltRange = (
                    Internal.getNodeFromName1(n_bc, "ElementRange")[1][0]
                )
                nfaces = np_eltRange[1] - np_eltRange[0] + 1
                np_eltRange[0] = offset + 1
                np_eltRange[1] = offset + nfaces
                ntotBCVertices += nfaces

    def PrepareNCFacesDataset(self):
        if self.conformal:
            return
        bcNames = [
            bc[0] for bc in Internal.getNodesFromType(self.pyTree, "BC_t")
        ]
        if "QuadNQuad" in bcNames:
            rm = Internal.getNodeFromName(self.pyTree, "QuadNQuad")
            old_name_hf = rm[0]
            Internal._renameNode(self.pyTree, old_name_hf, "NonConformalFaces")
            n_NCF = Internal.getNodeFromName(self.pyTree, "NonConformalFaces")
            if Internal.getNodeFromName(n_NCF, "PointList") is not None:
                lenNCF = Internal.getNodeFromName(n_NCF, "PointList")[1][
                    0
                ].shape[0]
                self.nsurfaceCells -= lenNCF
            elif Internal.getNodeFromName(n_NCF, "ElementRange") is not None:
                eltRange = Internal.getNodeFromName(n_NCF, "ElementRange")[1]
                if len(eltRange) == 1:
                    ERmin, ERmax = eltRange[0][:]
                elif len(eltRange) == 2:
                    ERmin, ERmax = eltRange[0], eltRange[1]
                else:
                    raise ValueError(
                        "PrepareNCFacesDataset: Problem with the "
                        "element range of non-conformal interfaces."
                    )
                lenNCF = ERmax - ERmin + 1
                self.nsurfaceCells -= lenNCF

    @ProfileTime
    def CreateNCFacesZone(self):
        if self.conformal:
            return None
        z_octreeFaces = None
        xCoord = Internal.getNodeFromName(self.pyTree, "CoordinateX")[1]
        yCoord = Internal.getNodeFromName(self.pyTree, "CoordinateY")[1]
        zCoord = Internal.getNodeFromName(self.pyTree, "CoordinateZ")[1]
        n_octreeFaces = Internal.getNodesFromName(
            self.pyTree, "NonConformalFaces"
        )
        if not n_octreeFaces:
            return None

        octreeFaces_EC_global = Internal.getNodeFromName(
            n_octreeFaces, "ElementConnectivity"
        )[1]
        lenNCF = len(octreeFaces_EC_global) // 4

        n_NCF = Internal.getNodesFromName(self.pyTree, "NonConformalFaces")
        for n in n_NCF:
            Internal._rmNode(self.pyTree, n)

        idx = numpy.unique(octreeFaces_EC_global, return_index=True)[1]
        octreeFaces_idx_nodes = (
            octreeFaces_EC_global[numpy.sort(idx)] - 1
        )  # indices loc2glob
        len_nodes_NCF = len(octreeFaces_idx_nodes)
        if len_nodes_NCF > 0:
            xCoord_nodes_NCF = xCoord[octreeFaces_idx_nodes]
            yCoord_nodes_NCF = yCoord[octreeFaces_idx_nodes]
            zCoord_nodes_NCF = zCoord[octreeFaces_idx_nodes]

            global_ncFaces = numpy.reshape(
                octreeFaces_EC_global - 1, (lenNCF, 4)
            )
            indices1 = numpy.arange(len_nodes_NCF, dtype=Internal.E_NpyInt)
            glob2loc = numpy.zeros(self.nvertices, dtype=Internal.E_NpyInt)
            glob2loc[octreeFaces_idx_nodes] = indices1
            locNCFaces = glob2loc[global_ncFaces]

            z_octreeFaces = Internal.newZone(
                name="NonConformalFaces",
                zsize=[[len(xCoord_nodes_NCF), len(octreeFaces_idx_nodes)]],
                ztype="Unstructured",
            )
            n_gc = Internal.newGridCoordinates(parent=z_octreeFaces)
            Internal.newDataArray(
                "CoordinateX", value=xCoord_nodes_NCF, parent=n_gc
            )
            Internal.newDataArray(
                "CoordinateY", value=yCoord_nodes_NCF, parent=n_gc
            )
            Internal.newDataArray(
                "CoordinateZ", value=zCoord_nodes_NCF, parent=n_gc
            )
            Internal.newElements(
                name="NonconformalFaces",
                etype=7,
                econnectivity=numpy.ravel(locNCFaces + 1),
                erange=[1, lenNCF],
                eboundary=0,
                parent=z_octreeFaces,
            )
        return z_octreeFaces

    @ProfileTime
    def PrepareCGNSConnectivities(self):
        if self.meshType == "Structured":
            return self.PrepareCGNSConnectivitiesStruct()
        return self.PrepareCGNSConnectivitiesUnstruct()

    @ProfileTime
    def PrepareCGNSConnectivitiesStruct(self):
        C._rmBCOfType(self.pyTree, "BCMatch")
        C._rmBCOfType(self.pyTree, "BCDegeneratedLine")

        novol = len(self.fsVolumeCellTypes)
        zones = Internal.getZones(self.pyTree)
        if Cmpi.master and self.verbose:
            print(f"Merging {len(zones)} CGNS zones into one zone.")
        z = zones[0]
        for noz in range(novol, len(zones)):
            z = T.join([z, zones[noz]])

        n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
        n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")

        zbcs = []
        bcTypes = []
        bcs = []
        for n_bc in n_bcs:
            if not any(suffix in n_bc[0] for suffix in [".TRI", ".QUAD"]):
                bcName = n_bc[0].replace(".", "")
            else:
                bcName = n_bc[0].split(".")[0]
            bcType = Internal.getValue(n_bc)
            if bcType == "FamilySpecified":
                fname = Internal.getNodeFromType1(n_bc, "FamilyName_t")
                if fname is not None:
                    bcName = Internal.getValue(fname)
                    bcType = f"FamilySpecified:{bcName}"
            if bcType not in bcTypes:
                self.bcsNames.append(bcName)
                bcTypes.append(bcType)
                bcs.append(n_bc)

        self.nsurfaceCells = 0
        for bcType in bcTypes:
            zbc = C.extractBCOfType(self.pyTree, bcType)
            zbc = C.convertArray2Hexa(zbc)
            zbc = T.join(zbc)
            self.nsurfaceCells += Internal.getValue(zbc)[0][1]
            zbcs.append(zbc)

        z = C.convertArray2Hexa(z)
        z = G.close(z)
        self.nvertices = int(Internal.getValue(z)[0][0])
        nBCs = len(bcTypes)
        for i in range(nBCs):
            _AddBC2ZoneLoc(z, self.bcsNames[i], bcTypes[i], zbcs[i])
        self.pyTree = C.newPyTree(["Unstructured", z])

        self.MergeUnstructSurfaceConnectivities()
        self.CreateBCZonePerSurfaceElementType()

    @ProfileTime
    def PrepareCGNSConnectivitiesUnstruct(self):
        """
        Merge connectivities such that there is at most one connectivity per
        element type
        """
        eltTypeList = []
        n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
        eltTypeList = [Internal.getValue(n_elt)[0] for n_elt in n_elts]
        foundMultipleEltNodeOfType = len(set(eltTypeList)) < len(eltTypeList)

        if Cmpi.size > 1 or foundMultipleEltNodeOfType:
            novol = len(self.fsVolumeCellTypes)
            zones = Internal.getZones(self.pyTree)
            if Cmpi.master and self.verbose:
                print(f"Merging {len(zones)} CGNS zones into one zone.")
            z = zones[0]
            for noz in range(novol, len(zones)):
                z = T.join([z, zones[noz]])

            # Fill the list of root BC names
            n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
            n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")
            for n_bc in n_bcs:
                if any(suffix in n_bc[0] for suffix in [".TRI", ".QUAD"]):
                    self.bcsNames.append(n_bc[0].split(".")[0])
                else:
                    self.bcsNames.append(n_bc[0].replace(".", ""))

            self.MergeUnstructSurfaceConnectivities()
            self.CreateBCZonePerSurfaceElementType()

    @ProfileTime
    def RecoverCGNSMeshInfo(self):
        if Cmpi.master and self.verbose:
            print("Fetching CGNS mesh info.")

        # Load CGNS mesh
        if self.pyTree is None:
            if Cmpi.size == 1:
                self.pyTree = C.convertFile2PyTree(self.meshName)
            else:
                self.pyTree = Cmpi.convertFile2PyTree(
                    self.meshName, proc=Cmpi.rank
                )
                bases = Internal.getBases(self.pyTree)
                if len(bases) != Cmpi.size:
                    import XCore.PyTree as XC

                    self.pyTree = XC.loadAndSplitElt(self.meshName)

        # Delete empty bases from the pyTree
        C._deleteEmptyBases(self.pyTree)

        # Get mesh info
        self.nvertices = 0
        self.nsurfaceCells = 0
        self.nvolumeCells = 0
        self.fsVolumeCellTypes = []
        self.fsSurfaceCellTypes = []

        # Reorder element types as in FSDM
        self.ReorderCells()

        zones = Internal.getZones(self.pyTree)
        if zones:
            z = zones[0]
            zoneDim = Internal.getZoneDim(z)
            self.meshType = zoneDim[0]

            if self.meshType == "Unstructured":
                self.nvertices = zoneDim[1]
                self.nvolumeCells = zoneDim[2]
                isNGon = zoneDim[3] == "NGON"
                if isNGon:
                    if FSMeshEnums.CT_Poly3D not in self.fsVolumeCellTypes:
                        self.fsVolumeCellTypes.append(FSMeshEnums.CT_Poly3D)
                    if FSMeshEnums.CT_Poly2D not in self.fsSurfaceCellTypes:   
                        self.fsSurfaceCellTypes.append(FSMeshEnums.CT_Poly2D)
                else:
                    n_elts = Internal.getNodesFromType(z, "Elements_t")
                    for n_elt in n_elts:
                        cgnsCellType = Internal.getValue(n_elt)[0]
                        fsCellType = FSCGNSConverter.CGNS2FS_CT[cgnsCellType]
                        if (
                            fsCellType in FSUnstructVolumeCellTypes
                            and fsCellType not in self.fsVolumeCellTypes
                        ):
                            self.fsVolumeCellTypes.append(fsCellType)
                        elif fsCellType in FSUnstructSurfaceCellTypes:
                            eltRange = Internal.getNodeFromName(
                                n_elt, "ElementRange"
                            )[1]
                            nfaces = eltRange[1] - eltRange[0] + 1
                            if nfaces == 0:
                                raise ValueError(
                                    "RecoverCGNSMeshInfo: Empty surface "
                                    "connectivity."
                                )
                            self.nsurfaceCells += nfaces
                            if fsCellType not in self.fsSurfaceCellTypes:
                                self.fsSurfaceCellTypes.append(fsCellType)

            elif self.meshType == "Structured":
                ni, nj, nk = zoneDim[1], zoneDim[2], zoneDim[3]
                self.nvertices = ni * nj * nk
                self.nvolumeCells = (ni - 1) * (nj - 1) * (nk - 1)
                if self.nvolumeCells > 0:
                    self.fsVolumeCellTypes.append(FSMeshEnums.CT_Hexa8)
                    self.fsSurfaceCellTypes.append(FSMeshEnums.CT_Quad4)

    @ProfileTime
    def RecoverCGNSCoordinates(self):
        if Cmpi.master and self.verbose:
            print("Fetching CGNS PyTree coordinates.")
        foundCoords = [False for _ in range(3)]
        coordName2Pos = {"CoordinateX": 0, "CoordinateY": 1, "CoordinateZ": 2}
        np_coordinates = numpy.empty((self.nvertices, 3))

        z = Internal.getZones(self.pyTree)[0]
        n_gridCoords = Internal.getNodeFromType1(z, "GridCoordinates_t")
        n_coords = Internal.getNodesFromType1(n_gridCoords, "DataArray_t")

        for n_coord in n_coords:
            i = coordName2Pos.get(n_coord[0], None)
            if i is None:
                continue
            np_coordinates[:, i] = n_coord[1]
            foundCoords[i] = True

        if not all(foundCoords):
            raise ValueError(
                "RecoverCGNSCoordinates: Coordinates missing "
                "in GridCoordinates_t."
            )
        self.SetNumpyCoordinates(np_coordinates)

    def InitCell2Proc(self, ncellsOfType=0):
        nprocs = Cmpi.size
        gath_cell2Proc = ArrayOps.AllGather(ncellsOfType, self.clac)
        fs_cell2Proc = FSIntArray(nprocs + 1)
        fs_cell2Proc[0] = 0
        for i in range(nprocs):
            fs_cell2Proc[i + 1] = fs_cell2Proc[i] + int(gath_cell2Proc[i])
        return fs_cell2Proc

    @ProfileTime
    def RecoverCGNSConnectivity(self):
        # If the CGNS PyTree contains at least 1 NGON zone, convert it to a
        # signed NGonv4
        zones = Internal.getZones(self.pyTree)
        for z in zones:
            dim = Internal.getZoneDim(z)
            if dim[0] == 'Unstructured' and dim[3] == 'NGON':
                Internal._adaptNGon32NGon4(self.pyTree)
                C._signNGonFaces(self.pyTree)
                break

        # Loop over Elements_t nodes and store references to the CGNS
        # connectivities (these are 1-based) in self.connectivityDict
        self.connectivityDict = {}
        n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
        for n_elt in n_elts:
            eltType = Internal.getValue(n_elt)[0]
            fsCellType = FSCGNSConverter.CGNS2FS_CT.get(eltType, None)
            if fsCellType is None:
                print(
                    f"Warning: FS element type {fsCellType} not supported."
                    "Skipping."
                )
                continue
            if fsCellType not in self.connectivityDict:
                self.connectivityDict[fsCellType] = {}
            n_ER = Internal.getNodeFromName1(n_elt, "ElementRange")
            n_EC = Internal.getNodeFromName1(n_elt, "ElementConnectivity")
            nelts = int(n_ER[1][1] - n_ER[1][0] + 1)
            if eltType == 22:  # NGON
                self.connectivityDict[FSMeshEnums.CT_Poly3D]["NGON"] = n_EC[1]
                n_indPG = Internal.getNodeFromName1(n_elt, "ElementStartOffset")
                self.connectivityDict[FSMeshEnums.CT_Poly3D]["indPG"] = n_indPG[1]
            elif eltType == 23:  # NFACE
                self.connectivityDict[FSMeshEnums.CT_Poly3D]["nCells"] = nelts
                self.connectivityDict[FSMeshEnums.CT_Poly3D]["NFACE"] = n_EC[1]
                n_indPH = Internal.getNodeFromName1(n_elt, "ElementStartOffset")
                self.connectivityDict[FSMeshEnums.CT_Poly3D]["indPH"] = n_indPH[1]
            else:
                # Basic Element
                nvpe = int(n_EC[1].size // nelts)
                self.connectivityDict[fsCellType]["nvpe"] = nvpe
                cellTypeDict = self.connectivityDict[fsCellType]
                cellTypeDict.setdefault("nCells", []).append(nelts)
                cellTypeDict.setdefault("ElementConnectivity", []).append(
                    n_EC[1].reshape(nelts, nvpe)
                )

    @ProfileTime
    def InitFSConnectivity(self, cellType):
        hasCellsOfType = cellType in self.connectivityDict
        if cellType == FSMeshEnums.CT_Poly2D and hasCellsOfType:
            # Construct boundary node lists and boundary node counts
            fsCellTypeDict = self.connectivityDict[cellType]
            poly2DCell2NodeList = fsCellTypeDict["ElementConnectivity"]
            poly2DCell2NodeCounts = fsCellTypeDict["NodeCounts"]

            # Store the boundary face node counts in a 1D array
            fs_poly2DCell2NodeCounts = FSIntArray(poly2DCell2NodeCounts.size)
            numpy.copyto(
                numpy.array(fs_poly2DCell2NodeCounts.Buffer(), copy=False),
                poly2DCell2NodeCounts,
                casting="same_kind",
            )

            # Store the boundary face nodes in a 1D array
            fs_poly2DCell2NodeList = FSIntArray(poly2DCell2NodeList.size)
            numpy.copyto(
                numpy.array(fs_poly2DCell2NodeList.Buffer(), copy=False),
                poly2DCell2NodeList,
                casting="same_kind",
            )

            # Init the unstructured polygonal mesh faces
            self.fsmesh.InitUnstructPolyCells(
                cellType,
                fs_poly2DCell2NodeCounts,
                fs_poly2DCell2NodeList
            )
        elif cellType == FSMeshEnums.CT_Poly3D:
            nPoly3D = self.connectivityDict[cellType]["nCells"]
            np_ngon = self.connectivityDict[cellType]["NGON"]
            np_nface = self.connectivityDict[cellType]["NFACE"]
            np_indPG = self.connectivityDict[cellType]["indPG"]
            np_indPH = self.connectivityDict[cellType]["indPH"]
            
            # Construct node lists and node counts
            poly3DCell2NodeCounts = []
            poly3DCell2NodeList = []
            face2NodeCounts = []
            face2NodeList = []
            for i in range(nPoly3D):
                nodes = []
                locFace2NodeList = []
                seen = set()

                # First pass: build the cell node list
                faceIds = np_nface[np_indPH[i]:np_indPH[i+1]]
                for fidx in faceIds:
                    f = abs(fidx) - 1
                    faceNodes = np_ngon[np_indPG[f]:np_indPG[f+1]] - 1
                    locFace2NodeList.append(faceNodes)
                    for n in faceNodes:
                        if n not in seen:
                            seen.add(n)
                            nodes.append(n)

                globalToLocal = {gn: ln for ln, gn in enumerate(nodes)}

                # Second pass: store faces using local node indices
                for faceNodes in locFace2NodeList:
                    face2NodeCounts.append(len(faceNodes))
                    localFaceNodes = [globalToLocal[n] for n in faceNodes]
                    face2NodeList.extend(localFaceNodes)

                poly3DCell2NodeList.extend(nodes)
                poly3DCell2NodeCounts.append(len(nodes))

            # Store the node counts for each polyhedron in a 1D array
            fs_poly3DCell2NodeCounts = FSIntArray(nPoly3D)
            numpy.copyto(
                numpy.array(fs_poly3DCell2NodeCounts.Buffer(), copy=False),
                numpy.asarray(poly3DCell2NodeCounts, dtype=Internal.E_NpyInt),
                casting="same_kind",
            )
            totalNumberOfCellNodes = numpy.sum(poly3DCell2NodeCounts)

            # Store the node list in a 1D array
            fs_poly3DCell2NodeList = FSIntArray(int(totalNumberOfCellNodes))
            numpy.copyto(
                numpy.array(fs_poly3DCell2NodeList.Buffer(), copy=False),
                numpy.asarray(poly3DCell2NodeList, dtype=Internal.E_NpyInt),
                casting="same_kind",
            )

            # Store the number of faces for each 3D polyhedron in a 1D array
            fs_cell2FaceCounts = FSIntArray(nPoly3D)
            numpy.copyto(
                numpy.array(fs_cell2FaceCounts.Buffer(), copy=False),
                numpy.diff(np_indPH),
                casting="same_kind",
            )
            totalNumberOfFaces = numpy.sum(fs_cell2FaceCounts.Buffer())

            # Store the number of nodes for each face in a 1D array
            fs_face2NodeCounts = FSIntArray(int(totalNumberOfFaces))
            numpy.copyto(
                numpy.array(fs_face2NodeCounts.Buffer(), copy=False),
                numpy.asarray(face2NodeCounts, dtype=Internal.E_NpyInt),
                casting="same_kind",
            )
            totalNumberOfFaceNodes = numpy.sum(fs_face2NodeCounts.Buffer())

            # Store the face nodes in a 1D array
            fs_face2NodeList = FSIntArray(int(totalNumberOfFaceNodes))
            numpy.copyto(
                numpy.array(fs_face2NodeList.Buffer(), copy=False),
                numpy.asarray(face2NodeList, dtype=Internal.E_NpyInt),
                casting="same_kind",
            )

            # Init the unstructured polyhedron mesh cells
            self.fsmesh.InitUnstructPolyCells(
                cellType,
                fs_poly3DCell2NodeCounts,
                fs_poly3DCell2NodeList
            )
            self.fsmesh.InitUnstructPolyCellFaces(
                cellType,
                fs_cell2FaceCounts,
                fs_face2NodeCounts,
                fs_face2NodeList
            )

        else:  # Basic-Elements
            if hasCellsOfType:
                nvpe = self.connectivityDict[cellType]["nvpe"]
                ncells = self.connectivityDict[cellType]["nCells"]
                offsets = numpy.concatenate(([0], numpy.cumsum(ncells)))
                fs_cell2Node = FSIntArray(sum(ncells), nvpe)
            else:
                fs_cell2Node = FSIntArray(0, FSCellInfo.NNodes(cellType))

            cell2NodeList = self.connectivityDict[cellType]["ElementConnectivity"]
            fsBuffer = numpy.array(fs_cell2Node.Buffer(), copy=False)

            if Cmpi.size > 1:
                if hasCellsOfType:
                    nnodesPrevious = self.cell2ProcDict[FSMeshEnums.CT_Node][
                        Cmpi.rank
                    ]
                    for i, np_cell2Node in enumerate(cell2NodeList):
                        numpy.copyto(
                            fsBuffer[offsets[i]:offsets[i+1]],
                            nnodesPrevious + (np_cell2Node - 1),
                            casting="same_kind",
                        )
                self.fsmesh.InitUnstructCells(
                    cellType, self.cell2ProcDict[cellType], fs_cell2Node, True
                )
            else:
                for i, np_cell2Node in enumerate(cell2NodeList):
                    numpy.copyto(
                        fsBuffer[offsets[i]:offsets[i+1]],
                        np_cell2Node - 1,  # 0-based vertex indexing in FS
                        casting="same_kind",
                    )
                self.fsmesh.InitUnstructCells(cellType, fs_cell2Node, True)

    @ProfileTime
    def InitFSMesh(self, z_nc=None):
        if self.fsmesh is None:
            self.fsmesh = FSMesh(self.clac)
        self.fsmesh.BeginInitialization()
        if Cmpi.size > 1:
            self.cell2ProcDict[FSMeshEnums.CT_Node] = self.InitCell2Proc(self.nvertices)
            self.fsmesh.InitUnstructNodes(
                self.cell2ProcDict[FSMeshEnums.CT_Node]
            )
            foundCellTypes = Cmpi.allgather(list(self.connectivityDict.keys()))
            foundCellTypes = set().union(*foundCellTypes)
            for fsCellType in foundCellTypes:
                ncellsOfType = 0
                if fsCellType in self.connectivityDict:
                    ncellsOfType = sum(self.connectivityDict[fsCellType]["nCells"])
                self.cell2ProcDict[fsCellType] = self.InitCell2Proc(ncellsOfType)
                self.InitFSConnectivity(fsCellType)
        else:
            self.fsmesh.InitUnstructNodes(self.nvertices)
            for fsCellType in self.fsVolumeCellTypes + self.fsSurfaceCellTypes:
                self.InitFSConnectivity(fsCellType)

        if not self.conformal:
            if Cmpi.size > 1:
                self.InitPseudoCell_QuadNQuadMPI(z_nc)
            else:
                self.InitPseudoCell_QuadNQuad(z_nc)

        self.fsmesh.EndInitialization()
        self.InitFSBCCoordinates(
            [FSDataName(self.coordsName)], FSMeshEnums.CT_Node
        )

        if self.nvertices > 0:
            fs_coordinates = self.fsmesh.GetUnstructDataset(
                FSDataName(self.coordsName)
            ).GetValues()
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                self.np_coordinates,
                casting="no",
            )

    @ProfileTime
    def InitIBMDatasets(self):
        if not(self.ibm and self.nvertices > 0):
            return
        # Flis wall distance initialization
        spatial_discretization = self.ibmParameters["spatial discretization"][
            "type"
        ]
        if spatial_discretization == "FV":
            N_IP_per_element = 1
        else:
            try:
                import QuadratureDG as Q
            except ImportError:
                raise ImportError("QuadratureDG module not found.")
            degree = self.ibmParameters["spatial discretization"]["degree"]
            if spatial_discretization == "DG":
                quadratureType = "GaussLegendre"
            elif spatial_discretization == "DGSEM":
                quadratureType = "GaussLobatto"
            else:
                raise ValueError(
                    "InitIBMDatasets: spatial discretization "
                    f"'{spatial_discretization}' not supported: options are "
                    "DG and DGSEM."
                )
            integrationDegree = 2 * degree - 1
            N_IP_per_element = Q.GetReferencePointsHexa(
                integrationDegree, quadratureType
            )[0]

        list_suffix_datasets = [""]
        list_suffix_datasets.extend(range(1, N_IP_per_element))
        for i in range(N_IP_per_element):
            n_flis = Internal.getNodeFromName(
                self.pyTree, "FlisWallDistance" + str(list_suffix_datasets[i])
            )
            np_flisDistance = Internal.getNodeFromName(
                n_flis, "TurbulentDistance"
            )[1]
            quantityName = "FlisWallDistance" + str(list_suffix_datasets[i])
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantityName
            quantitySpecs = FSDataSpecArray(1)
            quantitySpecs[0].Length()

            fs_volumeCellTypes = FSIntArray(len(self.fsVolumeCellTypes))
            numpy.copyto(
                numpy.array(fs_volumeCellTypes.Buffer(), copy=False),
                self.fsVolumeCellTypes,
                casting="same_kind",
            )
            self.fsmesh.InitUnstructDataset(
                quantityName,
                FSDatasetInfo(quantityNames, quantitySpecs, fs_volumeCellTypes),
            )
            fs_var = self.fsmesh.GetUnstructDataset(quantityName).GetValues()
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                np_flisDistance[:, None],
                casting="no",
            )

    @ProfileTime
    def RecoverPointList2BoundaryMarkers(self):
        ibmBCCoordsX = {}
        ibmBCCoordsY = {}
        ibmBCCoordsZ = {}
        ibmBCNames = []
        bcWallCoordsX = []
        bcWallCoordsY = []
        bcWallCoordsZ = []
        BC_wall_names = []
        self.bcEltsDict = {"TRI": [], "QUAD": []}

        n_zoneBC = Internal.getNodeFromType(self.pyTree, "ZoneBC_t")
        if n_zoneBC is None:
            return None
        pytree_bc_nodes = Internal.getNodesFromType(n_zoneBC, "BC_t")
        current_automatic_marker = 1
        
        if self.ibm and self.ibmParameters["IBM type"]["type"] == "local":
            wall_bMarkers = self.ibmParameters["IBM type"]["wall boundary markers"]

        # Loop on BCs
        for bc_node in pytree_bc_nodes:
            bcName = bc_node[0]
            # Get list of BC points: based on name to make the difference
            # between point list and point range
            bc_bMarker = None
            if Internal.getNodeFromType(bc_node, "IndexArray_t") is not None:
                pointlist_node = Internal.getNodeFromType(
                    bc_node, "IndexArray_t"
                )
                point_list = numpy.ravel(pointlist_node[1]) - 1
                bMarker_node = Internal.getNodeFromName(
                    bc_node, "BoundaryMarker"
                )
                if bMarker_node is not None:
                    bc_bMarker = Internal.getValue(bMarker_node)
                else:
                    # Get boundary marker: generate one if it does not exist
                    # otherwise we expect it to be in a user defined node named
                    # "BoundaryMarker"
                    bc_bMarker = current_automatic_marker
                    current_automatic_marker += 1

            elif Internal.getNodeFromType(bc_node, "IndexRange_t") is not None:
                n_er = Internal.getNodeFromType(bc_node, "IndexRange_t")
                if len(n_er[1]) == 1:
                    point_list = numpy.arange(n_er[1][0][0] - 1, n_er[1][0][1])
                elif len(n_er[1]) == 2:
                    point_list = numpy.arange(n_er[1][0] - 1, n_er[1][1])
                bMarker_node = Internal.getNodeFromName(
                    bc_node, "BoundaryMarker"
                )
                if bMarker_node is not None:
                    print(
                        "WARNING: Boundary markers in the CGNS pyTree will "
                        "not be taken into account. Automatic boundary "
                        "markers are defined instead."
                    )
                bc_bMarker = current_automatic_marker
                current_automatic_marker += 1

            # Fill boundary dicts
            if bc_bMarker is None: continue
            self.bcDict[bc_bMarker] = {"name": bcName, "facePL": point_list}

            bcNameSplit = bcName.split(".")
            if len(bcNameSplit) > 1:
                if bcNameSplit[1].startswith("TRI"):
                    self.bcEltsDict["TRI"].append(bc_bMarker)
                elif bcNameSplit[1].startswith("QUAD"):
                    self.bcEltsDict["QUAD"].append(bc_bMarker)

            bc_dataset_node = Internal.getNodeFromType(bc_node, "BCDataSet_t")
            if bc_dataset_node is not None:
                # raise ValueError("RecoverPointList2BoundaryMarkers: "
                #                  "empty BCDataset for IBM.")
                bc_data_nodes = Internal.getNodesFromType(
                    bc_dataset_node, "BCData_t"
                )
                for data_node in bc_data_nodes:
                    fs_bc_dataset_name = data_node[0]
                    if self.ibm and bcName.startswith("IBMWall"):
                        if bcName not in ibmBCCoordsX:
                            ibmBCCoordsX[bcName] = []
                            ibmBCCoordsY[bcName] = []
                            ibmBCCoordsZ[bcName] = []
                        ibmBCNames.append(fs_bc_dataset_name)
                        if self.flipYZAxes:
                            ibmBCCoordsX[bcName].append(data_node[2][0][1])
                            ibmBCCoordsY[bcName].append(data_node[2][2][1])
                            ibmBCCoordsZ[bcName].append(-data_node[2][1][1])
                        else:
                            ibmBCCoordsX[bcName].append(data_node[2][0][1])
                            ibmBCCoordsY[bcName].append(data_node[2][1][1])
                            ibmBCCoordsZ[bcName].append(data_node[2][2][1])

                    elif (
                        self.ibm
                        and self.ibmParameters["IBM type"]["type"] == "local"
                        and bc_bMarker in wall_bMarkers
                    ):
                        BC_wall_names.append(fs_bc_dataset_name)
                        if self.flipYZAxes:
                            bcWallCoordsX.append(data_node[2][0][1])
                            bcWallCoordsY.append(data_node[2][2][1])
                            bcWallCoordsZ.append(-data_node[2][1][1])
                        else:
                            bcWallCoordsX.append(data_node[2][0][1])
                            bcWallCoordsY.append(data_node[2][1][1])
                            bcWallCoordsZ.append(data_node[2][2][1])

        if self.ibm:
            if self.ibmParameters["IBM type"]["type"] == "global":
                return [ibmBCNames, ibmBCCoordsX, ibmBCCoordsY, ibmBCCoordsZ]
            return [  # 'local' formulation
                ibmBCNames,
                ibmBCCoordsX,
                ibmBCCoordsY,
                ibmBCCoordsZ,
                BC_wall_names,
                bcWallCoordsX,
                bcWallCoordsY,
                bcWallCoordsZ,
            ]
        return None

    @ProfileTime
    def InitFSBCs(self, ibmDatasets=None):
        # We now have our point list for each marker so we can init the cell
        # attribute in the fsmesh
        np_markerArray = numpy.zeros(
            self.nsurfaceCells, dtype=Internal.E_NpyInt
        )
        for marker in self.bcDict:
            np_facePL = self.bcDict[marker]["facePL"]
            np_markerArray[np_facePL - self.nvolumeCells] = marker

        offset = 0
        # Loop on surface cell types in the mesh and slice the array above to
        # get the data we need
        for cellType in self.fsSurfaceCellTypes:
            if cellType == FSMeshEnums.CT_Poly2D:
                continue  # TODO
            if (self.bcEltsDict["QUAD"] and self.bcEltsDict["TRI"]):
                cgnsCellNo = FSCGNSConverter.FS2CGNS_CT[cellType]
                cgnsCellType = Internal.eltNo2EltName(cgnsCellNo)[0]
                temp = set(self.bcEltsDict[cgnsCellType])
                res = [
                    i for i, val in enumerate(np_markerArray) if val in temp
                ]  # TODO REFACTOR
                np_marker_array_cell_type = np_markerArray[res]
            elif len(self.fsSurfaceCellTypes) == 1:
                np_marker_array_cell_type = np_markerArray
            else:
                ncells = self.fsmesh.GetNCells(cellType)
                res = numpy.arange(offset, offset + ncells)
                np_marker_array_cell_type = np_markerArray[res]
                offset += ncells

            fs_marker_array_cell_type = FSIntArray(
                np_marker_array_cell_type.shape[0]
            )
            numpy.copyto(
                numpy.array(fs_marker_array_cell_type.Buffer(), copy=False),
                np_marker_array_cell_type,
                casting="same_kind",
            )
            self.fsmesh.InitCellAttribute(
                FS_AT_CADGroupID, cellType, fs_marker_array_cell_type
            )

        # Then we attach our boundary marker to their name in the fsmesh
        ibmBMarkers = []
        ibmNames = []
        for marker in self.bcDict:
            bcName = self.bcDict[marker]["name"]
            self.fsmesh.SetCellAttributeValueName(
                FS_AT_CADGroupID, marker, bcName
            )
            if self.ibm and bcName.startswith("IBMWall"):
                ibmBMarkers.append(marker)
                ibmNames.append(bcName)

        if (
            self.ibm
            and isinstance(ibmDatasets, list)
            and len(ibmDatasets) >= 4
            and ibmNames
        ):
            ibmDataset1 = []
            ibmDataset2 = []
            ibmDataset3 = []
            pointListIBC = []

            for ibmBMarker, ibmName in zip(ibmBMarkers, ibmNames):
                ibmDataset1.append(ibmDatasets[1][ibmName])
                ibmDataset2.append(ibmDatasets[2][ibmName])
                ibmDataset3.append(ibmDatasets[3][ibmName])
                pointListIBC.append(self.bcDict[ibmBMarker]["facePL"])
            ibmDataset1 = numpy.concatenate(ibmDataset1, axis=1)
            ibmDataset2 = numpy.concatenate(ibmDataset2, axis=1)
            ibmDataset3 = numpy.concatenate(ibmDataset3, axis=1)
            pointListIBC = (
                numpy.concatenate(pointListIBC, dtype=Internal.E_NpyInt)
                - self.nvolumeCells
            )

            self.CreateBCCoordinatesDataset(
                ibmDataset1,
                ibmDataset2,
                ibmDataset3,
                ibmDatasets[0],
                self.nsurfaceCells,
                pointListIBC,
            )

            if (
                self.ibmParameters["IBM type"]["type"] == "local"
                and len(ibmDatasets) == 8
            ):
                wall_bMarkers = self.ibmParameters["IBM type"][
                    "wall boundary markers"
                ]
                self.CreateBCCoordinatesDataset(
                    ibmDatasets[5],
                    ibmDatasets[6],
                    ibmDatasets[7],
                    ibmDatasets[4],
                    self.nsurfaceCells,
                    self.bcDict[wall_bMarkers[0]]["facePL"]
                    - self.nvolumeCells,
                )

    @ProfileTime
    def InitFSBCsMPI(self, ibmDatasets=None):
        np_markerArray = numpy.zeros(
            self.nsurfaceCells, dtype=Internal.E_NpyInt
        )
        keys = list(self.bcDict.keys())
        values = [x["name"] for x in self.bcDict.values()]
        gath_values = Cmpi.allgather(values)

        bc_names_all = sorted(
            set([item for row in gath_values for item in row])
        )
        bc_markers_all = numpy.arange(1, len(bc_names_all) + 1).tolist()

        bMarker2BCName2 = {}
        bMarker2PL2 = {}
        for marker, name in zip(bc_markers_all, bc_names_all):
            if name in values:
                idx = keys[values.index(name)]
                bMarker2BCName2[marker] = self.bcDict[idx]["name"]
                bMarker2PL2[marker] = self.bcDict[idx]["facePL"]

        for marker in bMarker2BCName2:
            point_list = bMarker2PL2[marker] - self.nvolumeCells
            np_markerArray[point_list] = marker

        # Loop on surface cell types in the mesh and slice the array above
        # to get the data we need
        for cellType in self.fsSurfaceCellTypes:
            if (self.bcEltsDict["QUAD"] and self.bcEltsDict["TRI"]):
                cgnsCellNo = FSCGNSConverter.FS2CGNS_CT[cellType]
                cgnsCellType = Internal.eltNo2EltName(cgnsCellNo)[0]
                temp = set(self.bcEltsDict[cgnsCellType])
                res = [i for i, val in enumerate(np_markerArray) if val in temp]
                np_marker_array_cell_type = np_markerArray[res]
            else:
                np_marker_array_cell_type = np_markerArray
            fs_marker_array_cell_type = FSIntArray(
                np_marker_array_cell_type.shape[0]
            )
            numpy.copyto(
                numpy.array(fs_marker_array_cell_type.Buffer(), copy=False),
                np_marker_array_cell_type,
                casting="same_kind",
            )
            self.fsmesh.InitCellAttribute(
                FS_AT_CADGroupID, cellType, fs_marker_array_cell_type
            )

        # Then we attach our boundary marker to their name in the fsmesh
        # for marker in bMarker2BCName2.keys():
        ibmBMarkers = []
        ibmNames = []
        if self.ibm:
            fsdatanames = ["WallPointCoordinates", "DonorPointCoordinates"]
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            if self.fsSurfaceCellTypes:
                numpy.copyto(
                    numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting="same_kind",
                )
            self.InitFSBCCoordinates(fsdatanames, fs_surfaceCellTypes)

        for marker, name in zip(bc_markers_all, bc_names_all):
            self.fsmesh.SetCellAttributeValueName(
                FS_AT_CADGroupID, marker, name
            )
            if (
                self.ibm
                and marker in self.bcDict
                and self.bcDict[marker]["name"].startswith("IBMWall")
            ):
                ibmBMarkers.append(marker)
                ibmNames.append(self.bcDict[marker]["name"])

        if (
            self.ibm
            and isinstance(ibmDatasets, list)
            and len(ibmDatasets) >= 4
            and ibmNames
        ):
            ibmDataset1 = []
            ibmDataset2 = []
            ibmDataset3 = []
            pointListIBC = []

            for ibmBMarker, ibmName in zip(ibmBMarkers, ibmNames):
                ibmDataset1.append(ibmDatasets[1][ibmName])
                ibmDataset2.append(ibmDatasets[2][ibmName])
                ibmDataset3.append(ibmDatasets[3][ibmName])
                pointListIBC.append(self.bcDict[ibmBMarker]["facePL"])
            ibmDataset1 = numpy.concatenate(ibmDataset1, axis=1)
            ibmDataset2 = numpy.concatenate(ibmDataset2, axis=1)
            ibmDataset3 = numpy.concatenate(ibmDataset3, axis=1)
            pointListIBC = (
                numpy.concatenate(pointListIBC, dtype=Internal.E_NpyInt)
                - self.nvolumeCells
            )

            self.CreateBCCoordinatesDataset(
                ibmDataset1,
                ibmDataset2,
                ibmDataset3,
                ibmDatasets[0],
                self.nsurfaceCells,
                pointListIBC,
            )

    def InitFSBCCoordinates(self, bcNames, cellType):
        coordNames = FSStringArray(3)
        coordNames[0] = FSDataName.Coordinate().X()
        coordNames[1] = FSDataName.Coordinate().Y()
        coordNames[2] = FSDataName.Coordinate().Z()
        coordSpecs = FSDataSpecArray(3)
        coordSpecs[0].Length()
        coordSpecs[1].Length()
        coordSpecs[2].Length()

        for fsdataname in bcNames:
            self.fsmesh.InitUnstructDataset(
                fsdataname, FSDatasetInfo(coordNames, coordSpecs, cellType)
            )

    def CreateBCCoordinatesDataset(
        self,
        coords_x,
        coords_y,
        coords_z,
        BC_names,
        nb_cell_surf,
        point_list,
    ):  # TODO rewrite entirely for loop
        # point_list = numpy.asarray(point_list) TODO
        if not Cmpi.size > 1:
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            numpy.copyto(
                numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                self.fsSurfaceCellTypes,
                casting="same_kind",
            )
            self.InitFSBCCoordinates(BC_names, fs_surfaceCellTypes)

        # TODO
        # if len(coords_x) > 1:
        #     coords_x = numpy.concatenate(coords_x)
        #     coords_y = numpy.concatenate(coords_y)
        #     coords_z = numpy.concatenate(coords_z)
        # nVertices = coords_x.size
        # np_coordinates = numpy.column_stack((coords_x, coords_y, coords_z))

        for i in range(len(coords_x)):
            coord_x = coords_x[i]
            coord_y = coords_y[i]
            coord_z = coords_z[i]

            nVertices = coord_x.size
            np_coordinates = numpy.append(
                coord_x[:, None], coord_y[:, None], axis=1
            )
            np_coordinates = numpy.append(
                np_coordinates, coord_z[:, None], axis=1
            )
            fs_coordinates = FSFloatArray(nVertices, 3)
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                np_coordinates,
                casting="no",
            )

            dataset = numpy.zeros((nb_cell_surf, 3))
            # dataset[point_list] = np_coordinates # TODO
            for j in range(nVertices):
                dataset[point_list[j]][0] = np_coordinates[j][0]
                dataset[point_list[j]][1] = np_coordinates[j][1]
                dataset[point_list[j]][2] = np_coordinates[j][2]
            fsdataname = FSDataName(BC_names[i])

            fs_var = self.fsmesh.GetUnstructDataset(fsdataname).GetValues()
            fs_var.Fill(0.0)
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                dataset,
                casting="safe",
            )

    @ProfileTime
    def CheckFSMesh(self):
        isMeshOK = self.fsmesh.Check()
        if all(Cmpi.allgather(isMeshOK)):
            if Cmpi.master and self.verbose:
                print("FSMesh successfully created.")
        else:
            if not isMeshOK:
                raise ValueError(
                    f"[{Cmpi.rank}] CheckFSMesh: failed checking the FSMesh."
                )
            sys.exit(1)

    @ProfileTime
    def InitCGNSCoordinates(self):
        if Cmpi.master and self.verbose:
            print("Initializing CGNS PyTree coordinates.")
        self.pyTree = Internal.newCGNSTree()
        base = Internal.newCGNSBase("Base", 3, 3, parent=self.pyTree)
        self.nvertices = self.np_coordinates.shape[0]
        pyTree_zone = Internal.newZone(
            name=f"zone.{Cmpi.rank:d}",
            zsize=[[self.nvertices, self.nvolumeCells, 0]],
            ztype=self.meshType,
            family=None,
            parent=base,
        )

        # Create CGNS node for coordinates
        n_coordinates = Internal.newGridCoordinates(parent=pyTree_zone)
        Internal.newDataArray(
            "CoordinateX", value=self.np_coordinates[:, 0], parent=n_coordinates
        )
        Internal.newDataArray(
            "CoordinateY", value=self.np_coordinates[:, 1], parent=n_coordinates
        )
        Internal.newDataArray(
            "CoordinateZ", value=self.np_coordinates[:, 2], parent=n_coordinates
        )

    @ProfileTime
    def RecoverFSConnectivity(
        self, includeGhostCells=False, includeSurfaceData=True
    ):
        """
        Fetch FS mesh connectivity data from an FS mesh and initialize the
        corresponding class attributes

        Args:
            includeGhostCells: bool; whether to consider ghost cells

            includeSurfaceData: bool; whether to consider surface data
        """
        if Cmpi.master and self.verbose:
            print("Fetching FS mesh connectivity data.")
        self.connectivityDict = {}
        for cellType in self.fsVolumeCellTypes:
            nownedCells = self.fsmesh.GetNOwnedCells(cellType)
            fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
            np_cell2Node = numpy.array(
                fs_cell2Node.Buffer(),
                dtype=Internal.E_NpyInt,
                copy=False
            )
            if not includeGhostCells:
                np_cell2Node = np_cell2Node[:nownedCells]
            self.connectivityDict[cellType] = {
                "nCells": nownedCells,
                "ElementConnectivity": np_cell2Node
            }

        if includeSurfaceData:
            # Get list of BC marker ids
            fs_bMarkers = self.fsmesh.GetCellAttributeValuesWithNames(
                "CADGroupID"
            )
            try:
                np_bMarkers = numpy.array(fs_bMarkers.Buffer(), copy=False)
            except BufferError:
                np_bMarkers = []

            cellType2BMarkersDict = {}
            for cellType in self.fsSurfaceCellTypes:
                # Fill connectivityDict
                nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
                np_cell2Node = numpy.array(
                    fs_cell2Node.Buffer(),
                    dtype=Internal.E_NpyInt,
                    copy=False
                )[:nownedCells]
                self.connectivityDict[cellType] = {
                    "nCells": nownedCells,
                    "ElementConnectivity": np_cell2Node
                }

                if len(np_bMarkers) == 0:
                    continue

                # Fill indirection between cell types and boundary markers
                fs_bMarkerCellType = self.fsmesh.GetCellAttribute(
                    "CADGroupID", cellType
                )
                cellType2BMarkersDict[cellType] = numpy.array(
                    fs_bMarkerCellType.Buffer(),
                    dtype=Internal.E_NpyInt,
                    copy=False
                )[:nownedCells]  # no ghost cells

            for marker in np_bMarkers:
                # BC name truncation because of a limitation of the CGNS format
                # (a 9-char suffix may be added)
                fsbcname = self.bcDict[marker]["name"]
                fsbcname = fsbcname[-23:]
                self.bcDict[marker]["name"] = fsbcname

                # Fill bcDict and connectivityDict for this marker
                offset = 0
                for cellType in self.fsSurfaceCellTypes:
                    nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                    np_bFace2bMarker = cellType2BMarkersDict[cellType]
                    np_markerPositions = numpy.flatnonzero(np_bFace2bMarker == marker)
                    if np_markerPositions.size:
                        self.bcDict[marker].setdefault("cellTypes", []).append(cellType)
                        self.bcDict[marker].setdefault("facePLs", []).append(np_markerPositions + offset)
                        if -marker not in self.connectivityDict:
                            self.connectivityDict[-marker] = {}
                        markerDict = self.connectivityDict[-marker]
                        markerDict.setdefault("nCells", []).append(nownedCells)
                        markerDict.setdefault("facePositions", []).append(np_markerPositions)
                    offset += nownedCells

    @ProfileTime
    def InitCGNSConnectivity(self, includeSurfaceData=True):
        """Initialize CGNS mesh connectivity from FS mesh data"""
        if Cmpi.master and self.verbose:
            print("Initializing CGNS mesh connectivity from FS mesh info.")
        ntotElts = 0
        z = Internal.getZones(self.pyTree)[0]

        for fsCellType in self.fsVolumeCellTypes:
            np_cell2Node = self.connectivityDict[fsCellType]["ElementConnectivity"]
            nCellsOfType, nvpe = np_cell2Node.shape
            if nCellsOfType == 0:
                continue
            cgnsEltNo = FSCGNSConverter.FS2CGNS_CT[fsCellType]
            cgnsEltName = Internal.eltNo2EltName(cgnsEltNo)[0]
            Internal.newElements(
                name="GridElements_" + cgnsEltName,
                etype=cgnsEltName,
                econnectivity=1 + np_cell2Node.ravel(),
                erange=[ntotElts + 1, ntotElts + nCellsOfType],
                eboundary=0,
                parent=z
            )
            ntotElts += nCellsOfType

        if not includeSurfaceData:
            return

        for marker in self.bcDict:
            bcType = self.bcDict[marker]["type"]
            fsCellTypes = self.bcDict[marker].get("cellTypes", [])
            if not fsCellTypes:
                continue

            markerPosList = self.connectivityDict[-marker]["facePositions"]
            for fsCellType, np_markerPos in zip(fsCellTypes, markerPosList):
                np_cell2Node = (
                    self.connectivityDict[fsCellType]["ElementConnectivity"][
                        np_markerPos
                    ]
                )
                nCellsOfType, nvpe = np_cell2Node.shape
                if nCellsOfType == 0:
                    continue
                cgnsEltNo = FSCGNSConverter.FS2CGNS_CT[fsCellType]
                cgnsEltName = Internal.eltNo2EltName(cgnsEltNo)[0]
                eltRange = [ntotElts + 1, ntotElts + nCellsOfType]

                # If cgnsEltName is already in bcName, remove it
                bcName = self.bcDict[marker]["name"].split(".")[0]
                bcName = f"{bcName}.{cgnsEltName}_{marker}"
                Internal.newElements(
                    name=bcName,
                    etype=cgnsEltName,
                    erange=eltRange,
                    econnectivity=1 + np_cell2Node.ravel(),
                    eboundary=nCellsOfType,
                    parent=z
                )

                C._addBC2Zone(z, bcName, bcType, elementRange=eltRange)
                zoneBC = Internal.getNodeFromType(z, "ZoneBC_t")
                lastBCName = C.getLastBCName(bcName)
                n_bc = Internal.getNodeFromName(zoneBC, lastBCName)
                n_bc[0] = bcName
                boundaryStateDataset = Internal.createNode(
                    "BCDataSet", "BCDataSet_t", parent=n_bc, value="Null"
                )
                boundaryState = Internal.createNode(
                    "Boundary", "BCData_t", parent=boundaryStateDataset
                )
                boundaryState[2].append(
                    ["BoundaryMarker", marker, [], "UserDefinedData_t"]
                )
                ntotElts += nCellsOfType

    @ProfileTime
    def RecoverFSFlowSolution(self):
        """
        Fetch FS flow solution data and initialize that of the CGNS pyTree
        """
        datasets = set(str(d) for d in self.fsmesh.GetUnstructDatasetNames())
        if isinstance(self.datasets, set):
            datasets = datasets.intersection(self.datasets)
        if not datasets:
            return

        if Cmpi.master and self.verbose:
            print("Fetching FS flow solution data.")

        for dsName in datasets:
            if self.coordsName[:-1] in dsName:
                continue
            if Cmpi.master and self.verbose:
                print(f"  - Dataset: {dsName}.")
            unstructDataset = self.fsmesh.GetUnstructDataset(dsName)
            fs_cellTypes = unstructDataset.GetCellTypes()
            flowSolutionNames = [
                str(name) for name in unstructDataset.GetNames()
            ]
            fs_flowSolutionValues = unstructDataset.GetValues()

            # Compute the indices of the ghost cells such that they can be removed
            # from the flow solution dataset
            ntotCells = 0
            np_ghostCellsIndices = []
            for cellType in fs_cellTypes:
                nownedCells = self.fsmesh.GetNOwnedCells(cellType)
                nCells = self.fsmesh.GetNCells(cellType)
                np_ghostCellsIndices.append(
                    numpy.arange(
                        ntotCells + nownedCells,
                        ntotCells + nCells,
                        dtype=Internal.E_NpyInt,
                    )
                )
                ntotCells += nCells
            np_ghostCellsIndices = numpy.concatenate(np_ghostCellsIndices)

            np_flowSolutionValues = numpy.array(
                fs_flowSolutionValues.Buffer(), copy=True
            )
            np_flowSolutionValues = numpy.delete(
                np_flowSolutionValues, np_ghostCellsIndices, axis=0
            )

            if fs_cellTypes[0] in self.fsVolumeCellTypes:
                z = Internal.getZones(self.pyTree)[0]
                n_FS = Internal.newFlowSolution(
                    name=f"FlowSolution#{dsName}",
                    gridLocation="CellCenter",
                    parent=z,
                )
                for i, flowSolutionName in enumerate(flowSolutionNames):
                    Internal.newDataArray(
                        f"{flowSolutionName}"[:32],
                        value=np_flowSolutionValues[:, i],
                        parent=n_FS,
                    )
            else:
                zoneBC = Internal.getNodeFromType(self.pyTree, "ZoneBC_t")
                if zoneBC is not None:
                    n_bcs = Internal.getNodesFromType(zoneBC, "BC_t")
                    for i, n_bc in enumerate(n_bcs):
                        suffix = n_bc[0].split(".")[1]
                        cgnsCellName, marker = suffix.split("_")
                        cgnsCellType = Internal.eltName2EltNo(cgnsCellName)[0]
                        cellType = FSCGNSConverter.CGNS2FS_CT.get(cgnsCellType)
                        marker = int(marker)

                        cellTypes = self.bcDict[marker]["cellTypes"]
                        ct = cellTypes.index(cellType)
                        np_facePL = self.bcDict[marker]["facePLs"][ct]

                        bdrStateDataset = Internal.getNodeFromType(
                            n_bc, "BCDataSet_t"
                        )
                        bdrState = Internal.getNodeFromType(
                            bdrStateDataset, "BCData_t"
                        )
                        for j, flowSolutionName in enumerate(flowSolutionNames):
                            Internal.newDataArray(
                                flowSolutionName[:32],
                                value=np_flowSolutionValues[:, j][np_facePL],
                                parent=bdrState
                            )

    @ProfileTime
    def InitFSFlowSolution(self):
        """
        Fetch CGNS flow solution data and initialize that in FS
        """

        def CheckSetInsertion(s, item):
            return len(s) != (s.add(item) or len(s))

        def AddFSUnstructDataset(datasetName, loc, varNames, np_vars):
            if Cmpi.master and self.verbose:
                print(f"  - Dataset: {datasetName}.")
                print(
                    "\n".join(
                        f"    + Variable: {name} @ {loc}." for name in varNames
                    )
                )
            nvars = len(varNames)
            fsVarNames = FSStringArray(nvars)
            for i, name in enumerate(varNames):
                fsVarNames[i] = FSDataName(name)
            fsVarSpecs = FSDataSpecArray(nvars)
            if loc == "CellCenter":
                fs_loc = FSIntArray(len(self.fsVolumeCellTypes))
                numpy.copyto(
                    numpy.array(fs_loc.Buffer(), copy=False),
                    self.fsVolumeCellTypes,
                    casting="same_kind",
                )
            elif loc == "FaceCenter":
                fs_loc = FSIntArray(len(self.fsSurfaceCellTypes))
                numpy.copyto(
                    numpy.array(fs_loc.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting="same_kind",
                )
            else:
                fs_loc = FSIntArray(1)
                fs_loc[0] = FSMeshEnums.CT_Node

            self.fsmesh.InitUnstructDataset(
                datasetName, FSDatasetInfo(fsVarNames, fsVarSpecs, fs_loc)
            )
            try:
                fs_vars = numpy.asarray(
                    self.fsmesh.GetUnstructDataset(
                        datasetName
                    ).GetValues().Buffer()
                )
                if loc == "CellCenter":
                    for cgnsStart, fsdmStart, count in self.eltRangeMap.values():
                        numpy.copyto(
                            fs_vars[fsdmStart:fsdmStart+count, :],
                            np_vars[cgnsStart:cgnsStart+count, :],
                            casting="no"
                        )
                else:
                    numpy.copyto(fs_vars, np_vars, casting="no")
            except BufferError:
                pass

        if Cmpi.size > 1: # TODO
            return

        if not self.datasets:
            return
        elif self.datasets == "all":
            self.datasets = set(CGNS_CONTAINER_NAMES)
        if Cmpi.master and self.verbose:
            print("Fetching CGNS flow solution data.")

        z = Internal.getZones(self.pyTree)[0]
        n_flowSolns = Internal.getNodesFromType1(z, "FlowSolution_t")
        if n_flowSolns == []:
            return

        foundVarNames = set()
        for datasetName in self.datasets:
            if datasetName in CGNS_CONTAINER_NAMES:
                n = Internal.getNodeFromName1(z, datasetName)
                if n is None:
                    continue
                n_loc = Internal.getNodeFromType1(n, "GridLocation_t")
                if n_loc is None:
                    loc = "Node"
                else:
                    loc = Internal.getValue(n_loc)
                n_vars = Internal.getNodesFromType1(n, "DataArray_t")
                varNames = []
                np_vars = []
                for n_var in n_vars:
                    varName = Internal.getName(n_var)
                    if self.coordsName[:-1] in varName:
                        continue
                    if not CheckSetInsertion(foundVarNames, f"{loc}:{varName}"):
                        continue
                    varNames.append(varName)
                    np_vars.append(Internal.getValue(n_var))
                if not np_vars:
                    continue
                np_vars = numpy.stack(np_vars, axis=-1)
                AddFSUnstructDataset(datasetName, loc, varNames, np_vars)
            else:
                if ":" in datasetName:
                    varLoc, varName = datasetName.split(":")[:2]
                    if varLoc == "nodes":
                        varLoc = "Node"
                    elif varLoc == "centers":
                        varLoc = "CellCenter"
                else:
                    varLoc, varName = None, datasetName
                varNames = []
                np_vars = []
                n_flowSolns = Internal.getNodesFromType1(z, "FlowSolution_t")
                for n_flowSoln in n_flowSolns:
                    n_loc = Internal.getNodeFromType1(
                        n_flowSoln, "GridLocation_t"
                    )
                    if n_loc is None:
                        loc = "Node"
                    else:
                        loc = Internal.getValue(n_loc)
                    if varLoc is not None and varLoc != loc:
                        continue
                    n_var = Internal.getNodeFromName1(n_flowSoln, varName)
                    if n_var is None:
                        continue
                    if self.coordsName[:-1] in varName:
                        continue
                    if not CheckSetInsertion(foundVarNames, f"{loc}:{varName}"):
                        continue
                    np_var = Internal.getValue(n_var)[:, None]
                    AddFSUnstructDataset(
                        f"{loc}:{varName}", loc, [varName], np_var
                    )

    @ProfileTime
    def InitPseudoCell_QuadNQuad(self, z_nc):
        locNCFaces = (
            Internal.getNodeFromName(z_nc, "ElementConnectivity")[1] - 1
        )
        lenNCF = len(locNCFaces) // 4
        np_coordsX = Internal.getNodeFromName(z_nc, "CoordinateX")[1]
        np_coordsY = Internal.getNodeFromName(z_nc, "CoordinateY")[1]
        np_coordsZ = Internal.getNodeFromName(z_nc, "CoordinateZ")[1]

        np_coords = numpy.column_stack((np_coordsX, np_coordsY, np_coordsZ))
        ncFacesCentroids = ComputeQuadCentroids(
            np_coordsX, np_coordsY, np_coordsZ, locNCFaces
        )

        locNCFaces = numpy.reshape(locNCFaces, (lenNCF, 4))
        if self.dimPb == 2:
            createQuadNQuad = CreateQuad2Quad
        else:
            createQuadNQuad = CreateQuad4Quad
        locQNQList = createQuadNQuad(
            np_coords, locNCFaces, ncFacesCentroids
        )  # plane, tol

        Internal._rmNodesFromType(self.pyTree, "Elements_t")
        hook = C.createHook(self.pyTree, "nodes")
        ids = C.identifyNodes(hook, z_nc)
        ids = ids[ids != -1] - 1
        qnqList = ids[locQNQList]

        fs_cell2Node = FSIntArray(*qnqList.shape)
        numpy.copyto(
            numpy.array(fs_cell2Node.Buffer(), copy=False),
            qnqList,
            casting="same_kind",
        )
        if self.dimPb == 2:
            self.fsmesh.InitUnstructCells(
                FSMeshEnums.PCT_Quad2Quad, fs_cell2Node, False
            )
        else:
            self.fsmesh.InitUnstructCells(
                FSMeshEnums.PCT_Quad4Quad, fs_cell2Node, False
            )

    def InitPseudoCell_QuadNQuadMPI(self, z_nc):
        if Cmpi.master and self.verbose:
            print("Creating QuadNQuad pseudo connectivity.")
        rank = self.clac.ProcID()
        tol = 1e-10
        decimals = int(-numpy.log10(tol))
        if self.dimPb == 2:
            fsCellType = FSMeshEnums.PCT_Quad2Quad
            createQuadNQuad = CreateQuad2Quad
        else:
            fsCellType = FSMeshEnums.PCT_Quad4Quad
            createQuadNQuad = CreateQuad4Quad

        if z_nc is not None:
            z_nc[0] += str(rank)
            np_coordsX = Internal.getNodeFromName(z_nc, "CoordinateX")[1]
            np_coordsY = Internal.getNodeFromName(z_nc, "CoordinateY")[1]
            np_coordsZ = Internal.getNodeFromName(z_nc, "CoordinateZ")[1]
        else:
            np_coordsX = numpy.empty(0)
            np_coordsY = numpy.empty(0)
            np_coordsZ = numpy.empty(0)

        gath_np_coordsX = Cmpi.allgather(np_coordsX)
        gath_np_coordsY = Cmpi.allgather(np_coordsY)
        gath_np_coordsZ = Cmpi.allgather(np_coordsZ)
        lenNCF = len(np_coordsX)
        del np_coordsX, np_coordsY, np_coordsZ

        self.cell2ProcDict[fsCellType] = self.InitCell2Proc(lenNCF)
        if z_nc is not None:
            locNCFaces = (
                Internal.getNodeFromName(z_nc, "ElementConnectivity")[1]
                - 1
                + self.cell2ProcDict[fsCellType][rank]
            )
        else:
            locNCFaces = numpy.empty(0, dtype=Internal.E_NpyInt)

        gath_locNCFaces = Cmpi.allgather(locNCFaces)
        del locNCFaces

        locQNQList = []
        uniqueCoords = numpy.empty((0, 3))
        dedupMap = numpy.empty((0), dtype=Internal.E_NpyInt)

        if Cmpi.master:
            gath_np_coordsX = numpy.concatenate(gath_np_coordsX)
            gath_np_coordsY = numpy.concatenate(gath_np_coordsY)
            gath_np_coordsZ = numpy.concatenate(gath_np_coordsZ)
            gath_locNCFaces = numpy.concatenate(gath_locNCFaces)
            shape = (len(gath_np_coordsX), 1)
            np_coords = numpy.hstack(
                [
                    gath_np_coordsX.reshape(*shape),
                    gath_np_coordsY.reshape(*shape),
                    gath_np_coordsZ.reshape(*shape),
                ]
            )

            # Use lexicographical order to sort by z, then y, then x
            lexOrder = numpy.lexsort(
                numpy.around(np_coords, decimals=decimals).T
            )
            np_sortedCoords = np_coords[lexOrder]
            nvertices = len(np_sortedCoords)

            # Find unique (sorted) coordinates and their indices
            uniqueMask = numpy.ones(nvertices, dtype=bool)
            uniqueMask[1:] = numpy.any(
                numpy.abs(numpy.diff(np_sortedCoords, axis=0)) > tol, axis=1
            )
            uniqueIndices = numpy.where(uniqueMask)[0]
            uniqueCoords = np_sortedCoords[uniqueMask]

            # Compute the gap (ie, number of duplicates) for each unique vertex
            duplicatesCount = numpy.diff(numpy.append(uniqueIndices, nvertices))

            # Map each duplicate to its corresponding unique vertex index
            # dedupMap: original -> unique
            dedupMap = numpy.full(nvertices, -1, dtype=Internal.E_NpyInt)
            dedupMap[lexOrder] = numpy.repeat(
                numpy.arange(len(uniqueCoords)), duplicatesCount
            )
            # dedup2dup: unique -> representative original
            dedup2dup = lexOrder[uniqueIndices]

            lenNCF = len(gath_locNCFaces) // 4
            gath_locNCFaces = dedupMap[gath_locNCFaces]
            ncFacesCentroids = ComputeQuadCentroids(
                uniqueCoords[:, 0],
                uniqueCoords[:, 1],
                uniqueCoords[:, 2],
                gath_locNCFaces,
            )
            gath_locNCFaces = numpy.reshape(gath_locNCFaces, (lenNCF, 4))

            locQNQList = createQuadNQuad(
                uniqueCoords, gath_locNCFaces, ncFacesCentroids
            )
            locQNQList = dedup2dup[locQNQList]

        self.cell2ProcDict[fsCellType] = self.InitCell2Proc(len(locQNQList))
        Internal._rmNodesFromType(self.pyTree, "Elements_t")
        if z_nc is not None:
            hook = C.createHook(self.pyTree, "nodes")
            ids = C.identifyNodes(hook, z_nc)
        else:
            ids = numpy.empty(0, dtype=Internal.E_NpyInt)
        ids = ids[ids > -1] - 1 + self.cell2ProcDict[1][rank]
        gath_ids = numpy.concatenate(Cmpi.allgather(ids))

        if Cmpi.master:
            qnqList = gath_ids[locQNQList]
            fs_cell2Node = FSIntArray(*qnqList.shape)
            numpy.copyto(
                numpy.array(fs_cell2Node.Buffer(), copy=False),
                qnqList,
                casting="same_kind",
            )
        else:
            fs_cell2Node = FSIntArray(0, FSCellInfo.NNodes(fsCellType))
        self.fsmesh.InitUnstructCells(
            fsCellType, self.cell2ProcDict[fsCellType], fs_cell2Node, False
        )
        return None

    @ProfileTime
    def DeduplicateFSNodesMPI(self):
        tol = 1e-10
        decimals = int(-numpy.log10(tol))

        fs_coords = self.fsmesh.GetUnstructDataset("Coordinates").GetValues()
        np_coords = numpy.array(fs_coords.Buffer(), copy=True)
        np_coords = ArrayOps.Gather(np_coords, self.clac)

        uniqueCoords = numpy.empty((0, 3))
        dedupMap = numpy.empty((0), dtype=Internal.E_NpyInt)

        if Cmpi.master:
            # Use lexicographical order to sort by z, then y, then x
            lexOrder = numpy.lexsort(
                numpy.around(np_coords, decimals=decimals).T
            )
            np_sortedCoords = np_coords[lexOrder]
            nvertices = len(np_sortedCoords)

            # Find unique (sorted) coordinates and their indices
            uniqueMask = numpy.ones(nvertices, dtype=bool)
            uniqueMask[1:] = numpy.any(
                numpy.abs(numpy.diff(np_sortedCoords, axis=0)) > tol, axis=1
            )
            uniqueIndices = numpy.where(uniqueMask)[0]
            uniqueCoords = np_sortedCoords[uniqueMask]

            # Compute the gap (ie, number of duplicates) for each unique vertex
            duplicatesCount = numpy.diff(numpy.append(uniqueIndices, nvertices))

            # Map each duplicate to its corresponding unique vertex index
            dedupMap = numpy.full(nvertices, -1, dtype=Internal.E_NpyInt)
            dedupMap[lexOrder] = numpy.repeat(
                numpy.arange(len(uniqueCoords)), duplicatesCount
            )

        if Cmpi.master:
            print("Removing duplicated vertices from mesh connectivity.")
        cell2Proc_nodes = self.InitCell2Proc(len(uniqueCoords))
        dedupMap = ArrayOps.Broadcast(dedupMap, self.clac)
        fsCellTypesNC = self.fsVolumeCellTypes + self.fsSurfaceCellTypes
        if not self.conformal:
            if self.dimPb == 2:
                fsCellTypesNC.append(15)
            else:
                fsCellTypesNC.append(16)
        cell2NodeDict = {}
        for cellType in fsCellTypesNC:
            fs_cell2Node = self.fsmesh.GetCell2Node(cellType)
            np_cell2Node = numpy.array(fs_cell2Node.Buffer(), copy=True)
            cell2NodeDict[cellType] = dedupMap[np_cell2Node]
        gath_fsCellTypesNC = Cmpi.allgather(fsCellTypesNC)
        fsCellTypesNC = set(
            ct for cellTypes in gath_fsCellTypesNC for ct in cellTypes
        )

        fs_bMarkerList = self.fsmesh.GetCellAttributeValuesWithNames(
            "CADGroupID"
        )
        names = []
        fsCellTypeMarkersDict = {}
        npCellTypeMarkersDict = {}
        try:
            np_bMarkers = numpy.array(fs_bMarkerList.Buffer(), copy=True)
            for cellType in self.fsSurfaceCellTypes:
                fs_markers_array_cell_type = self.fsmesh.GetCellAttribute(
                    "CADGroupID", cellType
                )
                np_markers_array_cell_type = numpy.array(
                    fs_markers_array_cell_type.Buffer(), copy=True
                )
                npCellTypeMarkersDict[cellType] = np_markers_array_cell_type
                fs_markers_array_cell_type = FSIntArray(
                    len(np_markers_array_cell_type)
                )
                for i in range(len(np_markers_array_cell_type)):
                    fs_markers_array_cell_type[i] = int(
                        np_markers_array_cell_type[i]
                    )
                fsCellTypeMarkersDict[cellType] = fs_markers_array_cell_type

            fs_bMarkerList = FSIntArray(len(np_bMarkers))
            for i in range(len(np_bMarkers)):
                fs_bMarkerList[i] = int(np_bMarkers[i])

            for marker in fs_bMarkerList:
                names.append(
                    self.fsmesh.GetCellAttributeValueName("CADGroupID", marker)
                )
        except BufferError:
            pass

        if self.ibm:
            if self.fsVolumeCellTypes:
                flis_distance = self.fsmesh.GetUnstructDataset(
                    "FlisWallDistance"
                ).GetValues()
                np_flisDistance = numpy.array(flis_distance.Buffer(), copy=True)
            else:
                np_flisDistance = numpy.empty(0)

            fsDataNames = ["WallPointCoordinates", "DonorPointCoordinates"]
            datasets = []
            for varName in fsDataNames:
                if self.fsSurfaceCellTypes:
                    dataset = numpy.array(
                        self.fsmesh.GetUnstructDataset(varName)
                        .GetValues()
                        .Buffer(),
                        copy=True,
                    )
                else:
                    dataset = numpy.empty(0)
                datasets.append(dataset)

        self.fsmesh.Reset()
        self.fsmesh.BeginInitialization()
        self.fsmesh.InitUnstructNodes(cell2Proc_nodes)

        for cellType in fsCellTypesNC:
            ncellsOfType = 0
            if cellType in cell2NodeDict:
                np_cell2Node = cell2NodeDict[cellType]
                fs_cell2Node = FSIntArray(*np_cell2Node.shape)
                ncellsOfType = np_cell2Node.shape[0]
                cell2Proc = self.InitCell2Proc(ncellsOfType)
                if ncellsOfType > 0:
                    numpy.copyto(
                        numpy.array(fs_cell2Node.Buffer(), copy=False),
                        np_cell2Node,
                        casting="same_kind",
                    )
            else:
                cell2Proc = self.InitCell2Proc(0)
                fs_cell2Node = FSIntArray(0, FSCellInfo.NNodes(cellType))
            self.fsmesh.InitUnstructCells(
                cellType, cell2Proc, fs_cell2Node, True
            )

        self.fsmesh.EndInitialization()
        self.InitFSBCCoordinates(
            [FSDataName(self.coordsName)], FSMeshEnums.CT_Node
        )
        if uniqueCoords.shape[0] > 0:
            fs_coordinates = self.fsmesh.GetUnstructDataset(
                FSDataName(self.coordsName)
            ).GetValues()
            numpy.copyto(
                numpy.array(fs_coordinates.Buffer(), copy=False),
                uniqueCoords,
                casting="no",
            )

        for name, marker in zip(names, fs_bMarkerList):
            self.fsmesh.SetCellAttributeValueName(
                FS_AT_CADGroupID, marker, name
            )
        for cellType in self.fsSurfaceCellTypes:
            self.fsmesh.InitCellAttribute(
                FS_AT_CADGroupID, cellType, fsCellTypeMarkersDict[cellType]
            )

        if self.ibm:
            quantityName = "FlisWallDistance"
            quantityNames = FSStringArray(1)
            quantityNames[0] = quantityName
            quantitySpecs = FSDataSpecArray(1)
            quantitySpecs[0].Length()
            fs_volumeCellTypes = FSIntArray(len(self.fsVolumeCellTypes))
            if self.fsVolumeCellTypes:
                numpy.copyto(
                    numpy.array(fs_volumeCellTypes.Buffer(), copy=False),
                    self.fsVolumeCellTypes,
                    casting="same_kind",
                )

            self.fsmesh.InitUnstructDataset(
                quantityName,
                FSDatasetInfo(quantityNames, quantitySpecs, fs_volumeCellTypes),
            )
            if self.fsVolumeCellTypes:
                fs_var = self.fsmesh.GetUnstructDataset(
                    quantityName
                ).GetValues()
                numpy.copyto(
                    numpy.array(fs_var.Buffer(), copy=False),
                    np_flisDistance,
                    casting="no",
                )
            fs_surfaceCellTypes = FSIntArray(len(self.fsSurfaceCellTypes))
            if self.fsSurfaceCellTypes:
                numpy.copyto(
                    numpy.array(fs_surfaceCellTypes.Buffer(), copy=False),
                    self.fsSurfaceCellTypes,
                    casting="same_kind",
                )
            self.InitFSBCCoordinates(fsDataNames, fs_surfaceCellTypes)
            if self.fsSurfaceCellTypes:
                for varName, dataset in zip(fsDataNames, datasets):
                    fs_var = self.fsmesh.GetUnstructDataset(
                        varName
                    ).GetValues()
                    numpy.copyto(
                        numpy.array(fs_var.Buffer(), copy=False),
                        dataset,
                        casting="no",
                    )

    def FreeResources(self):
        """Delete class attributes that are no longer needed"""
        del (
            self.fsmesh,
            self.np_coordinates,
            self.connectivityDict
        )

    @ProfileTime
    def MergeBCsByMarker(self, tol=1e-11):
        if Cmpi.master and self.verbose:
            print("Merging BCs: one CGNS BC per boundary marker.")
        familyNames = []
        familyTypes = []
        fsMarkers = []

        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
        n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")

        for n_bc in n_bcs:
            bcName = Internal.getName(n_bc)
            bcType = Internal.getValue(n_bc)
            familyName, marker = bcName.split(".")[:2]
            marker = marker.split("_")[1]
            familyNameMarker = f"{familyName}_{marker}"
            if marker not in fsMarkers:
                fsMarkers.append(marker)
                familyNames.append(familyNameMarker)
                familyTypes.append(bcType)

            Internal.createChild(
                n_bc,
                "FamilyName",
                "FamilyName_t",
                value=familyNameMarker,
                pos=0,
            )
            Internal.setValue(n_bc, "FamilySpecified")

        zbcs = []
        FS = Internal.getNodesFromType(self.pyTree, "FlowSolution_t")
        Internal._rmNodesByType(self.pyTree, "FlowSolution_t")
        for familyName in familyNames:
            zbc = C.extractBCOfType(
                self.pyTree, "FamilySpecified:" + familyName
            )
            zbc = T.join(zbc)
            zbcs.append([zbc])
        C._recoverBCs(
            self.pyTree,
            (zbcs, familyNames, familyTypes),
            tol=tol,
            removeBC=True,
        )
        z = Internal.getZones(self.pyTree)[0]
        for FS_node in FS:
            Internal._addChild(z, FS_node, pos=-1)  # at the end

    def MergeBCsFamilySpecified(self):
        if Cmpi.master and self.verbose:
            print(
                "Specifying FamilySpecified BCs: one FamilyName per boundary marker."
            )
        BCs = []
        base = Internal.getNodeFromType1(self.pyTree, "CGNSBase_t")
        z = Internal.getZones(self.pyTree)[0]
        n_zoneBCs = Internal.getNodesFromType1(z, "ZoneBC_t")
        n_bcs = Internal.getNodesFromType1(n_zoneBCs, "BC_t")

        for n_bc in n_bcs:
            bcName = Internal.getName(n_bc)
            bcType = Internal.getValue(n_bc)
            familyName = bcName.split(".")[0]
            BCs.append((bcName, bcType, familyName))
            Internal.createChild(
                n_bc, "FamilyName", "FamilyName_t", value=familyName, pos=0
            )
            Internal.setValue(n_bc, "FamilySpecified")

        for bc in BCs:
            if (
                Internal.getNodesFromNameAndType(self.pyTree, bc[2], "Family_t")
                == []
            ):
                n_family = Internal.createNode(bc[2], "Family_t", parent=base)
                Internal.createChild(
                    n_family, "FamilyBC", "FamilyBC_t", value=bc[1], pos=0
                )

    @ProfileTime
    def ReorderCells(self):
        """Reorder CGNS volume and surface element types as in FSDM"""
        def ReorderVolumicCells__(cellType2RangeDict):
            fsdmOrder = [10, 12, 14, 17]  # TETRA, PYRA, PENTA, HEXA
            remap = {}
            fsdmStart = 0
            for cellType in fsdmOrder:
                if cellType not in cellType2RangeDict:
                    continue
                cgnsRange = cellType2RangeDict[cellType]
                cgnsStart = cgnsRange[0] - 1
                count = cgnsRange[1] - cgnsRange[0] + 1
                remap[cellType] = (cgnsStart, fsdmStart, count)
                fsdmStart += count
            return remap
        # Loop over volume and surface element types and store their range
        cellType2RangeDict = {}
        n_elts = Internal.getNodesFromType(self.pyTree, "Elements_t")
        for n_elt in n_elts:
            cellType = Internal.getValue(n_elt)[0]
            eltRange = Internal.getNodeFromName(n_elt, "ElementRange")[1]
            cellType2RangeDict[cellType] = eltRange

        # Sort element types by minimum range
        cellType2RangeDict = dict(
            sorted(cellType2RangeDict.items(), key=lambda item: item[1][0])
        )

        # Detect if the minimum element range of surface element types is less
        # than that of volume element types, in which case CGNS element types
        # must be rearranged: surface elements come last in FSDM
        areCellTypesOrdered = True
        cellTypeList = list(cellType2RangeDict.keys())
        for i, cellType in enumerate(cellTypeList):
            if cellType in [5, 7]:
                if any(ct not in [5, 7] for ct in cellTypeList[i + 1:]):
                    areCellTypesOrdered = False
                    break

        if areCellTypesOrdered:
            self.eltRangeMap = ReorderVolumicCells__(cellType2RangeDict)
            return None

        if Cmpi.master and self.verbose:
            print("Reording CGNS element types to follow that of FSDM.")

        # Loop over all cell types to determine the new element ranges
        ntotCells = 0
        newCellType2RangeDict = {}
        cellTypeList = [ct for ct in cellTypeList if ct not in [5, 7]] + [
            ct for ct in cellTypeList if ct in [5, 7]
        ]

        for cellType in cellTypeList:
            eltRange = cellType2RangeDict[cellType]
            nCells = eltRange[1] - eltRange[0] + 1
            newCellType2RangeDict[cellType] = [
                ntotCells + 1,
                ntotCells + nCells,
            ]
            ntotCells += nCells

        self.eltRangeMap = ReorderVolumicCells__(newCellType2RangeDict)

        # Loop over volume and surface element types to assign their new range
        for n_elt in n_elts:
            cellType = Internal.getValue(n_elt)[0]
            eltRange = Internal.getNodeFromName(n_elt, "ElementRange")
            eltRange[1] = newCellType2RangeDict[cellType]

        # Loop over all CGNS BC nodes and offset vertex point lists
        # using the difference between new and old
        n_bcs = Internal.getNodesFromType(self.pyTree, "BC_t")
        origRangeTri = cellType2RangeDict.get(5)
        newRangeTri = newCellType2RangeDict.get(5)
        origRangeQuad = cellType2RangeDict.get(7)
        newRangeQuad = newCellType2RangeDict.get(7)
        for n_bc in n_bcs:
            ptList = Internal.getNodeFromName(n_bc, "PointList")
            if ptList is None:
                continue
            np_vertexPL = ptList[1].copy()

            # Mark which faces are tris and which are quads
            if newRangeTri is not None and newRangeQuad is not None:
                isQuad = numpy.logical_and(
                    np_vertexPL >= origRangeQuad[0],
                    np_vertexPL <= origRangeQuad[1],
                )
                np_vertexPL[isQuad] += newRangeQuad[0] - origRangeQuad[0]
                np_vertexPL[~isQuad] += newRangeTri[0] - origRangeTri[0]
            elif newRangeQuad is not None:
                np_vertexPL[:] += newRangeQuad[0] - origRangeQuad[0]
            else:
                np_vertexPL[:] += newRangeTri[0] - origRangeTri[0]

            ptList[1][:] = np_vertexPL

    def Export(self, filename, **kwargs):
        """Export to file based on the filename extension"""
        validExtensions = ["cgns", "h5", "plt"]
        ext = filename.split(".")[-1]
        if ext == "cgns":
            self.ExportCGNS(filename=filename, **kwargs)
        elif ext == "h5":
            self.ExportFSMesh(filename=filename, **kwargs)
        elif ext == "plt":
            self.__Export2Tecplot(filename=filename)
        else:
            raise ValueError(
                "FSCGNSConverter.FSCGNSConverter.export: Input "
                "filename does not have a valid extension. It can "
                "either be {}.".format(", ".join(e for e in validExtensions))
            )

    def ExportFSMesh(self, filename="", verbose=False):
        """Export FSMesh to file"""
        if Cmpi.master and self.verbose:
            print("Export FS mesh.")
        if filename:
            if filename.endswith(".h5"):
                filename = filename[:-3]
        elif isinstance(self.meshName, str):
            # Mesh saved in the same dir. as the input mesh
            filename = self.meshName.split(".")[0]
        else:
            filename = "t"
        if not self.fsmesh.ExportMeshHDF5(Filename=filename + ".h5"):
            FSError.PrintAndExit()
        if verbose:
            self.fsmesh.PrintInfo()

    def ExportFSMesh2Tecplot(self, filename=""):
        """Export FS mesh to tecplot format"""
        if Cmpi.master and self.verbose:
            print("Export FS mesh to tecplot format.")
        ext = "plt"
        if filename:
            if filename.endswith(".pt"):
                ext = "pt"
            if any(filename.endswith(i) for i in [".h5", ".plt", ".pt"]):
                filename = filename.rsplit(".", 1)[0]
        elif isinstance(self.meshName, str):
            # Output saved in the same dir. as the input mesh
            filename = self.meshName.rsplit(".", 1)[0]
        else:
            filename = "t"

        volumeCellTypes = tuple(
            FSMeshEnums.CellTypeToString(vct) for vct in self.fsVolumeCellTypes
        )
        if not self.fsmesh.ExportMeshTECPLOT(
            Filename=f"{filename}_vol.{ext}",
            PrefixDatasetName=True,
            ExportCellTypes=volumeCellTypes,
        ):
            FSError.PrintAndExit()

        surfaceCellTypes = tuple(
            FSMeshEnums.CellTypeToString(sct) for sct in self.fsSurfaceCellTypes
        )
        if not self.fsmesh.ExportMeshTECPLOT(
            Filename=f"{filename}_surf.{ext}",
            PrefixDatasetName=True,
            ZonePerCellAttributeValue=True,
            CellAttribute=FS_AT_CADGroupID,
            UseCellAttributeValueName=True,
            ExportCellTypes=surfaceCellTypes,
        ): 
            FSError.PrintAndExit()

    def ExportCGNS(self, filename="", verbose=False):
        """Export CGNS to file"""
        if Cmpi.master and self.verbose:
            print("Export CGNS mesh.")
        if filename:
            if filename.endswith(".cgns"):
                filename = filename[:-5]
        elif isinstance(self.meshName, str):
            filename = self.meshName.rsplit(".", 1)[
                0
            ]  # Mesh saved in the same dir. as the input mesh
        else:
            filename = "t"
        Cmpi.convertPyTree2File(self.pyTree, filename + ".cgns")
        if verbose:
            Internal.printTree(self.pyTree)

    def ExportCGNS2Tecplot(self, filename=""):
        """Export CGNS to tecplot format"""
        if Cmpi.master and self.verbose:
            print("Export CGNS mesh to tecplot format.")
        if filename:
            if filename.endswith(".cgns"):
                filename = filename[:-5]
            elif filename.endswith(".plt"):
                filename = filename[:-4]
        elif isinstance(self.meshName, str):
            filename = self.meshName.split(".")[
                0
            ]  # Output saved in the same dir. as the input mesh
        else:
            filename = "t"
        if Cmpi.size == 1:
            C.convertPyTree2File(self.pyTree, f"{filename}.plt")
        else:
            C.convertPyTree2File(self.pyTree, f"{filename}_{Cmpi.rank:03d}.plt")

    # Lower-case aliases for Cassiopee users
    convert = Convert
    convert2CGNS = Convert2CGNS
    convert2FSDM = Convert2FSDM
    export = Export
    exportFSMesh = ExportFSMesh
    exportFSMesh2Tecplot = ExportFSMesh2Tecplot
    exportCGNS = ExportCGNS
    exportCGNS2Tecplot = ExportCGNS2Tecplot
