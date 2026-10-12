# FSCGNSConverter

FSCGNSConverter is a mesh conversion Flow-Simulator/Cassiopee module that
facilitates data exchange between the CODA ecosystem and ONERA's CGNS pre- and
post-processing tools [Cassiopee](https://github.com/onera/Cassiopee).
It is a newer iteration of FSMeshPyTreeConversion offering an improved API and
increased performance.

<br>

## Table of Contents
1. [API](#1-api)  
1.1 [Instantiation](#11-instantiation)  
1.2 [Class methods](#12-class-methods)  
1.2.1 [Conversion](#121-conversion)  
1.2.2 [Export](#122-export)  
1.3 [Class attributes](#13-class-attributes)  

2. [Examples](#2-examples)  
2.1 [Conversion from h5 to cgns](#21-conversion-from-h5-to-cgns)  
2.2 [Conversion from cgns to h5](#22-conversion-from-cgns-to-h5)  
2.2.1 [Using a filename](#221-using-a-filename)  
2.2.1 [From a CGNS tree](#222-from-a-cgns-tree)  

<br>

## 1. API

The API of the FSCGNSConverter class is given below and includes  
  - constructor arguments: a list of mandatory and optional parameters required to create an instance of the class;  
  - public methods/attributes: the functions/variables that can be called on the instance.

### 1.1 Instantiation

The names of all the input arguments for the `FSCGNSConverter` class are listed
hereafter:

```py
convObj = FSCGNSConverter(
    meshName=None,
    pyTree=None,
    clac=None,
    fsmesh=None,
    flipYZAxes=False,
    conformal=True,
    IBMParameters=None,
    dimPb=2,
    datasets='all',
    bcDict=None,
    coordsName="Coordinates",
    **kwargs
)
```

| Input argument | Name in FSMeshPyTreeConversion | Type | Default | Description |
|--------------|------------|--------------|--------------|-------------|
| meshName | mesh_name | str | None | Mesh filename |
| pyTree | pytree | CGNS PyTree / base / zone | None | Input CGNS mesh |
| clac | | object | None, ie, `FSClac` | FSDataManager's `clac` object |
| fsmesh | | object | None | FSDataManager's `fsmesh` object |
| flipYZAxes | invertPlanesYZ | bool | False | Whether to flip the y- and z-axes |
| conformal | | bool | True | Whether the mesh presents hanging nodes |
| IBMParameters | IBM_parameters | dict | None | Dictionary of IBM parameters |
| dimPb | | int | 2 | Problem dimension (only useful for IBM) |
| datasets | whichDatasets | str, list or set | 'all' | Names of the datasets to consider during conversion |
| bcDict | dict_BCs | dict | None | Dictionary mapping FS BC markers to pairs of CGNS BC names and types |
| coordsName | coords_name  | str | "Coordinates" | Name of the Coordinates field |
| verbose | x | bool | True | Whether to display the step-by-step script progression |

NB: Input arguments `inmemory`, `keepFlowSolution` and `IBM` were deleted.

### 1.2 Class methods

#### 1.2.1 Conversion

There is a single function to convert the mesh from one format to the other:

```py
convObj.Convert(**kwargs)
```

The resulting format was inferred during the class instantiation.  
When converting to CGNS, the keyworded arguments are:
 - `forOverset`: `bool`, Whether convert for overset cases (default: `False`)
 - `forFFDX`: `bool`, Whether convert for use in FFD / FFX (Multiple-Elements -> monozone NGon, normals pointing inwards) (default: `False`)

#### 1.2.2 Export

There is a single function to convert the mesh from one format to the other:

```py
convObj.Export(filename=outfile, **kwargs)
```

The output format is inferred from the filename's extension.  
When exporting the CGNS PyTree or the FS mesh, the keyworded arguments are:
 - `verbose`: `bool`, print CGNS tree / FS mesh info (default: `True`)

### 1.3 Class attributes

After calling the `Convert` function, the CGNS pyTree or the FS mesh can be obtained with:

```py
t = convObj.pyTree
fsmesh = convObj.fsmesh
```

<br>

## 2. Examples

CGNS files comprising both the mesh and solution fields can be converted to
h5 format, and vice-versa.

### 2.1 Conversion from h5 to cgns

In this example, an h5 file is converted to cgns format.

```py
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshName = "mesh.h5"
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(meshName=meshName, **caseConfig)
convObj.Convert()

# Get the CGNS tree or export the CGNS mesh
t = convObj.pyTree
convObj.Export(filename="t.cgns", verbose=True)
```

When there are boundary conditions, a mapping between FS marker indices and 
CGNS BC types is required in the form of a `bcDict` dictionary. A BC name can 
be renamed by providing a pair `[BCName, BCType]` instead of a `BCType` string.

```py
bcDict = {
    "1": ["myInlet", "BCInflow"],  # BC name given (max. 32 chars), has precedence over name in the input file
    "2": "BCWallInviscid",
    "3": "BCFarfield",
    "4": "BCOutflow"
}
```

### 2.2 Conversion from cgns to h5

#### 2.2.1 Using a filename 

In this example, a CGNS file is converted to h5 format.

```py
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshName = "mesh.cgns"
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(meshName=meshName, **caseConfig)
convObj.Convert()

# Get the FS mesh or export the FS mesh
fsmesh = convObj.fsmesh
convObj.Export(filename="mesh.h5", verbose=True)
```

#### 2.2.2 From a CGNS tree 

In this example, a CGNS tree is converted to h5 format and saved to tecplot format.

```py
import Converter.PyTree as C
from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

meshName = "mesh.cgns"
t = C.convertFile2PyTree(meshName)
caseConfig = {
    # list of optional arguments
}

convObj = FSCGNSConverter(pyTree=t, **caseConfig)
convObj.Convert()

# Save to tecplot format
convObj.Export(filename="mesh.plt")
```
