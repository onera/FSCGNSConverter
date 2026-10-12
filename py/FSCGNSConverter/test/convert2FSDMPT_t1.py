# - convert2FSDMPT (pyTree) -
import os
import Converter.PyTree as C
import Generator.PyTree as G
from KCore.test import getLocal

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter
from FSPlugins.test import testH5

LOCAL = getLocal()

# Simple box, Elements, from pyTree
N = 5
meshName = os.path.join(LOCAL, "out.h5")

# - BE -
# Tetra BE
t = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 4)

# Penta BE
t = G.cartPenta((0., 0., 0.), (1., 1., 1.), (N, N, N))
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 5)

# Pyra BE
t = G.cartPyra((0., 0., 0.), (1., 1., 1.), (N, N, N))
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 6)

# Hexa BE
t = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 7)

# - ME -
# Hexas
a = G.cartHexa((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartHexa((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity(a, b, boundary=0)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 8)

# All BE
a = G.cartTetra((0., 0., 0.), (1., 1., 1.), (N, N, N))
b = G.cartPenta((N-1., 0., 0.), (1., 1., 1.), (N, N, N))
c = G.cartPyra((2.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
d = G.cartHexa((3.*(N-1.), 0., 0.), (1., 1., 1.), (N, N, N))
t = C.mergeConnectivity([a, b, c, d], boundary=0)
convObj = FSCGNSConverter(pyTree=t)
convObj.convert()
testH5(convObj.clac, convObj.fsmesh, 9)
