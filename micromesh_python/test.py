# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os, io, sys
import pathlib
import argparse
import struct
import numpy as np

parser = argparse.ArgumentParser(description='Runs tests for micromesh_toolkit python bindings')
parser.add_argument('--moduleDir', required=True, type=pathlib.Path, help='Directory containing the .so (linux) or .pyd (windows) module')
parser.add_argument('--resultsDir', required=True, type=pathlib.Path, help='Path to write results to')
parser.add_argument('--meshDir', required=True, type=pathlib.Path, help='Asset directory for test scenes')

args = parser.parse_args()

sys.path.insert(0, str(args.moduleDir))

print("Entering " + os.path.dirname(os.path.realpath('__file__')), file=sys.stderr)

import micromesh_python as pymm

print("micromesh_python module objects:", file=sys.stderr)
for name in dir(pymm):
    print("  " + name, file=sys.stderr)

settings = pymm.BakerSettings()
settings.level = 3
settings.maxTraceLength = 0.0
settings.enableCompression = False
settings.minPSNR = 50.0
settings.subdivMethod = pymm.SubdivMethod.Uniform
settings.uniDirectional = True

baseMesh = pymm.Mesh()
baseMesh.vertexPositions = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
baseMesh.vertexNormals = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
baseMesh.vertexDirectionBounds = np.array([[0,1], [0,1], [0,1]], dtype=float)
baseMesh.vertexDirections = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
baseMesh.vertexTexcoords0 = np.array([[0,0],[1,0],[0,1]], dtype=float)
baseMesh.triangleVertices = np.array([[0,1,2]], dtype=np.uint)
baseMeshTransform = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=float)

referenceMesh = pymm.Mesh()
# Same vertices as the baseMesh except the last is raised up (+Z) by 1
referenceMesh.vertexPositions = np.array([[0,0,0],[1,0,0],[0,1,1]], dtype=float)
referenceMesh.vertexNormals = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
referenceMesh.vertexDirectionBounds = np.array([[0,1], [0,1], [0,1]], dtype=float)
referenceMesh.vertexDirections = np.array([[0,0,1], [0,0,1], [0,0,1]], dtype=float)
referenceMesh.vertexTexcoords0 = np.array([[0,0],[1,0],[0,1]], dtype=float)
referenceMesh.triangleVertices = np.array([[0,1,2]], dtype=np.uint)
referenceMeshTransform = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=float)

meshDir = pathlib.Path(args.meshDir).resolve()
resultsDir = pathlib.Path(args.resultsDir).resolve()

bakeInput = pymm.BakerInput()
bakeInput.settings = settings
bakeInput.baseMesh = baseMesh
bakeInput.baseMeshTransform = baseMeshTransform
bakeInput.referenceMesh = referenceMesh
bakeInput.referenceMeshTransform = referenceMeshTransform
bakeInput.heightmap.filepath = str(meshDir / "cracks" / "height.png")
bakeInput.heightmap.bias = 0.0
bakeInput.heightmap.scale = 1.0
bakeInput.normalMapFilepath = str(resultsDir / "normal.png")
bakeInput.normalMapResolution = 32
bakeInput.uvRemapFilepath = str(resultsDir / "uvremap.png")
bakeInput.uvRemapResolution = 32

textureInputs = ["height.png", "height.png"]
textureOutputs = ["outHeight.png", "outHeight2.png"]

resampledWidth = 256
resampledHeight = 256

resamplerInputs = []
for inputTexture, outputTexture in zip(textureInputs, textureOutputs):
    resamplerInput = pymm.ResamplerInput()
    resamplerInput.input.filepath = str(meshDir / "cracks" / inputTexture)
    resamplerInput.input.type = pymm.TextureType.Generic
    resamplerInput.output.filepath = str(resultsDir / outputTexture)
    resamplerInput.output.width = resampledWidth
    resamplerInput.output.height = resampledHeight
    resamplerInputs.append(resamplerInput)
if len(resamplerInputs) > 0:
    bakeInput.resamplerInput = resamplerInputs

context = pymm.createContext(verbosity=pymm.Verbosity.Info)

bakeOutput = pymm.bakeMicromesh(context, bakeInput)

print("Bake Finished")
displacementBytes = bakeOutput.values.tobytes()
displacements = struct.unpack('{}f'.format(bakeOutput.valueCount), displacementBytes)
print(displacements)

assert abs(displacements[2] - 1.0) < 0.01, "third vertex should have a displacement of 1.0"

for resamplerInput in resamplerInputs:
    assert os.path.exists(resamplerInput.output.filepath), f"resampled texture {resamplerInput.output.filepath} does not exist"
    size = os.path.getsize(resamplerInput.output.filepath)
    with io.open(resamplerInput.output.filepath, "rb") as file:
        data = file.read(26)
        # Ensure it's a (modern) PNG file
        assert (size >= 24) and data.startswith(b'\211PNG\r\n\032\n') and (data[12:16] == b'IHDR'), f"resampled texture isn't a .png"
        # Unpack the resolution from the magic header
        w, h = struct.unpack(">LL", data[16:24])
        assert (w == resampledWidth) and (h == resampledHeight), f"resampled texture {resamplerInput.output.filepath} size isn't correct"

sys.stdout.flush()
print("Leaving " + os.path.dirname(os.path.realpath('__file__')), file=sys.stderr)

context = None
