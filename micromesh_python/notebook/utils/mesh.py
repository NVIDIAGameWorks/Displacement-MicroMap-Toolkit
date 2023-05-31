import numpy as np
from numba import njit, prange
from pxr import Usd, UsdGeom, Sdf, Vt
from typing import Tuple

import micromesh_python as pymm

from nvMicromesh import DisplacementMicromapAPI

def print_mesh_stats(mesh : pymm.Mesh):
    print(f"triangleVertices: {mesh.triangleVertices.shape[0]}")
    print(f"vertexPositions: {mesh.vertexPositions.shape[0]}")
    print(f"vertexNormals: {mesh.vertexNormals.shape[0]}")
    print(f"vertexTexcoords0: {mesh.vertexTexcoords0.shape[0]}")
    print(f"vertexTangents: {mesh.vertexTangents.shape[0]}")
    print(f"vertexDirections: {mesh.vertexDirections.shape[0]}")
    print(f"vertexDirectionBounds: {mesh.vertexDirectionBounds.shape[0]}")
    print(f"vertexImportance: {mesh.vertexImportance.shape[0]}")
    print(f"triangleSubdivisionLevels: {mesh.triangleSubdivisionLevels.shape[0]}")
    print(f"trianglePrimitiveFlags: {mesh.trianglePrimitiveFlags.shape[0]}")

# Sanity check for mesh compatibility
def is_prim_compatible(prim : Usd.Prim) -> bool:
    if not prim.IsA(UsdGeom.Mesh):
        return False
    mesh = UsdGeom.Mesh(prim)
    primvar_api = UsdGeom.PrimvarsAPI(prim)
    
    # Ensure it is a triangle mesh
    vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.uint)
    if vertex_counts.min() != 3 or vertex_counts.max() != 3:
        return False
    
    # Ensure both normals and UV coordinates are per-vertex
    if mesh.GetNormalsInterpolation() != 'vertex':
        return False
    
    # Ensure uv-coordinates are also per-vertex
    if primvar_api.HasPrimvar("st"):
        if primvar_api.GetPrimvar("st").GetInterpolation() != 'vertex':
            return False

    return True

# Flip the t-component to/from USD standard
#   https://graphics.pixar.com/usd/release/spec_usdpreviewsurface.html#texture-coordinate-orientation-in-usd
@njit(parallel=True)
def t_flip(array : np.array) -> np.array:
    flipped_array = np.empty_like(array)
    for i in prange(array.shape[0]):
        flipped_array[i][0] = array[i][0]
        flipped_array[i][1] = 1 - array[i][1]
    return flipped_array

# Create a Micromesh SDK Python API mesh object from a USD mesh prim
def get_mesh(prim : Usd.Prim) -> Tuple[pymm.Mesh, np.array]:
    if not is_prim_compatible(prim):
        print(f"Primitive {prim.GetPrimPath()} is not compatible")

    mesh_prim = UsdGeom.Mesh(prim)
    primvar_api = UsdGeom.PrimvarsAPI(prim)

    mesh = pymm.Mesh()
    
    # USD Vt arrays are easily converted to numpy arrays
    mesh.vertexPositions = np.array(mesh_prim.GetPointsAttr().Get(), dtype=float)
    mesh.vertexNormals = np.array(mesh_prim.GetNormalsAttr().Get(), dtype=float)
    if mesh.vertexNormals.shape == () or mesh.vertexNormals.min() == np.nan or mesh.vertexNormals.max() == np.nan:
        raise RuntimeError(f"{prim.GetPrimPath()}: Meshes without normals are not supported")

    # Convert the UV coordinates
    if primvar_api.HasPrimvar("st"):
        uv_primvar = primvar_api.GetPrimvar("st")
        if uv_primvar.GetInterpolation() != 'vertex':
            raise RuntimeError(f"{prim.GetPrimPath()}: only per-vertex UV coordinates are supported")
        mesh.vertexTexcoords0 = np.array(primvar_api.GetPrimvar("st").Get(), dtype=float)
        mesh.vertexTexcoords0 = t_flip(mesh.vertexTexcoords0)
        
    mesh.triangleVertices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.uint)
    mesh.triangleVertices = mesh.triangleVertices.reshape((int(len(mesh_prim.GetFaceVertexIndicesAttr().Get())/3), 3))

    # Get the transform matrix
    xform_cache = UsdGeom.XformCache()
    transform = xform_cache.GetLocalToWorldTransform(prim)
    mesh_transform = np.array([transform.GetColumn(0),
                              transform.GetColumn(1),
                              transform.GetColumn(2),
                              transform.GetColumn(3)], dtype=float)

    return mesh, mesh_transform

def create_mesh(prim_path : str, mesh : pymm.Mesh, stage : Usd.Stage) -> Usd.Prim:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
        prim = mesh_prim.GetPrim()
    else:
        mesh_prim = UsdGeom.Mesh(prim)

    mesh_primvar_api = UsdGeom.PrimvarsAPI(prim)

    mesh_prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertexPositions))
    mesh_prim.SetNormalsInterpolation('vertex')
    mesh_prim.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertexNormals))
    mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(mesh.triangleVertices))
    mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full((mesh.triangleVertices.shape[0]), mesh.triangleVertices.shape[1])))
    if mesh_primvar_api.HasPrimvar("st") or mesh_primvar_api.CreatePrimvar("st", Sdf.ValueTypeNames.Float2Array):
        uv_primvar = mesh_primvar_api.GetPrimvar("st")
        uv_primvar.SetInterpolation('vertex')
        # Flip the t-component to USD standard
        #   https://graphics.pixar.com/usd/release/spec_usdpreviewsurface.html#texture-coordinate-orientation-in-usd
        uv_primvar.Set(Vt.Vec2fArray.FromNumpy(t_flip(mesh.vertexTexcoords0)))
        
    return prim
        
def store_micromesh_primvars(micromesh_data : pymm.MicromeshData, prim : Usd.Prim):
    min_subdiv_level = int(micromesh_data.minSubdivLevel)
    max_subdiv_level = int(micromesh_data.maxSubdivLevel)
    bias = float(micromesh_data.bias)
    scale = float(micromesh_data.scale)

    direction_bounds = Vt.Vec2fArray.FromNumpy(micromesh_data.vertexDirectionBounds)
    values = Vt.UCharArray.FromNumpy(micromesh_data.values)
    
    value_format = int(micromesh_data.valueFormat)
    value_layout = int(micromesh_data.valueLayout)
    value_frequency = int(micromesh_data.valueFrequency)
    value_count = int(micromesh_data.valueCount)
    value_byte_size = int(micromesh_data.valueByteSize)
    value_byte_alignment = int(micromesh_data.valueByteAlignment)
    
    triangle_value_offsets = Vt.UIntArray.FromNumpy(micromesh_data.triangleValueOffsets)
    triangle_subdiv_levels = Vt.UIntArray.FromNumpy(micromesh_data.triangleSubdivLevels)
    triangle_block_formats = Vt.UIntArray.FromNumpy(micromesh_data.triangleBlockFormats)
    histogram_entry_counts = Vt.UIntArray.FromNumpy(micromesh_data.histogramEntryCounts)
    histogram_entry_subdiv_levels = Vt.UIntArray.FromNumpy(micromesh_data.histogramEntrySubdivLevels)
    histogram_entry_block_formats = Vt.UIntArray.FromNumpy(micromesh_data.histogramEntryBlockFormats)
    triangle_flags = Vt.UCharArray.FromNumpy(micromesh_data.triangleFlags)
    
    triangle_flag_format = int(micromesh_data.triangleFlagFormat)
    triangle_flag_count = int(micromesh_data.triangleFlagCount)
    triangle_flag_byte_size = int(micromesh_data.triangleFlagByteSize)
    triangle_flag_byte_alignment = int(micromesh_data.triangleFlagByteAlignment)
    
    triangle_min_maxs = Vt.UCharArray.FromNumpy(micromesh_data.triangleMinMaxs)
    
    triangle_min_max_format = int(micromesh_data.triangleMinMaxFormat)
    triangle_min_max_count = int(micromesh_data.triangleMinMaxCount)
    triangle_min_max_byte_size = int(micromesh_data.triangleMinMaxByteSize)
    triangle_min_max_byte_alignment = int(micromesh_data.triangleMinMaxByteAlignment)
    
    if micromesh_data.vertexDirections is not None and micromesh_data.vertexDirections.shape[0] > 0:
        directions = Vt.Vec3fArray.FromNumpy(micromesh_data.vertexDirections)
    else:
        directions = None

    try:
        from nvMicromesh import DisplacementMicromapAPI

        if not prim.HasAPI(DisplacementMicromapAPI):
            DisplacementMicromapAPI.Apply(prim)

        micromesh_prim = DisplacementMicromapAPI(prim)

        micromesh_prim.GetPrimvarsMicromeshDirectionBoundsAttr().Set(direction_bounds)
        micromesh_prim.GetPrimvarsMicromeshDirectionsAttr().Set(directions)
        micromesh_prim.GetPrimvarsMicromeshFloatBiasAttr().Set(bias)
        micromesh_prim.GetPrimvarsMicromeshFloatScaleAttr().Set(scale)
        micromesh_prim.GetPrimvarsMicromeshHistogramBlockFormatsAttr().Set(histogram_entry_block_formats)
        micromesh_prim.GetPrimvarsMicromeshHistogramCountsAttr().Set(histogram_entry_counts)
        micromesh_prim.GetPrimvarsMicromeshHistogramSubdivLevelsAttr().Set(histogram_entry_subdiv_levels)
        micromesh_prim.GetPrimvarsMicromeshMinSubdivLevelAttr().Set(min_subdiv_level)
        micromesh_prim.GetPrimvarsMicromeshMaxSubdivLevelAttr().Set(max_subdiv_level)
        micromesh_prim.GetPrimvarsMicromeshTriangleFlagsAttr().Set(triangle_flags)
        micromesh_prim.GetPrimvarsMicromeshTriangleFlagsCountAttr().Set(triangle_flag_count)
        micromesh_prim.GetPrimvarsMicromeshTriangleFlagsByteSizeAttr().Set(triangle_flag_byte_size)
        micromesh_prim.GetPrimvarsMicromeshTriangleFlagsFormatAttr().Set(triangle_flag_format)
        micromesh_prim.GetPrimvarsMicromeshTriangleMinMaxsAttr().Set(triangle_min_maxs)
        micromesh_prim.GetPrimvarsMicromeshTriangleMinMaxsCountAttr().Set(triangle_min_max_count)
        micromesh_prim.GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr().Set(triangle_min_max_byte_size)
        micromesh_prim.GetPrimvarsMicromeshTriangleMinMaxsFormatAttr().Set(triangle_min_max_format)
        micromesh_prim.GetPrimvarsMicromeshTriangleValueOffsetsAttr().Set(triangle_value_offsets)
        micromesh_prim.GetPrimvarsMicromeshTriangleSubdivLevelsAttr().Set(triangle_subdiv_levels)
        micromesh_prim.GetPrimvarsMicromeshTriangleBlockFormatsAttr().Set(triangle_block_formats)
        micromesh_prim.GetPrimvarsMicromeshValuesAttr().Set(values)
        micromesh_prim.GetPrimvarsMicromeshValueFormatAttr().Set(value_format)
        micromesh_prim.GetPrimvarsMicromeshValueLayoutAttr().Set(value_layout)
        micromesh_prim.GetPrimvarsMicromeshValueFrequencyAttr().Set(value_frequency)
        micromesh_prim.GetPrimvarsMicromeshValueCountAttr().Set(value_count)
        micromesh_prim.GetPrimvarsMicromeshValueByteSizeAttr().Set(value_byte_size)
        micromesh_prim.GetPrimvarsMicromeshVersionAttr().Set(100)
    except:
        # Unable to load the USD schema module (perhaps it was not built)
        # Use the generic primvars API instead
        primvarsapi = UsdGeom.PrimvarsAPI(prim)
        primvarsapi.CreatePrimvar('micromesh:minSubdivLevel', Sdf.ValueTypeNames.UInt).Set(min_subdiv_level)
        primvarsapi.CreatePrimvar('micromesh:maxSubdivLevel', Sdf.ValueTypeNames.UInt).Set(max_subdiv_level)
        primvarsapi.CreatePrimvar('micromesh:directionBounds', Sdf.ValueTypeNames.Float2Array).Set(direction_bounds)
        primvarsapi.CreatePrimvar('micromesh:directions', Sdf.ValueTypeNames.Float3Array).Set(directions)
        primvarsapi.CreatePrimvar('micromesh:floatScale', Sdf.ValueTypeNames.Float).Set(scale)
        primvarsapi.CreatePrimvar('micromesh:floatBias', Sdf.ValueTypeNames.Float).Set(bias)
        primvarsapi.CreatePrimvar('micromesh:histogramCounts', Sdf.ValueTypeNames.UIntArray).Set(histogram_entry_counts)
        primvarsapi.CreatePrimvar('micromesh:histogramSubdivLevels', Sdf.ValueTypeNames.UIntArray).Set(histogram_entry_subdiv_levels)
        primvarsapi.CreatePrimvar('micromesh:histogramBlockFormats', Sdf.ValueTypeNames.UIntArray).Set(histogram_entry_block_formats)
        primvarsapi.CreatePrimvar('micromesh:triangleFlagsFormat', Sdf.ValueTypeNames.UInt).Set(triangle_flag_format)
        primvarsapi.CreatePrimvar('micromesh:triangleFlagsCount', Sdf.ValueTypeNames.UInt).Set(triangle_flag_count)
        primvarsapi.CreatePrimvar('micromesh:triangleFlagsByteSize', Sdf.ValueTypeNames.UInt).Set(triangle_flag_byte_size)
        primvarsapi.CreatePrimvar('micromesh:triangleFlags', Sdf.ValueTypeNames.UCharArray).Set(triangle_flags)
        primvarsapi.CreatePrimvar('micromesh:triangleMinMaxsFormat', Sdf.ValueTypeNames.UInt).Set(triangle_min_max_format)
        primvarsapi.CreatePrimvar('micromesh:triangleMinMaxsCount', Sdf.ValueTypeNames.UInt).Set(triangle_min_max_count)
        primvarsapi.CreatePrimvar('micromesh:triangleMinMaxsByteSize', Sdf.ValueTypeNames.UInt).Set(triangle_min_max_byte_size)
        primvarsapi.CreatePrimvar('micromesh:triangleMinMaxs', Sdf.ValueTypeNames.UCharArray).Set(triangle_min_maxs)
        primvarsapi.CreatePrimvar('micromesh:triangleValueOffsets', Sdf.ValueTypeNames.UIntArray).Set(triangle_value_offsets)
        primvarsapi.CreatePrimvar('micromesh:triangleSubdivLevels', Sdf.ValueTypeNames.UIntArray).Set(triangle_subdiv_levels)
        primvarsapi.CreatePrimvar('micromesh:triangleBlockFormats', Sdf.ValueTypeNames.UIntArray).Set(triangle_block_formats)
        primvarsapi.CreatePrimvar('micromesh:valueLayout', Sdf.ValueTypeNames.UInt).Set(value_layout)
        primvarsapi.CreatePrimvar('micromesh:valueFrequency', Sdf.ValueTypeNames.UInt).Set(value_frequency)
        primvarsapi.CreatePrimvar('micromesh:valueFormat', Sdf.ValueTypeNames.UInt).Set(value_format)
        primvarsapi.CreatePrimvar('micromesh:valueCount', Sdf.ValueTypeNames.UInt).Set(value_count)
        primvarsapi.CreatePrimvar('micromesh:valueByteSize', Sdf.ValueTypeNames.UInt).Set(value_byte_size)  
        primvarsapi.CreatePrimvar('micromesh:values', Sdf.ValueTypeNames.UCharArray).Set(values)
