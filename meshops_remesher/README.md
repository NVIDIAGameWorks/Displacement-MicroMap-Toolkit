# Remesher
The remesher prepares high resolution meshes for use with micromeshes. It currently decimates the input mesh and outputs target micromesh subdivision levels and displacement bounds. This information can then be used by the micromesh baker to generate a lightweight, micromesh-enabled model. The remesher fully executes on graphics hardware for interactive performance: the model below has been decimated in 700 ms using an NVIDIA GeForce RTX 3090Ti. 

The parameters described below can be found in the `OpRemesh_input` structure. 

Original 2M triangles      |  Decimated 20K triangles
:---------------------------------:|:---------------------------------------:
![](docs/remesher_intro_full.png)  |  ![](docs/remesher_intro_decimated.png)

## Features

* Edge collapse decimation
* Parallel execution
* Preservation of attributes discontinuities
* Outputs micromesh metadata to drive micromesh bakers

## Global Stopping Criteria

The remesher collapses all edges until such collapsed incur a maximum error. This error  is a mixture of resulting edge length, a user-defined, per-vertex importance metric, the resulting vertex valence and a maximum decimation level. 

### **Overall Error:  `errorThreshold`**
The maximum error value can be directly specified using this parameter. This value is dependent on the scale of the scene.

### **Maximum Triangle Count: `maxOutputTriangleCount`**
A simpler stopping criterion is a target triangle count for the decimated mesh. In practice this will gradually increase the error threshold until reaching the maximum triangle count. Note that due to the massively parallel nature of the underlying algorithms the resulting triangle count may be significantly lower than this maximum value.

Original 2M triangles | Target 500K &rarr; 355K | Target 50K &rarr; 40K | Target 5K &rarr; 4.5K | Target 1.5K &rarr; 1.4K
:--------------------:|:-----------------------:|:---------------------:|:---------------------:|:-----------------------:
![](docs/dragon_head_original.png) | ![](docs/dragon_head_500k_355k.png) | ![](docs/dragon_head_50k_40k.png) | ![](docs/dragon_head_5k_4k.png) | ![](docs/dragon_head_1k_1k.png)

## Local Stopping Criteria

The decision to collapse an edge is not only based on the above error metric, but also uses local criteria to determine whether the edge can be collapsed. 

### **Maximum decimation level:  `maxSubdivLevel`**
An edge can be collapsed only if the resulting triangles would not cover more than `4^maxSubdivLevel` triangles in the original mesh. 

### **Target displacement map resolution:  `heightmapTextureWidth` and `heightmapTextureHeight`**
If provided along with the texture coordinate index `heightmapTextureCoord` this extends the principle of `maxSubdivLevel` to the UV domain: an edge collapse will be blocked if it would result in triangles covering more than `4^maxSubdivLevel` texels.

### **Maximum vertex valence:  `maxVertexValence`**
Limits the valence of the vertices resulting from an edge collapse, which helps avoiding large triangle fans in the final result. Note this will not reduce the valence of the input vertices even if their valence exceeds this threshold.

Vertex Valence | Original Mesh | Unlimited valence | Max valence 11 
:-------------:|:-------------:|:-----------------:|:--------------:
| | ![](docs/valence_original.png) | ![](docs/valence_unlimited.png) | ![](docs/valence_11.png)

### **Vertex Importance**

The remesher supports per-vertex importance, whose value is integrated in the error calculation. The higher the importance value (between 0 and 1) of the vertices of an edge, the higher the collapse error metric. The per-vertex importance is provided as part of the `ResizableMeshView`, and can be [generated automatically](#vertex-importance-generation). 

The `importanceWeight` parameter defines the weight of the per-vertex importance in the estimate of the error incurred by edge collapse operations. 

| Importance Weight | 0.05 | 0.1 | 0.2
:-------------------:|:-:|:--:|:--:
| | ![](docs/importance_weight_5.png) | ![](docs/importance_weight_10.png) | ![](docs/importance_weight_20.png) 


If defined, `importanceThreshold` forbids edge collapses involving a vertex whose importance exceeds this threshold. This can be particularly useful to enforce the preservation of fine features.

Vertex Importance | Original Mesh | No Max | Max 0.9 
:-------------:|:-------------:|:-----------------:|:--------------:
| | ![](docs/max_importance_original.png) | ![](docs/max_importance_off.png) | ![](docs/max_importance_on.png)


## Attribute Preservation

The `preservedVertexAttributeFlags` value defines which vertex attributes discontinuities are preserved by the remeshing process. This typically allows preservation of UV islands, so the textures designed for the input mesh can still be used on the decimated mesh. 

This parameter supports combinations of a subset of `MeshAttributeFlagBits`: `eMeshAttributeVertexNormalBit`, `eMeshAttributeVertexTangentBit`, 
`eMeshAttributeVertexDirectionBit`, and `eMeshAttributeVertexTexcoordBit`. 


| Texture Map | Original 100K triangles | Decimated 20K triangles 
:-------------------:|:-:|:--:
|![](docs/seams_texture.jpg) | ![](docs/seams_original.png) | ![](docs/seams_decimated.png) 


## Micromesh Metadata

During decimation the remesher gathers information linking the input and decimated meshes, such as the input vertices covered by each decimated triangle and the size of the shell encompassing the input triangles. This data is then used to deduce a target subdivision level and displacement bounds for micromeshes.

| Original Mesh | Decimated Result | Target Subdivision Level | Displacement Bounds 
:--------------:|:----------------:|:------------------------:|:-------------------
|![](docs/micromesh_original_wireframe.png) | ![](docs/micromesh_decimated_wireframe.png) | ![](docs/micromesh_subd_level.png) | ![](docs/micromesh_bounds.png) 

The generated shell may bound the original mesh very tightly. To ensure a successfull micromesh baking the `directionBoundsFactor` may be used to enlarge the displacement bounds, avoiding floating-point approximation artifacts.


This metadata export can be enabled and disabled using the **`generateMicromeshInfo`** parameter.

## Limitations

### Mesh Topology
The input mesh is expected to have manifold topology. Topological issues such as T-junctions, overlapping triangles, or edges connected to more than 2 triangles will result in undefined behavior. 

Distinct connected components will be remeshed independently, even though they visually appear connected (e.g. assembled mechanical parts). After remeshing some visible gaps may appear between those connected components. 

### High Decimation Rates
By default the remesher attempts to preserve the volume of the input mesh by offsetting the position of the vertices along their displacement direction. While this reduces shrinking, decimating a mesh to a very low triangle count may result in excessive offsets. If needed this mechanism can be disabled by setting `fitToOriginalSurface` to `false`. 

### Displacement Bounds
The displacement bounds approximation may underestimate the required displacement, especially when the number of triangles in the decimated mesh is close to the one of the original mesh. Adjusting `directionBoundsFactor` accordingly works around this limitation.

### Memory
The remesher fully operates on graphics hardware, and requires some scratch memory. Decimating large meshes can therefore fail due to insufficient memory. In our tests we could decimate meshes up to 70M triangles on a 24GB RTX 3090Ti. 

# Vertex Importance Generation

The per-vertex importance used by the remesher can be either computed externally and provided in the input `ResizableMeshView`, or generated using `meshopsOpGenerateImportance`, whose parameters are defined in `OpGenerateImportance_modified`. 

The importance generation is based either on a local curvature metric obtained by tracing rays above and below each vertex, or using an input importance map. 

### **Maximum Curvature Ray Tracing Distance: `rayTracingDistance`**
The world-space tracing distance defines the scale of the features preserved during decimation. 

Tracing Distance | 0.05 | 0.1 | 0.2
:-------------------:|:-:|:--:|:--:
| | ![](docs/trace_distance_5.png) | ![](docs/trace_distance_10.png) | ![](docs/trace_distance_20.png) 


### **Curvature Power: `importancePower`**
 The power value is applied to the curvature values obtained by ray tracing, allowing for fine adjustment of the curvature contrast. 

Curvature Power | 0.5 | 1.0 | 5.0
:-------------------:|:-:|:--:|:--:
| | ![](docs/curvature_power_5.png) | ![](docs/curvature_power_10.png) | ![](docs/curvature_power_50.png) 


### **Importance Map: `importanceTexture`**
This texture object is an external importance source, mapped using the texture coordinates indicated by `importanceTextureCoord`. If non-null, this importance map supersedes the curvature estimation. 


| Importance Map | Per-Vertex Importance | Decimated Result 
:-------------------:|:-:|:--:
|![](docs/importance_map.png) | ![](docs/importance_map_preview.png) | ![](docs/importance_map_decimated.png) | ![](docs/importance_weight_20.png) 


# Acknowledgements

The models shown in this page are under Creative Commons licence. The "Plastic Dragon" and "Dark Finger Reef Crab" models are available on [Three D Scans](https://threedscans.com/). The "Panulirus Longpipes" model is available on [The Smithsonian 3D Digitization](https://3d.si.edu/).