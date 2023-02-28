# Baker

The baker, `meshopsOpBake()`, generates displacement distances to transform a
*base* mesh into a given *reference* mesh (after tessellation). The tessellation
scheme is specific to micromeshes. Base triangles are subdivided to a given
level into micro-triangles. The baker then raytraces from the micro-vertices
along per-triangle direction vectors until an intersection with the reference
mesh is found. `VK_KHR_acceleration_structure` is required for this. The
distances, relative to the direction vector length, are written to a BaryData
object. These are typically then compressed into a .bary file.

Distances will always be in the unit range, 0 to 1. If per-vertex direction
bounds do not exist, the BaryGroup's bias and scale is used to define their
range. This is uncommon as per-vertex bounds are required for micromeshes to
perform well.

It is common to use [`meshops_remesher`](../meshops_remesher) to generate the
base mesh. In such cases, textures may only exist for the reference mesh. These
cannot simply be applied to the base mesh even though the texture coordinates
may be similar, especially in the case of normal maps where tangent space
changes dramatically. [Resampling](#resampler) textures uses the same reference
geometry and can be performed at the same time as baking.

## Bounds fitting

Direction bounds are a per-vertex bias and scale pair affecting the displacement
start and direction. These form a minimum and maximum shell that any
displacement must be within. A tight shell will result in better raytracing
performance.

The baker can generate direction bounds. This is currently implemented in two
passes. The first computes per-triangle displacement ranges and the second
re-computes displacement distances after using those displacement ranges as
direction bounds. A final global fitting is performed as the second pass may
still produce displacements outside the unit range.

Note that changing the direction bounds can actually change the micro-vertex
displacement direction vector due to interpolation across the triangle. This is
why bounds fitting must be iterative. It also has an effect on texture sampling,
which is why resampling is encouraged.

## Heightmaps

Providing a heightmap for the reference mesh will result in tessellation and
displacement of the reference mesh before tracing. This can result in a
significant amount of geometry, in which case baking can happen in batches.

Enabling PN Triangle interpolation will smooth the reference mesh tessellation
while applying heightmap displacement.

# Resampler

Texture resampling happens during baking, to take advantage of the reference
geometry existing at the same time, particularly when it is generated on the fly
with heightmap displacement. The mesh is rendered in UV space over the output
texture and a ray is traced to the reference mesh for each pixel. The values
written depend on the texture type:

- **Generic** - E.g. color/diffuse texture.
- **Normal map** - Resample direction vectors based on the change in tangent
  space from one mesh to the other.

Textures may also be generated from scratch:

- **Offset map** - a texture that stores the same texture space offsets used to
  resample **Generic** textures. This can be used to access the original
  non-resampled textures at runtime by offsetting the texture coordinates prior
  to lookup.
- **Quaternion map** - a texture of quaternion rotations representing the change
  of basis used when resampling **Normal** textures. Similar to the offset map,
  but for normal maps.
- **Height map** - displacements to the reference mesh. This is similar to a
  micromap but at texel frequency rather than micro-vertex. Experimental as it
  conflicts with bounds fitting and has no scale support.

Note that the resampler operates using the same per-vertex displacement
directions and direction bounds, which means they are only valid after applying
them to the original mesh.
