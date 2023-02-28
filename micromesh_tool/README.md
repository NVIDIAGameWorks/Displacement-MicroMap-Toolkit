# micromesh_tool

`micromesh_tool` is a command line utility for processing and baking
micromeshes. The GUI, [`micromesh_toolkit`](../README.md#micromesh-toolkit), is
good for exploring micromaps visually. **glTF** is the only file format
currently supported. Operations can be joined without having to write
intermediate results, by chaining multiple commands wrapped in curly braces,
e.g. `{ operation --args ... }`.

Input and output top level arguments are required:

- `--input <source_scene.gltf>`
- `--output <destination_scene.gltf>`

It can be useful to enable `--verbose` to see more context for errors as
operations run.

The operations are:

- **generate** - Creates test meshes with textures. Use displacedtessellate to
  create real geometry from meshes with heightmaps. `--input` is not required
  when using generate.
- **pretessellate** - Tessellates a mesh to match the heightmap resolution plus
  a `--subdivLevelBias`. Useful when a mesh is too coarse for baking.
- **bake** - Creates an NVIDIA displacement micromap. Takes a base triangle mesh
  and computes distances needed to tessellate and displace it to match a
  reference mesh (--high). The result is written to a .bary file, referenced by
  the .gltf scene. Baking requires `VK_KHR_acceleration_structure` support.
- **displacedtessellate** - Tessellates and displaces a mesh with bary or
  heightmap displacement.
- **remesh** - Decimates a triangle mesh, optimizing for micromap friendly
  geometry.
- **optimize** - Trims and compresses displacement data to save space and
  improve performance
- **merge** - Merges multiple glTF files into one, with support for micromesh
  extensions.
- **print** - Prints a summary of the current scene data. Useful for inspecting
  intermediate pipeline state without writing the file to disk.

The **bake** operation takes two meshes as an argument. The base mesh is the
current scene in the pipeline. The reference mesh (typically with much more
detail) can be referenced with `--high <file>`. If the **remesh** operation
appears before **bake**, a copy of the scene is made before remeshing and passed
to **bake** in place of `--high`.

If the destination path is different from the source, textures will be copied
there.
