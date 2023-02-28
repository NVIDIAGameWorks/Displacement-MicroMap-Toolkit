# meshops_core

The `meshops` API is a layer above the `micromesh` SDK, that provides easier usage and higher level operations. 
The functions within typically operate on a single mesh provided in memory through an abstraction called [`meshops::MeshView`](include/meshops/meshops_mesh_view.h). 

Basic operations for this library can be found [here](include/meshops/meshops_operations.h).

While the `micromesh` core is storage agnostic, the `meshops` layer is design around using the `bary` container. As a result the library also hosts the `microutils` namespace and utilities, which handle some `micromesh` related operations on top of the bary file containers provided through `bary` or `baryutils`. The primary usecase are [compression related functions](include/microutils/microutils_compression.hpp).

This layer makes use of a Vulkan implementation for the various GPU-based operations and leverages `nvpro_core`'s `nvvk` utilities.
It is possible to use this API without explicitly accessing any Vulkan resources, or creating a custom Vulkan context. However, it does also support this to be embedded in an interactive viewer.

The headers in `include/meshops_internal` are strictly meant for other meshops libraries, and not meant as public interface for this api.

## Roadmap

Currently this library is still a bit more work in progress and subject to changes, until we have migrated all tools into `micromesh_tool` and refactored the `micromesh_viewer`. We might move or hide the microutils layer in future.

A few upcoming changes for `MeshView`:

- reduce template nesting, improve debugging
- remove bitangent vector and rely only on 4 component tangents
- add more texcoords and colors in MeshView
