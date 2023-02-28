# NVIDIA Displacement Micro-Map Toolkit

This toolkit provides libraries, samples and tools to create and view displaced micromeshes.
It is a work in progress and feedback is welcome!

We recommend to check the [Micro-Mesh Basics
slides](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/slides/Micro-Mesh_Basics.pdf)
first as well as the [`dmm_displacement`
mini-sample](/mini_samples/dmm_displacement/README.md).

There are also NVIDIA GTC presentations [Getting Started with Compressed Micro-Meshes \[S51410\]](https://register.nvidia.com/flow/nvidia/gtcspring2023/attendeeportal/page/sessioncatalog/session/1666430278669001BFSR)
 and [Interactive GPU-Based Remeshing of Large Meshes \[S51567\]](https://register.nvidia.com/flow/nvidia/gtcspring2023/attendeeportal/page/sessioncatalog/session/1666622202853001BIHK).

The NVIDIA Micro-Mesh technology covers both opacity micromaps and displacement micromaps.
This SDK currently solely covers displacements as a separate SDK exists for opacity.

![](docs/micromesh_reefcrab.png)

- [NVIDIA Displacement Micromap Toolkit](#nvidia-displacement-micromap-toolkit)
  - [Building](#building)
    - [Windows](#windows)
    - [Linux](#linux)
  - [Example Micromeshes](#example-micromeshes)
  - [SDK and Toolkit Structure](#sdk-and-toolkit-structure)
    - [**micromesh SDK**](#micromesh-sdk)
    - [**bary file**](#bary-file)
    - [**meshops API**](#meshops-api)
    - [**tools \& libraries**](#tools--libraries)
    - [**mini samples**](#mini-samples)
    - [**micromesh toolbox**](#micromesh-toolbox)
    - [**micromesh tool**](#micromesh-tool)
    - [**micromesh python**](#micromesh-python)
  - [Fundamental Data Structures](#fundamental-data-structures)
  - [Micromesh Asset Pipeline](#micromesh-asset-pipeline)
  - [Third-Party Licenses](#third-party-licenses)

## Building

This repository contains submodules. After cloning, initialize them with:

```
git submodule update --init --recursive --jobs 8
```

The one manual dependency is the [Vulkan SDK](https://vulkan.lunarg.com/). For
Windows, make sure to install the optional "glm" during the install or set the
cmake `GLM_INCLUDE_DIR` location if installed separately.

Further dependencies will be downloaded during the cmake configuration step.

### Windows

Tested with Visual Studio 2017, 2019 and 2022. Use
[cmake-gui](https://cmake.org/download/) to generate project files.

- Install the [Vulkan SDK](https://vulkan.lunarg.com/) with **glm** (set
  cmake `GLM_INCLUDE_DIR` if glm is installed separately)
- Clone this repository and initialize the submodules
- Run cmake-gui, source code = this repo, build folder = anywhere
- Configure and Generate. Vulkan paths should be set by the SDK installer.
- Open path\\to\\build\\micromesh_toolkit.sln with Visual Studio
- In Solution Explorer, right click `micromesh_toolbox` and "Set as Startup Project"
- Build -> Build Solution

See [docs/examples.md](docs/examples.md) for usage examples.

### Linux

Tested with gcc 11.2.1 and 11.3.0. This assumes build tools and cmake are
already installed.

- Install the [Vulkan SDK](https://vulkan.lunarg.com/)
- Install glm (e.g. `sudo apt install libglm-dev` or `sudo dnf install glm-devel`)
- Install X11, GLFW and nVidia-ML libs
  (`sudo apt-get install libx11-dev libxcb1-dev libxcb-keysyms1-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev 	libxxf86vm-dev libvulkan-dev libglfw3-dev libnvidia-ml-dev` )
- Clone this repository and initialize the submodules
- Run `source path/to/vulkan-sdk/1.3..<version>/setup-env.sh`
- Run `cmake -DVULKAN_BUILD_DEPENDENCIES=on -S . -B path/to/build`
- Run `make -C path/to/build -j`

See [docs/examples.md](docs/examples.md) for usage examples.

## Example Micromeshes

The toolkit provides some basic tools to process and bake displacement
micromaps. Think of it like creating a heightmap to apply to a low-poly mesh,
except the heightmap is highly compressed, supports hardware accelerated
raytracing and doesn't need UVs. Some example starting assets are provided here
to demonstrate toolkit usage. Download these to get started but feel free to try
your own.

- [Crab](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/reefcrab.7z)
- [Wall](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/wall.7z)
- [Rocks](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/rocks.7z)
- [PingPongPaddle](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/pingpongpaddle.7z)

Note that these tools currently only support assets in glTF format. Everything
else must be converted.

See [docs/examples.md](docs/examples.md) for typical asset processing examples
using these meshes.

## SDK and Toolkit Structure

### **micromesh SDK**

The [NVIDIA Displacement Micro-Map SDK](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/)
is the low-level API meant for embedding in other applications and tools.
It has a C-style API as well as an API agnostic GPU interface to facilitate this. 
As a result it is sometimes a bit less easy to use. All functionality is provided
through the `micromesh` namespace and it makes frequent use of the `micromesh::ArrayInfo`
structure, which allows it to pass data as a pointer & stride combination. All user visible data
is allocated by the user, so some operations are executed in two steps where a *micromeshOpSomethingBegin* returns
the sizing required, while *micromeshOpSomethingEnd* completes it. One can also abort such 
operations with *micromeshOpContextAbort*. The `micromesh::Context` therefore is stateful
but fairly lightweight, in case you want to create one per thread. Right now there is also some rudimentary
automatic threading within the context.

- [`micromesh_core`](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/micromesh_core/README.md):
  Library for basic data structures, utilities and operations to create or
  modify micromap and micromesh data.
- [`micromesh_displacement_compression`](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/micromesh_displacement_compression/README.md):
  Library that handles the compression of displacement micromaps.
- [`micromesh_displacement_remeshing`](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/micromesh_displacement_remeshing/README.md):
  Library for GPU-based remeshing.

### **bary file**

- [NVIDIA Displacement Micro-Map BaryFile](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-BaryFile/): library for the `.bary` file container that is used to store `micromaps`. The repository provides three libraries
  - `bary_core`: The core library defines the structs used in the container and provides basic functions that aid validation and serialization. C-style interface.
  - `bary_utils`: Utility library has C++ utilities that aid storing data with stl containers as well as loading or saving things via file operations.
  - `bary_tool`: A command-line tool that prints key information about the content of `.bary` files.

### **meshops API**

The `meshops` API is a layer above the `micromesh` SDK, that provides easier usage and higher level operations, 
such as baking. The functions typically operate on a single mesh provided in memory through an abstraction called `meshops::MeshView`. 
This layer makes use of a Vulkan implementation for the various GPU-based operations and leverages
the `nvpro_core` framework doing so.

This `meshops` layer is still a bit work in progress as we are migrating additional capabilities to it and might see a few more changes than the `micromesh` API.

- [`meshops_core`](/meshops_core/README.md): Provides the main framework for the `meshops` namespace
  and covers several [basic operations](/meshops_core/include/meshops/meshops_operations.h)
- [`meshops_bake`](/meshops_bake): Features a variety of operations based on raytracing from the base mesh to the high detail reference mesh.
  We want to point out this is not a commercial class baker and it primarily exists for sample purposes.
  - [`meshops_remesher`](/meshops_remesher): The remesher runs a novel
    GPU-accelerated mesh simplification algorithm and also generates other mesh
    properties useful for displacement micromap baking.
  - generate displacement micromaps for high detail reference meshes, which optionally can be tessellated on the fly to account for additional heightmap displacement if provided.
  - resample existing textures from the high detail reference mesh to the base mesh (also allows creation or resampling of tangent space normal maps).

### **tools & libraries**

All tools operate on glTF 2.0 files and do support additional micromap specific
glTF extensions from NVIDIA. The spcifications are available at
<https://github.com/NeilBickford-NV/glTF/tree/micro-mesh/extensions/2.0/Vendor>.

All micromap data is stored as `.bary` files which is a new container / file
format that was specifically designed to allow direct storage of data consumed
by the raytracing APIs without additional processing. See the open-source
[NVIDIA Displacement Micro-Map
BaryFile](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-BaryFile/)
for more details. 

- [`micromesh_tool`](/micromesh_tool/README.md): This is the main tool for `meshops` based operation.
  We intend to move all tooling operations here, but haven't completed this migration and refactoring yet.
  The tool operations currently operate on an independent scene state (load gltf freshly), in future there will be a persistent in-memory scene that the
  operations modify.
  The tool currently supports:
  - *baking* using `meshops_bake`
  - *pre-tessellation* using `meshops_core`, compute subdivision levels and adjust base mesh tessellation to account for subdiv level 5 limit 
  - *displaced-tessellation* using `meshops_core`, create a tessellated mesh that applies all displacements provided through the micromaps (flatten the mesh to traditional triangles)
  - *optimization*: trims and compresses micromeshes. Since only compressed micromaps are compatible with the ray tracing APIs, this can be used to prepare uncompressed micromaps for rendering.
  - *remeshing* using `meshops_remesher`, decimates a mesh to be used as a base mesh for baking.
- [`micromesh_python`](/micromesh_python): This is a python module with similar
  functionality to `micromesh_tool`, exposing toolkit operations from the
  `meshops_*` layer, but without the dependence on glTF. It operates on meshes
  with geometry and other attributes defined by more generic `numpy` arrays."

### **mini samples**
  
The mini-samples showcase our APIs in a minimal fashion, not relying on loading models or other files, but just procedurally generating content to reduce the
amount of complexity

- [`dmm_displacement`](/mini_samples/dmm_displacement/README.md): Learn how to raytrace displaced micromeshes 
  using the `VK_NV_displacement_micromap` extension. It also shows how to create the necessary data
  for it in a minimal fashion using the `micromesh` sdk. 
  The extension support is mandatory for this sample.

> At the time of writing `VK_NV_displacement_micromap` is in beta still, meaning
  the extension is subject to changes. Do not use its headers here in production
  code!

### **micromesh toolbox**

The [`micromesh_toolbox`](/micromesh_toolbox/README.md) is a graphical workbench
that allows inspecting micromeshes as well as interacting with some of the
tools. It relies on `VK_NV_mesh_shader` to allow rasterized display of
micromeshes. `VK_KHR_acceleration_structure` is required for baking micromaps.
If available, `VK_NV_displacement_micromap` is used to render micromeshes with
raytracing when choosing *Rendering -> RTX*. `VK_NV_displacement_micromap` was
introduced with the Ada Lovelace architecture based GPUs. If you see a message
about the missing extension, update to the latest driver (note: beta drivers are
available at https://developer.nvidia.com/vulkan-driver). A driver may not be
available at the time of writing.

> The rasterization of micromeshes, especially compressed is a bit of
  a more complex topic on its own. Therefore there will be a future
  dedicated sample that goes into the details of how it works
  and showcases more features, such as dynamic level-of-detail.
  We recommend to wait for this, rather than attempt to
  embed the code from the toolbox. The future sample will also
  provide more performant solutions and address compute-based
  rasterization as well.

> At the time of writing `VK_NV_displacement_micromap` is in beta still, meaning
  the extension is subject to changes. Do not use its headers here in production
  code!

### **micromesh tool**

The command line tool, [`micromesh_tool`](/micromesh_tool/README.md), can be
used for automation and defining persistent asset pipelines.

### **micromesh python**

The python bindings for the toolkit, summarized in a [`Jupyter Notebook`](/micromesh_python/notebook/micromesh.ipynb), can be used for integrating asset pipelines inside any application supporting a Python environment.

See [`micromesh_python`](/micromesh_python/README.md) for more details.

## Fundamental Data Structures

A **micromesh** is the result of subdividing a base triangle using a power of two scheme.
It is made of **microtriangles** and **microvertices**.

```
Triangle W,U,V

          V                     V                       V
          x                     x                       x
         / \                   / \ micro               / \
        /   \                 /   \ triangle          x _ x microvertex
       /     \               /     \                 / \ / \
      /       \             x _____ x               x _ x _ x
     /         \           / \     / \             / \ / \ / \
    /           \         /   \   /   \           x _ x _ x _ x
   /             \       /     \ /     \         / \ / \ / \ / \ 
  x _____________ x     x _____ x _____ x       x _ x _ x _ x _ x
W                  U   W                  U   W                   U

Subdivision level 0    Subdivision level 1    Subdivision level 2
  1 microtriangle        4 microtriangles       16 microtriangles
  aka base triangle
```

See [docs/data_structures.md](docs/data_structures.md) for more details on how this is used to generate displacement.

## Micromesh Asset Pipeline

We describe the core steps that would be performed to create a micromesh asset
from an original input mesh in the [Micro-Mesh Asset Pipeline
slide-deck](https://developer.download.nvidia.com/ProGraphics/nvpro-samples/slides/Micro-Mesh_Asset_Pipeline.pdf)
as well as in [docs/asset_pipeline.md](docs/asset_pipeline.md)

## Third-Party Licenses

This project embeds or includes (as submodules) several third-party open-source
libraries and/or code derived from them. All such libraries' licenses are
included in the PACKAGE-LICENSES folder of [this project](PACKAGE-LICENSES) or
of the [nvpro_core
dependency](https://github.com/nvpro-samples/nvpro_core/tree/master/PACKAGE-LICENSES).

## Support Contact

Feel free to file issues directly on the GitHub page or reach out to NVIDIA at
<displacedmicromesh-sdk-support@nvidia.com>
