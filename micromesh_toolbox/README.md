# Micromesh ToolBox

`micromesh_toolbox` is a graphical workbench that allows inspecting micromeshes
as well as interacting with some of the tools. Use it to visually explore mesh
processing operations, bake and view meshes with displacement micromaps (that
are stored in .bary files). glTF is currently the only file format supported.

This application requires driver support for the `VK_NV_mesh_shader` extension.
The `VK_NV_displacement_micromap` extension is required to see the displacement
in raytracing mode. This extension will first ship with [NVIDIA Vulkan Beta
drivers](https://developer.nvidia.com/vulkan-driver). Check the website's
information for which drivers support it. This extension will be accelerated
through special hardware features of the NVIDIA Ada Generation hardware.
However, the extension will be available to previous hardware generations as
well, albeit having slower ray-tracing performance.

HDR environment maps are supported for rendering. See
<https://github.com/nvpro-samples/vk_shaded_gltfscene> for some examples.

> The rasterization within this application is primarily for visualization and
  not as efficient or feature rich as one that will be part of a future
  vk_displacement_micromaps sample.
