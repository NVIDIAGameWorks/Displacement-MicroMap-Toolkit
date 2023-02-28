# micromesh_python

`micromesh_python` is a Python module for processing and baking micromeshes.
Uses may include content creation plugins or creating customized asset
 pipelines. The GUI, [`micromesh_toolkit`](../README.md#micromesh-toolkit), and
 [`micromesh_tool`](../README.md#micromesh-tool) are good for exploring
 micromaps visually or in command line environment.

Building this module requires an active Python 3 environment and Pybind11 -- point cmake variable `PYBIND11_LOCATION` to root of pybind11 header installation while configuring.  If both Python3 and Pybind11 are found then CMake will configure the micromesh_python project to build the bindings.

There is a [`Jupyter Notebook`](/micromesh_python/notebook/micromesh.ipynb) as an example of how to create assets compatible with micro-mesh technology using the `micromesh_python` module. Features briefly covered include:

- **pretessellate** - Tessellates a mesh to match the heightmap resolution.
- **bake** - Creates an NVIDIA displacement micromap given a low-resolution mesh and high-resolution mesh, heightmap, or both.
- **resample** - Resamples textures to be compatible with resulting micro-mesh.
- **displace** - Tessellates and displaces a mesh with micromap displacement.
- **remesh** - Decimates a triangle mesh, optimizing for micromap friendly
  geometry.
- **save** - Save the displacement micromap data out to a .bary file for future use.

The notebook can be run locally.  After building this repository, you can run `<path_to_build>/micromesh_python/install_jupyter.cmd` to install Jupyter Lab, and then `<path_to_build>/micromesh_python/start_jupyter.cmd` to launch the Jupyter server and automatically load the notebook in a browser.  The notebook contains cells to install (and uninstall) dependencies, as well as downloading assets not distributed with this repository.