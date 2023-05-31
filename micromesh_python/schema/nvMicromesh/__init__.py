# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

from pxr import Plug, Tf, UsdGeom

plugin_root = os.path.join(os.path.dirname(__file__), "resources")
Plug.Registry().RegisterPlugins(plugin_root)

# PreparePythonModule didn't make it's way into USD
# until 21.08 - older versions import the module
# manually and call PrepareModule
if hasattr(Tf, "PreparePythonModule"):
    Tf.PreparePythonModule()
else:
    from . import _nvMicromesh
    Tf.PrepareModule(_nvMicromesh, locals())

try:
    import __DOC

    __DOC.Execute(locals())
    del __DOC
except Exception:
    try:
        import __tmpDoc

        __tmpDoc.Execute(locals())
        del __tmpDoc
    except:
        pass

del Plug
del Tf
del UsdGeom
