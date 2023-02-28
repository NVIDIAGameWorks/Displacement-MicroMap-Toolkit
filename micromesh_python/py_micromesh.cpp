#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <Python.h>

#include <nvmath/nvmath.h>

#include <meshops/meshops_operations.h>
#include <meshops/meshops_mesh_view.h>
#include <meshops/meshops_vk.h>
#include <microutils/microutils.hpp>
#include <tool_meshops_objects.hpp>

#include <thread>

#include "py_enums.hpp"
#include "py_types.hpp"
#include "py_operations.hpp"

//
// Required due to some linklib requiring symbols
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE  // Include nvpro_core's stb_image instead of tinygltf's
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include "third_party/stb/stb_image.h"
#include "third_party/stb/stb_image_write.h"
#include "tiny_gltf.h"
// Should be removed if that is included in another static lib

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "nvvk/memallocator_vma_vk.hpp"


namespace py = pybind11;
using namespace meshops;

const char *micromesh_Result_str[] =
{
    "Success",
    "Failure",
    "Continue",
    "InvalidFrequency",
    "InvalidFormat",
    "InvalidBlockFormat",
    "InvalidRange",
    "InvalidValue",
    "InvalidLayout",
    "InvalidOperationOrder",
    "MismatchingInputEdgeValues",
    "MismatchingOutputEdgeValues",
    "UnsupportedVersion",
    "UnsupportedShaderCodeType"
};

const char * getResultString(const micromesh::Result& result)
{
    return micromesh_Result_str[(int)result];
}

class PyMeshopsContext
{
public:
    PyMeshopsContext(bool verbose=false, int maxThreadCount=-1) :
      m_context(nullptr), m_verbose(verbose), m_maxThreadCount(maxThreadCount)
    {
        createContext();
    }

    PyMeshopsContext(const PyMeshopsContext&) = delete;
    PyMeshopsContext& operator=(const PyMeshopsContext&) = delete;

    ~PyMeshopsContext()
    {
        destroyContext();
    }

    bool createContext()
    {
        destroyContext();

        ContextConfig meshopsContextConfig{};
        meshopsContextConfig.messageCallback = microutils::makeDefaultMessageCallback();

        if (m_maxThreadCount <= 0)
        {
            meshopsContextConfig.threadCount = std::thread::hardware_concurrency();
        }
        else
        {
            meshopsContextConfig.threadCount = m_maxThreadCount;
        }

        if (m_verbose)
        {
            meshopsContextConfig.verbosityLevel  = 999;
        }
        else
        {
            meshopsContextConfig.verbosityLevel  = 1;
        }
        meshopsContextConfig.requiresDeviceContext = true;

        nvvk::ContextCreateInfo contextInfo;
        std::vector<uint8_t> createInfoData;
        meshopsGetContextRequirements(meshopsContextConfig, contextInfo, createInfoData);

        contextInfo.verboseUsed = m_verbose;
        contextInfo.verboseCompatibleDevices = m_verbose;

        try
        {
            py::gil_scoped_release release;
            micromesh::Result result = meshopsContextCreate(meshopsContextConfig, &m_context);      

            if (result != micromesh::Result::eSuccess)
            {
                throw std::runtime_error("Error creating meshops context (" + std::string(getResultString(result)) + ")");
            }
        }
        catch (const std::exception e)
        {
            throw std::runtime_error("Error creating meshops context (" + std::string(e.what()) + ")");
        }
        catch (...)
        {
            throw std::runtime_error("Error creating meshops context (unknown)");
        }

        LOGI("Context created\n");

        return true;
    }

    bool destroyContext()
    {
        if (m_context != nullptr)
        {
            meshopsContextDestroy(m_context);
            LOGI("Context destroyed\n");
            m_context = nullptr;
        }

        return true;
    }

    meshops::Context context() const { return m_context; }

private:
    meshops::Context m_context;
    bool m_verbose;
    int m_maxThreadCount;
};

std::shared_ptr<PyMeshopsContext> getContext()
{
    std::cout << "Creating temporary context; call createContext/destroyContext to create a reusable context object and pass to operator" << std::endl;
    return std::make_shared<PyMeshopsContext>();
}

void setVerbosity(PyVerbosity verbosity)
{
    nvprintSetConsoleLogging(false, LOGBITS_ALL);
    nvprintSetConsoleLogging(true, verbosity);
}

PyMeshopsContext* createContext(PyVerbosity verbosity=PyVerbosity::eWarnings, int maxThreadCount=-1)
{
    setVerbosity(verbosity);

    return new PyMeshopsContext(verbosity == PyVerbosity::eInfo, maxThreadCount);
}

PyMicromeshData pyBaker(PyMeshopsContext *context, PyBakerInput& input)
{
    PyMicromeshData output;

    bake((context != nullptr ? context : getContext().get())->context(), input, output);

    return output;
}

PyMesh pyDisplaceMesh(PyMeshopsContext *context, PyMesh& input, PyMicromeshData& micromesh)
{
    PyMesh output;
    
    displace((context != nullptr ? context : getContext().get())->context(), input, micromesh, output);

    return output;
}

PyMesh pyRemesh(PyMeshopsContext *context, PyMesh& input, PyRemesherSettings& settings)
{
    PyMesh output;

    remesh((context != nullptr ? context : getContext().get())->context(), input, settings, output);

    return output;
}

PyMesh pyPreTessellate(PyMeshopsContext *context, PyMesh& input, PyPreTessellatorSettings& settings)
{
    PyMesh output;

    preTessellate((context != nullptr ? context : getContext().get())->context(), input, settings, output);

    return output;
}

bool pyWriteBary(PyMeshopsContext *context, const std::string& filepath, PyMesh& mesh, PyMicromeshData& micromesh, bool forceOverwrite=false)
{
    return writeBary((context != nullptr ? context : getContext().get())->context(), filepath, mesh, micromesh, forceOverwrite);
}

PyMicromeshData pyReadBary(PyMeshopsContext *context, const std::string& filepath, PyMesh& mesh)
{
    PyMicromeshData output;

    readBary((context != nullptr ? context : getContext().get())->context(), filepath, mesh, output);

    return output;
}

PYBIND11_MODULE(micromesh_python, m) {
    m.doc() = "meshops python bindings";

    py::class_<PyMeshopsContext>(m, "MeshopsContext")
        .def(py::init<>());

    m.def("createContext", &createContext, py::return_value_policy::take_ownership, py::arg("verbosity") = false, py::arg("maxThreadCount") = -1, "Create context");
    m.def("setVerbosity", &setVerbosity, py::arg("verbosity") = false, "Set logging verbosity");

    registerEnums(m);

    py::class_<PyMesh>(m, "Mesh")
        .def(py::init<>())
        .def_readwrite("triangleVertices", &PyMesh::triangleVertices)
        .def_readwrite("vertexPositions", &PyMesh::vertexPositions)
        .def_readwrite("vertexNormals", &PyMesh::vertexNormals)
        .def_readwrite("vertexTexcoords0", &PyMesh::vertexTexcoords0)
        .def_readwrite("vertexTangents", &PyMesh::vertexTangents)
        .def_readwrite("vertexDirections", &PyMesh::vertexDirections)
        .def_readwrite("vertexDirectionBounds", &PyMesh::vertexDirectionBounds)
        .def_readwrite("vertexImportance", &PyMesh::vertexImportance)
        .def_readwrite("triangleSubdivisionLevels", &PyMesh::triangleSubdivisionLevels)
        .def_readwrite("trianglePrimitiveFlags", &PyMesh::trianglePrimitiveFlags);
        //.def_readwrite("triangleMappings", &PyMesh::triangleMappings);

    py::class_<PyTexture> (m, "Texture")
        .def(py::init<>())
        .def_readwrite("type", &PyTexture::type)
        .def_readwrite("filepath", &PyTexture::filepath)
        .def_readwrite("format", &PyTexture::format)
        .def_readwrite("width", &PyTexture::width)
        .def_readwrite("height", &PyTexture::height)
        .def_readwrite("data", &PyTexture::data);

    py::class_<PyHeightMap, PyTexture>(m, "Heightmap")
        .def(py::init<>())
        .def_readwrite("bias", &PyHeightMap::bias)
        .def_readwrite("scale", &PyHeightMap::scale);

    py::class_<PyResamplerInput>(m, "ResamplerInput")
        .def(py::init<>())
        .def_readwrite("input", &PyResamplerInput::input)
        .def_readwrite("output", &PyResamplerInput::output);

    py::class_<PyBakerInput>(m, "BakerInput")
        .def(py::init<>())
        .def_readwrite("baseMesh", &PyBakerInput::baseMesh)
        .def_readwrite("baseMeshTransform", &PyBakerInput::baseMeshTransform)
        .def_readwrite("referenceMesh", &PyBakerInput::referenceMesh)
        .def_readwrite("referenceMeshTransform", &PyBakerInput::referenceMeshTransform)
        .def_readwrite("resamplerInput", &PyBakerInput::resamplerInput) // Disabled until fully supported
        .def_readwrite("heightmap", &PyBakerInput::heightmap)
        .def_readwrite("normalMapFilepath", &PyBakerInput::normalMapFilepath)
        .def_readwrite("normalMapResolution", &PyBakerInput::normalMapResolution)
        .def_readwrite("uvRemapFilepath", &PyBakerInput::uvRemapFilepath)
        .def_readwrite("uvRemapResolution", &PyBakerInput::uvRemapResolution)
        .def_readwrite("settings", &PyBakerInput::settings);

    py::class_<PyMicromeshData>(m, "MicromeshData")
        .def(py::init<>())
        .def_readwrite("vertexDirections", &PyMicromeshData::vertexDirections)
        .def_readwrite("vertexDirectionBounds", &PyMicromeshData::vertexDirectionBounds)
        .def_readwrite("minSubdivLevel", &PyMicromeshData::minSubdivLevel)
        .def_readwrite("maxSubdivLevel", &PyMicromeshData::maxSubdivLevel)
        .def_readwrite("bias", &PyMicromeshData::bias)
        .def_readwrite("scale", &PyMicromeshData::scale)
        .def_readwrite("values", &PyMicromeshData::values)
        .def_readwrite("valueFormat", &PyMicromeshData::valueFormat)
        .def_readwrite("valueLayout", &PyMicromeshData::valueLayout)
        .def_readwrite("valueFrequency", &PyMicromeshData::valueFrequency)
        .def_readwrite("valueCount", &PyMicromeshData::valueCount)
        .def_readwrite("valueByteSize", &PyMicromeshData::valueByteSize)
        .def_readwrite("valueByteAlignment", &PyMicromeshData::valueByteAlignment)
        .def_readwrite("triangleValueOffsets", &PyMicromeshData::triangleValueOffsets)
        .def_readwrite("triangleSubdivLevels", &PyMicromeshData::triangleSubdivLevels)
        .def_readwrite("triangleBlockFormats", &PyMicromeshData::triangleBlockFormats)
        .def_readwrite("histogramEntryCounts", &PyMicromeshData::histogramEntryCounts)
        .def_readwrite("histogramEntrySubdivLevels", &PyMicromeshData::histogramEntrySubdivLevels)
        .def_readwrite("histogramEntryBlockFormats", &PyMicromeshData::histogramEntryBlockFormats)
        .def_readwrite("triangleMinMaxs", &PyMicromeshData::triangleMinMaxs)
        .def_readwrite("triangleMinMaxFormat", &PyMicromeshData::triangleMinMaxFormat)
        .def_readwrite("triangleMinMaxCount", &PyMicromeshData::triangleMinMaxCount)
        .def_readwrite("triangleMinMaxByteSize", &PyMicromeshData::triangleMinMaxByteSize)
        .def_readwrite("triangleMinMaxByteAlignment", &PyMicromeshData::triangleMinMaxByteAlignment)
        .def_readwrite("triangleFlags", &PyMicromeshData::triangleFlags)
        .def_readwrite("triangleFlagFormat", &PyMicromeshData::triangleFlagFormat)
        .def_readwrite("triangleFlagCount", &PyMicromeshData::triangleFlagCount)
        .def_readwrite("triangleFlagByteSize", &PyMicromeshData::triangleFlagByteSize)
        .def_readwrite("triangleFlagByteAlignment", &PyMicromeshData::triangleFlagByteAlignment);

    py::class_<PyBakerSettings>(m, "BakerSettings")
        .def(py::init<>())
        .def_readwrite("level", &PyBakerSettings::level)
        .def_readwrite("maxTraceLength", &PyBakerSettings::maxTraceLength)
        .def_readwrite("uniDirectional", &PyBakerSettings::uniDirectional)
        .def_readwrite("fitDirectionBounds", &PyBakerSettings::fitDirectionBounds)
        .def_readwrite("memLimitBytes", &PyBakerSettings::memLimitBytes)
        .def_readwrite("uncompressedLayout", &PyBakerSettings::uncompressedLayout)
        .def_readwrite("uncompressedDisplacementFormat", &PyBakerSettings::uncompressedDisplacementFormat)
        .def_readwrite("uncompressedNormalFormat", &PyBakerSettings::uncompressedNormalFormat)
        .def_readwrite("subdivMethod", &PyBakerSettings::subdivMethod)
        .def_readwrite("adaptiveFactor", &PyBakerSettings::adaptiveFactor)
        .def_readwrite("normalReduceOp", &PyBakerSettings::normalReduceOp)
        .def_readwrite("tangentAlgorithm", &PyBakerSettings::tangentAlgorithm)
        .def_readwrite("enableCompression", &PyBakerSettings::enableCompression)
        .def_readwrite("lowTesselationBias", &PyBakerSettings::lowTessBias)
        .def_readwrite("highTesselationBias", &PyBakerSettings::highTessBias)
        .def_readwrite("minPSNR", &PyBakerSettings::minPSNR)
        .def_readwrite("compressedRasterData", &PyBakerSettings::compressedRasterData);    

    py::class_<PyRemesherSettings>(m, "RemesherSettings")
        .def(py::init<>())
        .def_readwrite("errorThreshold", &PyRemesherSettings::errorThreshold)
        .def_readwrite("maxOutputTriangleCount", &PyRemesherSettings::maxOutputTriangleCount, "Maximum output triangle count (if non-zero then decimationRatio is ignored)")
        .def_readwrite("curvaturePower", &PyRemesherSettings::curvaturePower)
        .def_readwrite("importanceWeight", &PyRemesherSettings::importanceWeight)
        .def_readwrite("curvatureMaxDist", &PyRemesherSettings::curvatureMaxDist)
        .def_readwrite("directionBoundsFactor", &PyRemesherSettings::directionBoundsFactor)
        .def_readwrite("curvatureMaxDistMode", &PyRemesherSettings::curvatureMaxDistMode)
        .def_readwrite("fitToOriginalSurface", &PyRemesherSettings::fitToOriginalSurface)
        .def_readwrite("maxSubdivLevel", &PyRemesherSettings::maxSubdivLevel)
        .def_readwrite("heightmapWidth", &PyRemesherSettings::heightmapWidth)
        .def_readwrite("heightmapHeight", &PyRemesherSettings::heightmapHeight)
        .def_readwrite("heightmapTexcoord", &PyRemesherSettings::heightmapTexcoord)
        .def_readwrite("importanceMap", &PyRemesherSettings::importanceMap)
        .def_readwrite("importanceTexcoord", &PyRemesherSettings::importanceTexcoord)
        .def_readwrite("decimationRatio", &PyRemesherSettings::decimationRatio)
        .def_readwrite("maxVertexValence", &PyRemesherSettings::maxVertexValence)
        .def_readwrite("importanceThreshold", &PyRemesherSettings::importanceThreshold)
        .def_readwrite("ignoreTexCoords", &PyRemesherSettings::ignoreTexCoords)
        .def_readwrite("ignoreNormals", &PyRemesherSettings::ignoreNormals)
        .def_readwrite("ignoreTangents", &PyRemesherSettings::ignoreTangents)
        .def_readwrite("ignoreDisplacementDirections", &PyRemesherSettings::ignoreDisplacementDirections)
        .def_readwrite("disableMicromeshData", &PyRemesherSettings::disableMicromeshData);

    py::class_<PyPreTessellatorSettings>(m, "PreTessellatorSettings")
        .def(py::init<>())
        .def_readwrite("maxSubdivLevel", &PyPreTessellatorSettings::maxSubdivLevel)
        .def_readwrite("heightmapWidth", &PyPreTessellatorSettings::heightmapWidth, "Maximum output triangle count (if non-zero then decimationRatio is ignored)")
        .def_readwrite("heightmapHeight", &PyPreTessellatorSettings::heightmapHeight)
        .def_readwrite("subdivLevelBias", &PyPreTessellatorSettings::subdivLevelBias)
        .def_readwrite("edgeLengthBased", &PyPreTessellatorSettings::edgeLengthBased);

    m.def("bakeMicromesh", &pyBaker, py::return_value_policy::move, py::arg("context"), py::arg("bakerInput"),
          "Create micromesh data using base mesh with reference mesh and/or heightmap");

    m.def("displaceMicromesh", &pyDisplaceMesh, py::return_value_policy::move, py::arg("context"), py::arg("inputMesh"), py::arg("inputMicromesh"),
          "Create micromesh data using base mesh with reference mesh and/or heightmap and produce displaced base mesh");

    m.def("remesh", &pyRemesh, py::return_value_policy::move, py::arg("context"), py::arg("inputMesh"), py::arg("settings"),
          "Remesh the input");

    m.def("preTessellate", &pyPreTessellate, py::return_value_policy::move, py::arg("context"), py::arg("inputMesh"), py::arg("settings"),
          "Pretessellate the input");

    m.def("writeBary", &pyWriteBary, py::arg("context"), py::arg("filepath"), py::arg("mesh"), py::arg("micromesh"), py::arg("forceOverwrite") = false,
          "Write the micromesh out to .bary format at the given filepath");
    m.def("readBary", &pyReadBary, py::arg("context"), py::arg("filepath"), py::arg("mesh"), "Read the .bary file at the given filepath");
}
