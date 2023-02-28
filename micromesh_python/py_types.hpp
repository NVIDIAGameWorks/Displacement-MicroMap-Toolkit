#pragma once

#include "py_conversions.hpp"
#include "py_enums.hpp"

class PyMesh
{
public:
    py::array_t<unsigned int> triangleVertices;
    py::array_t<float>        vertexPositions;
    py::array_t<float>        vertexNormals;
    py::array_t<float>        vertexTexcoords0;
    py::array_t<float>        vertexTangents;
    py::array_t<float>        vertexDirections;
    py::array_t<float>        vertexDirectionBounds;
    py::array_t<float>        vertexImportance;
    py::array_t<uint16_t>     triangleSubdivisionLevels;
    py::array_t<uint8_t>      trianglePrimitiveFlags;
    //py::array_t<uint32_t>     triangleMappings;

    void toMeshView(ResizableMeshView& meshView)
    {
        numpyArrayToArrayView<3, unsigned int, nvmath::vec3ui>(triangleVertices,   meshView.triangleVertices);
        numpyArrayToArrayView<3, float, nvmath::vec3f>(vertexPositions,            meshView.vertexPositions);
        numpyArrayToArrayView<3, float, nvmath::vec3f>(vertexNormals,              meshView.vertexNormals);
        numpyArrayToArrayView<2, float, nvmath::vec2f>(vertexTexcoords0,           meshView.vertexTexcoords0);
        numpyArrayToArrayView<4, float, nvmath::vec4f>(vertexTangents,             meshView.vertexTangents);
        numpyArrayToArrayView<3, float, nvmath::vec3f>(vertexDirections,           meshView.vertexDirections);
        numpyArrayToArrayView<2, float, nvmath::vec2f>(vertexDirectionBounds,      meshView.vertexDirectionBounds);
        numpyArrayToArrayView<1, float, float>(vertexImportance,                   meshView.vertexImportance);
        numpyArrayToArrayView<1, uint16_t, uint16_t>(triangleSubdivisionLevels,    meshView.triangleSubdivisionLevels);
        numpyArrayToArrayView<1, uint8_t, uint8_t>(trianglePrimitiveFlags,         meshView.trianglePrimitiveFlags);
        //numpyArrayToArrayView<1, uint32_t, uint32_t>(triangleMappings,             meshView.triangleMappings);
    }

    void fromMeshView(const ResizableMeshView& meshView)
    {
        arrayViewToNumpyArray<3, nvmath::vec3ui, unsigned int>(meshView.triangleVertices,  triangleVertices);
        arrayViewToNumpyArray<3, nvmath::vec3f, float>(meshView.vertexPositions,           vertexPositions);
        arrayViewToNumpyArray<3, nvmath::vec3f, float>(meshView.vertexNormals,             vertexNormals);
        arrayViewToNumpyArray<2, nvmath::vec2f, float>(meshView.vertexTexcoords0,          vertexTexcoords0);
        arrayViewToNumpyArray<4, nvmath::vec4f, float>(meshView.vertexTangents,            vertexTangents);
        arrayViewToNumpyArray<3, nvmath::vec3f, float>(meshView.vertexDirections,          vertexDirections);
        arrayViewToNumpyArray<2, nvmath::vec2f, float>(meshView.vertexDirectionBounds,     vertexDirectionBounds);
        arrayViewToNumpyArray<1, float, float>(meshView.vertexImportance,                  vertexImportance);
        arrayViewToNumpyArray<1, uint16_t, uint16_t>(meshView.triangleSubdivisionLevels,   triangleSubdivisionLevels);
        arrayViewToNumpyArray<1, uint8_t, uint8_t>(meshView.trianglePrimitiveFlags,        trianglePrimitiveFlags);
        //arrayViewToNumpyArray<1, const uint32_t, uint32_t>(meshView.triangleMappings,            triangleMappings);
    }
};

class PyTexture
{
public:
    std::string filepath;
    meshops::TextureType type = meshops::TextureType::eGeneric;
    PyTextureFormat format = PyTextureFormat::eRGBA16Unorm;
    uint32_t width = 0;
    uint32_t height = 0;
    py::array_t<uint8_t> data;
};

class PyHeightMap : public PyTexture
{
public:
    PyHeightMap()
    {
        type = meshops::TextureType::eHeightMap;
    }

    float scale = 0.0f;
    float bias  = 0.0f;
};

class PyResamplerInput
{
public:
    PyTexture input;
    PyTexture output;
};

class PyBakerSettings
{
public:
    // Output subdivision level to bake at. Each level produces 4x microtriangles.
    uint32_t level = 3;

    // If non-zero, overrides trace distance (in world space) otherwise defined by baseMeshView.vertexDirections and
    // baseMeshView.vertexDirectionBounds.
    float maxTraceLength = 0.0f;

    // Trace only in the direction of baseMeshView.vertexDirections if true. Otherwise traces backwards too.
    bool uniDirectional = false;

    bool fitDirectionBounds = false;

    // Rudimentary memory limit. Baking will be split into batches to maintain the limit.
    uint64_t          memLimitBytes                  = 4096ULL << 20;
    bary::ValueLayout uncompressedLayout             = bary::ValueLayout::eTriangleBirdCurve;
    bary::Format      uncompressedDisplacementFormat = bary::Format::eR16_unorm;
    bary::Format      uncompressedNormalFormat       = bary::Format::eRG16_snorm;

    void toSettings(meshops::OpBake_settings& settings)
    {
        settings.level = level;
        settings.maxTraceLength = maxTraceLength;
        settings.uniDirectional = uniDirectional;
        settings.fitDirectionBounds = fitDirectionBounds;
        settings.memLimitBytes = memLimitBytes;
        settings.uncompressedLayout = uncompressedLayout;
        settings.uncompressedDisplacementFormat = uncompressedDisplacementFormat;
        settings.uncompressedNormalFormat = uncompressedNormalFormat;
    }

    // Other baker settings not included in OpBake_settings but required for baking
    PySubdivMethod subdivMethod = PySubdivMethod::eUniform;
    float adaptiveFactor = 1.0f;
    meshops::TangentSpaceAlgorithm tangentAlgorithm = meshops::TangentSpaceAlgorithm::eDefault;
    NormalReduceOp normalReduceOp = NormalReduceOp::eNormalReduceLinear;

    bool enableCompression = true;
    float minPSNR = 50.0f;
    bool  compressedRasterData = false;
    int lowTessBias          = 0;
    int highTessBias         = 0;
};


class PyBakerInput
{
public:
    PyMesh* baseMesh = nullptr;

    // Column-major object-to-world space transform
    py::array_t<float> baseMeshTransform; // micromesh::Matrix_float_4x4

    // May be the same as the base mesh
    PyMesh* referenceMesh = nullptr;

    // Column-major object-to-world space transform
    py::array_t<float> referenceMeshTransform; // micromesh::Matrix_float_4x4

    py::list resamplerInput; // PyResamplerInput

    PyHeightMap heightmap;

    std::string normalMapFilepath;
    int normalMapResolution;

    std::string uvRemapFilepath;
    int uvRemapResolution;

    PyBakerSettings settings;
};

class PyMicromeshData
{
public:
    py::array_t<float>    vertexDirections;
    py::array_t<float>    vertexDirectionBounds;

    uint32_t              minSubdivLevel;
    uint32_t              maxSubdivLevel;

    float                 bias = 0.0f;
    float                 scale = 1.0f;

    py::array_t<uint8_t>  values;
    bary::Format          valueFormat;
    bary::ValueLayout     valueLayout;
    bary::ValueFrequency  valueFrequency;
    uint32_t              valueCount;
    uint32_t              valueByteSize;
    uint32_t              valueByteAlignment;

    py::array_t<uint8_t>  triangleFlags;
    bary::Format          triangleFlagFormat;
    uint32_t              triangleFlagCount;
    uint32_t              triangleFlagByteSize;
    uint32_t              triangleFlagByteAlignment;

    py::array_t<uint32_t> triangleValueOffsets;
    py::array_t<uint16_t> triangleSubdivLevels;
    py::array_t<uint16_t> triangleBlockFormats;

    py::array_t<uint32_t> histogramEntryCounts;
    py::array_t<uint32_t> histogramEntrySubdivLevels;
    py::array_t<uint32_t> histogramEntryBlockFormats;

    py::array_t<uint8_t>  triangleMinMaxs;
    bary::Format          triangleMinMaxFormat;
    uint32_t              triangleMinMaxCount;
    uint32_t              triangleMinMaxByteSize;
    uint32_t              triangleMinMaxByteAlignment;

    template <template <class T> class ArrayType>
    void fromBaryData(baryutils::BaryBasicData* basicData,
                      meshops::ArrayView<const nvmath::vec3f> * vertexDirections,
                      meshops::ArrayView<const nvmath::vec2f> * vertexDirectionBounds,
                      meshops::MeshAttributes<ArrayType>* meshView = nullptr)
    {
        if(basicData)
        {
            if(basicData->groups.size() < 1)
            {
                throw std::runtime_error("group size must be at least 1");
            }

            minSubdivLevel = basicData->groups[0].minSubdivLevel;
            maxSubdivLevel = basicData->groups[0].maxSubdivLevel;

            bias  = basicData->groups[0].floatBias.r;
            scale = basicData->groups[0].floatScale.r;

            vectorToNumpyArray<1, uint8_t, uint8_t>(basicData->values, values);
            valueFormat        = basicData->valuesInfo.valueFormat;
            valueLayout        = basicData->valuesInfo.valueLayout;
            valueFrequency     = basicData->valuesInfo.valueFrequency;
            valueCount         = basicData->valuesInfo.valueCount;
            valueByteSize      = basicData->valuesInfo.valueByteSize;
            valueByteAlignment = basicData->valuesInfo.valueByteAlignment;

            std::vector<uint32_t> triangleValueOffsets; triangleValueOffsets.reserve(basicData->triangles.size());
            std::vector<uint16_t> triangleSubdivLevels; triangleSubdivLevels.reserve(basicData->triangles.size());
            std::vector<uint16_t> triangleBlockFormats; triangleBlockFormats.reserve(basicData->triangles.size());

            for(size_t i = 0; i < basicData->triangles.size(); ++i)
            {
                triangleValueOffsets.emplace_back(basicData->triangles[i].valuesOffset);
                triangleSubdivLevels.emplace_back(basicData->triangles[i].subdivLevel);
                triangleBlockFormats.emplace_back(basicData->triangles[i].blockFormat);
            }

            vectorToNumpyArray<1, uint32_t, uint32_t>(triangleValueOffsets, this->triangleValueOffsets);
            vectorToNumpyArray<1, uint16_t, uint16_t>(triangleSubdivLevels, this->triangleSubdivLevels);
            vectorToNumpyArray<1, uint16_t, uint16_t>(triangleBlockFormats, this->triangleBlockFormats);

            std::vector<uint32_t> histogramEntryCounts;       histogramEntryCounts.reserve(basicData->histogramEntries.size());
            std::vector<uint32_t> histogramEntrySubdivLevels; histogramEntrySubdivLevels.reserve(basicData->histogramEntries.size());
            std::vector<uint32_t> histogramEntryBlockFormats; histogramEntryBlockFormats.reserve(basicData->histogramEntries.size());

            for(size_t i = 0; i < basicData->histogramEntries.size(); ++i)
            {
                histogramEntryCounts.emplace_back(basicData->histogramEntries[i].count);
                histogramEntrySubdivLevels.emplace_back(basicData->histogramEntries[i].subdivLevel);
                histogramEntryBlockFormats.emplace_back(basicData->histogramEntries[i].blockFormat);
            }

            vectorToNumpyArray<1, uint32_t, uint32_t>(histogramEntryCounts, this->histogramEntryCounts);
            vectorToNumpyArray<1, uint32_t, uint32_t>(histogramEntrySubdivLevels, this->histogramEntrySubdivLevels);
            vectorToNumpyArray<1, uint32_t, uint32_t>(histogramEntryBlockFormats, this->histogramEntryBlockFormats);

            vectorToNumpyArray<1, uint8_t, uint8_t>(basicData->triangleMinMaxs, triangleMinMaxs);
            triangleMinMaxFormat        = basicData->triangleMinMaxsInfo.elementFormat;
            triangleMinMaxCount         = basicData->triangleMinMaxsInfo.elementCount;
            triangleMinMaxByteSize      = basicData->triangleMinMaxsInfo.elementByteSize;
            triangleMinMaxByteAlignment = basicData->triangleMinMaxsInfo.elementByteAlignment;
        }

        if(meshView)
        {
            if(meshView->trianglePrimitiveFlags.size() > 0)
            {
                triangleFlagFormat        = bary::Format::eR8_uint;
                triangleFlagCount         = (uint32_t)meshView->trianglePrimitiveFlags.size();
                triangleFlagByteSize      = sizeof(uint8_t);
                triangleFlagByteAlignment = 4;

                arrayViewToNumpyArray<1, uint8_t, uint8_t>(meshView->trianglePrimitiveFlags, this->triangleFlags);
            }

            if (meshView->vertexDirections.size() > 0)
            {
                arrayViewToNumpyArray<3, const nvmath::vec3f, float>(meshView->vertexDirections, this->vertexDirections);
            }

            if (meshView->vertexDirectionBounds.size() > 0)
            {
                arrayViewToNumpyArray<2, const nvmath::vec2f, float>(meshView->vertexDirectionBounds, this->vertexDirectionBounds);
            }
        }

        if(vertexDirections && vertexDirections->size() > 0)
        {
            arrayViewToNumpyArray<3, const nvmath::vec3f, float>(*vertexDirections, this->vertexDirections);
        }

        if(vertexDirectionBounds && vertexDirectionBounds->size() > 0)
        {
            arrayViewToNumpyArray<2, const nvmath::vec2f, float>(*vertexDirectionBounds, this->vertexDirectionBounds);
        }
    }

    template <template <class T> class ArrayType>
    void toBaryData(baryutils::BaryBasicData* basicData,
                    meshops::ArrayView<nvmath::vec3f>* vertexDirections,
                    meshops::ArrayView<nvmath::vec2f>* vertexDirectionBounds,
                    meshops::MeshAttributes<ArrayType>* meshView = nullptr)
    {
        if(basicData)
        {
            numpyArrayToVector<1, uint8_t, uint8_t>(values, basicData->values);
            basicData->valuesInfo.valueFormat = valueFormat;
            basicData->valuesInfo.valueLayout = valueLayout;
            basicData->valuesInfo.valueFrequency = valueFrequency;
            basicData->valuesInfo.valueCount = valueCount;
            basicData->valuesInfo.valueByteSize = valueByteSize;
            basicData->valuesInfo.valueByteAlignment = valueByteAlignment;

            std::vector<uint32_t> triangleValueOffsets;
            std::vector<uint16_t> triangleSubdivLevels;
            std::vector<uint16_t> triangleBlockFormats;

            numpyArrayToVector<1, uint32_t, uint32_t>(this->triangleValueOffsets, triangleValueOffsets);
            numpyArrayToVector<1, uint16_t, uint16_t>(this->triangleSubdivLevels, triangleSubdivLevels);
            numpyArrayToVector<1, uint16_t, uint16_t>(this->triangleBlockFormats, triangleBlockFormats);

            basicData->triangles.resize(triangleValueOffsets.size());

            for(size_t i = 0; i < basicData->triangles.size(); ++i)
            {
                basicData->triangles[i].valuesOffset = triangleValueOffsets[i];
                basicData->triangles[i].subdivLevel  = triangleSubdivLevels[i];
                basicData->triangles[i].blockFormat  = triangleBlockFormats[i];
            }

            std::vector<uint32_t> histogramEntryCounts;
            std::vector<uint32_t> histogramEntrySubdivLevels;
            std::vector<uint32_t> histogramEntryBlockFormats;

            numpyArrayToVector<1, uint32_t, uint32_t>(this->histogramEntryCounts, histogramEntryCounts);
            numpyArrayToVector<1, uint32_t, uint32_t>(this->histogramEntrySubdivLevels, histogramEntrySubdivLevels);
            numpyArrayToVector<1, uint32_t, uint32_t>(this->histogramEntryBlockFormats, histogramEntryBlockFormats);

            basicData->histogramEntries.resize(histogramEntryCounts.size());

            for(size_t i = 0; i < basicData->histogramEntries.size(); ++i)
            {
                basicData->histogramEntries[i].count       = histogramEntryCounts[i];
                basicData->histogramEntries[i].subdivLevel = histogramEntrySubdivLevels[i];
                basicData->histogramEntries[i].blockFormat = histogramEntryBlockFormats[i];
            }

            basicData->groupHistogramRanges.resize(1);
            basicData->groupHistogramRanges[0].entryFirst = 0;
            basicData->groupHistogramRanges[0].entryCount = (uint32_t)basicData->histogramEntries.size();

            numpyArrayToVector<1, uint8_t, uint8_t>(triangleMinMaxs, basicData->triangleMinMaxs);
            basicData->triangleMinMaxsInfo.elementFormat = triangleMinMaxFormat;
            basicData->triangleMinMaxsInfo.elementCount = triangleMinMaxCount;
            basicData->triangleMinMaxsInfo.elementByteSize = triangleMinMaxByteSize;
            basicData->triangleMinMaxsInfo.elementByteAlignment = triangleMinMaxByteAlignment;

            if(basicData->groups.size() < 1)
            {
              throw std::runtime_error("group size must be at least 1");
            }

            basicData->minSubdivLevel = basicData->groups[0].minSubdivLevel = minSubdivLevel;
            basicData->maxSubdivLevel = basicData->groups[0].maxSubdivLevel = maxSubdivLevel;

            basicData->groups[0].floatBias.r  = bias;
            basicData->groups[0].floatScale.r = scale;

            basicData->groups[0].triangleCount = (unsigned int)basicData->triangles.size();
            basicData->groups[0].valueCount    = (unsigned int)basicData->values.size();
        }

        if(meshView)
        {
            numpyArrayToArrayView<1, uint8_t, uint8_t>(this->triangleFlags, meshView->trianglePrimitiveFlags);

            numpyArrayToArrayView<3, float, nvmath::vec3f>(this->vertexDirections, meshView->vertexDirections);
            numpyArrayToArrayView<2, float, nvmath::vec2f>(this->vertexDirectionBounds, meshView->vertexDirectionBounds);
        }

        if(vertexDirections)
        {
            numpyArrayToArrayView<3, float, nvmath::vec3f>(this->vertexDirections, *vertexDirections);
        }

        if(vertexDirectionBounds)
        {
            numpyArrayToArrayView<2, float, nvmath::vec2f>(this->vertexDirectionBounds, *vertexDirectionBounds);
        }
    }
};

class PyRemesherSettings
{
public:
  float       errorThreshold{100.f};
  uint32_t    maxOutputTriangleCount{0};
  float       curvaturePower{1.f};
  float       importanceWeight{200.f};
  float       curvatureMaxDist{0.05f};
  float       directionBoundsFactor{1.02f};
  PyRemesherCurvatureMaxDistanceMode
              curvatureMaxDistMode{PyRemesherCurvatureMaxDistanceMode::eSceneFraction};
  bool        fitToOriginalSurface{true};
  uint32_t    maxSubdivLevel{5};
  int32_t     heightmapWidth{-1};
  int32_t     heightmapHeight{-1};
  uint32_t    heightmapTexcoord{0};  // Texture coordinates used by the displacement map

  std::string importanceMap;          // Input filename of the optional importance map
  uint32_t    importanceTexcoord{0};  // Texture coordinates to use with the importance map

  float       decimationRatio{0.1f};
  uint32_t    maxVertexValence{20};
  float       importanceThreshold{1.f};
  bool        ignoreTexCoords{false};
  bool        ignoreNormals{false};
  bool        ignoreTangents{false};
  bool        ignoreDisplacementDirections{false};
  bool        disableMicromeshData{false};
};

class PyPreTessellatorSettings
{
public:
  uint32_t    maxSubdivLevel{0};
  int32_t     heightmapWidth{-1};
  int32_t     heightmapHeight{-1};
  int32_t     subdivLevelBias{-5};
  bool        edgeLengthBased{false};
};