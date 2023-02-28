#pragma once

#include "py_types.hpp"

#include <meshops_internal/heightmap.hpp>
#include <meshops_internal/meshops_vertexattribs.h>
#include <tool_meshops_objects.hpp>

#include "imageio/imageio.hpp"

void bake(meshops::Context context, PyBakerInput& bakerInput, PyMicromeshData& bakeOutput)
{
  if (!context)
  {
      throw std::runtime_error("no context available");
  }

  meshops::MeshData baseMesh;
  meshops::ResizableMeshView  baseMeshView(baseMesh, makeResizableMeshViewCallback(baseMesh));
  micromesh::Matrix_float_4x4 baseMeshTransform;
  meshops::MeshTopologyData   baseMeshTopology;

  meshops::MeshData           referenceMesh;
  meshops::ResizableMeshView  referenceMeshView(referenceMesh, makeResizableMeshViewCallback(referenceMesh));
  micromesh::Matrix_float_4x4 referenceMeshTransform;
  meshops::MeshTopologyData   referenceMeshTopology;

  std::vector<meshops::OpBake_resamplerInput> resamplerInput;
  std::vector<meshops::Texture>               resamplerOutput;

  if (bakerInput.baseMesh)
  {
      bakerInput.baseMesh->toMeshView(baseMeshView);

      numpyArrayToMatrix(bakerInput.baseMeshTransform, baseMeshTransform);

      meshops::OpBuildTopology_input buildTopologyInput;
      buildTopologyInput.meshView = baseMeshView;

      meshops::OpBuildTopology_output buildTopologyOutput;
      buildTopologyOutput.meshTopology = &baseMeshTopology;

      if (meshops::meshopsOpBuildTopology(context, 1, &buildTopologyInput, &buildTopologyOutput) != micromesh::Result::eSuccess)
      {
          throw std::runtime_error("unable to create base mesh topology");
      }
  }

  bool baseMeshIncludesTexCoords = baseMeshView.vertexTexcoords0.size() > 0;

  if (bakerInput.referenceMesh)
  {
      bakerInput.referenceMesh->toMeshView(referenceMeshView);
          
      numpyArrayToMatrix(bakerInput.referenceMeshTransform, referenceMeshTransform);

      meshops::OpBuildTopology_input buildTopologyInput;
      buildTopologyInput.meshView = referenceMeshView;

      meshops::OpBuildTopology_output buildTopologyOutput;
      buildTopologyOutput.meshTopology = &referenceMeshTopology;

      if (meshops::meshopsOpBuildTopology(context, 1, &buildTopologyInput, &buildTopologyOutput) != micromesh::Result::eSuccess)
      {
          throw std::runtime_error("unable to create reference mesh topology");
      }
  }

  std::vector<std::unique_ptr<micromesh_tool::MeshopsTexture>> meshopsTextures;

  // Reference mesh heightmap config
  meshops::OpBake_heightmap heightmapDesc;
  meshops::TextureConfig    heightmapConfig{};
  heightmapDesc.normalizeDirections           = true;
  heightmapDesc.usesVertexNormalsAsDirections = false;  // Smooth direction vectors give better results at hard edges
  heightmapDesc.scale = bakerInput.heightmap.scale;
  heightmapDesc.bias = bakerInput.heightmap.bias;

  if(!referenceMeshView.triangleSubdivisionLevels.empty())
  {
    heightmapDesc.maxSubdivLevel = *std::max_element(referenceMeshView.triangleSubdivisionLevels.begin(),
                                                      referenceMeshView.triangleSubdivisionLevels.end());
  }

  // Load the heightmap, if there is one
  if (!bakerInput.heightmap.filepath.empty() || bakerInput.heightmap.data.size() > 0)
  {
    size_t w = 0, h = 0, comp = 0;
    imageio::ImageIOData data = nullptr;
    if(bakerInput.heightmap.filepath.empty())
    {
      if (bakerInput.heightmap.format == PyTextureFormat::eR16Unorm ||
          bakerInput.heightmap.format == PyTextureFormat::eRGBA16Unorm ||
          bakerInput.heightmap.format == PyTextureFormat::eRGBA8Unorm)
      {
        std::vector<uint8_t> rawData;
        numpyArrayToVector<1, uint8_t, uint8_t>(bakerInput.heightmap.data, rawData);

        if (rawData.size() > 0)
        {
          w = bakerInput.heightmap.width;
          h = bakerInput.heightmap.height;

          int bitDepth = 1;
          switch (bakerInput.heightmap.format)
          {
            case PyTextureFormat::eRGBA8Unorm:
              bitDepth = 8;
              comp = 4;                
              break;
            case PyTextureFormat::eRGBA16Unorm:
              bitDepth = 16;
              comp = 4;
              break;
            case PyTextureFormat::eR16Unorm:
              bitDepth = 16;
              comp = 1;
              break;
          };

          size_t sizeCheck = w * h * comp * (bitDepth / CHAR_BIT);

          if (sizeCheck != rawData.size())
          {
            imageio::freeData(&data);
            std::stringstream s; s << "Error: heightmap texture image data inconsistent with width '" << bakerInput.heightmap.width << "', height '" << bakerInput.heightmap.height << "', and format '" << bakerInput.heightmap.format << "' provided";
            throw std::runtime_error(s.str());
          }

          data = imageio::allocateData(rawData.size());
          memcpy(data, rawData.data(), rawData.size());

          // Convert from 8-16 bit if necessary
          if (bakerInput.heightmap.format == PyTextureFormat::eRGBA8Unorm)
          {
            if(!imageio::convertFormat(&data, w, h, 4, 8, 1, 32))
            {
              imageio::freeData(&data);
              throw std::runtime_error("Error: failed to convert heightmap texture image data from RGBA8Unorm to R32");
            }
          }
          else if (bakerInput.heightmap.format == PyTextureFormat::eRGBA16Unorm)
          {
            if(!imageio::convertFormat(&data, w, h, 4, 16, 1, 32))
            {
              imageio::freeData(&data);
              throw std::runtime_error("Error: failed to convert heightmap texture image data from RGBA16Unorm to R32");
            }
          }
          else if (bakerInput.heightmap.format == PyTextureFormat::eR16Unorm)
          {
            if(!imageio::convertFormat(&data, w, h, 1, 16, 1, 32))
            {
              imageio::freeData(&data);
              throw std::runtime_error("Error: failed to convert heightmap texture image data from R16Unorm to R32");
            }
          }
        }
        else
        {
          throw std::runtime_error("Error: heightmap texture image data is empty");
        }
      }
      else
      {
          throw std::runtime_error("Error: heightmap texture image data format is not compatible (8 or 16-bit RGBAUnorm only)"); 
      }
    }
    else
    {
      if (!imageio::info(bakerInput.heightmap.filepath.c_str(), &w, &h, &comp))
      {
        std::stringstream s; s << "Error: heightmap texture image data in wrong format or could not read file at path '" << bakerInput.heightmap.filepath << "'";
        throw std::runtime_error(s.str());
      }

      data = imageio::loadF(bakerInput.heightmap.filepath.c_str(), &w, &h, &comp, 1);
    }

    size_t dataSize = w * h * sizeof(float);

    heightmapConfig.baseFormat = micromesh::Format::eR32_sfloat;
    heightmapConfig.width = static_cast<uint32_t>(w);
    heightmapConfig.height = static_cast<uint32_t>(h);
    heightmapConfig.mips = 1;
    heightmapConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;

    meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerHeightmapSource, heightmapConfig, dataSize, data));
    imageio::freeData(&data);
    if(!meshopsTextures.back()->valid())
    {
      throw std::runtime_error("Error: meshopsTextureCreate() failed to create heightmap texture\n");
    }

    heightmapDesc.texture = *meshopsTextures.back();
  }

  // Set up resampled textures
  if(bakerInput.resamplerInput.size() > 0)
  {
      if (baseMeshIncludesTexCoords)
      {
          for (py::handle handle: bakerInput.resamplerInput)
          {
              meshops::OpBake_resamplerInput input;
              PyResamplerInput pyResamplerInput = py::cast<PyResamplerInput>(handle);

              size_t w = 0, h = 0, comp = 0;
              imageio::ImageIOData data = nullptr;
              if(pyResamplerInput.input.filepath.empty())
              {
                if (pyResamplerInput.input.format == PyTextureFormat::eRGBA16Unorm ||
                    pyResamplerInput.input.format == PyTextureFormat::eRGBA8Unorm)
                {
                  std::vector<uint8_t> rawData;
                  numpyArrayToVector<1, uint8_t, uint8_t>(pyResamplerInput.input.data, rawData);

                  if (rawData.size() > 0)
                  {
                    w = pyResamplerInput.input.width;
                    h = pyResamplerInput.input.height;

                    int bitDepth = 1;
                    switch (pyResamplerInput.input.format)
                    {
                      case PyTextureFormat::eRGBA8Unorm:
                        bitDepth = 8;
                        comp = 4;
                        break;
                      case PyTextureFormat::eRGBA16Unorm:
                        bitDepth = 16;
                        comp = 4;
                        break;
                    };

                    size_t sizeCheck = w * h * comp * (bitDepth / CHAR_BIT);

                    if (sizeCheck != rawData.size())
                    {
                      imageio::freeData(&data);
                      std::stringstream s; s << "Error: resampler input texture image data inconsistent with width '" << pyResamplerInput.input.width << "', height '" << pyResamplerInput.input.height << "', and format '" << pyResamplerInput.input.format << "' provided";
                      throw std::runtime_error(s.str());
                    }

                    data = imageio::allocateData(rawData.size());
                    memcpy(data, rawData.data(), rawData.size());

                    // Convert from 8-16 bit if necessary
                    if (pyResamplerInput.input.format == PyTextureFormat::eRGBA8Unorm)
                    {
                      if(!imageio::convertFormat(&data, w, h, 4, 8, 4, 16))
                      {
                        imageio::freeData(&data);
                        throw std::runtime_error("Error: failed to convert resampler input texture image data");
                      }
                    }
                  }
                  else
                  {
                    throw std::runtime_error("Error: resampler input texture image data is empty");
                  }
                }
                else
                {
                    throw std::runtime_error("Error: resampler input texture image data format is not compatible (8 or 16-bit RGBAUnorm only)"); 
                }
              }
              else
              {
                if (!imageio::info(pyResamplerInput.input.filepath.c_str(), &w, &h, &comp))
                {
                  std::stringstream s; s << "Error: resampler input texture image data in wrong format or could not read file at path '" << pyResamplerInput.input.filepath << "'";
                  throw std::runtime_error(s.str());
                }

                data = imageio::load16(pyResamplerInput.input.filepath.c_str(), &w, &h, &comp, 4);
              }

              size_t dataSize = w * h * 4 * 2 * sizeof(char);

              meshops::TextureConfig config;
              config.baseFormat = micromesh::Format::eRGBA16_unorm;
              config.width = static_cast<uint32_t>(w);
              config.height = static_cast<uint32_t>(h);
              config.mips = 1;
              config.internalFormatVk = VK_FORMAT_R16G16B16A16_UNORM;

              meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingSource, config, dataSize, data));
              imageio::freeData(&data);
              if(!meshopsTextures.back()->valid())
              {
                throw std::runtime_error("Error: meshopsTextureCreate() failed to create resampled input texture\n");
              }

              if (pyResamplerInput.input.type != meshops::TextureType::eNormalMap)
              {
                input.textureType = meshops::TextureType::eGeneric;
              }
              else
              {
                input.textureType = meshops::TextureType::eNormalMap;
              }

              input.texture = *meshopsTextures.back();

              micromesh::MicromapValue fillInitDist{};
              fillInitDist.value_float[0] = std::numeric_limits<float>::max();

              meshops::TextureConfig distanceConfig;
              distanceConfig.baseFormat = micromesh::Format::eR32_sfloat;
              distanceConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
              distanceConfig.width      = uint32_t(pyResamplerInput.output.width);
              distanceConfig.height     = uint32_t(pyResamplerInput.output.height);
              distanceConfig.mips       = static_cast<uint32_t>(std::floor(std::log2(std::min(pyResamplerInput.output.width, pyResamplerInput.output.height)))) + 1;

              meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDistance, distanceConfig, &fillInitDist));
              if(!meshopsTextures.back()->valid())
              {
                  throw std::runtime_error("Error: meshopsTextureCreate() failed to create reampled distance texture");
              }

              input.distance = *meshopsTextures.back();

              meshops::TextureConfig outputConfig;
              outputConfig.baseFormat = micromesh::Format::eRGBA16_unorm;
              outputConfig.width = static_cast<uint32_t>(pyResamplerInput.output.width);
              outputConfig.height = static_cast<uint32_t>(pyResamplerInput.output.height);
              outputConfig.mips = 1;
              outputConfig.internalFormatVk = VK_FORMAT_R16G16B16A16_UNORM;

              size_t outputImageDataSize = outputConfig.width * outputConfig.height * 4 * 2 * sizeof(char);

              std::vector<uint8_t> outputImageData;
              outputImageData.resize(outputImageDataSize);

              meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDestination, outputConfig, outputImageDataSize, outputImageData.data()));
              if(!meshopsTextures.back()->valid())
              {
                throw std::runtime_error("Error: meshopsTextureCreate() failed to create resampled input texture");
              }

              resamplerOutput.push_back(*meshopsTextures.back());

              resamplerInput.emplace_back(input);
          }
      }
      else
      {
        LOGI("There are textures to be resampled but base mesh does not contain texture coordinates; ignoring\n");
      }
  }



  // Create normal map texture
  int outputNormalMapOutputIndex = -1;
  if(!bakerInput.normalMapFilepath.empty() && baseMeshIncludesTexCoords)
  {
      meshops::OpBake_resamplerInput input;

      micromesh::MicromapValue fillValue;
      fillValue.value_int32[0] = fillValue.value_int32[1] = fillValue.value_int32[2] = fillValue.value_int32[3] = 0;

      meshops::TextureConfig inputConfig;
      inputConfig.baseFormat       = micromesh::Format::eR32_sfloat;
      inputConfig.internalFormatVk    = VK_FORMAT_R32_SFLOAT;
      inputConfig.width            = uint32_t(bakerInput.normalMapResolution);
      inputConfig.height              = uint32_t(bakerInput.normalMapResolution);
      inputConfig.mips = static_cast<uint32_t>(std::floor(std::log2(bakerInput.normalMapResolution))) + 1;

      meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingSource, inputConfig, &fillValue));
      if(!meshopsTextures.back()->valid())
      {
        throw std::runtime_error("Error: meshopsTextureCreate() failed to create quaternion input texture");
      }

      input.textureType = meshops::TextureType::eQuaternionMap;
      input.texture = *meshopsTextures.back();

      micromesh::MicromapValue fillInitDist{};
      fillInitDist.value_float[0] = std::numeric_limits<float>::max();

      meshops::TextureConfig distanceConfig;
      distanceConfig.baseFormat = micromesh::Format::eR32_sfloat;
      distanceConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
      distanceConfig.width      = uint32_t(bakerInput.normalMapResolution);
      distanceConfig.height     = uint32_t(bakerInput.normalMapResolution);
      distanceConfig.mips       = static_cast<uint32_t>(std::floor(std::log2(bakerInput.normalMapResolution))) + 1;

      meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDistance, distanceConfig, &fillInitDist));
      if(!meshopsTextures.back()->valid())
      {
          throw std::runtime_error("Error: meshopsTextureCreate() failed to create quaternion distance texture\n");
      }

      input.distance = *meshopsTextures.back();

      resamplerInput.push_back(input);

      meshops::TextureConfig quaternionMapConfig;
      quaternionMapConfig.baseFormat  = micromesh::Format::eRGBA8_unorm;
      quaternionMapConfig.internalFormatVk = VK_FORMAT_R8G8B8A8_UNORM;
      quaternionMapConfig.width      = uint32_t(bakerInput.normalMapResolution);
      quaternionMapConfig.height     = uint32_t(bakerInput.normalMapResolution);
      quaternionMapConfig.mips = static_cast<uint32_t>(std::floor(std::log2(bakerInput.normalMapResolution))) + 1;

      meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDestination, quaternionMapConfig, &fillValue));
      if(!meshopsTextures.back()->valid())
      {
          throw std::runtime_error("Error: meshopsTextureCreate() failed to create quaternion output texture\n");
      }

      outputNormalMapOutputIndex = (int)resamplerOutput.size();
      resamplerOutput.push_back(*meshopsTextures.back());
  }


  // Create uv remap texture
  int outputUvRemapOutputIndex = -1;
  if(!bakerInput.uvRemapFilepath.empty() && baseMeshIncludesTexCoords)
  {
    meshops::OpBake_resamplerInput input;

    micromesh::MicromapValue fillValue;
    fillValue.value_int32[0] = fillValue.value_int32[1] = fillValue.value_int32[2] = fillValue.value_int32[3] = 0;

    meshops::TextureConfig inputConfig;
    inputConfig.baseFormat       = micromesh::Format::eR32_sfloat;
    inputConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
    inputConfig.width            = uint32_t(bakerInput.uvRemapResolution);
    inputConfig.height           = uint32_t(bakerInput.uvRemapResolution);
    inputConfig.mips             = static_cast<uint32_t>(std::floor(std::log2(bakerInput.uvRemapResolution))) + 1;

    meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingSource,
                                                                                inputConfig, &fillValue));
    if(!meshopsTextures.back()->valid())
    {
      throw std::runtime_error("Error: meshopsTextureCreate() failed to create offset input texture\n");
    }

    input.textureType = meshops::TextureType::eOffsetMap;
    input.texture     = *meshopsTextures.back();

    micromesh::MicromapValue fillInitDist{};
    fillInitDist.value_float[0] = std::numeric_limits<float>::max();

    meshops::TextureConfig distanceConfig;
    distanceConfig.baseFormat       = micromesh::Format::eR32_sfloat;
    distanceConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
    distanceConfig.width            = uint32_t(bakerInput.uvRemapResolution);
    distanceConfig.height           = uint32_t(bakerInput.uvRemapResolution);
    distanceConfig.mips             = static_cast<uint32_t>(std::floor(std::log2(bakerInput.uvRemapResolution))) + 1;

    meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDistance,
                                                                                distanceConfig, &fillInitDist));
    if(!meshopsTextures.back()->valid())
    {
      throw std::runtime_error("Error: meshopsTextureCreate() failed to create offset distance texture\n");
    }

    input.distance = *meshopsTextures.back();

    resamplerInput.push_back(input);

    meshops::TextureConfig offsetMapConfig;
    offsetMapConfig.baseFormat           = micromesh::Format::eRGBA16_unorm;
    offsetMapConfig.internalFormatVk     = VK_FORMAT_R16G16B16A16_UNORM;
    offsetMapConfig.width                = uint32_t(bakerInput.uvRemapResolution);
    offsetMapConfig.height               = uint32_t(bakerInput.uvRemapResolution);
    offsetMapConfig.mips = static_cast<uint32_t>(std::floor(std::log2(bakerInput.uvRemapResolution))) + 1;

    meshopsTextures.push_back(std::make_unique<micromesh_tool::MeshopsTexture>(context, meshops::eTextureUsageBakerResamplingDestination, offsetMapConfig, &fillValue));
    if(!meshopsTextures.back()->valid())
    {
      throw std::runtime_error("Error: meshopsTextureCreate() failed to create offset output texture\n");
    }

    outputUvRemapOutputIndex = (int)resamplerOutput.size();
    resamplerOutput.push_back(*meshopsTextures.back());
  }


  OpBake_settings settings;
  bakerInput.settings.toSettings(settings);


  micromesh_tool::BakeOperator bakeOperator(context);
  meshops::OpBake_properties   bakeProperties;
  meshops::meshopsBakeGetProperties(context, bakeOperator, bakeProperties);

    // Make sure subdivision levels get generated unless explicitly requesting uniform values
  bool uniformSubdivLevels = bakerInput.settings.subdivMethod == PySubdivMethod::eUniform;

  // Query the mesh attributes needed to bake
  meshops::OpBake_requirements meshRequirements;
  meshops::meshopsBakeGetRequirements(context, bakeOperator, settings, resamplerInput,
                                      uniformSubdivLevels, heightmapDesc.texture != nullptr,
                                      heightmapDesc.usesVertexNormalsAsDirections, meshRequirements);

  if(!uniformSubdivLevels)
  {
      // while the baker doesn't need the basemesh with triangle primitive flags, the resulting
      // mesh must be consistent for further processing / saving etc.
      meshRequirements.baseMeshAttribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit;
  }

  // If we want uniform subdiv levels, we should not pass in a per-triangle array.
  // If we want generated subdiv levels we need to clear and re-generate existing ones.
  bool generateSubdivLevels = bakerInput.settings.subdivMethod == PySubdivMethod::eAdaptive3D
                              || bakerInput.settings.subdivMethod == PySubdivMethod::eAdaptiveUV;
  if(bakerInput.settings.subdivMethod == PySubdivMethod::eUniform || generateSubdivLevels)
  {
    if(!baseMeshView.triangleSubdivisionLevels.empty())
    {
      LOGW("Warning: clearing base mesh's subdivision levels due to --subdivmode.\n");
    }
    baseMeshView.triangleSubdivisionLevels = {};

    if(!baseMeshView.trianglePrimitiveFlags.empty())
    {
      LOGW("Warning: clearing base mesh's primitive flags due to --subdivmode.\n");
    }
    baseMeshView.trianglePrimitiveFlags = {};
  }

  // Warn if the input subdiv level is all ones or zeroes
  if(!baseMeshView.triangleSubdivisionLevels.empty())
  {
    int maxSubdivLevel = *std::max_element(baseMeshView.triangleSubdivisionLevels.begin(),
                                            baseMeshView.triangleSubdivisionLevels.end());
    if(maxSubdivLevel < 2)
    {
      LOGW("Warning: max input subdivision level in the base mesh is only %i\n", maxSubdivLevel);
    }
  }

  // Base mesh
  {
      meshops::OpGenerateSubdivisionLevel_input baseSubdivSettings;
      baseSubdivSettings.maxSubdivLevel = settings.level;
      baseSubdivSettings.subdivLevelBias = bakerInput.settings.lowTessBias;
      baseSubdivSettings.relativeWeight = bakerInput.settings.adaptiveFactor;
      baseSubdivSettings.useTextureArea = bakerInput.settings.subdivMethod == PySubdivMethod::eAdaptiveUV;

      if(baseSubdivSettings.useTextureArea)
      {
          if(heightmapDesc.texture == nullptr)
          {
              throw std::runtime_error("Error: adaptiveUV given but the reference mesh has no heightmap");
          }
          baseSubdivSettings.textureWidth  = heightmapConfig.width;
          baseSubdivSettings.textureHeight = heightmapConfig.height;
      }
      uint32_t          maxGeneratedSubdivLevel;

      // Scoped here to ensure Python global interpreter lock is
      // released while this function does all its heavy lifting
      {
          py::gil_scoped_release release;
          micromesh::Result      result =
              generateMeshAttributes(context, meshRequirements.baseMeshAttribFlags, &baseSubdivSettings,
                                      baseMeshTopology, baseMeshView, maxGeneratedSubdivLevel,
                                      bakerInput.settings.normalReduceOp, bakerInput.settings.tangentAlgorithm);
          if(result != micromesh::Result::eSuccess)
          {
              LOGE("Error: generating attributes for base mesh failed\n");
              throw std::runtime_error("unable to generate attributes for base mesh");
          }
      }
  }

  // Reference mesh
  // Updates heightmapDesc.maxSubdivLevel if subdiv levels are generated (it is unlikely to already have them)
  {
      meshops::OpGenerateSubdivisionLevel_input referenceSubdivSettings;
      referenceSubdivSettings.maxSubdivLevel  = bakeProperties.maxHeightmapTessellateLevel;
      referenceSubdivSettings.subdivLevelBias = bakerInput.settings.highTessBias;
      referenceSubdivSettings.textureWidth    = heightmapConfig.width;
      referenceSubdivSettings.textureHeight   = heightmapConfig.height;
      referenceSubdivSettings.useTextureArea  = true;

      // Scoped here to ensure Python global interpreter lock is
      // released while this function does all its heavy lifting
      {
          py::gil_scoped_release release;

          micromesh::Result result =
              generateMeshAttributes(context, meshRequirements.referenceMeshAttribFlags, &referenceSubdivSettings,
                                      referenceMeshTopology, referenceMeshView, heightmapDesc.maxSubdivLevel,
                                      bakerInput.settings.normalReduceOp, bakerInput.settings.tangentAlgorithm);
          if(result != micromesh::Result::eSuccess)
          {
              LOGE("Error: generating attributes for reference mesh failed\n");
              throw std::runtime_error("unable to generate attributes for reference mesh");
          }
      }
  }

  baryutils::BaryBasicData baryUncompressedTemp;

  meshops::OpBake_input input;
  input.settings                                                 = settings;
  input.baseMeshView                                             = baseMeshView;
  input.baseMeshTopology                                         = baseMeshTopology;
  input.referenceMeshView                                        = referenceMeshView;
  input.referenceMeshTopology                                    = referenceMeshTopology;
  input.referenceMeshHeightmap                                   = heightmapDesc;
  
  input.resamplerInput                                           = meshops::ArrayView(resamplerInput);
  input.baseMeshTransform                                        = baseMeshTransform;
  input.referenceMeshTransform                                   = referenceMeshTransform;

  std::vector<baryutils::BaryContentData> baryContents;
  baryContents.emplace_back();

  OpBake_output output;

  output.resamplerTextures        = meshops::ArrayView(resamplerOutput);
  output.uncompressedDisplacement = bakerInput.settings.enableCompression ? &baryUncompressedTemp : &baryContents.back().basic;


  std::vector<nvmath::vec2f> vertexDirectionBounds;
  vertexDirectionBounds.resize(input.baseMeshView.vertexDirectionBounds.size());
  std::copy(input.baseMeshView.vertexDirectionBounds.begin(), input.baseMeshView.vertexDirectionBounds.end(), vertexDirectionBounds.begin());
  output.vertexDirectionBounds = vertexDirectionBounds;

  // Scoped here to ensure Python global interpreter lock is  
  // released while this function does all its heavy lifting
  {
      py::gil_scoped_release release;

      micromesh::Result result = meshops::meshopsOpBake(context, bakeOperator, input, output);
      if(result != micromesh::Result::eSuccess)
      {
          throw std::runtime_error("baking mesh failed");
      }
  }

  baryutils::BaryBasicData* outputData = nullptr;

  if(bakerInput.settings.enableCompression)
  {
      bary::BasicView uncompressedView = output.uncompressedDisplacement->getView();

      meshops::OpCompressDisplacementMicromap_input compressedInput;
      compressedInput.meshTopology             = baseMeshTopology;
      compressedInput.meshView                 = baseMeshView;
      compressedInput.settings.minimumPSNR     = bakerInput.settings.minPSNR;
      compressedInput.settings.validateInputs  = true;
      compressedInput.settings.validateOutputs = true;
      compressedInput.uncompressedDisplacement = &uncompressedView;
      compressedInput.uncompressedDisplacementGroupIndex = 0;

      meshops::OpCompressDisplacementMicromap_output compressedOutput;
      compressedOutput.compressedDisplacement = &baryContents.back().basic;
      compressedOutput.compressedDisplacementRasterMips = bakerInput.settings.compressedRasterData ? &baryContents.back().misc : nullptr;

      micromesh::Result result = meshops::meshopsOpCompressDisplacementMicromaps(context, 1, &compressedInput, &compressedOutput);
      if(result != micromesh::Result::eSuccess)
      {
          throw std::runtime_error("compressing mesh failed");
      }

      outputData     = compressedOutput.compressedDisplacement;
  }
  else
  {
      outputData = output.uncompressedDisplacement;
  }

  //
  // Save textures to disk (these could be condensed into one loop like in baker with a little bit of refactoring)
  //

  // Write out resampled textures
  size_t resampledTextureIndex = 0;
  for (py::handle handle: bakerInput.resamplerInput)
  {
    if (resampledTextureIndex >= resamplerOutput.size())
    {
      break;
    }

    PyResamplerInput pyResamplerInput = py::cast<PyResamplerInput>(handle);
    meshops::Texture tex = resamplerOutput[resampledTextureIndex];
    
    size_t rawDataSize = meshops::meshopsTextureGetMipDataSize(tex, 0);
    std::vector<uint8_t> rawData;
    rawData.resize(rawDataSize);
    meshops::meshopsTextureToData(context, tex, rawDataSize, &rawData[0]);

    imageio::ImageIOData data = imageio::allocateData(rawData.size());
    memcpy(data, rawData.data(), rawData.size());

    int outputDataSize = 0;
    int w = pyResamplerInput.output.width;
    int h = pyResamplerInput.output.height;
    // Convert from internal format to output format if necessary
    if (pyResamplerInput.output.format == PyTextureFormat::eRGBA8Unorm)
    {
      if(!imageio::convertFormat(&data, w, h, 4, 16, 4, 8))
      {
        imageio::freeData(&data);
        throw std::runtime_error("Error: failed to convert resampler output texture image data");
      }
      outputDataSize = w * h * 4;
    }
    else if (pyResamplerInput.output.format == PyTextureFormat::eRGBA16Unorm)
    {
      //if(!imageio::convertFormat(&data, w, h, 4, 16, 4, 8))
      //{
      //  imageio::freeData(&data);
      //  throw std::runtime_error("Error: failed to convert resampler output texture image data");
      //}
      outputDataSize = w * h * 4 * 2;
    }
    else if (pyResamplerInput.output.format == PyTextureFormat::eR16Unorm)
    {
      if(!imageio::convertFormat(&data, w, h, 4, 16, 1, 16))
      {
        imageio::freeData(&data);
        throw std::runtime_error("Error: failed to convert resampler output texture image data");
      }
      outputDataSize = w * h * 2;
    }

    if (!pyResamplerInput.output.filepath.empty())
    {
      if (!imageio::writePNG(pyResamplerInput.output.filepath.c_str(), pyResamplerInput.output.width,
                        pyResamplerInput.output.height, data, (VkFormat)pyResamplerInput.output.format))
      {
        std::stringstream s; s << "Error: failed to write resampled output texture (" << pyResamplerInput.output.filepath << ")";
        throw std::runtime_error(s.str());
      }
    }
    else
    {
      rawData.assign((uint8_t *)data, (uint8_t *)data + outputDataSize);
      vectorToNumpyArray<1, uint8_t, uint8_t>(rawData, pyResamplerInput.output.data);
    }

    resampledTextureIndex++;
  }

  // Write out quat map
  if(!bakerInput.normalMapFilepath.empty() && outputNormalMapOutputIndex >= 0
     && static_cast<size_t>(outputNormalMapOutputIndex) < resamplerOutput.size())
  {
    meshops::Texture tex = resamplerOutput[outputNormalMapOutputIndex];
    
    size_t dataSize = meshops::meshopsTextureGetMipDataSize(tex, 0);
    std::vector<uint8_t> data;
    data.resize(dataSize);
    meshops::meshopsTextureToData(context, tex, dataSize, &data[0]);

    if(!imageio::writePNG(bakerInput.normalMapFilepath.c_str(), bakerInput.normalMapResolution,
                          bakerInput.normalMapResolution, &data[0], VK_FORMAT_R8G8B8A8_UNORM))
    {
      std::stringstream s;
      s << "Error: failed to write normal map (" << bakerInput.normalMapFilepath << ")";
      throw std::runtime_error(s.str());
    }
  }

  // Write out undistort map
  if(!bakerInput.uvRemapFilepath.empty() && outputUvRemapOutputIndex >= 0 && outputUvRemapOutputIndex < resamplerOutput.size())
  {
    meshops::Texture tex = resamplerOutput[outputUvRemapOutputIndex];

    size_t               dataSize = meshops::meshopsTextureGetMipDataSize(tex, 0);
    std::vector<uint8_t> data;
    data.resize(dataSize);
    meshops::meshopsTextureToData(context, tex, dataSize, &data[0]);

    if(!imageio::writePNG(bakerInput.uvRemapFilepath.c_str(), bakerInput.uvRemapResolution,
                          bakerInput.uvRemapResolution, &data[0], VK_FORMAT_R16G16B16A16_UNORM))
    {
      std::stringstream s;
      s << "Error: failed to write UV remap/undistort/offset texture (" << bakerInput.uvRemapFilepath << ")";
      throw std::runtime_error(s.str());
    }
  }

  // Copy to output
  bakeOutput.fromBaryData(outputData, &input.baseMeshView.vertexDirections,
                          reinterpret_cast<meshops::ArrayView<const nvmath::vec2f>*>(&output.vertexDirectionBounds),
                          &baseMeshView);
}

void displace(meshops::Context context, PyMesh& inputMesh, PyMicromeshData& inputMicromesh, PyMesh& outputMesh)
{
  if(!context)
  {
      throw std::runtime_error("no context available");
  }

  meshops::MeshData           mesh;
  meshops::ResizableMeshView  meshView(mesh, makeResizableMeshViewCallback(mesh));
  inputMesh.toMeshView(meshView);

  baryutils::BaryBasicData baryBasicData;

  baryBasicData.groups.resize(1);

  meshops::ArrayView<nvmath::vec3f> vertexDirections;
  meshops::ArrayView<nvmath::vec2f> vertexDirectionBounds;

  inputMicromesh.toBaryData(&baryBasicData, &vertexDirections, &vertexDirectionBounds, &meshView);

  meshView.vertexDirections = vertexDirections;
  meshView.vertexDirectionBounds = vertexDirectionBounds;

  const bary::BasicView baryBasicView = baryBasicData.getView();

  meshops::OpDisplacedTessellate_input input{};
  input.meshView                   = meshView;
  input.baryDisplacement           = &baryBasicView;
  input.baryDisplacementGroupIndex = 0;
  input.baryDisplacementMapOffset  = 0;

  if(!meshView.hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBit))
  {
    LOGW("Warning: missing direction vectors. Using normals instead; there may be cracks.\n");
    input.meshView.vertexDirections = meshView.vertexNormals;
  }

  meshops::MeshData          tessellatedMesh;

  meshops::OpDisplacedTessellate_output output{};
  meshops::ResizableMeshView tessellatedMeshView(tessellatedMesh, makeResizableMeshViewCallback(tessellatedMesh));
  output.meshView = &tessellatedMeshView;

  micromesh::Result result = meshops::meshopsOpDisplacedTessellate(context, 1, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    throw std::runtime_error("displacing mesh failed");
  }

  outputMesh.fromMeshView(*output.meshView);

  fflush(stdout);
}

void remesh(meshops::Context context, PyMesh& inputMesh, PyRemesherSettings& settings, PyMesh& outputMesh)
{
  if (!context)
  {
      throw std::runtime_error("no context available");
  }

  meshops::MeshData mesh;
  meshops::ResizableMeshView  meshView(mesh, makeResizableMeshViewCallback(mesh));

  inputMesh.toMeshView(meshView);

  bool baseMeshIncludesTexCoords = meshView.vertexTexcoords0.size() > 0;

  micromesh_tool::GenerateImportanceOperator generateImportanceOperator(context);

  micromesh_tool::RemeshingOperator remeshingOperator(context);

  if(!generateImportanceOperator.valid())
  {
    throw std::runtime_error("Error: failed to create vertex importance operator\n");
  }

  meshops::MeshAttributeFlags requiredMeshAttributes =
      meshops::eMeshAttributeTriangleVerticesBit | meshops::eMeshAttributeTriangleSubdivLevelsBit
      | meshops::eMeshAttributeTrianglePrimitiveFlagsBit | meshops::eMeshAttributeVertexPositionBit
      | meshops::eMeshAttributeVertexNormalBit | meshops::eMeshAttributeVertexTangentBit
      | meshops::eMeshAttributeVertexDirectionBit | meshops::eMeshAttributeVertexDirectionBoundsBit
      | meshops::eMeshAttributeVertexImportanceBit | meshops::eMeshAttributeVertexTexcoordBit;

  // Allocate storage for output attributes, if missing
  const meshops::MeshAttributeFlags combinedMeshAttributes = (~meshView.getMeshAttributeFlags()) & requiredMeshAttributes;
  bool hadDirections = ((meshView.getMeshAttributeFlags() & meshops::eMeshAttributeVertexDirectionBit)
                        == meshops::eMeshAttributeVertexDirectionBit);
  meshView.resize(combinedMeshAttributes, meshView.triangleCount(), meshView.vertexCount());

  {
    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;

    if(meshopsGenerateVertexDirections(context, meshView) != micromesh::Result::eSuccess)
    {
      throw std::runtime_error("Error: could not generate valid per-vertex directions\n");
    }
  }

  uint64_t originalTriangleCount = meshView.triangleCount();

  meshops::DeviceMeshSettings deviceMeshSettings;
  deviceMeshSettings.usageFlags  = meshops::DeviceMeshUsageBlasBit;
  deviceMeshSettings.attribFlags = requiredMeshAttributes;
  meshops::DeviceMesh deviceMesh;
  micromesh::Result   result = micromesh::Result::eSuccess;
  {
    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;

    result = meshopsDeviceMeshCreate(context, meshView, deviceMeshSettings, &deviceMesh);
    if(result != micromesh::Result::eSuccess)
    {
      std::stringstream s; s << "Error: cannot create device mesh (" << micromesh::micromeshResultGetName(result) << ")";
      throw std::runtime_error(s.str());
    }
  }

  meshops::OpGenerateImportance_modified importanceParameters{};
  importanceParameters.deviceMesh             = deviceMesh;
  importanceParameters.meshView               = meshView;
  importanceParameters.importanceTextureCoord = ~0u;
  importanceParameters.importancePower        = settings.curvaturePower;

  meshops::Texture importanceMap;

  if(!settings.importanceMap.empty())
  {
    size_t               width = 0, height = 0, components = 0;
    size_t               requiredComponents = 1;
    imageio::ImageIOData importanceData =
        imageio::loadGeneral(settings.importanceMap.c_str(), &width, &height, &components, requiredComponents, 8);

    if(width == 0 || height == 0 || components == 0)
    {
      std::stringstream s; s << "Error: cannot load importance map '" << settings.importanceMap << "'"; 
      throw std::runtime_error(s.str());
    }

    meshops::TextureConfig config{};
    config.width            = static_cast<uint32_t>(width);
    config.height           = static_cast<uint32_t>(height);
    config.baseFormat       = micromesh::Format::eR8_unorm;
    config.internalFormatVk = VK_FORMAT_R8_UNORM;

    result = meshops::meshopsTextureCreateFromData(context,
                                                    meshops::TextureUsageFlagBit::eTextureUsageRemesherImportanceSource,
                                                    config, width * height, importanceData, &importanceMap);
    if(result != micromesh::Result::eSuccess)
    {
      std::stringstream s; s << "Error: cannot create meshops importance map texture '" << micromesh::micromeshResultGetName(result) << "'"; 
      throw std::runtime_error(s.str());
    }
    importanceParameters.importanceTexture      = importanceMap;
    importanceParameters.importanceTextureCoord = ((settings.importanceTexcoord == ~0u) ? 0 : settings.importanceTexcoord);
  }

  if(settings.curvatureMaxDistMode == PyRemesherCurvatureMaxDistanceMode::eWorldSpace)
  {
    importanceParameters.rayTracingDistance = settings.curvatureMaxDist;
  }
  if(settings.curvatureMaxDistMode == PyRemesherCurvatureMaxDistanceMode::eSceneFraction)
  {   
    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;

    meshops::ContextConfig contextConfig;
    result = meshops::meshopsContextGetConfig(context, &contextConfig);
    if(result != micromesh::Result::eSuccess)
    {
      std::stringstream s; s << "Error: cannot get meshops config '" << micromesh::micromeshResultGetName(result) << "'"; 
      throw std::runtime_error(s.str());
    }

    float scale                             = meshopsComputeMeshViewExtent(context, meshView);
    importanceParameters.rayTracingDistance = settings.curvatureMaxDist * scale;
  }
  { 
    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;

    result = meshops::meshopsOpGenerateImportance(context, generateImportanceOperator, 1, &importanceParameters);
    if(result != micromesh::Result::eSuccess)
    {
      std::stringstream s; s << "Error: cannot generate vertex importance '" << micromesh::micromeshResultGetName(result) << "'";
      throw std::runtime_error(s.str());
    }
  }
  if(!settings.importanceMap.empty())
  {
    meshops::meshopsTextureDestroy(context, importanceMap);
  }


  meshops::OpRemesh_input input{};
  input.errorThreshold        = settings.errorThreshold;
  input.maxOutputTriangleCount = settings.maxOutputTriangleCount;
  input.generateMicromeshInfo = !settings.disableMicromeshData;
  if(settings.heightmapWidth > 0 && settings.heightmapHeight > 0)
  {
    input.heightmapTextureCoord = (settings.heightmapTexcoord != ~0u) ? settings.heightmapTexcoord : 0;
  }
  else
  {
    input.heightmapTextureCoord = 0;
  }

  input.heightmapTextureWidth  = settings.heightmapWidth;
  input.heightmapTextureHeight = settings.heightmapHeight;
  input.importanceThreshold    = settings.importanceThreshold;
  input.importanceWeight       = settings.importanceWeight;

  if (settings.maxOutputTriangleCount == 0)
  {
    if(settings.decimationRatio > 0.f && settings.decimationRatio < 1.f)
    {
      input.maxOutputTriangleCount = static_cast<uint32_t>(static_cast<float>(meshView.triangleCount()) * settings.decimationRatio);
    }
    else
    {
      input.maxOutputTriangleCount = ~0u;
    }
  }
  
  input.maxSubdivLevel                = settings.maxSubdivLevel;
  input.maxVertexValence              = settings.maxVertexValence;
  input.progressiveRemeshing          = false;
  input.preservedVertexAttributeFlags = 0u;

  if(!settings.ignoreDisplacementDirections)
  {
    input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexDirectionBit;
  }
  if(!settings.ignoreNormals)
  {
    input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexNormalBit;
  }
  if(!settings.ignoreTangents)
  {
    input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexTangentBit;
  }
  if(!settings.ignoreTexCoords)
  {
    input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexTexcoordBit;
  }

  meshops::OpRemesh_modified modified{};
  modified.deviceMesh = deviceMesh;
  modified.meshView   = &meshView;

  {
    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;
    result = meshops::meshopsOpRemesh(context, remeshingOperator, 1, &input, &modified);
    if(result != micromesh::Result::eSuccess)
    {
      std::stringstream s; s << "Error: cannot remesh '" << micromesh::micromeshResultGetName(result) << "'";
      throw std::runtime_error(s.str());
    }
  }

  meshops::meshopsDeviceMeshDestroy(context, deviceMesh);

  uint64_t finalTriangleCount = meshView.triangleCount();
  LOGI("  Triangles: %zu -> %zu\n", originalTriangleCount, finalTriangleCount);

  outputMesh.fromMeshView(meshView);
}

void preTessellate(meshops::Context context, PyMesh& inputMesh, PyPreTessellatorSettings& settings, PyMesh& outputMesh)
{
  if (!context)
  {
      throw std::runtime_error("no context available");
  }

  meshops::MeshData mesh;
  meshops::ResizableMeshView  meshView(mesh, makeResizableMeshViewCallback(mesh));

  inputMesh.toMeshView(meshView);

  bool baseMeshIncludesTexCoords = meshView.vertexTexcoords0.size() > 0;

  size_t totalTriangles    = 0;
  size_t totalNewTriangles = 0;

  meshops::MeshTopologyData meshTopology;
  if(micromesh_tool::buildTopologyData(context, meshView, meshTopology) != micromesh::Result::eSuccess)
  {
    throw std::runtime_error("Error: failed to build mesh topology");
  }

  // Generate subdivision levels and edge flags
  meshops::OpGenerateSubdivisionLevel_input baseSubdivSettings;

  if(settings.edgeLengthBased && settings.maxSubdivLevel == 0)
  {
    throw std::runtime_error("Error: must choose non-zero maxSubdivLevel when edgeLengthBased is enabled");
  }

  if (settings.maxSubdivLevel == 0)
  {
    baseSubdivSettings.maxSubdivLevel = baryutils::BaryLevelsMap::MAX_LEVEL;
  }
  else
  {
    baseSubdivSettings.maxSubdivLevel = settings.maxSubdivLevel;
  }

  baseSubdivSettings.useTextureArea  = !settings.edgeLengthBased;
  baseSubdivSettings.subdivLevelBias = settings.subdivLevelBias;
  baseSubdivSettings.textureWidth    = settings.heightmapWidth;
  baseSubdivSettings.textureHeight   = settings.heightmapHeight;

  uint32_t          maxGeneratedSubdivLevel;
  micromesh::Result result =
      generateMeshAttributes(context,
                              meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
                                  | meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit
                                  | meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit,
                              &baseSubdivSettings, meshTopology, meshView, maxGeneratedSubdivLevel,
                              NormalReduceOp::eNormalReduceNormalizedLinear, meshops::TangentSpaceAlgorithm::eDefault);
  if(result != micromesh::Result::eSuccess)
  {
    throw std::runtime_error("Error: generating attributes for mesh failed");
  }

  // Tessellate based on the generated subdivision levels
  {
    meshops::OpPreTessellate_input input{};
    input.maxSubdivLevel = maxGeneratedSubdivLevel;
    input.meshView       = meshView;
    meshops::OpPreTessellate_output output{};
    output.meshView = &meshView;

    // Scoped here to ensure Python global interpreter lock is  
    // released while this function does all its heavy lifting
    py::gil_scoped_release release;

    result = meshops::meshopsOpPreTessellate(context, 1, &input, &output);
    if(result != micromesh::Result::eSuccess)
    {
      throw std::runtime_error("Error: failed to tessellate mesh");
    }
  }

  // Subdiv levels were generated for tessellation input but should not be saved
  assert(meshView.triangleSubdivisionLevels.empty());
  assert(meshView.trianglePrimitiveFlags.empty());

  LOGI("  Triangles: %zu -> %zu\n", meshView.triangleCount(), meshView.triangleCount());

  outputMesh.fromMeshView(meshView);

  // Remove triangleSubdivisionLevels and trianglePrimitives flags as workaround to above not being empty
  std::vector<uint16_t> emptyTriangleSubdivisionLevels;
  std::vector<uint8_t> emptyTrianglePrimitiveFlags;
  vectorToNumpyArray<1, uint16_t, uint16_t>(emptyTriangleSubdivisionLevels, outputMesh.triangleSubdivisionLevels);
  vectorToNumpyArray<1, uint8_t, uint8_t>(emptyTrianglePrimitiveFlags, outputMesh.trianglePrimitiveFlags);
}

bool writeBary(meshops::Context context, std::string filename, PyMesh& inputMesh, PyMicromeshData& inputMicromesh, bool forceOverwrite=false)
{
  if(!context)
  {
      throw std::runtime_error("no context available");
  }

  meshops::MeshData           mesh;
  meshops::ResizableMeshView  meshView(mesh, makeResizableMeshViewCallback(mesh));
  inputMesh.toMeshView(meshView);

  baryutils::BaryBasicData baryBasicData;

  baryBasicData.groups.resize(1);

  meshops::ArrayView<nvmath::vec3f> vertexDirections;
  meshops::ArrayView<nvmath::vec2f> vertexDirectionBounds;

  inputMicromesh.toBaryData(&baryBasicData, &vertexDirections, &vertexDirectionBounds, &meshView);

  meshView.vertexDirections = vertexDirections;
  meshView.vertexDirectionBounds = vertexDirectionBounds;

  bary::ContentView baryContentView;

  baryContentView.basic = baryBasicData.getView();
  //baryContentView.mesh  = meshView;
  
  bary::StandardPropertyType errorProp;
  baryutils::BarySaver       saver;
  bary::Result               result = saver.initContent(&baryContentView, &errorProp);
  if(result != bary::Result::eSuccess)
  {
    std::stringstream s; s << "Error: Failure initializing content for '" << filename << "'";
    throw std::runtime_error(s.str());
  }

  result = saver.save(filename);
  if(result != bary::Result::eSuccess)
  {
    std::stringstream s; s << "Error: Failure writing '" << filename << "'";
    throw std::runtime_error(s.str());
  }

  return true;
}

bool readBary(meshops::Context context, std::string filename, PyMesh& inputMesh, PyMicromeshData& outputMicromesh)
{
    baryutils::BaryFile        bfile;
    bary::StandardPropertyType errorProp = bary::StandardPropertyType::eUnknown;

    baryutils::BaryFileOpenOptions openOptions = {0};
    openOptions.fileApi.userData               = nullptr;
    openOptions.fileApi.read                   = nullptr; // Support later
    openOptions.fileApi.release                = nullptr;

    bary::Result result = bfile.open(filename, &openOptions, &errorProp);

    meshops::MeshData           mesh;
    meshops::ResizableMeshView  meshView(mesh, makeResizableMeshViewCallback(mesh));
    inputMesh.toMeshView(meshView);

    baryutils::BaryBasicData baryBasicData;

    if(result == bary::Result::eErrorVersion)
    {
        bfile.close();
        throw std::runtime_error("Error: .bary has unsupported version");
    }
    else if(result == bary::Result::eSuccess)
    {
      baryBasicData.setData(bfile.getBasic());
    }
    else
    {
        return false;
    }

    bfile.close();

    outputMicromesh.fromBaryData(&baryBasicData, (meshops::ConstArrayView<nvmath::vec3f>*)nullptr, (meshops::ConstArrayView<nvmath::vec2f> *)nullptr, (meshops::ResizableMeshView *)&meshView);

    return true;
}