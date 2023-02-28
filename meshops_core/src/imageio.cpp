/*
* SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

// This has _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES enabled so that MSVC turns
// stb_image_write's sprintf call into an sprintf_s call, avoiding the CRT
// deprecation warning.
#ifndef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#include "imageio/imageio.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"
// Currently we only use libpng for writing, but we have the stb_image_write
// implementation here so that other parts of the code can use it.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb/stb_image_write.h"
#include "png.h"

#define TINYEXR_IMPLEMENTATION
#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#pragma warning(disable : 4018)  // C4018 : '<' : signed / unsigned mismatch
#include "tinyexr.h"
#pragma warning(default : 4018)
#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

#include <nvh/filemapping.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

namespace imageio {
// In practice, all our image loaders use malloc() and free().

ImageIOData allocateData(size_t byteLength)
{
  return malloc(byteLength);
}

void freeData(ImageIOData* pData)
{
  if(pData)
  {
    free(*pData);
    *pData = nullptr;
  }
}

void freeDataNoClear(ImageIOData data)
{
  freeData(&data);
}

//-------------------------------------------
// Template code generation for convertFormat

template <class Out, class In>
void convertElement(Out* o, const In* i)
{
  *o = *i;
}

// For quantization, we use centered quantization, from
// http://cbloomrants.blogspot.com/2020/09/topics-in-quantization-for-games.html.

template <>
void convertElement<uint8_t, uint16_t>(uint8_t* o, const uint16_t* i)
{
  *o = static_cast<uint8_t>((*i) >> 8);
}

template <>
void convertElement<uint8_t, float>(uint8_t* o, const float* i)
{
  *o = uint8_t(std::min(std::max(0.f, *i), 1.0f) * 255.0f + 0.5f);
}

template <>
void convertElement<uint16_t, uint8_t>(uint16_t* o, const uint8_t* i)
{
  const uint8_t value = *i;
  *o                  = static_cast<uint16_t>((value << 8) + value);
}

template <>
void convertElement<uint16_t, float>(uint16_t* o, const float* i)
{
  *o = uint16_t(std::min(std::max(0.f, *i), 1.0f) * 65535.0f + 0.5f);
}

template <>
void convertElement<float, uint8_t>(float* o, const uint8_t* i)
{
  *o = float(*i) / 255.0f;
}

template <>
void convertElement<float, uint16_t>(float* o, const uint16_t* i)
{
  *o = float(*i) / 65535.0f;
}

template <size_t InputComponents, class InputType, size_t OutputComponents, class OutputType>
bool convertFormat_monomorphized_full(ImageIOData* inout_image, const size_t width, const size_t height)
{
  // If an earlier function failed, don't try to access null data.
  if(inout_image == nullptr)
  {
    return false;
  }

  if(InputComponents == OutputComponents && std::is_same_v<InputType, OutputType>)
  {
    // Nothing to do
    return true;
  }

  const InputType* input_elements = reinterpret_cast<const InputType*>(*inout_image);
  const size_t     num_pixels     = width * height;
  ImageIOData      output         = allocateData(num_pixels * OutputComponents * sizeof(OutputType));
  if(!output)
  {
    // Memory allocation failed!
    return false;
  }
  OutputType* output_elements = reinterpret_cast<OutputType*>(output);

  for(size_t pixel = 0; pixel < num_pixels; pixel++)
  {
    for(size_t c = 0; c < OutputComponents; c++)
    {
      OutputType* output_element = output_elements + (pixel * OutputComponents + c);
      if(c < InputComponents)
      {
        const InputType* input_element = input_elements + (pixel * InputComponents + c);
        convertElement<OutputType, InputType>(output_element, input_element);
      }
      else
      {
        if(c < 3)
        {
          // Red, green, or blue: Fill with 0
          const float rgb = 0.0f;
          convertElement<OutputType, float>(output_element, &rgb);
        }
        else
        {
          // Alpha: Fill with 1
          const float a = 1.0f;
          convertElement<OutputType, float>(output_element, &a);
        }
      }
    }
  }

  // Replace the input with the output:
  freeData(inout_image);
  *inout_image = output;
  return true;
}

template <size_t InputComponents, size_t OutputComponents>
bool convertFormat_monomorphized_both_components(ImageIOData* image,
                                                 const size_t width,
                                                 const size_t height,
                                                 const size_t input_bit_depth,
                                                 const size_t output_bit_depth)
{
  if(input_bit_depth == 8)
  {
    switch(output_bit_depth)
    {
      case 8:
        return convertFormat_monomorphized_full<InputComponents, uint8_t, OutputComponents, uint8_t>(image, width, height);
      case 16:
        return convertFormat_monomorphized_full<InputComponents, uint8_t, OutputComponents, uint16_t>(image, width, height);
      case 32:
        return convertFormat_monomorphized_full<InputComponents, uint8_t, OutputComponents, float>(image, width, height);
      default:
        break;
    }
  }
  else if(input_bit_depth == 16)
  {
    switch(output_bit_depth)
    {
      case 8:
        return convertFormat_monomorphized_full<InputComponents, uint16_t, OutputComponents, uint8_t>(image, width, height);
      case 16:
        return convertFormat_monomorphized_full<InputComponents, uint16_t, OutputComponents, uint16_t>(image, width, height);
      case 32:
        return convertFormat_monomorphized_full<InputComponents, uint16_t, OutputComponents, float>(image, width, height);
      default:
        break;
    }
  }
  else if(input_bit_depth == 32)
  {
    switch(output_bit_depth)
    {
      case 8:
        return convertFormat_monomorphized_full<InputComponents, float, OutputComponents, uint8_t>(image, width, height);
      case 16:
        return convertFormat_monomorphized_full<InputComponents, float, OutputComponents, uint16_t>(image, width, height);
      case 32:
        return convertFormat_monomorphized_full<InputComponents, float, OutputComponents, float>(image, width, height);
      default:
        break;
    }
  }
  assert(!"Invalid input_bit_depth and output_bit_depth combination! These must each be 8, 16, or 32.");
  return false;
}

template <size_t InputComponents>
bool convertFormat_monomorphized_input_components(ImageIOData* image,
                                                  const size_t width,
                                                  const size_t height,
                                                  const size_t input_bit_depth,
                                                  const size_t output_components,
                                                  const size_t output_bit_depth)
{
  switch(output_components)
  {
    case 1:
      return convertFormat_monomorphized_both_components<InputComponents, 1>(image, width, height, input_bit_depth, output_bit_depth);
    case 2:
      return convertFormat_monomorphized_both_components<InputComponents, 2>(image, width, height, input_bit_depth, output_bit_depth);
    case 3:
      return convertFormat_monomorphized_both_components<InputComponents, 3>(image, width, height, input_bit_depth, output_bit_depth);
    case 4:
      return convertFormat_monomorphized_both_components<InputComponents, 4>(image, width, height, input_bit_depth, output_bit_depth);
  }
  assert(!"Invalid value for output_components! This must be 1, 2, 3, or 4.");
  return false;
}

bool convertFormat(ImageIOData* image,
                   const size_t width,
                   const size_t height,
                   const size_t input_components,
                   const size_t input_bit_depth,
                   const size_t output_components,
                   const size_t output_bit_depth)
{
  switch(input_components)
  {
    case 1:
      return convertFormat_monomorphized_input_components<1>(image, width, height, input_bit_depth, output_components, output_bit_depth);
    case 2:
      return convertFormat_monomorphized_input_components<2>(image, width, height, input_bit_depth, output_components, output_bit_depth);
    case 3:
      return convertFormat_monomorphized_input_components<3>(image, width, height, input_bit_depth, output_components, output_bit_depth);
    case 4:
      return convertFormat_monomorphized_input_components<4>(image, width, height, input_bit_depth, output_components, output_bit_depth);
  }
  assert(!"Invalid value for input_components! This must be 1, 2, 3, or 4.");
  return false;
}

bool info(const char* filename, size_t* width, size_t* height, size_t* components)
{
  int       x, y, comp;
  const int stb_result = stbi_info(filename, &x, &y, &comp);
  if(stb_result)
  {
    *width      = static_cast<size_t>(x);
    *height     = static_cast<size_t>(y);
    *components = static_cast<size_t>(comp);
  }
  return stb_result;
}

bool infoFromMemory(const void* data, const size_t byteLength, size_t* width, size_t* height, size_t* components)
{
  if(byteLength > std::numeric_limits<int>::max())
  {
    return false;
  }
  int       x, y, comp;
  const int stb_result =
      stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(data), static_cast<int>(byteLength), &x, &y, &comp);
  if(stb_result)
  {
    *width      = static_cast<size_t>(x);
    *height     = static_cast<size_t>(y);
    *components = static_cast<size_t>(comp);
  }
  return stb_result;
}

bool is16Bit(const char* filename)
{
  return stbi_is_16_bit(filename);
}

enum class Loader
{
  eSTBImage,
  eLibPNG,
  eTinyEXR
};

Loader determineLoader(const void* data, const size_t byteLength)
{
  // Look at the magic number in the first four bytes of the file to
  // determine the file format.
  if(byteLength < 4)
  {
    return Loader::eSTBImage;
  }

  const uint8_t*               bytes = reinterpret_cast<const uint8_t*>(data);
  const std::array<uint8_t, 4> magicNumber{bytes[0], bytes[1], bytes[2], bytes[3]};

  if(magicNumber == std::array<uint8_t, 4>{0x89, 0x50, 0x4e, 0x47})
  {
    return Loader::eLibPNG;
  }
  else if(magicNumber == std::array<uint8_t, 4>{0x76, 0x2f, 0x31, 0x01})
  {
    return Loader::eTinyEXR;
  }

  return Loader::eSTBImage;
}

ImageIOData loadWithStbImage(const void*  data,
                             const size_t byteLength,
                             size_t*      width,
                             size_t*      height,
                             size_t*      components,
                             const size_t required_components,
                             const size_t required_bit_depth)
{
  int         x, y, comp;
  ImageIOData result = nullptr;

  if(byteLength > std::numeric_limits<int>::max())
  {
    // Too long for stb_image!
    return result;
  }
  const int      byteLengthInt = static_cast<int>(byteLength);
  const stbi_uc* bytes         = reinterpret_cast<const stbi_uc*>(data);

  int bit_depth_from_stb_image;
  if(required_bit_depth == 8)
  {
    result                   = stbi_load_from_memory(bytes, byteLengthInt, &x, &y, &comp, 0);
    bit_depth_from_stb_image = 8;
  }
  else if(required_bit_depth == 16 || (required_bit_depth == 32 && !stbi_is_hdr_from_memory(bytes, byteLengthInt)))
  {
    // stbi_loadf handles 16-bit files by converting to 8-bit, then to
    // 32-bit float. We want to avoid this quantization step, so we call
    // stbi_load_16 here in that case:
    result                   = stbi_load_16_from_memory(bytes, byteLengthInt, &x, &y, &comp, 0);
    bit_depth_from_stb_image = 16;
  }
  else
  {
    assert(required_bit_depth == 32);
    result                   = stbi_loadf_from_memory(bytes, byteLengthInt, &x, &y, &comp, 0);
    bit_depth_from_stb_image = 32;
  }

  *width      = x;
  *height     = y;
  *components = comp;

  // We call convertFormat near the end of each load function, instead of
  // inside loadGeneralFromMemory, because of the bit depth change that can
  // occur above.
  if(required_components > 0 || (static_cast<size_t>(bit_depth_from_stb_image) != required_bit_depth))
  {
    if(!convertFormat(&result, *width, *height, *components, static_cast<size_t>(bit_depth_from_stb_image),
                      required_components, required_bit_depth))
    {
      freeData(&result);
      return nullptr;
    }

    *components = required_components;
  }

  return result;
}

// Rather than a function to access a buffer directly, libpng uses a read
// callback through png_set_read_fn. This struct contains the user data for
// both read and write callbacks.
struct PNGMemoryStreamState
{
  const void* data;
  size_t      position;
  size_t      size;
};

void pngReadCallback(png_structp context, png_bytep outBytes, png_size_t byteCountToRead)
{
  PNGMemoryStreamState* state = reinterpret_cast<PNGMemoryStreamState*>(png_get_io_ptr(context));
  if(state == nullptr)
    return;
}

ImageIOData loadWithLibPNG(const void*  data,
                           const size_t byteLength,
                           size_t*      width,
                           size_t*      height,
                           size_t*      components,
                           const size_t required_components,
                           const size_t required_bit_depth)
{
  // Here, we use the simplified libpng API, based on libpng's example.c.
  ImageIOData result = nullptr;
  png_image   image{};
  image.version = PNG_IMAGE_VERSION;

  if(!png_image_begin_read_from_memory(&image, data, byteLength))
  {
    std::cerr << "PNG image load failed: " << image.message << "\n";
    return result;
  }

  size_t read_components{};
  size_t read_bit_depth{};
  if(required_components == 0)
  {
    read_components = PNG_IMAGE_SAMPLE_CHANNELS(image.format);
    read_bit_depth  = PNG_IMAGE_SAMPLE_COMPONENT_SIZE(image.format) * 8;
    // libpng's 2-component format is {gray, alpha}, which isn't quite what
    // we want in convertFormat. So force that to RGBA:
    if(read_components == 2)
    {
      image.format = image.format | PNG_FORMAT_FLAG_COLOR;
    }
    // Make sure it's not color-mapped.
    image.format = image.format & (~PNG_FORMAT_FLAG_COLORMAP);
  }
  else if(required_components == 1)
  {
    if(required_bit_depth > 8)
    {
      // Use PNG "2-byte format" when 16 or 32 bits are requested
      image.format = PNG_FORMAT_LINEAR_Y;
    }
    else
    {
      // Note that we could use LINEAR_Y to perform gamma conversion here,
      // but instead, we ignore the color space information of the file.
      image.format = PNG_FORMAT_GRAY;
    }
  }
  else if(required_components <= 3)
  {
    image.format = PNG_FORMAT_RGB;
  }
  else
  {
    image.format = PNG_FORMAT_RGBA;
  }

  read_components = PNG_IMAGE_SAMPLE_CHANNELS(image.format);
  read_bit_depth  = PNG_IMAGE_SAMPLE_COMPONENT_SIZE(image.format) * 8;

  *width  = image.width;
  *height = image.height;

  // Allocate memory:
  result = allocateData((*width) * (*height) * read_components * (read_bit_depth / 8));
  if(!result)
  {
    std::cerr << "Allocating memory to load a PNG image failed.\n";
  }
  else
  {
    if(!png_image_finish_read(&image, nullptr /* background */, result, 0 /* row_stride */, nullptr /* colormap */))
    {
      std::cerr << "PNG image load failed: " << image.message << "\n";
    }
  }

  png_image_free(&image);

  if(required_components > 0 || read_bit_depth != required_bit_depth)
  {
    if(!convertFormat(&result, *width, *height, read_components, read_bit_depth, required_components, required_bit_depth))
    {
      freeData(&result);
      return nullptr;
    }
    *components = required_components;
  }

  return result;
}

ImageIOData loadWithTinyEXR(const void*  data,
                            const size_t byteLength,
                            size_t*      width,
                            size_t*      height,
                            size_t*      components,
                            const size_t required_components,
                            const size_t required_bit_depth)
{
  int         x, y;
  const char* tinyEXR_error = nullptr;
  ImageIOData result        = nullptr;

  const int tinyEXR_return_code = LoadEXRFromMemory(reinterpret_cast<float**>(&result), &x, &y,
                                                    reinterpret_cast<const unsigned char*>(data), byteLength, &tinyEXR_error);
  if(tinyEXR_return_code != TINYEXR_SUCCESS)
  {
    if(tinyEXR_error)
    {
      std::cerr << "Loading with TinyEXR failed; TinyEXR gave error " << tinyEXR_error << std::endl;
      FreeEXRErrorMessage(tinyEXR_error);
    }
    return nullptr;
  }

  *width      = x;
  *height     = y;
  *components = 4;

  if(required_components > 0 || required_bit_depth != 32)
  {
    if(!convertFormat(&result, *width, *height, *components, 32, required_components, required_bit_depth))
    {
      freeData(&result);
      return nullptr;
    }
    *components = required_components;
  }

  return result;
}

ImageIOData loadGeneral(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components, const size_t required_bit_depth)
{
  ImageIOData          result = nullptr;
  nvh::FileReadMapping readMapping;
  readMapping.open(filename);
  if(readMapping.valid())
  {
    result = loadGeneralFromMemory(readMapping.data(), readMapping.size(), width, height, components,
                                   required_components, required_bit_depth);
    readMapping.close();
  }
  return result;
}

ImageIOData loadGeneralFromMemory(const void*  data,
                                  const size_t byteLength,
                                  size_t*      width,
                                  size_t*      height,
                                  size_t*      components,
                                  const size_t required_components,
                                  const size_t required_bit_depth)
{
  const Loader loader = determineLoader(data, byteLength);

  switch(loader)
  {
    case Loader::eSTBImage:
      return loadWithStbImage(data, byteLength, width, height, components, required_components, required_bit_depth);
    case Loader::eLibPNG:
      return loadWithLibPNG(data, byteLength, width, height, components, required_components, required_bit_depth);
    case Loader::eTinyEXR:
      return loadWithTinyEXR(data, byteLength, width, height, components, required_components, required_bit_depth);
    default:
      break;
  }

  assert(!"Should be unreachable!");
  return nullptr;
}

ImageIOData load8(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneral(filename, width, height, components, required_components, 8);
}

ImageIOData load8FromMemory(const void* data, const size_t byteLength, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneralFromMemory(data, byteLength, width, height, components, required_components, 8);
}

ImageIOData load16(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneral(filename, width, height, components, required_components, 16);
}

ImageIOData load16FromMemory(const void* data, const size_t byteLength, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneralFromMemory(data, byteLength, width, height, components, required_components, 16);
}

ImageIOData loadF(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneral(filename, width, height, components, required_components, 32);
}

ImageIOData loadFFromMemory(const void* data, const size_t byteLength, size_t* width, size_t* height, size_t* components, const size_t required_components)
{
  return loadGeneralFromMemory(data, byteLength, width, height, components, required_components, 32);
}

bool writePNG(const char* filename, size_t width, size_t height, const void* data, VkFormat vkFormat)
{
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if(width > std::numeric_limits<uint32_t>::max())
  {
    std::cerr << "Could not write PNG file: a width of " << width << " pixels was too large to fit in a 32-bit unsigned int!\n";
    return false;
  }
  image.width = static_cast<uint32_t>(width);
  if(height > std::numeric_limits<uint32_t>::max())
  {
    std::cerr << "Could not write PNG file: a height of " << height << " pixels was too large to fit in a 32-bit unsigned int!\n";
    return false;
  }
  image.height = static_cast<uint32_t>(height);

  if(vkFormat == VK_FORMAT_R8G8B8A8_UNORM)
  {
    image.format = PNG_FORMAT_RGBA;
  }
  else if(vkFormat == VK_FORMAT_R16G16B16A16_UNORM)
  {
    image.format = PNG_FORMAT_LINEAR_RGB_ALPHA;
  }
  else if(vkFormat == VK_FORMAT_R16_UNORM)
  {
    image.format = PNG_FORMAT_LINEAR_Y;
  }
  else
  {
    std::cerr << "Could not write PNG file: writePNG() does not include a case for Vulkan format " << vkFormat
              << ". If this corresponds to a PNG_FORMAT, consider adding it.\n";
    return false;
  }

  // Trade off compression size for compression speed:
  image.flags |= PNG_IMAGE_FLAG_FAST;

  // And write it! png_image_write_to_file returns true on success.
  return png_image_write_to_file(&image, filename, 0 /* convert_to_8bit */, data, 0 /* row_stride */, nullptr /* colormap */);
}

}  // namespace imageio