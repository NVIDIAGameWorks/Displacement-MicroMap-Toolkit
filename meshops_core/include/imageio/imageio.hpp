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

#pragma once

// The ImageIO functions provide an stb_image-like interface over stb_image,
// tinyexr, and libpng, to provide both the flexibility of stb_image
// and tinyexr, and libpng's speed and 16-bit-per-component support.

#include <stdint.h>
#include <vulkan/vulkan_core.h>

namespace imageio {

// ImageIOData comes from one of the various loaders, and must be freed
// using freeData(). Yhese functions return pointers rather than vectors
// because it seems like that should avoid copying the uncompressed data
// more often. But maybe there's a better way.
using ImageIOData = void*;
ImageIOData allocateData(size_t byteLength);
void        freeData(ImageIOData* pData);

// Reads the header of an image and outputs the file's width, height, and
// number of components, without decompressing the image. Returns true
// on success.
[[nodiscard]] bool info(const char* filename, size_t* width, size_t* height, size_t* components);
[[nodiscard]] bool infoFromMemory(const void* data, const size_t byteLength, size_t* width, size_t* height, size_t* components);

[[nodiscard]] bool is16Bit(const char* filename);

// For all load() functions, required_components must currently be 0 (same
// number of components as the input), 1, 2, 3, or 4.
// On failure, they return nullptr.

// General function: loads as 8-bit UNORM, 16-bit UNORM, or 32-bit SFLOAT depending
// on the value of required_bit_depth.
[[nodiscard]] ImageIOData loadGeneral(const char*  filename,
                                      size_t*      width,
                                      size_t*      height,
                                      size_t*      components,
                                      const size_t required_components,
                                      const size_t required_bit_depth);
[[nodiscard]] ImageIOData loadGeneralFromMemory(const void*  data,
                                                const size_t byteLength,
                                                size_t*      width,
                                                size_t*      height,
                                                size_t*      components,
                                                const size_t required_components,
                                                const size_t required_bit_depth);

// Loads an image as 8-bit-per-component UNORM data.
[[nodiscard]] ImageIOData load8(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components);
[[nodiscard]] ImageIOData load8FromMemory(const void*  data,
                                          const size_t byteLength,
                                          size_t*      width,
                                          size_t*      height,
                                          size_t*      components,
                                          const size_t required_components);
// Loads an image as 16-bit-per-component UNORM data.
[[nodiscard]] ImageIOData load16(const char* filename, size_t* width, size_t* height, size_t* components, const size_t required_components);
[[nodiscard]] ImageIOData load16FromMemory(const void*  data,
                                           const size_t byteLength,
                                           size_t*      width,
                                           size_t*      height,
                                           size_t*      components,
                                           const size_t required_components);
// Loads an image as 32-bit-per-component SFLOAT data.
[[nodiscard]] ImageIOData loadF(const char* filename, size_t* width, size_t* height, size_t* components, size_t required_components);
[[nodiscard]] ImageIOData loadFFromMemory(const void*  data,
                                          const size_t byteLength,
                                          size_t*      width,
                                          size_t*      height,
                                          size_t*      components,
                                          const size_t required_components);

// Writes a PNG file. Unlike stb_image, this specifies the input format
// using a VkFormat, which must be either VK_FORMAT_R8G8B8A8_UNORM,
// VK_FORMAT_R16G16B16A16_UNORM, or VK_FORMAT_R16_UNORM.
// Returns true on success.
[[nodiscard]] bool writePNG(const char* filename, size_t width, size_t height, const void* data, VkFormat vkFormat);

// Converts raw image data to have a given number of components and bit depth.
// The number of components must be 1, 2, 3, or 4, and the bit depth must be
// 8, 16, or 32.
//
// We add new RGB components as needed by setting them to 0, and a new alpha
// component (if needed) by setting it to 1.
//
// Returns true on success and false on error (e.g. if reallocation fails).
// This will reallocate the data pointed by `image` if the input and output
// formats are different. If `image` was not nullptr, then it is always a valid
// pointer to ImageIOData when this function returns.
//
// Example:
// ```
// // Convert `data to RGBA8`
// if(imageio::convertFormat(image,
//     width, height,
//     input_components, input_bit_depth,
//     4, 8))
// {
//   // Do something with `image`...
// }
// else
// {
//   // Print a failure message or handle failure otherwise
// }
// imageio::freeData(image);
// ```
[[nodiscard]] bool convertFormat(ImageIOData* image,
                                 const size_t width,
                                 const size_t height,
                                 const size_t input_components,
                                 const size_t input_bit_depth,
                                 const size_t output_components,
                                 const size_t output_bit_depth);

}  // namespace imageio