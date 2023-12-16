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

#include "imgui.h"
#include "ui_micromesh_tools.hpp"


void uiBaker(tool_bake::ToolBakeArgs& bake_args, ViewerSettings::GlobalToolSettings& toolSettings, GLFWwindow* glfwWindow)
{
  using PE = ImGuiH::PropertyEditor;

  // *** Note: we are currently only supporting compressed data ***
  bake_args.compressed = true;

  static const char* ImageFilter = "Images|*.jpg;*.png;*.tga;*.bmp;*.psd;*.gif;*.hdr;*.pic;*;pnm;*.exr";

  if(ImGui::SmallButton("Reset##Baker"))
    bake_args = {};
  ImGuiH::tooltip("Reset values to default");

  PE::begin();

  PE::entry(
      "Subdivision Level",
      [&] {
        ImGui::Text("%i", bake_args.level);
        return false;
      },
      "Level of subdivision, set by the global Bake Subdiv Level.");
  PE::entry(
      "Max Ray Trace Length (percent)",
      [&]() { return ImGui::InputFloat("Max Ray Trace Length (percent)", &bake_args.maxDisplacement); },
      "Maximum distance to trace rays from the low scene to look up data from the high scene, as a percentage of the "
      "radius of the axis-aligned bounding box of the high scene.\n"
      "Reduce the value to prevent the displaced mesh from trying to match unrelated parts of the hi-res mesh.");
  std::array<const char*, 5> items = {"Custom (else uniform)", "Uniform", "Adaptive3D", "AdaptiveUV", "Custom"};
  std::array<tool_bake::ToolBakeArgs::BakingMethod, 5> itemEnums = {
      tool_bake::ToolBakeArgs::eCustomOrUniform, tool_bake::ToolBakeArgs::eUniform,
      tool_bake::ToolBakeArgs::eAdaptive3D, tool_bake::ToolBakeArgs::eAdaptiveUV, tool_bake::ToolBakeArgs::eCustom};
  static int subdivMode = 0;
  PE::entry(
      "Subdivision Mode",
      [&]() { return ImGui::Combo("Subdivision Mode", &subdivMode, items.data(), static_cast<int>(items.size())); },
      "Uniform: all triangles use the same level\nAdaptive3D: will reduce the level of some triangles based "
      "on their coverage\nAdaptiveUV: same as Adaptive3D but using UV\nCustom: Uses subdivision levels from the low "
      "file's subdivisionLevels property");
  bake_args.method = itemEnums[subdivMode];
  {
    const tool_bake::ToolBakeArgs::BakingMethod castMethod =
        static_cast<tool_bake::ToolBakeArgs::BakingMethod>(bake_args.method);
    if(castMethod == tool_bake::ToolBakeArgs::eAdaptive3D || castMethod == tool_bake::ToolBakeArgs::eAdaptiveUV)
    {
      PE::entry(
          "Adaptive Factor", [&]() { return ImGui::InputFloat("Adaptive Factor", &bake_args.adaptiveFactor); },
          "Multiplication factor for adaptive subdivision levels. For instance, a factor of 2 doubles the "
          "microtriangle resolution (i.e. increases all subdivision levels by 1).");
    }
    if(castMethod == tool_bake::ToolBakeArgs::eUniform)
    {
      PE::entry(
          "Override Length", [&]() { return ImGui::Checkbox("##OLength", &bake_args.overrideDirectionLength); },
          "Don't use direction vector length, but Max Displacement");
    }
  }

  PE::entry(
      "Maximum Distance Factor", [&]() { return ImGui::InputFloat("##MaxDistanceFactor", &bake_args.maxDistanceFactor); },
      "Factor applied to the maximum tracing distance, useful when the displacement bounds define a tight "
      "shell around the original geometry");


  PE::entry(
      "Fit Direction Bounds", [&]() { return ImGui::Checkbox("##FitDirectionBounds", &bake_args.fitDirectionBounds); },
      "Compute direction vector bounds for tighter BVH");
  PE::entry(
      "Use PN Triangles", [&]() { return ImGui::Checkbox("##UsePNTriangles", &bake_args.heightmapPNtriangles); },
      "Use smooth Point-Normal Triangle surfaces (Vlachos 2001) when tessellating a high-res mesh with heightmaps.");
  PE::entry("Heightmap Subdiv. Bias",
            [&] { return ImGui::SliderInt("##HeightmapSubdivBias", (int*)&bake_args.highTessBias, -5, 5); });
  PE::entry(
      "Generate Heightmap Directions",
      [&] { return ImGui::Checkbox("##DirectionsGen", &bake_args.heightmapDirectionsGen); },
      "Computes smooth heightmap displacement direction vectors. Mesh normals are used otherwise.");
  ImGui::BeginDisabled(!bake_args.heightmapDirectionsGen);
  PE::entry("Direction Generation Method", [&] {
    static const std::array<const char*, 3> reduce_names = {"Linear", "Normalized Linear", "Tangent"};
    return ImGui::Combo("##Op", (int*)&bake_args.heightmapDirectionsOp, reduce_names.data(),
                        static_cast<int>(reduce_names.size()));
  });
  ImGuiH::tooltip(
      "Linear = angle-weighted average of adjacent face normals; Normalized Linear = average + normalize to unit "
      "length; Tangent = preserve sharp edges");
  ImGui::EndDisabled();
  PE::entry(
      "Discard Input Bounds", [&]() { return ImGui::Checkbox("##DiscardBounds", &bake_args.discardInputBounds); },
      "Discards any input direction vector bounds. They will be re-created if Fit Direction Bounds is enabled.");
  PE::entry(
      "Apply Direction Bounds", [&]() { return ImGui::Checkbox("##ApplyBounds", &bake_args.applyDirectionBounds); },
      "Applies any direction bounds to the positions and direction vectors after baking. This saves some space but "
      "loses the ability to render the original geometry without micromaps applied");

  // *** Note: we are currently only supporting compressed data ***
  //PE::entry(
  //    "Compressed", [&]() { return ImGui::Checkbox("##Compressed", &bake_args.compressed); },
  //    "Save the bary file compressed");

  if(true /*bake_args.compressed*/)
  {
    PE::entry("Min PSNR", [&]() { return ImGui::InputFloat("Min PSNR", &bake_args.minPSNR); });
    PE::entry(
        "Add Rasterization Mips",
        [&]() { return ImGui::Checkbox("##Add Rasterization Mips", &bake_args.compressedRasterData); },
        "Add uncompressed mips for rasterization");
  }

  PE::entry(
      "Quaternion Textures Stem",
      [&]() { return ImGuiH::InputText("##QTS", &bake_args.quaternionTexturesStem, ImGuiInputTextFlags_None); },
      "Add text here to generate a quaternion texture named {text}.{mesh index}.png for each mesh.");
  PE::entry(
      "Offset Textures Stem",
      [&]() { return ImGuiH::InputText("##OTS", &bake_args.offsetTexturesStem, ImGuiInputTextFlags_None); },
      "Add text here to generate an offset texture named {text}.{mesh index}.png for each mesh.");
  PE::entry(
      "Height Textures Stem",
      [&]() { return ImGuiH::InputText("##OTS", &bake_args.heightTexturesStem, ImGuiInputTextFlags_None); },
      "Add text here to generate a heightmap texture named {text}.{mesh index}.png for each mesh.");
  PE::entry(
      "Normal Textures Stem",
      [&]() { return ImGuiH::InputText("##OTS", &bake_args.normalTexturesStem, ImGuiInputTextFlags_None); },
      "Add text here to generate a normalmap texture named {text}.{mesh index}.png for each mesh.");

  // Resampling
  {
    // Corresponds to TexturesToResample
    std::array<const char*, 3> resampling_modes = {"None", "Only normal maps", "All textures"};
    PE::entry(
        "Resample Textures",
        [&]() {
          return ImGui::Combo("Resample Textures", reinterpret_cast<int*>(&bake_args.texturesToResample),
                              resampling_modes.data(), static_cast<int>(resampling_modes.size()));
        },
        "Selects textures to resample/re-bake. For instance, one can use this to bake normal maps, or to fix "
        "parallax "
        "issues from remeshing. It traces from the low mesh to the high mesh (with any displacement applied), "
        "looks up "
        "the high mesh's attributes at the intersection, and writes into the output's textures.\n"  //
        "None: The output will use the low file's textures\n"                                       //
        "Only normal maps: Normals will be traced and rotated to tangent spaces on the low mesh\n"  //
        "All textures: All textures including normal maps and other attributes will be resampled.");

    if(bake_args.texturesToResample != tool_bake::TexturesToResample::eNone)
    {
      PE::entry(
          "Resample Resolution",
          [&]() { return ImGui::InputInt("Resample Resolution", &bake_args.resampleResolution.x); },
          "The resolution in pixels of each side of each of the output resampled textures. 0 means the resampler will "
          "try to match the resolutions of the inputs.");

      // GUI for `resampleExtraTextures`; this is a variable-length vector!
      int numExtraTextures = static_cast<int>(bake_args.resampleExtraTextures.size());
      PE::entry(
          "Resample Extra Textures", [&]() { return ImGui::InputInt("Resample Extra Textures", &numExtraTextures); },
          "You can add additional textures to be resampled from a hi-res mesh to a lo-res mesh here, even if they "
          "aren't part of a glTF material.");
      // Limit numExtraTextures to a reasonable number
      numExtraTextures = std::max(0, std::min(numExtraTextures, 9999));
      bake_args.resampleExtraTextures.resize(numExtraTextures);

      // TODO: Can we put this in a group box of some sort?
      for(int i = 0; i < numExtraTextures; i++)
      {
        const std::string commonPrefix    = "  " + std::to_string(i) + ": ";
        const std::string meshLabel       = commonPrefix + "Mesh Index";
        const std::string meshLabelHidden = "##" + meshLabel;
        const std::string inLabel         = commonPrefix + "Input File";
        const std::string inLabelHidden   = "##" + inLabel;
        const std::string inButtonHidden  = "...##FI" + std::to_string(i);
        const std::string outLabel        = commonPrefix + "Output File (Optional)";
        const std::string outLabelHidden  = "##" + outLabel;
        const std::string outButtonHidden = "...##FO" + std::to_string(i);
        const std::string nrmLabel        = commonPrefix + "Is Normal Map";
        const std::string nrmLabelHidden  = "##" + nrmLabel;

        auto& tex = bake_args.resampleExtraTextures[i];

        PE::entry(
            meshLabel, [&]() { return ImGui::InputInt(meshLabelHidden.c_str(), &tex.meshIdx); },
            "The index of the hi-res and lo-res mesh to use to resample this texture");

        PE::entry(
            inLabel, [&]() { return ImGuiH::InputText(inLabelHidden.c_str(), &tex.inURI, ImGuiInputTextFlags_None); },
            "The input texture, for the hi-res mesh.");
        if(ImGui::SmallButton(inButtonHidden.c_str()))
        {
          tex.inURI = NVPSystem::windowOpenFileDialog(glfwWindow, "Choose Input Image", ImageFilter);
        }

        PE::entry(
            outLabel, [&]() { return ImGuiH::InputText(outLabelHidden.c_str(), &tex.outURI, ImGuiInputTextFlags_None); },
            "The file to write the resampled texture to. If not specified, the resampler generates a file name.");
        if(ImGui::SmallButton(outButtonHidden.c_str()))
        {
          tex.outURI = NVPSystem::windowOpenFileDialog(glfwWindow, "Choose Output Image to Overwrite", ImageFilter);
        }

        PE::entry(
            nrmLabel, [&]() { return ImGui::Checkbox(nrmLabelHidden.c_str(), &tex.isNormalMap); },
            "Is this a normal map? If so, we'll transform normals, instead of copying colors. A useful trick is "
            "that "
            "if you put a blank normal map (all pixels have 8-bit RGB color (127/255, 127/255, 255/255)) into the "
            "input field and check this checkbox, the resampler will bake a normal map for the micromesh!");
      }
    }
  }

  PE::end();
}
