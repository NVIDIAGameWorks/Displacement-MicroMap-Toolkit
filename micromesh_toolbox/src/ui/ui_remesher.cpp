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

#include "ui_micromesh_tools.hpp"


void uiRemesher(ViewerSettings::GlobalToolSettings& toolSettings, tool_remesh::ToolRemeshArgs& remesh_args)
{
  using PE = ImGuiH::PropertyEditor;

  if(ImGui::SmallButton("Reset##Remesher"))
    remesh_args = {};
  ImGuiH::tooltip("Reset values to default");

  PE::begin();

  // Decimation target computation: 0 = user-defined decimation rate, 1 = deduce from toolSettings.subdivLevel
  PE::entry(
      "Decimation target",
      [&] {
        ImGui::RadioButton("Rate", &toolSettings.decimateRateFromSubdivLevel, 0);
        ImGui::RadioButton("Bake Subdiv level", &toolSettings.decimateRateFromSubdivLevel, 1);
        return true;
      },
      "Decimation rate is either explicitly defined using 'Rate', or deduced from the main baking subdiv level");

  if(toolSettings.decimateRateFromSubdivLevel == 0)
  {
    PE::entry(
        "Decimation Ratio",
        [&] {
          return ImGui::SliderFloat("##erro", &remesh_args.decimationRatio, 1.e-4f, 1.f - 1.e-4f, "% .4f", ImGuiSliderFlags_Logarithmic);
        },
        "Ratio between the remeshed and input triangle counts. With a ratio of 0.1 the remesher will produce a mesh "
        "containing at most 10% of the original triangle count");
  }
  else
  {
    PE::entry(
        "Decimation Ratio",
        [&] {
          ImGui::Text("% .4f", remesh_args.decimationRatio);
          return false;
        },
        "Ratio between the remeshed and input triangle counts, deduced from the bake subdivision level");
  }
  PE::entry(
      "Curvature Power", [&] { return ImGui::DragFloat("##erro", &remesh_args.curvaturePower); },
      "Power applied to the per-vertex importance, used to tweak importance contrast");
  PE::entry(
      "Vertex Importance Weight", [&] { return ImGui::DragFloat("##erro", &remesh_args.importanceWeight); },
      "Weight given to the per-vertex importance in the error calculation. The higher, the more triangles will be "
      "preserved on curved areas");

  PE::entry(
      "Curvature Max Dist", [&] { return ImGui::SliderFloat("##erro", &remesh_args.curvatureMaxDist, 0.001f, 1.f); },
      "Maximum raytracing distance (fraction of the scene size) used when estimating the per-vertex importance "
      "using "
      "the local mesh curvature. ");
  PE::entry(
      "Direction Bounds Factor",
      [&] { return ImGui::DragFloat("##erro", &remesh_args.directionBoundsFactor, 0.01f, 1.f, 2.f); },
      "The remesher generates very tight displacement bounds, which may result in rounding issues in the micromesh "
      "baker. This factor increases those bounds.");
  PE::entry(
      "Fit To Original Surface", [&] { return ImGui::Checkbox("##erro", &remesh_args.fitToOriginalSurface); },
      "If checked, the remesher tries to preserve the mesh volume during decimation");

  PE::entry(
      "Max Decimation Level",
      [&] {
        ImGui::Text("%i", remesh_args.maxSubdivLevel);
        return false;
      },
      "If not -1, controls the maximum subdivision level generated during remeshing: a triangle may not be further "
      "collapsed if its implicit subdivision level reaches 4^level. That is based on the greater of either the "
      "heightmap resolution of its area, or the number of source triangles that are represented by the output "
      "triangle. Set by the global Bake Subdiv Level.");
  PE::entry(
      "Max Vertex Valence", [&] { return ImGui::InputInt("##erro", (int*)&remesh_args.maxVertexValence); },
      "Maximum vertex valence resulting from decimation operations.");
  PE::entry(
      "Vertex Importance Threshold", [&] { return ImGui::SliderFloat("##erro", &remesh_args.importanceThreshold, 0, 1); },
      "Maximum importance vertex of the vertices involved in edge collapse operations.");
  PE::entry(
      "Ignore Tex Coords", [&] { return ImGui::Checkbox("##erro", &remesh_args.ignoreTexCoords); },
      "Ignore the texture coordinate discontinuities.");
  PE::entry(
      "Ignore Normals", [&] { return ImGui::Checkbox("##erro", &remesh_args.ignoreNormals); },
      "Ignore the shading normal discontinuities.");
  PE::entry(
      "Ignore Tangents", [&] { return ImGui::Checkbox("##erro", &remesh_args.ignoreTangents); },
      "Ignore the tangent space discontinuities.");
  PE::entry(
      "Ignore Displacement Directions",
      [&] { return ImGui::Checkbox("##erro", &remesh_args.ignoreDisplacementDirections); },
      "Ignore the displacement direction discontinuities.");

  PE::end();
}
