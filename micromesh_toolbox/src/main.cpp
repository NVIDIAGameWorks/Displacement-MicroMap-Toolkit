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

// VMA LEAK FINDER
// - Uncomment to show allocation info in the log
// - Call findLeak() with the value showing in the leak report
//

//#define VMA_DEBUG_LOG(format, ...)                                                                                     \
//  do                                                                                                                   \
//  {                                                                                                                    \
//    nvprintfLevel(LOGLEVEL_INFO, format, __VA_ARGS__);                                                                 \
//    nvprintfLevel(LOGLEVEL_INFO, "\n");                                                                                \
//  } while(false)

#define VMA_IMPLEMENTATION

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <thread>

#include "nvh/nvprint.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_logger.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/hdr_env_dome.hpp"
#include "nvvkhl/sky.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "elements/element_nvml.hpp"
#include "toolbox_viewer.hpp"
#include "debug_util.h"

std::shared_ptr<nvvkhl::ElementCamera> g_elem_camera;  // Is accessed elsewhere in the App

#include "elements/element_testing.hpp"
#include "resources/micromesh_ico_256.h"
#include "resources/micromesh_ico_32.h"
#include "resources/micromesh_ico_64.h"
#include "stb_image.h"


#ifdef USE_NSIGHT_AFTERMATH
#include "aftermath/nsight_aftermath_gpu_crash_tracker.h"
// Global marker map
::GpuCrashTracker::MarkerMap       g_markerMap;
std::unique_ptr<::GpuCrashTracker> g_aftermathTracker;

// Display errors
#ifdef _WIN32
#define ERR_EXIT(err_msg, err_class)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    MessageBox(nullptr, err_msg, err_class, MB_OK);                                                                    \
    exit(1);                                                                                                           \
  } while(0)
#else
#define ERR_EXIT(err_msg, err_class)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    printf("%s\n", err_msg);                                                                                           \
    fflush(stdout);                                                                                                    \
    exit(1);                                                                                                           \
  } while(0)
#endif

#endif  // USE_NSIGHT_AFTERMATH
#include "vulkan_nv/vk_nv_micromesh_prototypes.h"


#ifdef USE_NSIGHT_AFTERMATH

// cmake defines NVVK_CHECK to point to toolboxCheckResult
bool toolboxCheckResult(VkResult result, const char* /*file*/, int32_t /*line*/, const char* /* message */)
{
  if(result == VK_SUCCESS)
  {
    return false;
  }
  if(result == VK_ERROR_DEVICE_LOST)
  {
    // Device lost notification is asynchronous to the NVIDIA display
    // driver's GPU crash handling. Give the Nsight Aftermath GPU crash dump
    // thread some time to do its work before terminating the process.
    auto tdr_termination_timeout = std::chrono::seconds(5);
    auto t_start                 = std::chrono::steady_clock::now();
    auto t_elapsed               = std::chrono::milliseconds::zero();

    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

    while(status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed
          && status != GFSDK_Aftermath_CrashDump_Status_Finished && t_elapsed < tdr_termination_timeout)
    {
      // Sleep 50ms and poll the status again until timeout or Aftermath finished processing the crash dump.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

      auto t_end = std::chrono::steady_clock::now();
      t_elapsed  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    }

    if(status != GFSDK_Aftermath_CrashDump_Status_Finished)
    {
      std::stringstream err_msg;
      err_msg << "Unexpected crash dump status: " << status;
      ERR_EXIT(err_msg.str().c_str(), "Aftermath Error");
    }

    std::stringstream err_msg;
    err_msg << "Aftermath file dumped under:\n\n";
    err_msg << std::filesystem::current_path().string();

    // Terminate on failure
#ifdef _WIN32
    err_msg << "\n\n\nSave path to clipboard?";
    int ret = MessageBox(nullptr, err_msg.str().c_str(), "Nsight Aftermath", MB_YESNO | MB_ICONEXCLAMATION);
    if(ret == IDYES)
    {
      ImGui::SetClipboardText(std::filesystem::current_path().string().c_str());
    }
#else
    printf("%s\n", err_msg.str().c_str());
#endif

    exit(1);
  }
  return false;
}
#endif


//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int argc, char** argv)
{
#ifdef _WIN32
  fixAbortOnWindows();
#endif

  static nvvkhl::SampleAppLog g_logger;
  nvprintSetCallback([](int level, const char* fmt) { g_logger.addLog(level, "%s", fmt); });

#ifdef USE_NSIGHT_AFTERMATH
  nvvk::setCheckResultHook(toolboxCheckResult);
#endif

  // This is not absolutely required, but having this early, loads the Vulkan DLL, which delays
  // the window to show up by ~1.5 seconds, but on the other hands, reduce the time the window
  // displays a white background.
  int glfw_valid = GLFW_TRUE;
  glfw_valid &= glfwInit();
  glfw_valid &= glfwVulkanSupported();
  if(!glfw_valid)
  {
    std::string err_message("Vulkan is not supported on this computer.");
#if _WIN32
    MessageBox(nullptr, err_message.c_str(), "Vulkan error", MB_OK);
#endif
    LOGE("%s", err_message.c_str());
    return 1;
  }

  nvvkhl::ApplicationCreateInfo spec;

  // Parsing arguments
  bool        print_help{false};
  bool        verbose{false};
  bool        validation{false};
  bool        testing{false};
  std::string in_filename;
  std::string in_hdr;
  std::string in_config;

  nvh::CommandLineParser args("ToolBox: Tool to remesh and bake micromeshes");
  args.addArgument({"-f", "--filename"}, &in_filename, "Input filename");
  args.addArgument({"-h", "--help"}, &print_help, "Print Help");
  args.addArgument({"--hdr"}, &in_hdr, "Input HDR");
  args.addArgument({"--test"}, &testing, "Developer option for automated testing");
  args.addArgument({"--config"}, &in_config, "Override the default path to the .ini config file");
  args.addArgument({"-v", "--verbose"}, &verbose, "Set verbosity [true|false] default: false");
  args.addArgument({"--validation"}, &validation, "Set Vulkan validation layers [true|false]");
  args.addArgument({"--width"}, &spec.width, "Width of application");
  args.addArgument({"--height"}, &spec.height, "Height of application");
  args.addArgument({"--vsync"}, &spec.vSync, "Turning vSync on/off. [true|false] default: true");

  const bool result = args.parse(argc, argv);
  if((!result || print_help) && !testing)
  {
    args.printHelp();
    return 1;
  }

  if(verbose)
    g_logger.setLogLevel(LOGBITS_ALL);


  spec.name             = PROJECT_NAME " Example";
  spec.vkSetup          = nvvk::ContextCreateInfo(validation);
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  VkPhysicalDeviceFragmentShaderBarycentricFeaturesNV baryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_NV};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT floatFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtom64Features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
  VkPhysicalDeviceMeshShaderFeaturesNV meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_NV};
  //VkPhysicalDevice16BitStorageFeatures storageFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};

  spec.vkSetup.addDeviceExtension(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, false, &imageAtom64Features);
  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &ray_query_features);
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  spec.vkSetup.addDeviceExtension(VK_NV_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, false, &baryFeatures);
  spec.vkSetup.addDeviceExtension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, false, &floatFeatures);
  spec.vkSetup.addDeviceExtension(VK_NV_MESH_SHADER_EXTENSION_NAME, false, &meshFeatures);
  spec.vkSetup.addDeviceExtension(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME, false);
  spec.vkSetup.addDeviceExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME, false);
  //spec.vkSetup.addDeviceExtension(VK_KHR_16BIT_STORAGE_EXTENSION_NAME, false, &storageFeatures);

  // Request micromap extensions
  static VkPhysicalDeviceOpacityMicromapFeaturesEXT mmOpacityFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
  static VkPhysicalDeviceDisplacementMicromapFeaturesNV mmDisplacementFeatures = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV};
  spec.vkSetup.addDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, true, &mmOpacityFeatures);
  spec.vkSetup.addDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME, true, &mmDisplacementFeatures);

#ifdef USE_NSIGHT_AFTERMATH
  // Enable NV_device_diagnostic_checkpoints extension to be able to use Aftermath event markers.
  spec.vkSetup.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
  // Enable NV_device_diagnostics_config extension to configure Aftermath features.
  VkDeviceDiagnosticsConfigCreateInfoNV aftermath_info{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV};
  aftermath_info.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV
                         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV
                         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV;
  spec.vkSetup.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, false, &aftermath_info);

  // #Aftermath - Initialization
  g_aftermathTracker = std::make_unique<::GpuCrashTracker>(g_markerMap);
  g_aftermathTracker->initialize();

  LOGW(
      "\n-------------------------------------------------------------------"
      "\nWARNING: Aftermath extensions enabled. This may affect performance."
      "\n-------------------------------------------------------------------\n\n");
#endif

#ifdef _DEBUG
  if(!validation)
  {
    LOGW(
        "Warning: debug build is run without validation active (use `--validation true` if desired).\n"
        "However, until proper support it will cause crashes with VK_NV_displacement_micromap usage\n")
  }

  // #debug_printf
  VkValidationFeaturesEXT                    features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  std::vector<VkValidationFeatureEnableEXT>  enables{VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  std::vector<VkValidationFeatureDisableEXT> disables{};
  features.enabledValidationFeatureCount  = static_cast<uint32_t>(enables.size());
  features.pEnabledValidationFeatures     = enables.data();
  features.disabledValidationFeatureCount = static_cast<uint32_t>(disables.size());
  features.pDisabledValidationFeatures    = disables.data();
  spec.vkSetup.instanceCreateInfoExt      = &features;
#endif

  // Request for extra Queue for loading in parallel
  spec.vkSetup.addRequestedQueue(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1, 1.0F);


  // Setting up the layout of the application
  spec.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID microPipeID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Micromesh Pipeline", microPipeID);
    ImGuiID microOpID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Micromesh Operations", microOpID);
    ImGuiID logID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", logID);
    ImGuiID nvmlID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.3F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("NVML Monitor", nvmlID);
    ImGuiID statID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.2F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("Statistics", statID);
  };

  // Ignore specific warnings/debugs due new extension not supported by validation layer
  spec.ignoreDbgMessages.push_back(0x901f59ec);  // vkCreateDevice: pCreateInfo->pNext chain includes a structure with unknown VkStructureType
  spec.ignoreDbgMessages.push_back(0xdd73dbcf);  // vkGetPhysicalDeviceProperties2: pProperties->pNext chain includes a structure with unknown VkStructureType
  spec.ignoreDbgMessages.push_back(0x9f0bb94d);  // vkCmdBuildMicromapsEXT/vkGetMicromapBuildSizesEXT: value of (*)->type unknown
  spec.ignoreDbgMessages.push_back(0xb80964e5);  // vkCreateMicromapEXT: value of pCreateInfo->type unknown
  spec.ignoreDbgMessages.push_back(0xa7bb8db6);  // SPIR-V Capability (StorageInputOutput16)
  spec.ignoreDbgMessages.push_back(0x715035dd);  // storageInputOutput16 is not enabled
  spec.ignoreDbgMessages.push_back(0x06e224e9);  // yet another StorageInputOutput16 message
  spec.ignoreDbgMessages.push_back(0x22d5bbdc);  // vkCreateRayTracingPipelinesKHR: value of pCreateInfos[0].flags contains flag bits
  spec.ignoreDbgMessages.push_back(0xf69d66f5);  // vkGetAccelerationStructureBuildSizesKHR: pInfos[0].pGeometries[0].geometry.triangles.pNext chain includes a structure with unknown VkStructureType

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Adding micromap function prototypes
  bool hasDisplacementMicromeshExt = app->getContext()->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME);
  if(hasDisplacementMicromeshExt)
    load_VK_EXT_opacity_micromap_prototypes(app->getContext()->m_device, vkGetDeviceProcAddr);


#ifdef _DEBUG
  //------
  // #debug_printf
  // Vulkan message callback - for receiving the printf in the shader
  // Note: there is already a callback in nvvk::Context, but by defaut it is not printing INFO severity
  //       this callback will catch the message and will make it clean for display.
  auto dbg_messenger_callback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                   const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData) -> VkBool32 {
    // Get rid of all the extra message we don't need
    std::string clean_msg = callbackData->pMessage;
    clean_msg             = clean_msg.substr(clean_msg.find_last_of('|') + 1);
    nvprintfLevel(LOGLEVEL_DEBUG, "%s", clean_msg.c_str());  // <- This will end up in the Logger
    return VK_FALSE;                                         // to continue
  };

  // #debug_printf : Creating the callback
  VkDebugUtilsMessengerEXT           dbg_messenger{};
  VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  dbg_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  dbg_messenger_create_info.pfnUserCallback = dbg_messenger_callback;
  NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(app->getContext()->m_instance, &dbg_messenger_create_info, nullptr, &dbg_messenger));
#endif

  // Set the Application ICON
  GLFWimage images[3];
  images[0].pixels =
      stbi_load_from_memory(g_micromesh_ico_256, sizeof(g_micromesh_ico_256), &images[0].width, &images[0].height, 0, 4);
  images[1].pixels =
      stbi_load_from_memory(g_micromesh_ico_64, sizeof(g_micromesh_ico_64), &images[1].width, &images[1].height, 0, 4);
  images[2].pixels =
      stbi_load_from_memory(g_micromesh_ico_32, sizeof(g_micromesh_ico_32), &images[2].width, &images[2].height, 0, 4);
  glfwSetWindowIcon(app->getWindowHandle(), 3, images);
  stbi_image_free(images[0].pixels);

  // Imgui Style override
  ImGui::GetStyle().DisabledAlpha = 0.2F;

  if(!in_config.empty())
  {
    ImGuiIO& io    = ImGui::GetIO();
    io.IniFilename = in_config.c_str();
  }

  // Create Elements of the application
  auto toolbox_viewer = std::make_shared<ToolboxViewer>();
  g_elem_camera       = std::make_shared<nvvkhl::ElementCamera>();

  app->addElement(g_elem_camera);                                              // Controlling the camera movement
  app->addElement(toolbox_viewer);                                             // Our sample
  app->addElement(std::make_unique<nvvkhl::ElementLogger>(&g_logger, false));  // Add logger window
  app->addElement(std::make_unique<nvvkhl::ElementNvml>(false));               // Add logger window
  app->addElement(std::make_unique<ElementTesting>(argc, argv, toolbox_viewer->settings()));  // --test

  // Loading HDR and scene; default or command line
  toolbox_viewer->onFileDrop(in_hdr.c_str());
  toolbox_viewer->waitForLoad();
  toolbox_viewer->onFileDrop(in_filename.c_str());
  toolbox_viewer->waitForLoad();

  // Start Application: which will loop and call on"Functions" for all Elements
  app->run();

  // Cleanup
  vkDeviceWaitIdle(app->getContext()->m_device);
#ifdef _DEBUG
  vkDestroyDebugUtilsMessengerEXT(app->getContext()->m_instance, dbg_messenger, nullptr);
#endif
  toolbox_viewer.reset();
  app.reset();

#ifdef USE_NSIGHT_AFTERMATH
  g_aftermathTracker.reset();
#endif

  return 0;
}

#if _WIN32
// Avoiding console on Windows
int __stdcall WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ char*, _In_ int nShowCmd)
{
  return main(__argc, __argv);
}
#endif