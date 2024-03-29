#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>
#include <math.h>
#include <iostream>
#include "../kernels/shared.h"
#include "Geometry.h"
#include "ImageWriter.h"
#include "MeshReader.h"
#include "Scene.h"
#include "TriangleMesh.h"
#include "assert.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#    include <GLES2/gl2.h>
#endif
#include <include/GLFW/glfw3.h> // Will drag system OpenGL headers

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}



void launchKernel(hipFunction_t func, int nx, int ny, void** args, hipStream_t stream = 0, size_t threadPerBlockX = 8, size_t threadPerBlockY = 8, size_t threadPerBlockZ = 1)
{
    size_t nBx = (nx + threadPerBlockX - 1) / threadPerBlockX;
    size_t nBy = (ny + threadPerBlockY - 1) / threadPerBlockY;
    HIP_ASSERT(hipModuleLaunchKernel(
                   func, (uint32_t) nBx, (uint32_t) nBy, 1, (uint32_t) threadPerBlockX, (uint32_t) threadPerBlockY, (uint32_t) threadPerBlockZ, 0, stream, args, 0) == hipSuccess,
               "Launch kernel");
}

struct GeometryData
{
    float3* vertices{nullptr};
    uint3* triangles{nullptr};
    float3* vertex_normals{nullptr};
    float3* triangle_normals{nullptr};
    uint32_t nUniqueTriangles{0};
    uint32_t nUniqueVertices{0};
    uint32_t nTriangles{0};
    uint32_t nVertices{0};
    uint32_t nDeformations{0};
    uint32_t geometryID{0};
    uint32_t instanceID{0};
};


#include "RenderCases.h"

int main(int argc, char const* argv[])
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

     // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

      // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

     // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

     while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");          // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);             // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*) &clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window",
                         &show_another_window); // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

      // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();



    std::cout << "Current working directory: " << fs::current_path() << "\n";
    HIP_ASSERT(hipInit(0) == hipSuccess, "hipInit");

    int deviceCount{-1};
    HIP_ASSERT(hipGetDeviceCount(&deviceCount) == hipSuccess, "hipGetDeviceCount");
    std::cout << "Device count: " << deviceCount << std::endl;

    int deviceId{0};
    HIP_ASSERT(hipSetDevice(deviceId) == hipSuccess, "hipSetDevice");

    hipCtx_t hipContext{nullptr};
    HIP_ASSERT(hipCtxCreate(&hipContext, 0, deviceId) == hipSuccess, "hipCtxCreate");

    hipStream_t stream{nullptr};
    HIP_ASSERT(hipStreamCreate(&stream) == hipSuccess, "hipStreamCreate");

    hipDeviceProp_t deviceProperties;
    HIP_ASSERT(hipGetDeviceProperties(&deviceProperties, deviceId) == hipSuccess, "hipGetDeviceProperties");
    std::cout << "Device: " << deviceProperties.name << std::endl;

    constexpr int hiprtApiVersion{2003};
    hiprtContextCreationInput input;
    input.ctxt = hipContext;
    input.device = deviceId;
    input.deviceType = hiprtDeviceAMD;

    hiprtContext rtContext{nullptr};
    HIP_ASSERT(hiprtCreateContext(hiprtApiVersion, input, rtContext) == hiprtSuccess, "hiprtCreateContext");

    /*Render<CASE_TYPE::GEOMETRY_HIT_DISTANCE>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");
    Render<CASE_TYPE::GEOMETRY_DEBUG>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");
    Render<CASE_TYPE::GEOMETRY_DEBUG_WITH_CAMERA>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "geometry_hit_distance.png");

    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SAMPLING>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "sampling_mb.png");
    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_SLERP>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "trannsform_slerp.png");
    Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_AO_SLERP_2_INSTANCES>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "slerp_2_instances.png");
    Render<CASE_TYPE::SCENE_AMBIENT_OCCLUSION>(rtContext, stream, "../../scenes/cornellbox/cornellbox.obj", "../../scens/cornellbox/", "cb.png");
    */

    //Render<CASE_TYPE::SCENE_TRANSFORMATION_MB_DEFORMATION>(rtContext, stream, "../../scenes/sphere/s.obj", "../../scens/sphere/", "scene_transform_MB_deformation.png");

    HIP_ASSERT(hiprtDestroyContext(rtContext) == hiprtSuccess, "hiprtDestroyContext");
    HIP_ASSERT(hipStreamDestroy(stream) == hipSuccess, "hipStreamDestroy");
    HIP_ASSERT(hipCtxDestroy(hipContext) == hipSuccess, "hipCtxDestroy");

    std::cout << "Finished\n";
    return 0;
}
