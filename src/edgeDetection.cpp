/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>

#include <chrono>
#include <iostream>

#include "algo.hpp"
#include "cli.hpp"
#include "io.hpp"

bool printfNPPinfo() {
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int process_video(std::string infilename, std::string outfilename) {
    cv::VideoCapture capture(infilename);
    int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

    cv::VideoWriter writer(outfilename, fourcc, fps, cv::Size(frameWidth, frameHeight));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open output video file: " << outfilename << std::endl;
        return -1;
    }

    cv::Mat frame;

    npp::ImageCPU_8u_C4 oHostSrc(frameWidth, frameHeight);
    npp::ImageCPU_8u_C4 oHostDst(oHostSrc.width(), oHostSrc.height());
    EdgeFilter_8u_C4 filter(frameWidth, frameHeight);

    // measure runtime: start
    auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    while (true) {
        capture >> frame;
        if (frame.empty()) break;
        loadFromFrame(frame, oHostSrc);
        filter.filter(oHostSrc, oHostDst);
        saveToFrame(oHostDst, frame);
        writer.write(frame);
        ++count;
    }
    // measure runtime: end
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " nanoseconds" << std::endl;
    std::cout << "per frame: " << duration / count << " nanoseconds" << std::endl;

    capture.release();
    writer.release();

    return 0;
}

int process_png(std::string infilename, std::string outfilename) {
    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C4 oHostSrc;

    // load gray-scale image from disk
    loadImage(infilename, oHostSrc);

    EdgeFilter_8u_C4 filter{oHostSrc.width(), oHostSrc.height()};
    // declare a host image for the result
    npp::ImageCPU_8u_C4 oHostDst(oHostSrc.width(), oHostSrc.height());
    // measure runtime: start
    auto start = std::chrono::high_resolution_clock::now();
    filter.filter(oHostSrc, oHostDst);

    // measure runtime: end
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " nanoseconds" << std::endl;

    saveImage(outfilename, oHostDst);
    std::cout << "Saved image: " << outfilename << std::endl;

    return 0;
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo() == false) {
        exit(EXIT_SUCCESS);
    }

    Cli cli{argc, argv};
    std::string filename = cli.fileName;
    std::string resultFilename = cli.resultFilename;

    if (cli.fileExtension == ".mp4") {
        return process_video(filename, resultFilename);
    }
    if (cli.fileExtension == ".png") {
        return process_png(filename, resultFilename);
    }

    return 0;
}
