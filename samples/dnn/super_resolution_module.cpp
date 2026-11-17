/*
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

This sample demonstrates the high-level dnn_superres module:
  - single image super-resolution,
  - batch processing of several images in one forward pass,
  - video / camera frame upscaling.

Usage examples:
  ./super_resolution_module --model=ESPCN_x4.onnx --scale=4 --input=image.png
  ./super_resolution_module --model=ESPCN_x4.onnx --scale=4 --input=img1.png,img2.png,img3.png
  ./super_resolution_module --model=ESPCN_x4.onnx --scale=4 --input=video.mp4
*/

#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>

#include "common.hpp"

using namespace cv;
using namespace cv::dnn_superres;

namespace {

// Self-contained usage extracted by the documentation system (@snippet).
static void documentationSnippets(const Mat& sampleImage)
{
    //! [Load and upsample a single image]
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->readModel("ESPCN_x4.onnx");              // format guessed from the extension
    sr->setScale(4);                             // x4 model
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);

    Mat upscaled;
    sr->upsample(sampleImage, upscaled);
    //! [Load and upsample a single image]

    //! [Batch super-resolution]
    std::vector<Mat> frames = {sampleImage, sampleImage};
    std::vector<Mat> results;
    sr->upsampleMulti(frames, results);          // one batched forward pass
    //! [Batch super-resolution]
}

std::vector<std::string> splitPaths(const std::string& value, char delimiter)
{
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= value.size())
    {
        size_t end = value.find(delimiter, begin);
        if (end == std::string::npos)
            end = value.size();
        const std::string token = value.substr(begin, end - begin);
        if (!token.empty())
            result.push_back(token);
        begin = end + 1;
    }
    return result;
}

std::string baseNameWithoutExtension(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    const size_t begin = slash == std::string::npos ? 0 : slash + 1;
    size_t end = path.find_last_of('.');
    if (end == std::string::npos || end < begin)
        end = path.size();
    return path.substr(begin, end - begin);
}

int runImages(Ptr<DnnSuperRes>& sr, const std::vector<std::string>& paths,
              const std::string& outputDir, int chunkSize)
{
    std::vector<Mat> images;
    images.reserve(paths.size());
    for (const std::string& path : paths)
    {
        const std::string fullPath = findFile(path);
        Mat image = imread(fullPath);
        if (image.empty())
        {
            std::cerr << "Cannot read image: " << path << std::endl;
            return -1;
        }
        images.push_back(image);
    }

    if (images.size() == 1)
    {
        Mat result;
        sr->upsample(images[0], result);
        images.assign(1, result);
    }
    else
    {
        std::vector<Mat> results;
        sr->upsampleMulti(images, results, chunkSize);
        images = std::move(results);
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        if (!outputDir.empty())
        {
            utils::fs::createDirectories(outputDir);
            const std::string outPath = outputDir + "/sr_" +
                    baseNameWithoutExtension(paths[i]) + ".png";
            imwrite(outPath, images[i]);
            std::cout << "Saved " << outPath << std::endl;
        }
        else
        {
            imshow("Super-Resolution " + std::to_string(i), images[i]);
        }
    }
    if (outputDir.empty())
        waitKey(0);
    return 0;
}

int runVideo(Ptr<DnnSuperRes>& sr, const std::string& inputPath)
{
    VideoCapture capture;
    if (inputPath == "0")
        capture.open(0);
    else
        capture.open(findFile(inputPath));
    if (!capture.isOpened())
    {
        std::cerr << "Cannot open video: " << inputPath << std::endl;
        return -1;
    }

    namedWindow("Input", WINDOW_NORMAL);
    namedWindow("Super-Resolution", WINDOW_NORMAL);

    Mat frame;
    while (capture.read(frame))
    {
        Mat result;
        const int64 t0 = getTickCount();
        sr->upsample(frame, result);
        const double ms = (getTickCount() - t0) * 1000.0 / getTickFrequency();

        putText(result, format("%.1f ms", ms), Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        imshow("Input", frame);
        imshow("Super-Resolution", result);
        if (waitKey(1) == 27)
            break;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    (void)documentationSnippets;
    const std::string keys =
        "{ help h       |        | Print help message }"
        "{ model m      |        | Path to an ONNX, TFLite or OpenVINO IR super-resolution model }"
        "{ scale s      | 4      | Model upscaling factor (2, 3, 4, ...) }"
        "{ input i      |        | Image(s), video file or camera index 0. Multiple images are comma-separated }"
        "{ output o     |        | Optional output directory; when omitted, results are shown in windows }"
        "{ chunk        | 0      | Max images per forward pass for batch input (0 = whole batch at once) }"
        "{ rgb          | false  | Swap BGR to RGB before forwarding, required by some models }"
        "{ scalefactor  | 1.0    | Pixel value scale factor used in preprocessing }"
        "{ backend      | default| default, openvino, opencv, vkcom, cuda or webnn }"
        "{ target       | cpu    | cpu, opencl, opencl_fp16, vpu, vulkan, cuda or cuda_fp16 }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const std::string modelPath = parser.get<std::string>("model");
    const std::string input = parser.get<std::string>("input");
    if (modelPath.empty() || input.empty())
    {
        std::cerr << "Both --model and --input are required. Use --help for details." << std::endl;
        return -1;
    }

    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    try
    {
        sr->readModel(findModel(modelPath, ""));
    }
    catch (const Exception& e)
    {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }
    sr->setScale(parser.get<int>("scale"));
    sr->setPreprocessing(parser.get<double>("scalefactor"), Scalar(),
                         parser.get<bool>("rgb"));
    sr->setPreferableBackend(getBackendID(parser.get<std::string>("backend")));
    sr->setPreferableTarget(getTargetID(parser.get<std::string>("target")));

    const std::vector<std::string> paths = splitPaths(input, ',');
    const bool isVideo = paths.size() == 1 &&
            (input == "0" ||
             input.find(".mp4") != std::string::npos ||
             input.find(".avi") != std::string::npos ||
             input.find(".mov") != std::string::npos ||
             input.find(".mkv") != std::string::npos);

    return isVideo ? runVideo(sr, input)
                   : runImages(sr, paths, parser.get<std::string>("output"),
                               parser.get<int>("chunk"));
}
