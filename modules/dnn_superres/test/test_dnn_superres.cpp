/*
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.
*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv::dnn;
using namespace cv::dnn_superres;

// Builds a minimal fully-convolutional "super-resolution" network entirely in
// memory: bilinear resize by `factor` followed by a 1x1 convolution that keeps
// the channel count. This keeps the tests independent of external model files.
static Net buildToySRNet(int factor, int channels = 3)
{
    Net net;

    LayerParams resizeParams;
    resizeParams.type = "Resize";
    resizeParams.name = "upsample";
    resizeParams.set("width", 0.0f);
    resizeParams.set("height", 0.0f);
    resizeParams.set("zoom_factor_x", static_cast<float>(factor));
    resizeParams.set("zoom_factor_y", static_cast<float>(factor));
    resizeParams.set("interpolation", String("bilinear"));
    resizeParams.set("half_pixel_centers", true);
    net.addLayerToPrev(resizeParams.name, resizeParams.type, resizeParams);

    LayerParams convParams;
    convParams.type = "Convolution";
    convParams.name = "conv1x1";
    convParams.set("kernel_size", 1);
    convParams.set("num_output", channels);
    convParams.set("bias_term", false);

    std::vector<int> weightsShape = {channels, channels, 1, 1};
    Mat weights(weightsShape, CV_32F, Scalar(0));
    for (int c = 0; c < channels; ++c)
        weights.ptr<float>(c, c)[0] = 1.0f;
    convParams.blobs.push_back(weights);
    net.addLayerToPrev(convParams.name, convParams.type, convParams);

    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    return net;
}

static Mat makeInputImage(int width, int height, int channels)
{
    Mat image(height, width, CV_MAKETYPE(CV_8U, channels));
    randu(image, Scalar::all(0), Scalar::all(255));
    return image;
}

TEST(DnnSuperRes, Lifecycle)
{
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    ASSERT_FALSE(sr->isLoaded());

    sr->setScale(4);
    sr->setAlgorithm("toy");
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);
    sr->setInputSize(Size());
    sr->setPreferableBackend(DNN_BACKEND_OPENCV);
    sr->setPreferableTarget(DNN_TARGET_CPU);

    EXPECT_EQ(4, sr->getScale());
    EXPECT_EQ(String("toy"), sr->getAlgorithm());
    EXPECT_EQ(Size(), sr->getInputSize());

    Net toyNet = buildToySRNet(4);
    sr->getNet() = toyNet;
    ASSERT_TRUE(sr->isLoaded());

    EXPECT_THROW(sr->readModel("model.unknown"), cv::Exception);
}

TEST(DnnSuperRes, UpsampleSingle)
{
    const int factor = 2;
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(factor);
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);

    Mat input = makeInputImage(16, 12, 3);
    Mat output;
    sr->upsample(input, output);

    EXPECT_EQ(CV_8UC3, output.type());
    EXPECT_EQ(Size(input.cols * factor, input.rows * factor), output.size());

    Mat reference;
    resize(input, reference, Size(), factor, factor, INTER_LINEAR);
    EXPECT_LE(cv::norm(output, reference, NORM_INF), 2.0);
}

TEST(DnnSuperRes, UpsampleGrayscale)
{
    const int factor = 3;
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(factor, 3);
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);

    Mat input = makeInputImage(10, 8, 1);
    Mat output;
    sr->upsample(input, output);

    EXPECT_EQ(CV_8UC1, output.type());
    EXPECT_EQ(Size(input.cols * factor, input.rows * factor), output.size());
}

TEST(DnnSuperRes, UpsampleBatch)
{
    const int factor = 2;
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(factor);
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);

    const int imageCount = 5;
    std::vector<Mat> inputs;
    for (int i = 0; i < imageCount; ++i)
        inputs.push_back(makeInputImage(16, 12, 3));

    std::vector<Mat> results;
    sr->upsampleMulti(inputs, results);

    ASSERT_EQ(imageCount, static_cast<int>(results.size()));
    for (const Mat& result : results)
    {
        EXPECT_EQ(CV_8UC3, result.type());
        EXPECT_EQ(Size(32, 24), result.size());
    }

    // Chunked processing must produce identical results to one-shot batching.
    std::vector<Mat> chunkedResults;
    sr->upsampleMulti(inputs, chunkedResults, 2);
    ASSERT_EQ(results.size(), chunkedResults.size());
    for (size_t i = 0; i < results.size(); ++i)
        EXPECT_EQ(0, cv::norm(results[i], chunkedResults[i], NORM_INF));

    // In-place overload.
    std::vector<Mat> inplace = inputs;
    sr->upsampleMultiInplace(inplace);
    ASSERT_EQ(imageCount, static_cast<int>(inplace.size()));
    for (const Mat& result : inplace)
        EXPECT_EQ(Size(32, 24), result.size());
}

TEST(DnnSuperRes, BatchRejectsMixedSizes)
{
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(2);

    std::vector<Mat> inputs = {makeInputImage(16, 16, 3), makeInputImage(8, 16, 3)};
    std::vector<Mat> results;
    EXPECT_THROW(sr->upsampleMulti(inputs, results), cv::Exception);
}

TEST(DnnSuperRes, AutoScaleDetection)
{
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(4);
    sr->setPreprocessing(1.0 / 255.0, Scalar(), false);

    EXPECT_EQ(0, sr->getScale());
    Mat input = makeInputImage(8, 8, 3);
    Mat output;
    sr->upsample(input, output);
    EXPECT_EQ(4, sr->getScale());
    EXPECT_EQ(Size(32, 32), output.size());
}

TEST(DnnSuperRes, SwapRBPreprocessing)
{
    const int factor = 2;
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();
    sr->getNet() = buildToySRNet(factor);
    sr->setPreprocessing(1.0 / 255.0, Scalar(), true);

    Mat input = makeInputImage(8, 8, 3);
    Mat output;
    sr->upsample(input, output);
    EXPECT_EQ(Size(16, 16), output.size());

    Mat reference;
    resize(input, reference, Size(16, 16), 0.0, 0.0, INTER_LINEAR);
    std::vector<Mat> refChannels, outChannels;
    split(reference, refChannels);
    split(output, outChannels);
    // The preprocessing BGR->RGB swap is undone on the output side, so the
    // final channel order must match the plain resize reference.
    for (int c = 0; c < 3; ++c)
        EXPECT_LE(cv::norm(outChannels[c], refChannels[c], NORM_INF), 2.0);
}

TEST(DnnSuperRes, FormatDetection)
{
    Ptr<DnnSuperRes> sr = DnnSuperRes::create();

    // Files do not exist: parsing must fail, but format selection happens first.
    EXPECT_THROW(sr->readModel("missing.onnx"), cv::Exception);
    EXPECT_THROW(sr->readModel("missing.tflite"), cv::Exception);
    EXPECT_THROW(sr->readModel("missing.xml"), cv::Exception);
    EXPECT_THROW(sr->readModel("missing.weights"), cv::Exception);
}

}} // namespace
