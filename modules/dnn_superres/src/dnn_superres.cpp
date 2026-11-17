/*
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.
*/

#include "precomp.hpp"

namespace cv {
namespace dnn_superres {

using namespace cv::dnn;

namespace {

class DnnSuperResImpl CV_FINAL : public DnnSuperRes
{
public:
    DnnSuperResImpl()
        : scale_(0), algo_("unknown"), blobScale_(1.0),
          mean_(), swapRB_(false), inputSize_(),
          preferableBackend_(DNN_BACKEND_DEFAULT),
          preferableTarget_(DNN_TARGET_CPU)
    {}

    void readModel(const String& model, SRModelFormat format) CV_OVERRIDE
    {
        const SRModelFormat resolved =
                format == SR_FORMAT_AUTO ? detectFormat(model) : format;
        switch (resolved)
        {
        case SR_FORMAT_ONNX:
            net_ = readNetFromONNX(model);
            break;
        case SR_FORMAT_TFLITE:
            net_ = readNetFromTFLite(model);
            break;
        case SR_FORMAT_IR:
            net_ = readNetFromModelOptimizer(model);
            break;
        default:
            CV_Error(Error::StsBadArg, "Unsupported super-resolution model format");
        }
        applyBackendAndTarget();
    }

    void readModel(const std::vector<uchar>& buffer, SRModelFormat format) CV_OVERRIDE
    {
        switch (format)
        {
        case SR_FORMAT_ONNX:
            net_ = readNetFromONNX(buffer);
            break;
        case SR_FORMAT_TFLITE:
            net_ = readNetFromTFLite(buffer);
            break;
        default:
            CV_Error(Error::StsBadArg,
                     "Buffer loading supports SR_FORMAT_ONNX and SR_FORMAT_TFLITE only");
        }
        applyBackendAndTarget();
    }

    void readModel(const std::vector<uchar>& bufferModelConfig,
                   const std::vector<uchar>& bufferWeights) CV_OVERRIDE
    {
        net_ = readNetFromModelOptimizer(bufferModelConfig, bufferWeights);
        applyBackendAndTarget();
    }

    void upsample(InputArray img, OutputArray result) CV_OVERRIDE
    {
        CV_Assert(!net_.empty());
        std::vector<Mat> inputs(1, img.getMat());
        std::vector<Mat> outputs;
        runBatch(inputs, outputs);
        outputs[0].copyTo(result);
    }

    void upsampleMulti(InputArrayOfArrays imgs,
                       CV_OUT std::vector<Mat>& results,
                       int chunkSize = 0) CV_OVERRIDE
    {
        CV_Assert(!net_.empty());
        std::vector<Mat> inputs;
        imgs.getMatVector(inputs);
        CV_Assert(!inputs.empty());

        results.clear();
        results.reserve(inputs.size());

        const int total = static_cast<int>(inputs.size());
        const int step = chunkSize > 0 ? std::min(chunkSize, total) : total;
        for (int begin = 0; begin < total; begin += step)
        {
            const int end = std::min(begin + step, total);
            std::vector<Mat> chunk(inputs.begin() + begin, inputs.begin() + end);
            std::vector<Mat> chunkResults;
            runBatch(chunk, chunkResults);
            results.insert(results.end(), chunkResults.begin(), chunkResults.end());
        }
    }

    void upsampleMultiInplace(std::vector<Mat>& imgs, int chunkSize = 0) CV_OVERRIDE
    {
        CV_Assert(!net_.empty());
        std::vector<Mat> results;
        upsampleMulti(imgs, results, chunkSize);
        imgs = std::move(results);
    }

    void setScale(int scale) CV_OVERRIDE { scale_ = scale; }
    int getScale() const CV_OVERRIDE { return scale_; }

    void setAlgorithm(const String& algo) CV_OVERRIDE { algo_ = algo; }
    String getAlgorithm() const CV_OVERRIDE { return algo_; }

    void setPreprocessing(double scale, const Scalar& mean,
                          bool swapRB = false) CV_OVERRIDE
    {
        blobScale_ = scale;
        mean_ = mean;
        swapRB_ = swapRB;
    }

    void setInputSize(const Size& size) CV_OVERRIDE { inputSize_ = size; }
    Size getInputSize() const CV_OVERRIDE { return inputSize_; }

    void setPreferableBackend(int backendId) CV_OVERRIDE
    {
        preferableBackend_ = backendId;
        if (!net_.empty())
            net_.setPreferableBackend(backendId);
    }

    void setPreferableTarget(int targetId) CV_OVERRIDE
    {
        preferableTarget_ = targetId;
        if (!net_.empty())
            net_.setPreferableTarget(targetId);
    }

    dnn::Net& getNet() CV_OVERRIDE { return net_; }
    bool isLoaded() const CV_OVERRIDE { return !net_.empty(); }

private:
    static SRModelFormat detectFormat(const String& path)
    {
        const size_t dot = path.find_last_of('.');
        String ext = dot == String::npos ? String() : path.substr(dot + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == "onnx")
            return SR_FORMAT_ONNX;
        if (ext == "tflite")
            return SR_FORMAT_TFLITE;
        if (ext == "xml")
            return SR_FORMAT_IR;
        CV_Error_(Error::StsBadArg,
                  ("Cannot infer model format from extension '%s'; pass an explicit SRModelFormat",
                   ext.c_str()));
    }

    void applyBackendAndTarget()
    {
        net_.setPreferableBackend(preferableBackend_);
        net_.setPreferableTarget(preferableTarget_);
    }

    static Mat prepareInput(const Mat& image)
    {
        if (image.channels() == 1)
        {
            Mat converted;
            cvtColor(image, converted, COLOR_GRAY2BGR);
            return converted;
        }
        return image;
    }

    void runBatch(const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        CV_Assert(!net_.empty());
        CV_Assert(!inputs.empty());
        const Size baseSize = inputs[0].size();
        const int baseChannels = inputs[0].channels();
        CV_Assert(inputs[0].depth() == CV_8U);
        for (const Mat& image : inputs)
        {
            CV_Assert(image.type() == inputs[0].type());
            CV_Assert(image.size() == baseSize);
        }

        std::vector<Mat> prepared;
        prepared.reserve(inputs.size());
        for (const Mat& image : inputs)
            prepared.push_back(prepareInput(image));

        const Size blobSize = inputSize_.area() > 0 ? inputSize_ : Size();
        Mat blob = blobFromImages(prepared, blobScale_, blobSize, mean_,
                                  swapRB_, false, CV_32F);

        net_.setInput(blob);
        Mat rawOutput = net_.forward();

        CV_Assert(rawOutput.dims == 4);
        const int batch = rawOutput.size[0];
        CV_Assert(batch == static_cast<int>(inputs.size()));
        const int outChannels = rawOutput.size[1];

        if (scale_ <= 0)
        {
            const int heightRatio = cvRound(static_cast<double>(rawOutput.size[2]) /
                                            std::max(1, blobSize.height > 0 ? blobSize.height : baseSize.height));
            const int widthRatio = cvRound(static_cast<double>(rawOutput.size[3]) /
                                           std::max(1, blobSize.width > 0 ? blobSize.width : baseSize.width));
            scale_ = std::max(1, std::max(heightRatio, widthRatio));
        }

        // Recover 8-bit pixel values. Models are typically identity-scaled with
        // respect to their input: when the blob scale is 1/255 the network
        // outputs values in roughly [0; 1], while with a unit scale they stay
        // in the 0..255 range. Only fall back to range inspection for exotic
        // configurations.
        double valueScale;
        if (std::fabs(blobScale_) > 1e-12)
            valueScale = 1.0 / blobScale_;
        else
        {
            double minValue = 0.0, maxValue = 0.0;
            const Mat flatOutput = rawOutput.reshape(1,
                                                      static_cast<int>(rawOutput.total(0, 2)));
            minMaxLoc(flatOutput, &minValue, &maxValue);
            valueScale = maxValue <= 1.0 + 1e-4 && minValue >= -1e-4 ? 255.0 : 1.0;
        }

        outputs.clear();
        outputs.reserve(inputs.size());
        const Size expectedSize(baseSize.width * scale_, baseSize.height * scale_);

        for (int n = 0; n < batch; ++n)
        {
            std::vector<Mat> channelMats(outChannels);
            for (int c = 0; c < outChannels; ++c)
            {
                channelMats[c] = Mat(rawOutput.size[2], rawOutput.size[3], CV_32F,
                                     const_cast<float*>(rawOutput.ptr<float>(n, c)));
            }
            Mat hwc;
            merge(channelMats, hwc);

            if (swapRB_ && outChannels == 3)
                cvtColor(hwc, hwc, COLOR_RGB2BGR);

            Mat image8u;
            hwc.convertTo(image8u, CV_8U, valueScale);

            if (outChannels == 1 && baseChannels == 3)
            {
                Mat colored;
                cvtColor(image8u, colored, COLOR_GRAY2BGR);
                image8u = colored;
            }
            else if (outChannels == 3 && baseChannels == 1)
            {
                Mat gray;
                cvtColor(image8u, gray, COLOR_BGR2GRAY);
                image8u = gray;
            }

            if (image8u.size() != expectedSize)
                resize(image8u, image8u, expectedSize, 0.0, 0.0, INTER_LINEAR);

            outputs.push_back(image8u);
        }
    }

    dnn::Net net_;
    int scale_;
    String algo_;
    double blobScale_;
    Scalar mean_;
    bool swapRB_;
    Size inputSize_;
    int preferableBackend_;
    int preferableTarget_;
};

} // namespace

Ptr<DnnSuperRes> DnnSuperRes::create()
{
    return makePtr<DnnSuperResImpl>();
}

Ptr<DnnSuperRes> DnnSuperRes::create(const String& model, SRModelFormat format, int scale)
{
    Ptr<DnnSuperRes> superres = makePtr<DnnSuperResImpl>();
    superres->setScale(scale);
    superres->readModel(model, format);
    return superres;
}

DnnSuperRes::~DnnSuperRes() = default;
} // namespace dnn_superres
} // namespace cv
