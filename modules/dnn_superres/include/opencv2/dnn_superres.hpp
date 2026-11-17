/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2026, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability, or tort
// (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_DNN_SUPERRES_HPP
#define OPENCV_DNN_SUPERRES_HPP

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"

/**
@defgroup dnn_superres DNN-based Image Super-Resolution

This module offers a high-level API for running deep-learning based
super-resolution models through the @ref dnn module. It takes care of the
typical boilerplate (model loading, input preprocessing, output
postprocessing and scaling factor bookkeeping) and adds batch processing
on top of the regular single-image workflow.

Supported model formats are the ones supported by the DNN module:
- ONNX (`.onnx`, see cv::dnn::readNetFromONNX)
- TensorFlow Lite (`.tflite`, see cv::dnn::readNetFromTFLite)
- OpenVINO Model Optimizer IR (`.xml` + `.bin`, see cv::dnn::readNetFromModelOptimizer)

@note
The module only performs inference. The models themselves have to be obtained
separately (for example FSRCNN, ESPCN, EDSR, LapSRN or SeeMoreDetails).
 */

namespace cv {
namespace dnn_superres {

//! @addtogroup dnn_superres
//! @{

/** @brief Serialization format of a super-resolution model.
 */
enum SRModelFormat
{
    //! Format is guessed from the file extension (default).
    SR_FORMAT_AUTO = 0,
    //! ONNX format, typically stored in `.onnx` files.
    SR_FORMAT_ONNX = 1,
    //! TensorFlow Lite format, stored in `.tflite` files.
    SR_FORMAT_TFLITE = 2,
    //! OpenVINO Model Optimizer IR: an `.xml` graph with an optional `.bin` weights file.
    SR_FORMAT_IR = 3
};

/** @brief A class that runs DNN-based super-resolution on images.

The class wraps cv::dnn::Net together with the preprocessing/postprocessing
conventions shared by common super-resolution architectures. A typical
workflow is:

@code
    cv::Ptr<cv::dnn_superres::DnnSuperRes> sr =
            cv::dnn_superres::DnnSuperRes::create();
    sr->readModel("ESPCN_x4.onnx");
    sr->setScale(4);

    cv::Mat img = cv::imread("input.png");
    cv::Mat result;
    sr->upsample(img, result);
@endcode
 */
class CV_EXPORTS_W DnnSuperRes
{
public:
    CV_WRAP virtual ~DnnSuperRes();

    /** @brief Empty constructor, use readModel() to load a network.
     */
    CV_WRAP static Ptr<DnnSuperRes> create();

    /** @brief Construct an instance and load a model in one call.
     *  @param model Path to the model file.
     *  @param format Model format, SR_FORMAT_AUTO guesses it from the file extension.
     *  @param scale Upscaling factor of the model (2, 3, 4, ...).
     */
    CV_WRAP static Ptr<DnnSuperRes> create(CV_WRAP_FILE_PATH const String& model,
                                           SRModelFormat format = SR_FORMAT_AUTO,
                                           int scale = 0);

    /** @brief Load a super-resolution model from a file.
     *
     *  @param model Path to the model file. For IR models this is the `.xml`
     *               graph; the `.bin` weights are looked up next to it.
     *  @param format Model format. SR_FORMAT_AUTO selects the reader based on
     *               the file extension (`.onnx`, `.tflite`, `.xml`).
     */
    CV_WRAP virtual void readModel(CV_WRAP_FILE_PATH const String& model,
                                   SRModelFormat format = SR_FORMAT_AUTO) = 0;

    /** @brief Load an ONNX or TFLite model from an in-memory buffer.
     *  @param buffer Serialized model bytes.
     *  @param format Either SR_FORMAT_ONNX or SR_FORMAT_TFLITE.
     */
    CV_WRAP virtual void readModel(const std::vector<uchar>& buffer,
                                   SRModelFormat format) = 0;

    /** @brief Load an OpenVINO IR model from separate graph and weight buffers.
     */
    virtual void readModel(const std::vector<uchar>& bufferModelConfig,
                           const std::vector<uchar>& bufferWeights) = 0;

    /** @brief Run super-resolution on a single image.
     *  @param img Input image, 8-bit BGR or single-channel.
     *  @param result Output image, same channel count as the input, scaled by
     *                 getScale() in both dimensions.
     */
    CV_WRAP virtual void upsample(InputArray img, OutputArray result) = 0;

    /** @brief Run super-resolution on a batch of equally sized images.
     *
     *  All images are forwarded through the network in a single blob, which
     *  is usually significantly faster than calling upsample() in a loop.
     *
     *  @param imgs Input images. Every image must have the same size and
     *               channel count.
     *  @param results Upscaled images, one entry per input image.
     *  @param chunkSize Maximum number of images forwarded at once. Use 0 (the
     *                    default) to forward the whole batch in one call. Lower
     *                    values trade throughput for a smaller memory footprint.
     */
    CV_WRAP virtual void upsampleMulti(InputArrayOfArrays imgs,
                                       CV_OUT std::vector<Mat>& results,
                                       int chunkSize = 0) = 0;

    /** @brief Convenience overload that processes images in-place.
     */
    CV_WRAP virtual void upsampleMultiInplace(std::vector<Mat>& imgs,
                                              int chunkSize = 0) = 0;

    /** @brief Set the model upscaling factor (for example 4 for x4 models).
     */
    CV_WRAP virtual void setScale(int scale) = 0;
    /** @brief Returns the model upscaling factor.
     */
    CV_WRAP virtual int getScale() const = 0;

    /** @brief Set an arbitrary algorithm label (informational only).
     */
    CV_WRAP virtual void setAlgorithm(const String& algo) = 0;
    /** @brief Returns the algorithm label.
     */
    CV_WRAP virtual String getAlgorithm() const = 0;

    /** @brief Configure input preprocessing applied before forwarding.
     *
     *  Defaults match the common FSRCNN/ESPCN/EDSR/LapSRN zoo models:
     *  scale = 1.0, mean = (0, 0, 0), swapRB = false.
     *
     *  @param scale Multiplier applied to the pixel values (blob scalefactor).
     *  @param mean Per-channel mean subtracted from the pixels.
     *  @param swapRB When true, BGR images are converted to RGB before forwarding.
     */
    CV_WRAP virtual void setPreprocessing(double scale, const Scalar& mean,
                                          bool swapRB = false) = 0;

    /** @brief Set the fixed network input size.
     *
     *  When left empty (the default) images are forwarded at their original
     *  resolution, which is what fully convolutional super-resolution models
     *  expect. Models exported with a static input shape require a fixed size.
     */
    CV_WRAP virtual void setInputSize(const Size& size) = 0;
    /** @brief Returns the configured fixed input size (empty when dynamic).
     */
    CV_WRAP virtual Size getInputSize() const = 0;

    /** @brief Choose the DNN computation backend, see cv::dnn::Net::setPreferableBackend.
     */
    CV_WRAP virtual void setPreferableBackend(int backendId) = 0;
    /** @brief Choose the DNN target device, see cv::dnn::Net::setPreferableTarget.
     */
    CV_WRAP virtual void setPreferableTarget(int targetId) = 0;

    /** @brief Provides direct access to the underlying network for advanced tuning.
     */
    virtual dnn::Net& getNet() = 0;
    /** @brief Returns true once a model has been loaded successfully.
     */
    CV_WRAP virtual bool isLoaded() const = 0;
};

//! @}

} // namespace dnn_superres
} // namespace cv

#include "opencv2/dnn_superres/version.hpp"

#endif // OPENCV_DNN_SUPERRES_HPP
