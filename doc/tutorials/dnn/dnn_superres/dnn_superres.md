# DNN-based image super-resolution {#tutorial_dnn_superres}

@tableofcontents

@prev_tutorial{tutorial_table_of_content_dnn}

|    |    |
| -: | :- |
| Original author | OpenCV Team |
| Compatibility | OpenCV >= 5.2 |

## Introduction

Super-resolution models reconstruct a high-resolution image from a
lower-resolution one. A number of popular architectures
(FSRCNN, ESPCN, EDSR, LapSRN, SeeMoreDetails and similar) are fully
convolutional, so they can be executed directly by the @ref dnn module.

The @ref dnn_superres module wraps the usual inference boilerplate -
model loading, blob preprocessing, output postprocessing and scaling-factor
bookkeeping - behind a small class, cv::dnn_superres::DnnSuperRes, and adds
batch processing on top of the single-image workflow.

## Supported model formats

Every format supported by the DNN module is accepted:

- **ONNX** (`.onnx`), loaded through cv::dnn::readNetFromONNX - the recommended format;
- **TensorFlow Lite** (`.tflite`), loaded through cv::dnn::readNetFromTFLite;
- **OpenVINO Model Optimizer IR** (`.xml` plus an optional `.bin`), loaded through
  cv::dnn::readNetFromModelOptimizer.

When the file extension is one of `.onnx`, `.tflite` or `.xml`, the correct
reader is selected automatically. Pass an explicit value from
cv::dnn_superres::SRModelFormat when reading from an in-memory buffer or when
the file has a non-standard name.

## Single image super-resolution

The minimal workflow consists of loading a model, declaring its upscaling
factor and calling `upsample()`:

@snippet dnn/super_resolution_module.cpp Load and upsample a single image

The same workflow in Python:

@snippet dnn/super_resolution_module.py Load and upsample a single image

The output image has the same channel count as the input and is scaled by the
model factor (for example an x4 model turns a 320x240 frame into 1280x960).

### Preprocessing

Zoo models differ in how they expect input pixels to be normalized. Configure
this once with `setPreprocessing()`:

- `scale` - multiplier applied to every pixel (use `1.0 / 255` for models that
  consume the `[0; 1]` range, the default is `1.0`);
- `mean` - per-channel mean subtraction (zero by default);
- `swapRB` - set to `true` for models trained on RGB inputs when you pass BGR
  images (the channel order is restored on the output side).

Models exported with a fixed input shape additionally require
`setInputSize()`; fully convolutional models accept any size and leave the
input size empty.

## Batch processing

Processing several equally sized images one by one repeats a lot of framework
overhead. `upsampleMulti()` packs every image into one blob and forwards the
whole batch in a single call:

@snippet dnn/super_resolution_module.cpp Batch super-resolution

The Python counterpart returns a list of result images:

@snippet dnn/super_resolution_module.py Batch super-resolution

The optional `chunkSize` argument limits how many images are forwarded at
once, which lets you trade throughput for memory on large batches. Chunked and
one-shot processing produce identical results.

## Videos and camera streams

A video is simply a sequence of frames, so the same class works with
cv::VideoCapture. Read the complete workflow in
`samples/dnn/super_resolution_module.cpp`; the key call is still `upsample()`
applied to every captured frame.

## Backend and target selection

`setPreferableBackend()` and `setPreferableTarget()` forward directly to the
underlying cv::dnn::Net, so GPU acceleration (CUDA, OpenCL, Vulkan) or
OpenVINO can be selected exactly as for any other DNN workflow. Advanced users
can reach the network itself through `getNet()`.

## Running the sample

The accompanying samples accept images, comma-separated image lists and video
files:

```bash
# One image
./super_resolution_module --model=ESPCN_x4.onnx --scale=4 --input=input.png

# Three images forwarded as one batch, results written to disk
./super_resolution_module --model=ESPCN_x4.onnx --scale=4 \
    --input=a.png,b.png,c.png --output=results/

# A video file (or pass 0 for the default camera)
./super_resolution_module --model=ESPCN_x4.onnx --scale=4 --input=clip.mp4
```

```bash
python super_resolution_module.py --model=ESPCN_x4.onnx --scale=4 \
    --input=a.png,b.png,c.png --chunk=2 --output=results/
```
