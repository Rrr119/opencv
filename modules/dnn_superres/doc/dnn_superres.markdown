# DNN-based Super-Resolution {#dnn_superres_overview}

## Overview

The `dnn_superres` module provides a high-level interface for running
deep-learning super-resolution models on top of the @ref dnn inference engine.
It removes the repetitive glue code required to feed an image through a network
and convert its output back into a regular `cv::Mat`.

Main entry point: cv::dnn_superres::DnnSuperRes.

## Features

- Single-image upscaling with `DnnSuperRes::upsample`.
- Batched upscaling with `DnnSuperRes::upsampleMulti`, including optional
  chunking to bound memory usage.
- Model loading from ONNX, TensorFlow Lite and OpenVINO IR files, plus
  in-memory buffers. The format is guessed from the file extension by default.
- Configurable preprocessing (pixel scale, mean subtraction, BGR/RGB swap) and
  optional fixed input size for models exported with a static shape.
- Direct pass-through of DNN backend and target selection (CPU, OpenCL, CUDA,
  Vulkan, OpenVINO, ...).

## Typical workflow

```cpp
cv::Ptr<cv::dnn_superres::DnnSuperRes> sr =
        cv::dnn_superres::DnnSuperRes::create();
sr->readModel("ESPCN_x4.onnx");
sr->setScale(4);

cv::Mat upscaled;
sr->upsample(cv::imread("input.png"), upscaled);
```

See the @ref tutorial_dnn_superres tutorial for batch processing, video
support and the complete C++/Python samples.
