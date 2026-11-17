#!/usr/bin/env python3
"""
This file is part of OpenCV project.
It is subject to the license terms in the LICENSE file found in the top-level directory
of this distribution and at http://opencv.org/license.html.

Super-resolution sample built on the high-level cv.dnn_superres API.

Usage examples:
    python super_resolution_module.py --model=ESPCN_x4.onnx --input=image.png
    python super_resolution_module.py --model=ESPCN_x4.onnx \\
        --input=img1.png,img2.png,img3.png --chunk=2 --output=results/
    python super_resolution_module.py --model=ESPCN_x4.onnx --input=video.mp4
"""

import argparse
import os

import cv2 as cv
from common import findModel, get_backend_id, get_target_id


def build_superres(args):
    sr = cv.dnn_superres.DnnSuperRes_create()
    sr.readModel(findModel(args.model, ""))
    sr.setScale(args.scale)
    sr.setPreprocessing(args.scalefactor, (0.0, 0.0, 0.0), args.rgb)
    sr.setPreferableBackend(get_backend_id(args.backend))
    sr.setPreferableTarget(get_target_id(args.target))
    return sr


def snippets():
    #! [Load and upsample a single image]
    sr = cv.dnn_superres.DnnSuperRes_create()
    sr.readModel("ESPCN_x4.onnx")        # format guessed from the extension
    sr.setScale(4)
    sr.setPreprocessing(1.0 / 255.0, (0.0, 0.0, 0.0), False)

    image = cv.imread("input.png")
    upscaled_image = sr.upsample(image)
    #! [Load and upsample a single image]

    #! [Batch super-resolution]
    frames = [cv.imread("frame1.png"), cv.imread("frame2.png")]
    upscaled_frames = sr.upsampleMulti(frames, 0)   # one batched forward pass
    #! [Batch super-resolution]
    return upscaled_image, upscaled_frames


def process_images(sr, paths, output_dir, chunk):
    images = []
    for path in paths:
        image = cv.imread(cv.samples.findFile(path))
        if image is None:
            raise SystemExit(f"Cannot read image: {path}")
        images.append(image)

    if len(images) == 1:
        results = [sr.upsample(images[0])]
    else:
        results = sr.upsampleMulti(images, chunk)

    for path, result in zip(paths, results):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"sr_{name}.png")
            cv.imwrite(out_path, result)
            print(f"Saved {out_path}")
        else:
            cv.imshow(f"Super-Resolution {path}", result)
    if not output_dir:
        cv.waitKey(0)


def process_video(sr, source):
    capture = cv.VideoCapture(0 if source == "0" else cv.samples.findFile(source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {source}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t0 = cv.getTickCount()
        result = sr.upsample(frame)
        milliseconds = (cv.getTickCount() - t0) * 1000.0 / cv.getTickFrequency()
        cv.putText(result, f"{milliseconds:.1f} ms", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv.imshow("Input", frame)
        cv.imshow("Super-Resolution", result)
        if cv.waitKey(1) == 27:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Path to an ONNX, TFLite or OpenVINO IR super-resolution model.")
    parser.add_argument("--scale", type=int, default=4, help="Model upscaling factor.")
    parser.add_argument("--input", required=True,
                        help="Image(s), video file or camera index 0. "
                             "Separate multiple images with commas.")
    parser.add_argument("--output", default="",
                        help="Optional output directory; results are displayed when omitted.")
    parser.add_argument("--chunk", type=int, default=0,
                        help="Maximum images per forward pass for batch input (0 = whole batch).")
    parser.add_argument("--rgb", action="store_true",
                        help="Swap BGR to RGB before forwarding, required by some models.")
    parser.add_argument("--scalefactor", type=float, default=1.0,
                        help="Pixel value scale factor used in preprocessing.")
    parser.add_argument("--backend", default="default",
                        choices=("default", "openvino", "opencv", "vkcom", "cuda", "webnn"))
    parser.add_argument("--target", default="cpu",
                        choices=("cpu", "opencl", "opencl_fp16", "ncs2_vpu",
                                 "hddl_vpu", "vulkan", "cuda", "cuda_fp16"))
    args = parser.parse_args()

    sr = build_superres(args)
    paths = [part for part in args.input.split(",") if part]
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    is_video = args.input == "0" or args.input.lower().endswith(video_extensions)

    if is_video:
        process_video(sr, args.input)
    else:
        process_images(sr, paths, args.output, args.chunk)


if __name__ == "__main__":
    main()
