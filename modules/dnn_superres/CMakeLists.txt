set(the_description "DNN-based image super-resolution")

ocv_define_module(dnn_superres
    opencv_core
    opencv_imgproc
    opencv_dnn
    WRAP
        python
        java
        objc
        js
)
