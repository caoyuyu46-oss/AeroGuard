# AeroGuard-UAVB sample

This 200-image example dataset contains 150 training images, 25 validation images,
and 25 test images. The original split assignment is retained. Each selected image
belongs to a different recorded source group.

The sample contains 100 drone-only images, 66 bird-only images, 33 background images,
and one image containing both classes. It covers six source collections, four
drone-size bins, different brightness and contrast ranges, and dense bird scenes.

Labels use YOLO detection format: class, normalized center x, normalized center y,
normalized width, normalized height. Class 0 is bird; class 1 is drone. Background
images have empty label files. Image and annotation bytes are unchanged.

`manifest.csv` records provenance and file checksums. `summary.json` records the
measured sample composition. `DATA_LICENSE.txt` describes the scope of dataset rights.

The sample is intended for running the software and checking data interfaces.
It does not replace the 24,000-image benchmark or support comparison with the
manuscript's reported experimental results.
