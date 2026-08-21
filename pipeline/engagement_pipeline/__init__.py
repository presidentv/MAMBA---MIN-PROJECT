"""Reusable components for the DAiSEE student-engagement preprocessing pipeline.

The package is deliberately split so that each concern can be swapped out without
touching the phase drivers:

    config      typed, serialisable configuration for both phases
    dataset     discovery of clips + label loading (layout-driven, never hardcoded)
    video       frame sampling with an OpenCV primary path and ffmpeg fallback
    models      resolution/verification of the MediaPipe Tasks model bundles
    faces       thin wrappers over MediaPipe Tasks FaceDetector / FaceLandmarker
    alignment   canonical eye-line alignment + normalisation
    pose        solvePnP head-pose estimation and rotation-matrix decomposition
    mrs         Motion Reliability Score components and their combination
    io_utils    mirrored output paths, parquet/csv writing
    logging_utils  console + file logging
"""

__version__ = "1.0.0"
