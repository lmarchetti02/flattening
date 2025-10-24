import ctypes
import numpy as np
from pathlib import Path


def _load_library() -> ctypes.CDLL:
    """Load the shared C library.

    Returns:
        ctypes.CDLL: The shared library.
    """
    lib_path = next(Path(__file__).parent.glob("interpolate*.so"))
    print(lib_path)
    return ctypes.CDLL(lib_path)


def _interpolate(projections: np.ndarray, normalized_angles: np.ndarray) -> np.ndarray:
    """Performs the linear interpolation necessary to map the pixel values
    on the virtual flat detector.

    Args:
        projections (np.ndarray[ndim=3]): A set of 2D projections.
        normalized_angles (np.ndarray[ndim=1]): The angles corresponding to the
            columns of the curved detector.

    Returns:
        np.ndarray[ndim=3]: The set of flattened projections.
    """
    lib = _load_library()

    # Define C function signature
    lib.interpolation_loop.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # proj
        ctypes.POINTER(ctypes.c_double),  # normalized_angles
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_int,  # num_angles
        ctypes.c_int,  # num_rows
        ctypes.c_int,  # orig_num_detectors
        ctypes.c_int,  # num_cols
    ]
    lib.interpolation_loop.restype = None  # void function
    num_proj, num_rows, orig_num_detectors = projections.shape
    num_cols = len(normalized_angles)

    out = np.zeros((num_proj, num_rows, num_cols), dtype=np.float32)

    # convert to C-ordered arrays (flatten)
    proj_c = np.ascontiguousarray(projections, dtype=np.float32)
    ang_c = np.ascontiguousarray(normalized_angles, dtype=np.float64)
    out_c = np.ascontiguousarray(out, dtype=np.float32)

    # run C function
    lib.interpolation_loop(
        proj_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ang_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(num_proj),
        ctypes.c_int(num_rows),
        ctypes.c_int(orig_num_detectors),
        ctypes.c_int(num_cols),
    )

    return out_c


def flatten_detector(
    proj: np.ndarray,
    DSD: float,
    arclength: float,
    oversample: int = 1,
) -> np.ndarray:
    """Flatten the projections.

    Args:
        proj (np.ndarray[ndim=3]): The set of 2D curved projections.
        DSD (float): The distance from detector to source (in mm).
        arclength (float): The angular span of the detector (in rad).
        oversample (int, optional): The amount of oversampling on the flat plane.
            Defaults to 1.

    Returns:
        np.ndarray[ndim=3]: The set of flattened projections.
    """
    orig_num_detectors = proj.shape[-1]
    pixel_arclength = arclength / orig_num_detectors

    # Calculate the size of the detector projected from the arc onto the plane
    # This is slightly different for odd and even detectors
    if orig_num_detectors % 2:
        # Odd number of detectors - one on the centreline
        detector_size = np.tan(pixel_arclength) * DSD
    else:
        detector_size = np.tan(pixel_arclength / 2) * DSD * 2

    detector_size = detector_size / oversample
    total_detector_size = np.tan(arclength / 2) * DSD * 2

    # Calculate the equal distance detector positions on the plane detector
    if orig_num_detectors % 2:
        # Odd number of detectors - one on the centreline
        detectors = np.arange(detector_size, total_detector_size / 2, detector_size)
        detectors = np.concatenate((np.flip(-detectors), [0], detectors))
    else:
        detectors = np.arange(detector_size / 2, total_detector_size / 2, detector_size)
        detectors = np.concatenate((np.flip(-detectors), detectors))

    # Calculate the angle from the source to detector positions on the plane detector
    angles = np.arctan2(detectors, DSD)

    # Normailse to give this angle relative to the projection detector
    normalized_angles = angles / pixel_arclength + orig_num_detectors / 2

    # Interpolate
    flattened_proj = _interpolate(proj, normalized_angles)

    return flattened_proj


if __name__ == "__main__":
    pass
