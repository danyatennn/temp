DISPLAY_RES = (900, 1200)

ROI_TOP = 80
ROI_LEFT = 100
ROI_HEIGHT = 200
ROI_WIDTH = int(ROI_HEIGHT * DISPLAY_RES[1] / DISPLAY_RES[0])


def crop_roi(x):
    """
    Crop the region of interest from a batch of images.

    Loss and metrics are computed on this region, which corresponds to the
    object displayed on the screen during capture (the rest is black border).
    The values match the alignment used by the dataset preprocessor.

    Args:
        x (Tensor): images of shape (..., H, W).
    Returns:
        Tensor: cropped region of shape (..., ROI_HEIGHT, ROI_WIDTH).
    """
    return x[..., ROI_TOP : ROI_TOP + ROI_HEIGHT, ROI_LEFT : ROI_LEFT + ROI_WIDTH]
