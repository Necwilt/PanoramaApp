import os
import sys
import tempfile
from typing import OrderedDict
import PIL
from PyQt5 import QtGui
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, Qt, pyqtSignal, pyqtSlot
import cv2
import argparse
import glob
import time
import qimage2ndarray
from PyQt5.QtWidgets import QAction, QFrame, QHBoxLayout, QLayout, QMessageBox, QPushButton, QSizePolicy, QSlider, QWidget, QVBoxLayout, QLabel, QMainWindow, QApplication, QFileDialog, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from pano_ui import Ui_MainWindow
from PIL import Image

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv2.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv2.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv2.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv2.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv2.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv2.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv2.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv2.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv2.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    FEATURES_FIND_CHOICES['SURF'] = cv2.xfeatures2d_SURF.create
except (AttributeError, cv2.error) as e:
    print("SURF not available")
try:
    FEATURES_FIND_CHOICES['ORB'] = cv2.ORB.create
except (AttributeError, cv2.error) as e:
    print("ORB not available")
try:
    FEATURES_FIND_CHOICES['SIFT'] = cv2.SIFT_create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['BRISK'] = cv2.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['AKAZE'] = cv2.AKAZE_create
except AttributeError:
    print("AKAZE not available")

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv2.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv2.detail_GraphCutSeamFinder(
    'COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv2.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv2.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv2.detail.SeamFinder_createDefault(
    cv2.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv2.detail.SeamFinder_createDefault(
    cv2.detail.SeamFinder_NO)

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv2.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv2.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = OrderedDict()
WAVE_CORRECT_CHOICES['horiz'] = cv2.detail.WAVE_CORRECT_HORIZ
WAVE_CORRECT_CHOICES['no'] = None
WAVE_CORRECT_CHOICES['vert'] = cv2.detail.WAVE_CORRECT_VERT

BLEND_CHOICES = ('multiband', 'feather', 'no',)


class InvalidDirectoryError(Exception):
    pass


class StitchingFailedError(Exception):
    pass


class SalvageFailedError(Exception):
    pass


class VerticalScrolledFrame(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.viewport().setAutoFillBackground(False)
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.setWidget(self.content)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add_image(self, image_data):
        # Преобразуем массив numpy в изображение PIL
        image = Image.fromarray(image_data)

        # Преобразуем изображение в объект QImage
        image_data = image.tobytes("raw", "RGB")
        qimage = QImage(
            image_data, image.size[0], image.size[1], QImage.Format_RGB888)

        # Convert QImage to a supported format
        qimage = qimage.convertToFormat(QImage.Format_RGB888)

        # Создаем объект QPixmap из QImage
        pixmap = QPixmap.fromImage(qimage)

        # Создаем метку с изображением
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        # Добавляем метку в вертикальный макет
        self.layout.addWidget(label)


def show_image(header, image):
    """
    Displays an image in a window with the specified header.

    Args:
        header (str): Header for the image window.
        image (numpy.ndarray): Image array to be displayed.
    """
    # print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()


def write_image(directory, image):
    """
    Saves an image to the specified directory.

    Args:
        directory (str): Path to the directory where the image will be saved.
        image (numpy.ndarray): Image array to be saved.
    """
    # print("[Console] Saving image")
    cv2.imwrite(directory, image)


def get_images(directory):
    """
    Loads images from the specified directory.

    Args:
        directory (str): Path to the directory containing the images.

    Returns:
        List of loaded images (list of numpy.ndarray).

    Raises:
        InvalidDirectoryError: If the directory is invalid or no images are found.
    """
    try:
        print("[Console] Accessing folder")
        image_paths = glob.glob(directory)
        print(image_paths)
        if len(image_paths) == 0:
            raise InvalidDirectoryError(
                "[ERROR] Invalid directory or no images found.")
        images = []
        # Add images to memory
        print("[Console] Loading Images")
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Unable to load image: {image_path}")
            else:
                images.append(image)
        print(f"[INFO] Loaded {len(images)} image(s)")
        return images
    except Exception as e:
        print(str(e))
        raise e


def get_gray_image(image):
    """
    Converts an RGB image to grayscale.

    Args:
        image (numpy.ndarray): RGB image array.

    Returns:
        Grayscale image (numpy.ndarray).
    """
    image_uint8 = cv2.convertScaleAbs(image)
    gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    return gray_image

def draw_matched_points(image1, image2, keypoints1, keypoints2, matches):
    """
    Draws matched keypoints between two images.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        keypoints1 (list): Keypoints in the first image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): Matched keypoints between the first and second images.

    Returns:
        Image with matched keypoints drawn.
    """
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def save_and_show_matched_points_images(image1, image2, keypoints1, keypoints2, matches, output_path):
    """
    Saves and displays images with matched keypoints drawn.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        keypoints1 (list): Keypoints in the first image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): Matched keypoints between the first and second images.
        output_path (str): Path to save the images.
    """
    matched_image = draw_matched_points(
        image1, image2, keypoints1, keypoints2, matches)
    write_image(output_path, matched_image)
    show_image("Matched Points", matched_image)


def get_matcher(_try_cuda, _matcher_type, features, _match_conf, _range_width):
    try_cuda = _try_cuda
    matcher_type = _matcher_type
    if _match_conf is None:
        if features == 'ORB':
            match_conf = 0.3
        else:
            match_conf = 0.65
    else:
        match_conf = _match_conf
    range_width = _range_width
    if matcher_type == "affine":
        matcher = cv2.detail_AffineBestOf2NearestMatcher(
            False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv2.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    else:
        matcher = cv2.detail_BestOf2NearestRangeMatcher(
            range_width, try_cuda, match_conf)
    return matcher


def get_compensator(_expos_comp_type, _expos_comp_nr_feeds, _expos_comp_block_size):
    expos_comp_type = EXPOS_COMP_CHOICES[_expos_comp_type]
    expos_comp_nr_feeds = _expos_comp_nr_feeds
    expos_comp_block_size = _expos_comp_block_size
    # expos_comp_nr_filtering = args.expos_comp_nr_filtering
    if expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS:
        compensator = cv2.detail_ChannelsCompensator(expos_comp_nr_feeds)
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif expos_comp_type == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
        compensator = cv2.detail_BlocksChannelsCompensator(
            expos_comp_block_size, expos_comp_block_size,
            expos_comp_nr_feeds
        )
        # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv2.detail.ExposureCompensator_createDefault(
            expos_comp_type)
    return compensator


def get_keypoints_and_matches_ORB(image1, image2):
    """
    Finds keypoints and matches between two images.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        Keypoints and matches between the two images.
    """
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Нахождение соответствующих точек
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Отбор сильных соответствий с помощью теста соотношения Лоу
    good = []
    for match_pair in matches:
        m, n = match_pair
        if m.distance < 0.8 * n.distance:
            good.append(m)

    return keypoints1, keypoints2, good

def get_keypoints_and_matches_AKAZE(image1, image2):
    """
    Finds keypoints and matches between two images using AKAZE algorithm.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        Keypoints and matches between the two images.
    """
    akaze = cv2.AKAZE_create()
    keypoints1, descriptors1 = akaze.detectAndCompute(image1, None)
    keypoints2, descriptors2 = akaze.detectAndCompute(image2, None)

    # Create a brute force matcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches


def get_keypoints_and_matches_SIFT(image1, image2):
    """
    Finds keypoints and matches between two images using SIFT algorithm.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        Keypoints and matches between the two images.
    """
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Create a brute force matcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches


def get_keypoints_and_matches_SURF(image1, image2):
    """
    Finds keypoints and matches between two images using SURF algorithm.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        Keypoints and matches between the two images.
    """
    surf = cv2.xfeatures2d.SURF_create()
    keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(image2, None)

    # Create a brute force matcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def get_keypoints_and_matches_BRISK(image1, image2):
    """
    Finds keypoints and matches between two images using BRISK algorithm.

    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.

    Returns:
        Keypoints and matches between the two images.
    """
    brisk = cv2.BRISK_create()
    keypoints1, descriptors1 = brisk.detectAndCompute(image1, None)
    keypoints2, descriptors2 = brisk.detectAndCompute(image2, None)

    # Create a brute force matcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches


def get_threshold_image(gray_image):  # выравнивание яркости???
    """
    Applies thresholding to a grayscale image.

    Args:
        gray_image (numpy.ndarray): Grayscale image array.

    Returns:
        Thresholded image (numpy.ndarray).
    """
    # print("[Console] Thresholding image")
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]


def get_image_2D_dim(image):
    """
    Returns the dimensions (height and width) of an image.

    Args:
        image (numpy.ndarray): Image array.

    Returns:
        Tuple containing the height and width of the image.
    """
    return image.shape[:2]


def get_mask_image(image):
    """
    Creates a mask image by converting the input image to grayscale, applying thresholding, and blurring.

    Args:
        image (numpy.ndarray): Image array.

    Returns:
        Mask image (numpy.ndarray).
    """
    # print("[Console] Masking image")
    gray_image = get_gray_image(image)
    # Threshold + Blur + Threshold = Remove all the random black pixel in the white part of the first threshold
    threshold_image = get_threshold_image(gray_image)
    threshold_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)
    threshold_image = get_threshold_image(threshold_image)
    return threshold_image


def crop_image(image, factor):
    """
    Crops an image inward proportionally based on the specified factor.

    Args:
        image (numpy.ndarray): Image array to be cropped.
        factor (float): Proportion of the image to be cropped (0.0 to 1.0).

    Returns:
        Tuple containing the crop location (lower height, upper height, left width, right width)
        and the cropped image (numpy.ndarray).
    """
    (h, w) = get_image_2D_dim(image)
    # Crop horizontally (width)
    amount_crop = w * (1 - factor)
    w_right = int(w - amount_crop // 2)
    w_left = int(amount_crop // 2)
    # Crop vertically (height)
    amount_crop = h * (1 - factor)
    h_upper = int(h - amount_crop // 2)
    h_lower = int(amount_crop // 2)
    return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]


def is_black_ver_line(image, start_h, end_h, w):
    """
    Checks if there is a black pixel in a straight vertical line within the specified image region.

    Args:
        image (numpy.ndarray): Image array.
        start_h (int): Starting height of the region.
        end_h (int): Ending height of the region.
        w (int): Width of the vertical line.

    Returns:
        True if a black pixel is found, False otherwise.
    """
    for value in range(start_h, end_h):
        if all(image[value, w] == [0, 0, 0]):
            return True
    return False


def is_black_hor_line(image, start_w, end_w, h):
    """
    Checks if there is a black pixel in a straight horizontal line within the specified image region.

    Args:
        image (numpy.ndarray): Image array.
        start_w (int): Starting width of the region.
        end_w (int): Ending width of the region.
        h (int): Height of the horizontal line.

    Returns:
        True if a black pixel is found, False otherwise.
    """
    for value in range(start_w, end_w):
        if all(image[h, value] == [0, 0, 0]):
            return True
    return False


def is_black_pixel_outline(threshold_image):
    """
    Checks if there are black pixels on the four sides of the thresholded image.

    Args:
        threshold_image (numpy.ndarray): Thresholded image array.

    Returns:
        True if black pixels are found on the outline, False otherwise.
    """
    (height, width) = get_image_2D_dim(threshold_image)
    # Lower side (0, w)
    if is_black_hor_line(threshold_image, 0, width, 0):
        return True
    # Upper side (h, w)
    if is_black_hor_line(threshold_image, 0, width, height - 1):
        return True
    # Left side (h, 0)
    if is_black_ver_line(threshold_image, 0, height, 0):
        return True
    # Right side (h, w)
    if is_black_ver_line(threshold_image, 0, height, width - 1):
        return True
    return False


def expand_from_crop_image(image, crop_location):
    """
    Expands the cropped image by searching for the nearest black pixels on each side.

    Args:
        image (numpy.ndarray): Image array to be expanded.
        crop_location (tuple): Crop location (lower height, upper height, left width, right width).

    Returns:
        Tuple containing the expanded location and the expanded image (numpy.ndarray).
    """
    # print("[Console] Salvaging usable cropped portions")
    since = time.time()
    height, width = get_image_2D_dim(image)
    h_lower, h_upper, w_left, w_right = crop_location
    mask_img = get_mask_image(image)
    # Left side (h, 0)
    for w in range(w_left, -1, -1):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_left = w + 5
            break
    # Right side (h, w)
    for w in range(w_right, width):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_right = w - 5
            break
    # Lower side (0, w)
    for h in range(h_lower, -1, -1):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_lower = h + 5
            break
    # Upper side (w, 0)
    for h in range(h_upper, height):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_upper = h - 5
            break
    if crop_location is not (h_lower, h_upper, w_left, w_right):
        elapsed = time.time() - since
        # print(f"[INFO] Salvaging usable image portion success in {
        #       elapsed:2f}s")
        return (h_lower, h_upper, w_left, w_right), image[
            h_lower:h_upper, w_left:w_right
        ]
    else:
        print("[INFO] Salvage failed")
        return (None, None)


def remove_black_outline(image):
    """
    Crops the image inward proportionally until all the black pixel outlines are removed.

    Args:
        image (numpy.ndarray): Image array to be processed.

    Returns:
        Tuple containing the crop location and the cropped image (numpy.ndarray).
    """
    # print("Обрезание изображения")
    since = time.time()
    mask = get_mask_image(image)
    # Cropping image
    is_cropped = False
    for crop_factor in range(100, -1, -1):
        crop_factor = 0.01 * crop_factor
        trial_mask = crop_image(mask, crop_factor)[1]
        if not is_black_pixel_outline(trial_mask):
            # print(f"[Console] Crop image with factor of {crop_factor}")
            is_cropped = True
            break
    elapsed = time.time() - since
    # Showing result
    if is_cropped:
        # print(f"Обрезание успешно выполнено за {elapsed:2f}s")
        return crop_image(image, crop_factor)
    else:
        QMessageBox.warning("Ошибка", "Неподходящее изображение для обрезания")
        return None


def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32(
        [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate(
        (list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [
                             0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(
        img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1],
               translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img


def crop_pano(pano_image):
    # Определяем границы панорамного изображения
    gray = cv2.cvtColor(pano_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    # Обрезаем панорамное изображение
    cropped_pano = pano_image[y:y+h, x:x+w]
    return cropped_pano


def numpy_to_qpixmap(numpy_img):
    height, width, channel = numpy_img.shape
    channel_count = numpy_img.shape[2]
    bytes_per_line = channel_count * width
    # Преобразуем BGR изображение в RGB
    numpy_img_rgb = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    # Преобразуем изображение numpy в формат QImage
    qimage = QImage(numpy_img_rgb.data.tobytes(), width, height,
                    bytes_per_line, QImage.Format_RGB888)
    # Преобразуем QImage в QPixmap
    qpixmap = QPixmap.fromImage(qimage)

    return qpixmap


def save_pixmap_as_image(pixmap, file_path):
    """
    Сохраняет QPixmap в файл в формате изображения.

    Args:
        pixmap (QPixmap): Изображение в формате QPixmap.
        file_path (str): Путь к файлу, в который будет сохранено изображение.
    """
    # Преобразование QPixmap обратно в изображение numpy
    image_np = pixmap.toImage()
    image_np.save(file_path)  # Сохранение изображения в файл


stitching_algorithms = {
    "SURF": get_keypoints_and_matches_SURF,
    "ORB": get_keypoints_and_matches_ORB,
    "SIFT": get_keypoints_and_matches_SIFT,
    "BRISK": get_keypoints_and_matches_BRISK,
    "AKAZE": get_keypoints_and_matches_AKAZE   
}


def numpy_array_to_qpixmap(numpy_img):
    """
    Конвертирует массив NumPy обратно в формат QPixmap.
    """
    height, width, channel = numpy_img.shape
    bytes_per_line = channel * width
    qimage = QImage(numpy_img.data, width, height,
                    bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


def qpixmap_to_cv2(qpixmap):
    """
    Конвертирует QPixmap обратно в сырое изображение OpenCV.
    """
    # Преобразуем QPixmap в формат QImage
    qimage = qpixmap.toImage()
    # Получаем параметры изображения
    width = qimage.width()
    height = qimage.height()
    # Создаем буфер для данных изображения
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    qimage.save(buffer, "BMP")
    # Читаем данные изображения из буфера
    data = buffer.data()
    # Проверяем, соответствует ли размер данных ожидаемому размеру изображения
    expected_size = width * height * 3  # RGBA format
    if len(data) != expected_size:
        raise ValueError(f"Unexpected size of image data: {
                         len(data)}, expected: {expected_size}")
    # Создаем массив numpy из данных
    arr = np.frombuffer(data).reshape(
        height, width, 3)  # 4 channels: RGBA
    # Преобразуем изображение в формат BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def cv2_to_qpixmap(cv_image):
    """
    Конвертирует изображение из формата OpenCV в формат QPixmap.
    """
    height, width, channel = cv_image.shape
    bytes_per_line = 3 * width
    q_image = QImage(cv_image.data, width, height,
                     bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)


def apply_adjustments(image, brightness, contrast, clarity, shadows, exposure, temperature):
    # Применяем преобразования к изображению
    # Например, можно использовать методы из библиотеки OpenCV для редактирования изображения
    adjusted_image = image.copy()
    # Применяем преобразования
    # ...
    return adjusted_image


class ImageEditorWindow(QMainWindow):
    # Определяем сигнал для передачи измененного изображения
    image_changed = pyqtSignal(QPixmap)

    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Image Editor")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Устанавливаем фиксированный размер окна
        self.setFixedWidth(800)

        # Сохраняем изображение как атрибут класса
        self.image = image
        self.original_image = image.copy()
        self.modified_image = image.copy()

        # Создаем QLabel для отображения изображения
        self.image_label = QLabel()
        self.image_label.setPixmap(image)

        # Создаем кнопку для сохранения изменений
        self.save_button = QPushButton("Сохранить изменения")
        self.save_button.clicked.connect(self.save_changes)

        # Создаем кнопку для сброса изменений
        self.reset_button = QPushButton("Сбросить изменения")
        self.reset_button.clicked.connect(self.reset_changes)

        # Создаем горизонтальный макет для кнопок
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_button)

        # Добавляем кнопки в основной макет
        self.layout.addLayout(button_layout)

        # Создаем элементы управления для редактирования изображения
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.clarity_slider = QSlider(Qt.Horizontal)
        self.shadows_slider = QSlider(Qt.Horizontal)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.temperature_slider = QSlider(Qt.Horizontal)

        # Создаем макет для размещения элементов управления
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("Яркость"))
        control_layout.addWidget(self.brightness_slider)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Контрастность"))
        control_layout.addWidget(self.contrast_slider)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Резкость"))
        control_layout.addWidget(self.clarity_slider)
        self.clarity_slider.setMinimum(-100)
        self.clarity_slider.setMaximum(100)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Яркость теней"))
        control_layout.addWidget(self.shadows_slider)
        self.shadows_slider.setMinimum(-100)
        self.shadows_slider.setMaximum(100)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Экспозиция"))
        control_layout.addWidget(self.exposure_slider)
        self.exposure_slider.setMinimum(-100)
        self.exposure_slider.setMaximum(100)

        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Температура"))
        control_layout.addWidget(self.temperature_slider)
        self.temperature_slider.setMinimum(-100)
        self.temperature_slider.setMaximum(100)


        control_layout.addWidget(self.save_button)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.reset_button)

        # Растягиваем вертикальный макет, чтобы убрать пустое пространство внизу
        control_layout.addStretch(1)

        # Создаем основной виджет и устанавливаем макет
        editor_widget = QWidget()
        editor_layout = QHBoxLayout(editor_widget)
        editor_layout.addWidget(self.image_label)
        editor_layout.addLayout(control_layout)

        # Устанавливаем основной виджет в окно редактора
        self.setCentralWidget(editor_widget)

        # Подключаем сигналы valueChanged от каждого ползунка к слотам для обновления меток
        self.brightness_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                self.clarity_slider.value(),
                value,
                self.contrast_slider.value(),
                self.exposure_slider.value(),
                self.shadows_slider.value(),
                self.temperature_slider.value()
            ))
        self.contrast_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                self.clarity_slider.value(),
                self.brightness_slider.value(),
                value,
                self.exposure_slider.value(),
                self.shadows_slider.value(),
                self.temperature_slider.value()
            )
        )
        self.clarity_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                value,
                self.brightness_slider.value(),
                self.contrast_slider.value(),
                self.exposure_slider.value(),
                self.shadows_slider.value(),
                self.temperature_slider.value()
            )
        )
        self.shadows_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                self.clarity_slider.value(),
                self.brightness_slider.value(),
                self.contrast_slider.value(),
                self.exposure_slider.value(),
                value,
                self.temperature_slider.value()
            )
        )
        self.exposure_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                self.clarity_slider.value(),
                self.brightness_slider.value(),
                self.contrast_slider.value(),
                value,
                self.shadows_slider.value(),
                self.temperature_slider.value()
            )
        )
        self.temperature_slider.valueChanged.connect(
            lambda value: self.apply_filters(
                self.clarity_slider.value(),
                self.brightness_slider.value(),
                self.contrast_slider.value(),
                self.exposure_slider.value(),
                self.shadows_slider.value(),
                value
            )
        )

    def save_changes(self):
        # Проверяем, есть ли обработчик событий для измененного изображения
        if self.image_changed is not None:
            # После сохранения изменений отправляем новое изображение через сигнал
            pixmap = self.image_label.pixmap()
            self.image_changed.emit(pixmap)
        else:
            print("Ошибка: обработчик событий для измененного изображения не установлен.")
        # Закрываем окно
        self.close()

    def reset_changes(self):
        # Сбрасываем изменения, возвращаем изображение к первоначальному виду
        self.image_label.setPixmap(self.original_image)

        # Сбрасываем положение всех ползунков к исходным значениям
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.clarity_slider.setValue(0)
        self.shadows_slider.setValue(0)
        self.exposure_slider.setValue(0)
        self.temperature_slider.setValue(0)

    def qpixmap_to_cv2(self, qpixmap):
        qimage = qpixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.constBits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # 4 channels: RGBA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def cv2_to_qpixmap(self, cv_image):
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def apply_filters(self, clarity_value, brightness_value, contrast_value, exposure_value, shadows_value, temperature_value):
        cv_image = self.qpixmap_to_cv2(self.image)

        # Применяем каждый фильтр к изображению поочередно
        cv_image = self.apply_clarity(clarity_value, cv_image)
        cv_image = self.qpixmap_to_cv2(cv_image)
        cv_image = self.apply_brightness(brightness_value, cv_image)
        cv_image = self.qpixmap_to_cv2(cv_image)
        cv_image = self.apply_contrast(contrast_value, cv_image)
        cv_image = self.qpixmap_to_cv2(cv_image)
        cv_image = self.apply_exposure(exposure_value, cv_image)
        cv_image = self.qpixmap_to_cv2(cv_image)
        cv_image = self.apply_shadows_brightness(shadows_value, cv_image)
        cv_image = self.qpixmap_to_cv2(cv_image)
        cv_image = self.apply_temperature(temperature_value, cv_image)

        updated_image = cv_image
        self.image_label.setPixmap(updated_image)

    def apply_clarity(self, value, img):
        # Преобразование значения ползунка в диапазон [1, 101]
        kernel_size = int(value / 2) + 51

        # Проверка на нечетность
        if kernel_size % 2 == 0:
            kernel_size += 1

        # cv_image = self.qpixmap_to_cv2(self.image)
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return self.cv2_to_qpixmap(sharpened)

    def apply_brightness(self, value, img):
        # Конвертируем изображение в формат OpenCV
        # cv_image = self.qpixmap_to_cv2(self.image)
        # Применяем коррекцию яркости
        corrected_image = cv2.convertScaleAbs(img, beta=value)
        # Преобразуем изображение обратно в формат QPixmap и возвращаем его
        return self.cv2_to_qpixmap(corrected_image)

    def apply_contrast(self, value, img):
        # Конвертируем изображение в формат OpenCV
        # cv_image = self.qpixmap_to_cv2(self.image)

        # Конвертируем изображение в цветовое пространство HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Получаем текущее значение насыщенности изображения
        current_saturation = hsv_image[..., 1]

        # Рассчитываем изменение насыщенности относительно базового значения
        delta = current_saturation * (value / 100)

        # Применяем изменение насыщенности
        adjusted_saturation = current_saturation + delta

        # Ограничиваем значения насыщенности от 0 до 255
        adjusted_saturation = np.clip(adjusted_saturation, 0, 255)

        # Обновляем значение насыщенности в изображении
        hsv_image[..., 1] = adjusted_saturation.astype(np.uint8)

        # Конвертируем обратно в формат BGR
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return self.cv2_to_qpixmap(bgr_image)

    def apply_exposure(self, value, img):
        # Конвертируем изображение в формат OpenCV
        # cv_image = self.qpixmap_to_cv2(self.image)

        # Преобразовываем изображение в цветовое пространство LAB
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Разделяем канал яркости
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Применяем коррекцию экспозиции к каналу яркости
        exposure_corrected_l_channel = np.clip(
            l_channel * (1 + value / 100), 0, 255).astype(np.uint8)

        # Объединяем обновленный канал яркости с остальными каналами
        lab_corrected_image = cv2.merge(
            (exposure_corrected_l_channel, a_channel, b_channel))

        # Конвертируем обратно в формат BGR
        bgr_corrected_image = cv2.cvtColor(
            lab_corrected_image, cv2.COLOR_LAB2BGR)

        return self.cv2_to_qpixmap(bgr_corrected_image)

    def apply_shadows_brightness(self, value, img):
        # Конвертируем изображение в формат OpenCV
        # cv_image = self.qpixmap_to_cv2(self.image)

        # Преобразуем изображение в оттенки серого
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Создаем маску для темных областей
        _, mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

        # Применяем коррекцию яркости только к темным областям
        adjusted_image = img.copy().astype(np.int16)  # Преобразуем к int16
        adjusted_image[mask == 0] += value
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(
            np.uint8)  # Обрезаем значения и преобразуем к uint8

        return self.cv2_to_qpixmap(adjusted_image)

    def apply_temperature(self, value, img):
        # Конвертируем изображение в формат OpenCV
        # cv_image = self.qpixmap_to_cv2(self.image)

        # Создаем копию изображения
        result = img.copy()

        # Вычисляем уровень коррекции для каждого канала
        level = value / 2

        # Применяем уровень коррекции к каналам R и G и пропорционально каналу B
        result[..., 2] = np.clip(result[..., 2] - level, 0, 255)  # Канал R
        result[..., 1] = np.clip(result[..., 1] - level * 0.5, 0, 255)  # Канал G
        result[..., 0] = np.clip(result[..., 0] + level * 0.5, 0, 255)  # Канал B

        # Преобразуем обработанное изображение обратно в QPixmap и возвращаем
        return self.cv2_to_qpixmap(result)

class App(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Panorama App")

        # image_path = "./_internal/imageplaceholder.png"
        # pixmap = QPixmap(image_path)
        # self.resultImg.setPixmap(pixmap)

        self.counter = False
        
        self.editor_window = None
        self.current_label = None
        self.image_labels = []
        self.analyticDock.setVisible(False)
        self.dockWidgetContents_2.setLayout(self.dock_layout)

        self.label.setVisible(False)
        self.comboBox.setVisible(False)
        self.label_2.setVisible(False)
        self.match_conf.setVisible(False)
        self.label_3.setVisible(False)
        self.conf_thresh.setVisible(False)
        self.label_4.setVisible(False)
        self.wave_correct.setVisible(False)
        self.label_5.setVisible(False)
        self.warp.setVisible(False)

        self.preset1.triggered.connect(self.preset_choose)
        self.preset2.triggered.connect(self.preset_choose)
        self.preset3.triggered.connect(self.preset_choose)
        self.preset4.triggered.connect(self.preset_choose)
        self.preset5.triggered.connect(self.preset_choose)
        self.presetDefault.triggered.connect(self.preset_choose)

        # Создаем вертикальный прокручиваемый фрейм
        self.scrolled_frame = VerticalScrolledFrame()
        self.verticalLayout.addWidget(self.scrolled_frame)

        # Подключаем сигнал кнопки к слоту
        self.runButton.clicked.connect(self.stitch_images)

        # Подключаем пункт меню для загрузки изображений
        self.loadImages.triggered.connect(self.load_images)

        # Подключаем пункт меню для очистки загруженных снимков
        self.clearImages.triggered.connect(self.clear_images)

        # Подключаем пункт меню сохранения изображения
        self.saveImg.triggered.connect(self.save_Img)

        # Включить режим редактирования
        self.editEnable.triggered.connect(self.toggle_edit_mode)

        # Включить режим отладки
        self.debugEnable.triggered.connect(self.toggle_debug_mode)

        # Включить аналитический режим
        self.analyticEnable.triggered.connect(self.toggle_analytic_mode)

        # Включить ручной режим
        self.manualEnable.triggered.connect(self.toggle_manual_mode)

        # Выход из приложения
        self.exit.triggered.connect(self.exitApp)

    def update_image(self, pixmap):

        self.current_label.setPixmap(pixmap)

    def label_clicked(self, label):
        """
        Обработчик события клика мыши на QLabel.
        """
        if self.editEnable.isChecked():
            pixmap = label.pixmap()
            if pixmap is not None:
                self.current_label = label  # Сохраняем ссылку на метку
                self.open_editor_window(pixmap)

    def open_editor_window(self, pixmap):
        """
        Открывает окно редактора для заданного изображения.
        """
        if self.editor_window is None or not self.editor_window.isVisible():
            self.editor_window = ImageEditorWindow(pixmap)
            self.editor_window.image_changed.connect(self.update_image)
            self.editor_window.show()

    def exitApp(self):
        QApplication.quit()

    def load_images(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.jpg *.jpeg *.png *.bmp)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            for file_path in file_paths:
                # Считываем изображение с помощью OpenCV
                cv_image = cv2.imread(file_path)
                if cv_image is not None:
                    # Добавляем изображение в вертикальную прокручиваемую область
                    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    self.scrolled_frame.add_image(cv_image_rgb)
                else:
                    print(f"Failed to read image from {file_path}")

        for i in range(self.scrolled_frame.layout.count()):
            widget = self.scrolled_frame.layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widget.mousePressEvent = lambda event, label=widget: self.label_clicked(
                    label)

    def toggle_edit_mode(self):
        """
        Включает или выключает режим редактирования в зависимости от текущего состояния элемента меню.
        """
        # Получаем текущее состояние элемента меню
        edit_enabled = self.editEnable.isChecked()

        # Изменяем режим редактирования в зависимости от текущего состояния
        if edit_enabled:
            # Включаем режим редактирования
            QMessageBox.information(
                self, 'Edit Mode', 'Включен режим редактирования')
        else:
            # Выключаем режим редактирования
            QMessageBox.information(
                self, 'Edit Mode', 'Режим редактирования выключен')
            if self.editor_window is not None and self.editor_window.isVisible():
                self.editor_window.close()

    def toggle_debug_mode(self):
        debug_enabled = self.debugEnable.isChecked()

        if debug_enabled:
            QMessageBox.information(
                self, 'Debug Mode', 'Включен режим отладки')
        else:
            QMessageBox.information(
                self, 'Debug Mode', 'Режим отладки выключен')
            
    def toggle_analytic_mode(self):
        analytic_enabled = self.analyticEnable.isChecked()

        if analytic_enabled:
            self.analyticDock.setVisible(True)
            QMessageBox.information(
                self, 'Analytic Mode', 'Включен аналитический режим')
        else:
            self.analyticDock.setVisible(False)
            QMessageBox.information(
                self, 'Analytic Mode', 'Аналитический режим выключен')
            
    def toggle_manual_mode(self):
        manual_mode_enabled = self.manualEnable.isChecked()
        if manual_mode_enabled:
            self.label.setVisible(True)
            self.comboBox.setVisible(True)
            self.label_2.setVisible(True)
            self.match_conf.setVisible(True)
            self.label_3.setVisible(True)
            self.conf_thresh.setVisible(True)
            self.label_4.setVisible(True)
            self.wave_correct.setVisible(True)
            self.label_5.setVisible(True)
            self.warp.setVisible(True)
            QMessageBox.information(
                self, 'Manual Mode', 'Включен ручной режим')
        else:
            self.analyticDock.setVisible(False)
            QMessageBox.information(
                self, 'Manual Mode', 'Ручной режим выключен')
            self.comboBox.setCurrentIndex(0)
            self.match_conf.setValue(0.65)
            self.conf_thresh.setValue(1.0)
            self.wave_correct.setCurrentIndex(0)
            self.warp.setCurrentIndex(0)

            self.label.setVisible(False)
            self.comboBox.setVisible(False)
            self.label_2.setVisible(False)
            self.match_conf.setVisible(False)
            self.label_3.setVisible(False)
            self.conf_thresh.setVisible(False)
            self.label_4.setVisible(False)
            self.wave_correct.setVisible(False)
            self.label_5.setVisible(False)
            self.warp.setVisible(False)

    def clear_images(self):
        # Удаляем все изображения из вертикального прокручиваемого фрейма
        for i in reversed(range(self.scrolled_frame.layout.count())):
            widget = self.scrolled_frame.layout.itemAt(i).widget()
            self.scrolled_frame.layout.removeWidget(widget)
            widget.deleteLater()

    def load_image_from_scroll_area(self, widget):
        """
        Loads an image from the scroll area widget.

        Args:
            widget (QWidget): Widget containing the image.

        Returns:
            numpy.ndarray: Image data in numpy array format.
        """
        # Get the QPixmap from the widget
        pixmap = widget.pixmap()

        # Convert the QPixmap to numpy format
        image_np_rgb = qimage2ndarray.byte_view(pixmap.toImage())

        image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGBA2RGB)

        return image_np_rgb

    def save_Img(self):
        """
        Сохраняет QPixmap в файл в формате изображения.

        Args:
            pixmap (QPixmap): Изображение в формате QPixmap.
            file_path (str): Путь к файлу, в который будет сохранено изображение.
        """
        # Получаем текущий QPixmap из QLabel
        pixmap = self.resultImg.pixmap()

        # # Проверяем, существует ли QPixmap
        # if pixmap is None or pixmap.toImage() == self.placeholder.toImage():
        #     QMessageBox.critical(
        #         self, "Ошибка", "Нет изображения для сохранения.")
        #     return

        if self.counter == True:
            # Открываем диалоговое окно сохранения файла
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить изображение", "", "Изображения (*.png *.jpg *.bmp)")
        else:
            QMessageBox.critical(
                self, "Ошибка", "Нет изображения для сохранения.")
            return
        # Проверяем, был ли выбран файл
        if file_path:
            try:
                # Вызываем метод сохранения pixmap как изображения
                save_pixmap_as_image(pixmap, file_path)
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Ошибка при сохранении изображения: {str(e)}")
            else:
                QMessageBox.information(self, 'Уведомление', 'Изображение успешно сохранено')

    def preset_choose(self):
        sender = self.sender()
        if isinstance(sender, QAction):  # Проверяем, что отправитель - QAction
            if sender.objectName() == 'preset1':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(0)
                self.warp.setCurrentIndex(0)
            elif sender.objectName() == 'preset2':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(1)
                self.warp.setCurrentIndex(0)
            elif sender.objectName() == 'preset3':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(0)
                self.warp.setCurrentIndex(6)
            elif sender.objectName() == 'preset4':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(0)
                self.warp.setCurrentIndex(4)
            elif sender.objectName() == 'preset5':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(0)
                self.warp.setCurrentIndex(2)
            elif sender.objectName() == 'presetDefault':
                self.comboBox.setCurrentIndex(0)
                self.match_conf.setValue(0.65)
                self.conf_thresh.setValue(1.0)
                self.wave_correct.setCurrentIndex(0)
                self.warp.setCurrentIndex(0)

    def stitch_alg(self):
        OUT_PATH = "./output"  # Путь к каталогу для сохранения результатов
        CROP = self.crop.isChecked()
        # Получаем значения параметров из интерфейса
        selected_algorithm = self.comboBox.currentText()
        match_conf = self.match_conf.value()
        conf_thresh = self.conf_thresh.value()
        selected_wave_correct = self.wave_correct.currentText()
        if selected_wave_correct == "Горизонтальная":
            wave_correct = 'horiz'
        elif selected_wave_correct == "Вертикальная":
            wave_correct = 'vert'
        elif selected_wave_correct == "Выключить":
            wave_correct = 'no'
        warp_type = self.warp.currentText()
        if warp_type == "Сферическая":
            warp_type = 'spherical'
        elif warp_type == "Цилиндрическая":
            warp_type = "cylindrical"
        elif warp_type == "Аффинная":
            warp_type = "affine"
        elif warp_type == "Рыбий глаз":
            warp_type = "fisheye"
        elif warp_type == "Стереографическая":
            warp_type = "stereographic"
        elif warp_type == "paniniA1.5B1":
            warp_type = "paniniA1.5B1"
        elif warp_type == "paniniA2B1":
            warp_type = "paniniA2B1"

        wave_correct = WAVE_CORRECT_CHOICES[wave_correct]
        work_megapix = 0.6
        seam_megapix = 0.1
        compose_megapix = -1
        ba_refine_mask = 'xxxxx'
        save_graph = None
        if save_graph is None:
            save_graph = False
        else:
            save_graph = True
        blend_type = 'multiband'
        blend_strength = 5
        _timelapse = None
        finder = FEATURES_FIND_CHOICES[selected_algorithm]()
        seam_work_aspect = 1
        matcher_type = 'homography'
        estimator_type = 'homography'
        rangewidth = -1
        ba = 'ray'
        seam = 'gc_color'
        save_graph = None
        full_img_sizes = []
        features = []
        img_names = []
        expos_comp_type = 'gain_blocks'
        expos_comp_nr_feeds = 1
        expos_comp_block_size = 32
        try_cuda = False
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        
        if _timelapse is not None:
            timelapse = True
            if _timelapse == "as_is":
                timelapse_type = cv2.detail.Timelapser_AS_IS
            elif _timelapse == "crop":
                timelapse_type = cv2.detail.Timelapser_CROP
            else:
                QMessageBox.warning(self, "Ошибка", "Неподходящий timelapse метод.")
        else:
            timelapse = False

        images = []
       
        # Итерируем по меткам с изображениями в прокручиваемой области
        for i in range(self.scrolled_frame.layout.count()):
            widget = self.scrolled_frame.layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                pixmap = widget.pixmap()
                if pixmap is not None:
                    # Загружаем изображение из виджета прокрутки
                    image_np_rgb = self.load_image_from_scroll_area(widget)
                    images.append(image_np_rgb)

        if len(images) < 2:
            QMessageBox.information(
                self, 'Ошибка', 'Для создания панорамы необходимо как минимум 2 изображения.')
            return

        # Проверяем, выбран ли допустимый алгоритм сшивки
        if selected_algorithm not in stitching_algorithms:
            QMessageBox.information(
                self, 'Ошибка', 'Выбранный алгоритм не поддерживается.')
            return

        for i in range(len(images) - 1):
            image1 = images[i]
            image2 = images[i + 1]

            # Получаем keypoints и matches используя выбранный алгоритм
            keypoints1, keypoints2, matches = stitching_algorithms[selected_algorithm](image1, image2)
        
        if self.debugEnable.isChecked():
            for i in range(len(images) - 1):
                image1 = images[i]
                image2 = images[i + 1]

                # Получаем keypoints и matches используя выбранный алгоритм
                keypoints1, keypoints2, matches = stitching_algorithms[selected_algorithm](image1, image2)        
                save_and_show_matched_points_images(image1, image2, keypoints1, keypoints2, matches, f"{OUT_PATH}/match_{i}_{i+1}.jpg")

        main_since = time.time() #замеряем начало
        if self.analyticEnable.isChecked():
            self.analyticOutput.clear()
            for i in range(len(images) - 1):
                image1_index = i + 1
                image2_index = i + 2
                image1 = images[i]
                image2 = images[i + 1]

                # Получаем keypoints и matches используя выбранный алгоритм
                keypoints1, keypoints2, matches = stitching_algorithms[selected_algorithm](image1, image2)
                info_str = f"Количество точек на изображении №{image1_index}: {len(keypoints1)}\n"
                info_str += f"Количество точек на изображении №{image2_index}: {len(keypoints2)}\n"
                info_str += f"Количество отобранных хороших особых точек: {len(matches)}\n\n"
                
                # Добавляем информацию в QPlainTextEdit
                self.analyticOutput.insertPlainText(info_str)
       
        processing_time = time.time()
        resized_images = []  # Список для хранения измененных изображений
        for image in images:
            full_img = image
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(
                        work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv2.resize(src=full_img, dsize=None, fx=work_scale,
                                fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                if seam_megapix > 0:
                    seam_scale = min(1.0, np.sqrt(
                        seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                else:
                    seam_scale = 1.0
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat = cv2.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            resized_img = cv2.resize(src=full_img, dsize=None, fx=seam_scale,
                                    fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
            resized_images.append(resized_img)
        processing_time_elapsed  = time.time() - processing_time

        if self.analyticEnable.isChecked():
            self.analyticOutput.insertPlainText(f"Время обработки изображений: {processing_time_elapsed:.3f} секунд\n")

        match_time = time.time()
        matcher = get_matcher(try_cuda, matcher_type, finder, match_conf, rangewidth)
        p = matcher.apply2(features)
        matcher.collectGarbage()

        match_time_elapsed = time.time() - match_time
        if self.analyticEnable.isChecked():
            self.analyticOutput.insertPlainText(f"Время вычисления особых точек: {match_time_elapsed:.3f} секунд\n")

        indices = cv2.detail.leaveBiggestComponent(features, p, conf_thresh)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(images[indices[i]])
            img_subset.append(resized_images[indices[i]])
            full_img_sizes_subset.append(full_img_sizes[indices[i]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(img_names)
        if num_images < 2:
            QMessageBox.warning(self, "Ошибка", "Требуется больше изображений.")

        homography_time = time.time()
        estimator = ESTIMATOR_CHOICES[estimator_type]()
        try:
            b, cameras = estimator.apply(features, p, None)
            if not b:
                QMessageBox.warning(self, "Ошибка", "Ошибка при создании гомографии.")
                return
        except Exception as e:
            return

        homography_time_elapsed = time.time() - homography_time
        # if self.analyticEnable.isChecked():
        #     self.analyticOutput.insertPlainText(f"Время создания гомографии методом RANSAC: {homography_time_elapsed:.3f} секунд\n")
        # else:
        #     print(f"Время создания гомографии методом RANSAC: {homography_time_elapsed:.3f} секунд")

        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = BA_COST_CHOICES[ba]()
        adjuster.setConfThresh(conf_thresh)
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjustment_time = time.time()
        adjuster.setRefinementMask(refine_mask)

        adjustment_time_elapsed = time.time() - adjustment_time
        # print(f"Время настройки параметров камеры: {adjustment_time_elapsed:.3f} секунд")

        try:
            b, cameras = adjuster.apply(features, p, cameras)
            if not b:
                QMessageBox.warning(self, "Ошибка", "Не удалось настроить параметры камеры.")
        except Exception as e:
            return

        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (
                focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        if wave_correct is not None:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv2.detail.waveCorrect(rmats, wave_correct)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv2.UMat(
                255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        # warper could be nullptr?
        warper = cv2.PyRotationWarper(
            warp_type, warped_image_scale * seam_work_aspect)
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(
                images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(
                masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        compensator_time = time.time()
        compensator = get_compensator(expos_comp_type, expos_comp_nr_feeds, expos_comp_block_size)
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
        compensator_elapsed = time.time() - compensator_time
        if self.analyticEnable.isChecked():
            self.analyticOutput.insertPlainText(f"Время создания компенсатора экспозиции: {compensator_elapsed:.3f} секунд\n")

        seam_finder_time = time.time()
        seam_finder = SEAM_FIND_CHOICES[seam]
        masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
        seam_finder_elapsed = time.time() - seam_finder_time
        if self.analyticEnable.isChecked():
            self.analyticOutput.insertPlainText(f"Время работы алгоритма RANSAC: {seam_finder_elapsed:.3f} секунд\n")

        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = None
        
        Rotation_Warper_elapsed = 0
        blender_time_elapsed = 0
        for idx, image in enumerate(img_names):
            full_img = image
            Rotation_Warper_time = time.time()
            if not is_compose_scale_set:
                if compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(
                        compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                        int(round(full_img_sizes[i][1] * compose_scale)))
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv2.INTER_LINEAR_EXACT)
            else:
                img = full_img
            Rotation_Warper_elapsed += time.time() - Rotation_Warper_time

            _img_size = (img.shape[1], img.shape[0])
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(
                img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(
                mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv2.dilate(masks_warped[idx], None)
            seam_mask = cv2.resize(
                dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
            mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
            blender_time = time.time()
            if blender is None and not timelapse:
                blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
                dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                if blend_width < 1:
                    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv2.detail_MultiBandBlender()
                    blender.setNumBands(
                        (np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
                elif blend_type == "feather":
                    blender = cv2.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            elif timelapser is None and timelapse:
                timelapser = cv2.detail.Timelapser_createDefault(timelapse_type)
                timelapser.initialize(corners, sizes)
            if timelapse:
                ma_tones = np.ones(
                    (image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                timelapser.process(image_warped_s, ma_tones, corners[idx])
                fixed_file_name = f"fixed_image_{idx}.jpg"

                cv2.imwrite(fixed_file_name, timelapser.getDst())
            else:
                blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
            blender_time_elapsed += time.time() - blender_time
            Rotation_Warper_elapsed += 0.0001
        if self.analyticEnable.isChecked():
            self.analyticOutput.insertPlainText(f"Время перспективной трансформации изображений с учетом вращения: {Rotation_Warper_elapsed:.6f} секунд\n")
            self.analyticOutput.insertPlainText(f"Время смешивания изображений: {blender_time_elapsed:.3f} секунд\n")

        if not timelapse:
            result = None
            result_mask = None
            result, result_mask = blender.blend(result, result_mask)
            zoom_x = 600.0 / result.shape[1]
            dst = cv2.normalize(src=result, dst=None, alpha=255.,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)

            if CROP:
                crop_location, cropped_image = remove_black_outline(result)
                expand_location, expanded_img = expand_from_crop_image(result, crop_location)
                main_elapsed = time.time() - main_since
                if expanded_img is not None:
                    if self.analyticEnable.isChecked():
                        self.analyticOutput.insertPlainText(f"Выполнено за {main_elapsed:2f}s\n")
                    write_image(OUT_PATH + "/pano.jpg", expanded_img)
                    zoom_x = 1171.0 / expanded_img.shape[1]
                    dst = cv2.normalize(src=expanded_img, dst=None, alpha=255.,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
                else:
                    QMessageBox.warning(self, "Ошибка", "Не удалось выполнить обрезку")
            else:
                main_elapsed = time.time() - main_since
                if self.analyticEnable.isChecked():
                    self.analyticOutput.insertPlainText(f"Выполнено за {main_elapsed:2f}s\n")
                write_image(OUT_PATH + "/pano.jpg", result)
                zoom_x = 5000.0 / result.shape[1]
                dst = cv2.normalize(src=result, dst=None, alpha=255.,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        self.counter = True
        return dst

    def stitch_images(self):
        panorama = self.stitch_alg()
        if panorama is not None:
            panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            q_pixmap = cv2_to_qpixmap(panorama_rgb)
            self.resultImg.setPixmap(q_pixmap)
        else:
            QMessageBox.information(
                self, 'Ошибка', 'Не удалось создать панораму.')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
