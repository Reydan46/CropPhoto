import cv2
import matplotlib.pyplot as mpplt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
from time import time
import warnings
from PIL import Image
from PIL import ImageOps

import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)

c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter("[%(asctime)s - %(funcName)20s() ] %(message)s", datefmt='%d.%m.%Y %H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# logger.debug('logger.debug')
# logger.info('logger.info')
# logger.warning('logger.warning')
# logger.error('logger.error')
# logger.exception('logger.exception')

warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class CoordXYWH:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def is_null(self):
        if self.x == 0 and self.y == 0 and self.w == 0 and self.h == 0:
            return True
        else:
            return False

    def set_coord(self, coord):
        self.x, self.y, self.w, self.h = coord


class CropPhoto:
    def __init__(self, face_cascade_path: str = ''):
        self.img = np.array([])
        self.face_cascade = None
        self.eye_cascade = None
        self.face = CoordXYWH()
        self.title = ''
        self.scale_img = 1
        self.scale_face = 1
        self.brim = 0

        if face_cascade_path:
            self.load_face_cascade_file(face_cascade_path)

    def __init__params(self):
        self.img = np.array([])
        self.face = CoordXYWH()
        self.title = ''
        self.scale_img = 1
        self.scale_face = 1
        self.brim = 0

    def load_img_file(self, img_path: str):
        t = time()
        if img_path and os.path.exists(img_path):
            self.__init__params()
            img = Image.open(img_path)

            if hasattr(ImageOps, 'exif_transpose'):
                # Very recent versions of PIL can do exit transpose internally
                img = ImageOps.exif_transpose(img)
            else:
                # Otherwise, do the exif transpose ourselves
                img = self.exif_transpose(img)

            img = img.convert('RGB')

            self.img = np.array(img)

            # self.img = mpplt.imread(img_path)
        else:
            logger.error(f'Image file not found!')
        logger.debug(f'Time: {round(time() - t, 4)}')

    @staticmethod
    def load_cascade_file(cascade_path: str):
        t = time()
        if os.path.exists(cascade_path):
            return cv2.CascadeClassifier(cascade_path)
        else:
            logger.error(f'Cascade file not found!')
        logger.debug(f'Time: {round(time() - t, 4)}')

    def load_face_cascade_file(self, face_cascade_path: str):
        self.face_cascade = self.load_cascade_file(face_cascade_path)

    def set_title(self, title):
        self.title = title

    def set_image(self, img):
        self.__init__params()
        self.img = img

    def get_image(self):
        return self.img

    def get_height(self, input_img=None):
        # Return height
        if input_img is not None:
            img = input_img
        else:
            img = self.img

        if img.size > 0:
            return img.shape[0]
        else:
            return 0

    def get_width(self, input_img=None):
        # Return width
        if input_img is not None:
            img = input_img
        else:
            img = self.img

        if img.size > 0:
            return img.shape[1]
        else:
            return 0

    def get_size(self, input_img=None):
        # Return size (width,height)
        if input_img is not None:
            img = input_img
        else:
            img = self.img

        if img.size > 0:
            return self.get_width(img), self.get_height(img)
        else:
            return 0, 0

    def exif_transpose(self, img):
        if not img:
            return img

        exif_orientation_tag = 274

        # Check for EXIF data (only present on some files)
        if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
            exif_data = img._getexif()
            orientation = exif_data[exif_orientation_tag]

            # Handle EXIF Orientation
            if orientation == 1:
                # Normal image - nothing to do!
                pass
            elif orientation == 2:
                # Mirrored left to right
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotated 180 degrees
                img = img.rotate(180)
            elif orientation == 4:
                # Mirrored top to bottom
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                # Mirrored along top-left diagonal
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                # Rotated 90 degrees
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                # Mirrored along top-right diagonal
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                # Rotated 270 degrees
                img = img.rotate(90, expand=True)

        return img

    def __find_faces(self, scale_factor: float, min_neighbors: int, input_img=None):
        t = time()
        if input_img is not None:
            img = input_img
        else:
            img = self.img

        faces = []
        if img.size == 0:
            logger.error(f'Image is not set!')
        elif not self.face_cascade:
            logger.error(f'Face cascade is not set!')
        else:
            try:
                faces = self.face_cascade.detectMultiScale(img, scale_factor, min_neighbors,
                                                           cv2.CASCADE_DO_ROUGH_SEARCH)
            except cv2.error:
                logger.exception(f'Find Face error!')

        logger.debug(f'Time: {round(time() - t, 4)}')

        return faces

    def find_faces(self, rotate: bool = True, scale_factor: float = 1.25, min_neighbors: int = 1):
        t = time()
        faces = self.__find_faces(scale_factor, min_neighbors)

        if (len(faces) == 0) and rotate:
            angles = []
            for i in range(1, 16):
                angles += [-i, i]
            for angle in angles:
                img = self.rotate_angle(angle, np.array(self.img, copy=True))
                faces = self.__find_faces(scale_factor, min_neighbors, img)
                if len(faces) != 0:
                    self.img = img
                    break

        if len(faces) != 0:
            self.face.set_coord(faces[0])
            logger.info(f'Face ({self.title}): X {self.face.x} Y {self.face.y} W {self.face.w} H {self.face.h}')
        else:
            logger.error(f'Face is not set!')

        logger.debug(f'Time: {round(time() - t, 4)}')

        return len(faces)

    def get_scale(self, etalon_size_w: int, etalon_size_h: int, size_w: int, size_h: int, max_size_w: int,
                  max_size_h: int):
        try:
            scale = 1 / (((etalon_size_w / size_w) + (etalon_size_h / size_h)) / 2)
            if (scale * self.get_width()) > max_size_w:
                scale = max_size_w / self.get_width()
            if (scale * self.get_height()) > max_size_h:
                scale = max_size_h / self.get_height()
        except:
            scale = 1
        return scale

    def get_scale_face(self, size_w: int = 365, size_h: int = 365, max_size_w: int = 12000, max_size_h: int = 8000):
        if not self.face.is_null():
            self.scale_face = self.get_scale(self.face.w, self.face.h, size_w, size_h, max_size_w, max_size_h)
            logger.info(f'Scale face: {self.scale_face}')
        else:
            logger.error(f'Face is not set!')

    def get_scale_img(self, size_w: int = 12000, size_h: int = 8000):
        if self.img.size > 0:
            self.get_scale(self.get_width(), self.get_height(), size_w, size_h)
        else:
            logger.error(f'Image is not set!')

    def get_brim(self, out_size_img_w: int, out_size_img_h: int):
        def min_or_null(x):
            if x > 0:
                return x
            else:
                return 0

        t = time()
        if self.face.is_null():
            logger.error(f'Face is not set!')
        elif self.img.size == 0:
            logger.error(f'Image is not set!')
        elif self.get_width() < (self.face.x + self.face.w) or \
                self.get_height() < (self.face.y + self.face.h) or \
                out_size_img_w < self.face.w or \
                out_size_img_h < self.face.h:
            logger.error(f'Bad size face!')
        else:
            self.brim = min(
                self.face.x, self.face.y,
                min_or_null(self.get_width() - self.face.x - self.face.w),
                min_or_null(self.get_height() - self.face.y - self.face.h),
                (out_size_img_w - self.face.w) // 2,
                (out_size_img_h - self.face.h) // 2
            )
            logger.info(f'Brim: {self.brim}')
        logger.debug(f'Time: {round(time() - t, 4)}')

    @staticmethod
    def mark_find(plot, coord: CoordXYWH):
        t = time()
        if not coord.is_null():
            plot.plot(
                [coord.x, coord.x, coord.x + coord.w, coord.x + coord.w, coord.x],
                [coord.y, coord.y + coord.h, coord.y + coord.h, coord.y, coord.y]
            )
        logger.debug(f'Time: {round(time() - t, 4)}')

    def mark_face(self, plot):
        if not self.face.is_null():
            self.mark_find(plot, self.face)
        else:
            logger.error(f'Face is not set!')

    def mark_face_with_brim(self, plot):
        if not self.face.is_null():
            face_with_brim = CoordXYWH()
            face_with_brim.x = self.brim
            face_with_brim.y = self.brim
            face_with_brim.h = self.face.h
            face_with_brim.w = self.face.w
            self.mark_find(plot, face_with_brim)
        else:
            logger.error(f'Face is not set!')

    def rotate_90(self):
        # Rotate Image (angle -90)
        t = time()
        if self.img.size > 0:
            logger.info(f'Rotate image to angle: -90')
            self.img = np.rot90(self.img)
        else:
            logger.error(f'Image is not set!')
        logger.debug(f'Time: {round(time() - t, 4)}')

    def rotate_angle(self, angle: int = 0, input_img=None):
        t = time()
        if input_img is not None:
            img = input_img
        else:
            img = self.img

        if img.size > 0:
            logger.info(f'Rotate image to angle: {angle}')
            matrix = cv2.getRotationMatrix2D((self.get_width(img) // 2, self.get_height(img) // 2), angle, 1)
            img = cv2.warpAffine(
                img,
                matrix,
                (self.get_width(img), self.get_height(img)),
                flags=cv2.INTER_LINEAR
            )
            if input_img is not None:
                self.img = img
        else:
            logger.error(f'Image is not set!')

        logger.debug(f'Time: {round(time() - t, 4)}')

        return img

    def show_in_plt(self, plot, title: str = '', show_title: bool = True, axis: bool = True):
        # Show Image in plot with Title
        t = time()
        if not plot:
            logger.error(f'Plot is not set!')
        if self.img.size == 0:
            logger.error(f'Image is not set!')
        else:
            if not axis:
                plot.axis('off')
            if title:
                self.set_title(title)
            if show_title:
                plot.set_title(f'{self.title}\n({self.get_width()}x{self.get_height()})')
            plot.imshow(self.img)
        logger.debug(f'Time: {round(time() - t, 4)}')

    def resize(self, width: int, height: int):
        t = time()
        if self.img.size > 0:
            logger.info(f'Resize image: {self.get_width()}x{self.get_height()} > {width}x{height}')
            self.img = cv2.resize(self.img, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            logger.error(f'Image is not set!')
        logger.debug(f'Time: {round(time() - t, 4)}')

    def resize_limit(self, max_width: int = 12000, max_height: int = 8000):
        width = self.get_width()
        height = self.get_height()
        if width > max_width:
            height = round(height / (width / max_width))
            width = max_width
        if height > max_height:
            width = round(width / (height / max_height))
            height = max_height
        if self.get_width() != width or self.get_height() != height:
            self.resize(width, height)

    def scaling(self, scale: float = 1):
        if self.img.size == 0:
            logger.error(f'Image is not set!')
        elif self.scale_face == 1:
            logger.error(f'Scale is 1 (original)')
        else:
            logger.info(f'Scaling image ({scale}) > Resize')
            self.resize(int(self.get_width() * scale), int(self.get_height() * scale))

    def crop(self, x_min: int, x_max: int, y_min: int, y_max: int):
        t = time()
        if self.img.size > 0:
            logger.info(f'Crop: X_min {x_min} Y_min {y_min} X_max {x_max} Y_max {y_max}')
            self.img = self.img[
                       y_min:y_max,
                       x_min:x_max
                       ]
        else:
            logger.error(f'Image is not set!')
        logger.debug(f'Time: {round(time() - t, 4)}')

    def crop_image_with_brim(self):
        if not self.face.is_null():
            self.crop(
                self.face.x - self.brim,
                self.face.x + self.face.w + self.brim,
                self.face.y - self.brim,
                self.face.y + self.face.h + self.brim,
            )
        else:
            logger.error(f'Face is not set!')

    def crop_image_to_1c(self, width: int, height: int):
        self.crop(
            (height - width) // 2,
            width + ((height - width) // 2),
            0,
            height,
        )

    def save_image_to_file(self, img_path: str = '', filetype: str = 'jpg', dpi: int = 72):
        t = time()
        if self.img.size > 0:
            mpimg.imsave(img_path, self.img, format=filetype, dpi=dpi)
        else:
            logger.error(f'Image is not set!')
        logger.debug(f'Time: {round(time() - t, 4)}')
