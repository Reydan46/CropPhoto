import argparse
import os
import sys
import matplotlib
import matplotlib.pyplot as mpplt
from cropphoto import CropPhoto, logger
import tkinter as tk
from tkinter import filedialog

sys.stdout.reconfigure(encoding='utf-8')


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()


# Инициализация парсера
parser = MyParser()
parser.add_argument(nargs='+', type=str, action='store', dest='img_path',
                    help='Путь до изображения(ний)')
parser.add_argument('-p', type=bool, action=argparse.BooleanOptionalAction, dest='plt_show',
                    help='Показать панель отладки с просмотром этапов')
parser.add_argument('-z', type=bool, action=argparse.BooleanOptionalAction, dest='zoom',
                    help='Подгонять изображения к нужному выходному размеру')
parser.add_argument('-c', type=bool, action=argparse.BooleanOptionalAction, dest='correct_position',
                    help='Выводить запрос на корректировку координат лица')
parser.add_argument('-r', type=bool, action=argparse.BooleanOptionalAction, dest='rotate',
                    help='Поворачивать изображения при НЕнахождении лица')
parser.add_argument('-v', type=bool, action=argparse.BooleanOptionalAction, dest='view',
                    help='Показать только панель с отладкой (без сохранения результатов)')
parser.add_argument('-sf', type=float, action='store', dest='scale_factor', default=1.28,
                    help='Изменить scale factor поиска лица (по умолчанию 1.28)')
parser.add_argument('-zf', type=bool, action=argparse.BooleanOptionalAction, dest='zoom_face',
                    help='Изменить масштаб нахождения лица (по умолчанию - нет)')

# Отключение кнопок панели отладки
matplotlib.rcParams['toolbar'] = 'None'

plt_custom = None


def run(input_crop_photo,
        img_filepath: str = '',
        zoom: bool = False,
        plt_show: bool = False,
        rotate: bool = False,
        view: bool = False,
        correct_position: bool = False,
        scale_factor: float = 1.28,
        zoom_face: bool = False,
        input_out_size_img_w: int = 648,
        input_out_size_img_h: int = 648,
        input_out_size_img_1c_w: int = 100,
        input_out_size_img_1c_h: int = 130,
        input_face_size_img_w: int = 365,
        input_face_size_img_h: int = 365,
        ):
    global plt_custom

    if zoom_face:
        zoom_face_float = float(input('Enter face zoom:'))
        input_face_size_img_h = int(input_face_size_img_h * zoom_face_float)
        input_face_size_img_w = int(input_face_size_img_w * zoom_face_float)

    if plt_show or view:
        # Создание subplots для вывода изображений (оригинал, поворот, лицо, ресайз, лицо после ресайза, кроп)
        logger.info('Create subplots')
        if rotate:
            num_plots = 6
            fig, (
                plt_orig, plt_rotate, plt_resize, plt_crop, plt_output_ad, plt_output_1c
            ) = mpplt.subplots(1, num_plots, figsize=((257 * num_plots) // 72, 300 // 72), dpi=72,
                               subplot_kw={'aspect': 1})
        else:
            num_plots = 5
            fig, (
                plt_orig, plt_resize, plt_crop, plt_output_ad, plt_output_1c
            ) = mpplt.subplots(1, num_plots, figsize=((257 * num_plots) // 72, 300 // 72), dpi=72,
                               subplot_kw={'aspect': 1})
            plt_rotate = None
        fig.canvas.manager.set_window_title("CropPhoto")

    else:
        plt_orig, plt_rotate, plt_resize, plt_crop, plt_output_ad, plt_output_1c = None, None, None, None, None, None

    # Загрузка оригинального изображения
    logger.info(f'Load image: {img_filepath}')
    input_crop_photo.load_img_file(img_filepath)

    # Ограничение размера изображения
    logger.info('Check and resize image (limit)')
    input_crop_photo.resize_limit()

    if plt_show or view:
        # Отображение исходного изображения на plot
        logger.info('Show original image in plot')
        input_crop_photo.show_in_plt(plt_orig, "Original")

    # Поиск лица на изображении
    logger.info('Find faces')
    if input_crop_photo.find_faces(rotate=rotate, scale_factor=scale_factor) > 0:
        if plt_show or view:
            if rotate:
                # Отображение повёрнутого изображения на plot
                logger.info('Show rotated image in plot')
                input_crop_photo.show_in_plt(plt_rotate, "Rotate")
                # Выделение лица на plot
                logger.info('Mark face in rotated image')
                input_crop_photo.mark_face(plt_rotate)
            else:
                # Выделение лица на plot
                logger.info('Mark face in original image')
                input_crop_photo.mark_face(plt_orig)

        # Вычисление масштабирования изображения (по эталонному размеру лица)
        logger.info('Get scale face')
        input_crop_photo.get_scale_face(input_face_size_img_w, input_face_size_img_h)

        # Изменение размера изображения по вычесленному масштабированию
        logger.info('Resize image with scale')
        input_crop_photo.scaling(input_crop_photo.scale_face)
        if plt_show or view:
            # Отображение изображения после масштабирования на plot
            logger.info('Show resized image in plot')
            input_crop_photo.show_in_plt(plt_resize, "Resize")
        # Поиск лица на изображении
        logger.info('Find faces')
        if input_crop_photo.find_faces(rotate=rotate, scale_factor=scale_factor) > 0:
            if correct_position:
                input_crop_photo.correct_position_face()
            if plt_show or view:
                # Выделение лица на plot
                logger.info('Mark face in resized image')
                input_crop_photo.mark_face(plt_resize)
            # Вычисление нужной рамки вокруг лица
            logger.info('Get brim size for face')
            input_crop_photo.get_brim(input_out_size_img_w, input_out_size_img_h)
            # Обрезание изображения с вычисленной рамкой
            logger.info('Crop resized image with brim')
            input_crop_photo.crop_image_with_brim()
            if plt_show or view:
                # Отображение изображения после обрезки на plot
                logger.info('Show crop image in plot')
                input_crop_photo.show_in_plt(plt_crop, "Crop")
                # Выделение лица на plot
                logger.info('Mark face in crop image')
                input_crop_photo.mark_face_with_brim(plt_crop)
            # Изменение размера изображения к указанному или возврат к нормальному разрешению
            if zoom:
                logger.info('Zoom crop image')
                input_crop_photo.resize(input_out_size_img_w, input_out_size_img_h)
            else:
                # Если изображение было уменьшено, то предотвращаем его увеличение
                if input_crop_photo.scale_face > 1:
                    logger.info('Restore original scale')
                    input_crop_photo.scaling((1 / input_crop_photo.scale_face))
            if plt_show or view:
                # Отображение выходного изображения AD
                logger.info('Show AD image in plot')
                input_crop_photo.show_in_plt(plt_output_ad, "AD")
            if not view:
                # Сохранение полученного изображения для AD
                logger.info('Save AD image in file')
                input_crop_photo.save_image_to_file(f"{os.path.splitext(img_filepath)[0]}_AD.jpg")
            # Обрезка фото для 1С
            logger.info('Resize AD image for 1C')
            input_crop_photo.resize(max(input_out_size_img_1c_w, input_out_size_img_1c_h),
                                    max(input_out_size_img_1c_w, input_out_size_img_1c_h))
            logger.info('Crop AD image for 1C')
            input_crop_photo.crop_image_to_1c(input_out_size_img_1c_w, input_out_size_img_1c_h)
            if plt_show or view:
                # Отображение выходного изображения 1C
                logger.info('Show 1C image in plot')
                input_crop_photo.show_in_plt(plt_output_1c, "1C")
            if not view:
                # Сохранение полученного изображения для 1С
                logger.info('Save 1C image in file')
                input_crop_photo.save_image_to_file(f"{os.path.splitext(img_filepath)[0]}_1C.png", filetype="PNG",
                                                    quality=100)
        else:
            logger.error('Faces not found!')
    else:
        logger.error('Faces not found!')
    if plt_show or view:
        logger.info('Show plots')
        mpplt.ioff()
        mpplt.show(block=True)


if __name__ == "__main__":
    # Размер выходного изображения
    out_size_img_w = 648
    out_size_img_h = 648
    out_size_img_1c_w = 100
    out_size_img_1c_h = 130
    # Размер лица, для приведения размера фотографии к общему виду (эталонный размер)
    face_size_img_w = 365
    face_size_img_h = 365
    face_size_img_w = 340
    face_size_img_h = 340

    args = parser.parse_args()

    files = []
    ext = ['.bmp', '.gif', '.jp2', '.jpg', '.jpeg', '.pcx', '.png', '.ppm', '.tga']
    if args.img_path:
        for item in args.img_path:
            if os.path.isfile(item) and os.path.splitext(item)[-1].lower() in ext:
                files += [item]
            elif os.path.isdir(item):
                for subitem in os.listdir(item):
                    file = os.path.join(item, subitem)
                    if os.path.isfile(file) and os.path.splitext(file)[-1].lower() in ext:
                        files += [file]
    else:
        root = tk.Tk()
        root.withdraw()
        filetypes = (
            ('Изображения', ['*' + i for i in ext]),
        )
        img_paths = filedialog.askopenfilename(
            title='Выберите фотографии...',
            filetypes=filetypes,
            multiple=True,
        )
        root.quit()
        root.destroy()
        if img_paths:
            for file in img_paths:
                files += [file]

    crop_photo = CropPhoto(
        face_cascade_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_alt.xml"),
    )

    for file in files:
        try:
            run(crop_photo, file, args.zoom, args.plt_show, args.rotate, args.view, args.correct_position,
                args.scale_factor, args.zoom_face, out_size_img_w, out_size_img_h, out_size_img_1c_w, out_size_img_1c_h,
                face_size_img_w, face_size_img_h)
        except:
            logger.error(f'Failed to process image: {file}')
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fail.txt"), 'a').write(f'{file}\n')
