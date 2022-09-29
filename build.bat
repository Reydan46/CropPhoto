pyinstaller --upx-dir=upx --onefile --add-data "haarcascade_frontalface_alt.xml;." --icon=cropphoto.ico -n CropPhoto -y -c --log-level=WARN main.py
