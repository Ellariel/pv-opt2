pyinstaller gui.py --noconfirm --console --hidden-import "msgpack" --hidden-import "setproctitle" --hidden-import "ray._private.memory_monitor" --additional-hooks-dir=hooks --add-data="./templates;./templates" --add-data="./static;./static"
xcopy consumption dist\consumption /i /e /y
xcopy production dist\production /i /e /y
xcopy solution dist\solution /i /e /y
xcopy static dist\static /i /e /y
xcopy templates dist\templates /i /e /y
xcopy /y components.pickle dist\
xcopy /y components.xlsx dist\
xcopy /y cache.pickle dist\
xcopy /y pvgis_cache.faas dist\