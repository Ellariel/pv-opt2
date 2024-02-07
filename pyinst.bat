pyinstaller gui.py --noconfirm --console --hidden-import "msgpack" --hidden-import "setproctitle" --hidden-import "ray._private.memory_monitor" --additional-hooks-dir=hooks --add-data="./templates;./templates" --add-data="./static;./static"
xcopy consumption dist\gui\consumption /i /e /y
xcopy production dist\gui\production /i /e /y
xcopy solution dist\gui\solution /i /e /y
xcopy static dist\gui\static /i /e /y
xcopy templates dist\gui\templates /i /e /y
xcopy /y components.pickle dist\gui\
xcopy /y components.xlsx dist\gui\
xcopy /y cache.pickle dist\gui\
xcopy /y pvgis_cache.faas dist\gui\