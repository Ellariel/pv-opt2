python -m nuitka gui.py --standalone --include-package=openpyxl,tqdm,msgpack,ray,setproctitle --nofollow-import-to=IPython
xcopy consumption gui.dist\consumption /i /e /y
xcopy production gui.dist\production /i /e /y
xcopy solution gui.dist\solution /i /e /y
xcopy static gui.dist\static /i /e /y
xcopy templates gui.dist\templates /i /e /y
xcopy /y components.pickle gui.dist
xcopy /y components.xlsx gui.dist
xcopy /y cache.pickle gui.dist
xcopy /y pvgis_cache.faas gui.dist