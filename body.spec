# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['body.py'],
    pathex=[],
    binaries=[],
    datas=[('/home/yashraj/YashrajTiwari/proxie-studio/python/env/lib/python3.12/site-packages/streamlit/static', 'streamlit/static'), ('/home/yashraj/YashrajTiwari/proxie-studio/python/env/lib/python3.12/site-packages/mediapipe/python/mediapipe/', 'mediapipe/'), ('/home/yashraj/YashrajTiwari/proxie-studio/python/env/lib/python3.12/site-packages/opencv-python/opencv/', 'opencv/')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='body',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
