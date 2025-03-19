import PyInstaller.__main__

PyInstaller.__main__.run([
    "main.py",
    "--onefile",
    "--hidden-import=tensorflow",
    "--hidden-import=tensorflow.python",
    "--hidden-import=tensorflow_core",
    "--hidden-import=tensorflow_core.python",
    "--hidden-import=tensorflow._api.v2",
    "--add-data=pages;pages",
    "--add-data=static;static",
    "--add-data=config.py;.",
    "--add-data=model.py;.",
    "--name=WeatherNet"
])
