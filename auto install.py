import importlib
import subprocess
import sys

# package_name : module_name
REQUIRED = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "optuna": "optuna",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "psutil": "psutil",
    "chardet": "chardet",
    # 若使用 GPU 功能需安裝下列套件
    "cupy": "cupy",
    "cudf": "cudf",
}

for pkg, module in REQUIRED.items():
    try:
        importlib.import_module(module)
    except ImportError:
        print(f"Installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

print("All packages are available.")
