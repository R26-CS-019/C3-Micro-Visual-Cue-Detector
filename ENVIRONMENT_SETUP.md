# Recovered Python environment

The project environment is `.venv`, created with Python 3.12.10.
Dependencies were identified by scanning all 44 Python source files. The four
TensorFlow, NumPy and Pillow versions come from `research_suite/requirements.txt`.
Keras 3.15.1 matches the version recorded in both saved model archives; the
research suite's older Keras 3.13.2 pin fails direct loading in training scripts.
The root
`requirements.txt` also includes training plots/metrics, image hashing, browser
capture and the FastAPI server. `requirements-lock.txt` records every installed
package version.

Activate in PowerShell:

```powershell
cd 'C:\Users\oshan\Desktop\full  pp2 module'
.\.venv\Scripts\Activate.ps1
```

In VS Code, use **Python: Select Interpreter**, then select
`C:\Users\oshan\Desktop\full  pp2 module\.venv\Scripts\python.exe`.
Activation is optional; the executable can be used directly:

```powershell
& '.\.venv\Scripts\python.exe' --version
.\research_suite\run.cmd --help
```

To recreate later:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
& '.\.venv\Scripts\python.exe' -m playwright install chromium
```

`research_suite/run.cmd` defaults to this environment; `PP2_PYTHON` can override
it. Some original training and integration scripts contain absolute paths to
the previous Desktop folders. Review those paths before running them from this
combined project folder. Native Windows TensorFlow 2.20 uses CPU; GPU training
requires a separate supported setup such as WSL2.
