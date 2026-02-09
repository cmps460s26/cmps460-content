# CMPS 460 Python Environment Setup

The recommended way to run notebooks for this course is using **Google Colab in VS Code**.

## Option 1: Google Colab in VS Code (Recommended)

This option gives you the power of cloud computing ☁️ with the convenience of your local editor 💻.

1.  **Install Extensions:** In VS Code, install the following extensions:
    *   **Jupyter** (by Microsoft)
    *   **Google Colab** (by Google)
2.  **Open Notebook:** Open your `.ipynb` file.
3.  **Connect:** Click **Select Kernel** (top-right of the notebook editor) > **Colab** > **Select Google Account** (sign in if prompted).

---

## Option 2: Google Colab on the Web

You can also run notebooks directly in your browser without VS Code.

1.  Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Upload** your notebook (`.ipynb` file).
3.  **Install Libraries:** Libraries are pre-installed in Colab. If you need to install libraries, add and run this code cell at the top of your notebook:
    ```python
    !pip install -r https://raw.githubusercontent.com/cmps460s26/cmps460-content/refs/heads/main/examples/env-setup/requirements.txt
    ```

---

## Option 3: Local Python Installation

If you prefer running everything locally on your machine, follow these steps:

### 1. Install Python
Download and install the latest Python from [python.org](https://www.python.org/downloads/).
*   **Important:** Check the box **"Add Python to path"** during installation.

### 2. Setup Environment
Open your terminal (PowerShell/Command Prompt/Terminal) in the folder containing `requirements.txt` and run:

```bash
# 1. Create Virtual Environment
python -m venv cmps460-env       # Windows
python3 -m venv cmps460-env      # macOS/Linux

# 2. Activate Environment
cmps460-env\Scripts\activate     # Windows
source cmps460-env/bin/activate  # macOS/Linux

# 3. Install Libraries
pip install -r requirements.txt
```

### 3. Run Notebooks
*   **VS Code:** Install the **Jupyter** extension. Open a `.ipynb` file, click **Select Kernel** > **Python Environments** > `cmps460-env`.
*   **Browser:** Run `jupyter lab` in your activated terminal.

---

## Updating Libraries
To update your environment with the latest course requirements, run:
```bash
pip install --upgrade -r requirements.txt
```
