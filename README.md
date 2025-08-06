# CoderPush-Human-Detection
## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/minhtuan-ne/CoderPush-Human-Detection
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

### 3. Install FFmpeg

#### Ubuntu

Open a terminal and run:

```bash
sudo apt update
sudo apt install ffmpeg
````

To verify installation:

```bash
ffmpeg -version
```

---

#### macOS (Homebrew)

If you haven't installed Homebrew yet, first install it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install FFmpeg:

```bash
brew install ffmpeg
```

To verify installation:

```bash
ffmpeg -version
```


### 4. Install dependencies

```bash
pip install -r requirements/requirements.txt
```

### 5. Set up environment variables

Copy the example environment file:

```bash
cp .env/.env.example .env/.env
```

Then open `.env/.env` and fill in the actual values for your configuration.


### 6. Run the server

```bash
python src/api/app.py
```

> App runs at: `http://localhost:7860`

