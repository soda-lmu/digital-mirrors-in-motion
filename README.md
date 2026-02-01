# Unveiling Digital Mirrors

Interactive tool for analyzing body pose patterns and gender classification from images and videos.

Based on research from the paper: [Unveiling Digital Mirrors: Decoding Gendered Body Poses in Instagram Imagery](https://doi.org/10.1016/j.chb.2024.108464) published in [Computers in Human Behavior](https://www.sciencedirect.com/journal/computers-in-human-behavior).

## 📁 Project Structure

```
├── tool/                    # Main application
│   ├── frontend/           # Web interface (HTML/CSS/JavaScript)
│   ├── src/                # Python backend & data processing
│   └── serve.py            # Development server
├── docs/                   # Documentation
│   ├── TOOL_DOCUMENTATION.md   # Complete tool guide
│   └── PAPER_DATA.md           # Research data info
└── .gitignore             # Git configuration
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
cd tool
python serve.py
```

Then open `http://localhost:8000` in your browser.

## 📖 Documentation

- **[TOOL_DOCUMENTATION.md](docs/TOOL_DOCUMENTATION.md)** - Complete guide on how to use the tool and understand the analysis logic
- **[PAPER_DATA.md](docs/PAPER_DATA.md)** - Information about the research dataset and original analysis

## 🎯 Features

- **Image Analysis**: Upload images to analyze pose patterns and gender classification
- **Video Analysis**: Upload videos to analyze frame-by-frame pose patterns with gender distribution
- **Interactive Visualization**: View pose skeletons and match patterns
- **Gender Classification**: Based on 150 pose cluster prototypes from research

## 📊 How It Works

The tool uses K-means clustering on normalized body poses to classify movement patterns as masculine, feminine, or non-binary. Each frame or image is matched against the closest prototype cluster, with gender percentages calculated from the prototype distributions.

For detailed technical explanation, see [TOOL_DOCUMENTATION.md](docs/TOOL_DOCUMENTATION.md).
