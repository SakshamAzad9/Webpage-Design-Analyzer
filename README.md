# Webpage-Design-Analyzer

# 🎨 Webpage Design Analyzer

A simple AI-powered webpage design analysis tool using LLaVA vision models through Ollama. Upload webpage screenshots and get comprehensive design feedback, suggestions, and analysis.

## ✨ Features

- **🎨 Design Style Analysis**: Identifies design patterns, aesthetics, and visual appeal  
- **🎯 Color Scheme Evaluation**: Analyzes color harmony, contrast, and accessibility  
- **📝 Typography Assessment**: Reviews font choices, hierarchy, and readability  
- **📐 Layout Analysis**: Evaluates spacing, alignment, and visual hierarchy  
- **⚠️ Error Detection**: Identifies design mistakes and usability issues  
- **💡 Smart Suggestions**: AI-powered improvement recommendations  
- **📊 Multiple Analysis Types**: Comprehensive, Quick Scan, Accessibility Focus  
- **🤖 Model Selection**: Support for different LLaVA model sizes  
- **💾 Export Results**: Download as TXT or JSON format  

## 🚀 Quick Setup

### Prerequisites

1. **Python 3.8+** installed on your system  
2. **Git** for cloning the repository  

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/webpage-design-analyzer.git
cd webpage-design-analyzer

# Create and activate virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install streamlit ollama pillow pandas python-dotenv requests numpy matplotlib seaborn
````

### Install Ollama

**Windows:**

* Download from [https://ollama.com/download/windows](https://ollama.com/download/windows)
* Run the installer

**macOS:**

```bash
brew install ollama
# OR download from https://ollama.com/download/macos
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama and pull LLaVA model

```bash
# Start Ollama service
ollama serve

# In a new terminal, pull LLaVA model (choose one)
ollama pull llava:7b    # Fastest, good for testing (~4GB)
ollama pull llava:13b   # Recommended balance (~7GB)
ollama pull llava:34b   # Best quality, requires more VRAM (~19GB)
```

### Run the application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

## 📁 Project Structure

```
webpage-design-analyzer/
├── app.py                # Main Streamlit application (single file)
├── README.md             # This file
├── requirements.txt      # Python dependencies (optional)
├── .env.example          # Environment variables example
├── screenshots/          # Sample screenshots (optional)
└── analysis_results/     # Saved results (auto-created)
```

### Single File Structure

The current implementation uses a **single-file approach** (`app.py`) that contains:

* **Frontend UI**: Streamlit interface with modern styling
* **Image Processing**: Upload, validation, and optimization
* **AI Analysis**: Integration with Ollama and LLaVA models
* **Results Display**: Structured JSON parsing and visualization
* **Export Features**: Download results as TXT/JSON
* **Settings Management**: Model selection and analysis types

## 🛠️ How It Works

### 1. Image Upload & Validation

* Upload PNG, JPG, JPEG, or WEBP files
* Automatic validation for size and format
* Image optimization for AI analysis

### 2. AI Analysis Process

```
Screenshot → Image Processing → LLaVA Model → Structured Analysis → Results Display
```

### 3. Analysis Types Available

#### **Comprehensive Analysis**

* Overall design score (1-10)
* Style classification
* Color scheme evaluation
* Typography assessment
* Layout structure analysis
* Error detection
* Improvement suggestions
* Accessibility issues

#### **Quick Scan**

* Quick overall score
* Top 3 strengths and issues
* Priority fixes
* Style categorization

#### **Accessibility Focus**

* WCAG compliance assessment
* Color contrast evaluation
* Text readability analysis
* Navigation accessibility
* Screen reader compatibility

## 🎯 Usage Guide

### Basic Usage

```bash
streamlit run app.py
```

Configure settings in the sidebar:

* Select LLaVA model
* Choose analysis type
* Customize prompt if needed

Upload a screenshot → Click **“🔍 Analyze Design”** → Review structured results → Export results as TXT or JSON.

### Sample Analysis Output

```json
{
  "overall_score": "8/10",
  "style": "Modern minimalistic design with professional aesthetics",
  "color_scheme": {
    "primary_colors": ["#2563eb", "#ffffff", "#f8fafc"],
    "contrast_rating": "9/10",
    "accessibility": "Excellent contrast ratios throughout"
  },
  "typography": {
    "font_hierarchy": "Clear hierarchy with proper size differentiation",
    "readability": "Excellent readability with sufficient line spacing",
    "consistency": "Consistent font usage across all elements"
  },
  "errors": ["Minor spacing inconsistency in navigation"],
  "improvements": [
    "Consider adding more visual emphasis to call-to-action buttons",
    "Increase padding around form elements for better mobile experience"
  ]
}
```

## 🔧 Configuration

Create a `.env` file for custom settings:

```env
# Default model to use
DEFAULT_MODEL=llava:13b

# Analysis timeout (seconds)
ANALYSIS_TIMEOUT=120

# Maximum image size (MB)
MAX_IMAGE_SIZE=10

# Enable debug mode
DEBUG_MODE=false
```

### Model Comparison

| Model     | Size   | Speed  | Quality | VRAM Required |
| --------- | ------ | ------ | ------- | ------------- |
| llava:7b  | \~4GB  | Fast   | Good    | \~8GB         |
| llava:13b | \~7GB  | Medium | Better  | \~16GB        |
| llava:34b | \~19GB | Slow   | Best    | \~32GB        |

## 🚨 Troubleshooting

#### **"ModuleNotFoundError: No module named 'ollama'"**

```bash
pip install ollama
```

#### **"Ollama not connected"**

```bash
ollama serve
curl http://localhost:11434/api/tags
```

#### **"No LLaVA models found"**

```bash
ollama pull llava:13b
ollama list
```

#### **"Analysis timeout" or slow performance**

* Try a smaller model (`llava:7b`)
* Reduce image size before upload
* Check available system RAM/VRAM

#### **"Image validation failed"**

* Ensure image format is PNG, JPG, JPEG, or WEBP
* Check file size (default max: 10MB)

### Performance Tips

* For faster analysis: Use `llava:7b`
* For better quality: Use `llava:13b` or `llava:34b`
* Optimize images: Resize to 1920x1080 or smaller
* 8GB+ RAM recommended

## 🔄 Updates & Upgrades

### Updating Dependencies

```bash
pip install --upgrade streamlit ollama pillow
```

### Updating Models

```bash
ollama pull llava:13b
ollama rm old_model_name
```

### Adding New Features

Modify `app.py`:

* **Custom prompts**: Edit the `get_prompt_templates()` function
* **New analysis types**: Add entries to the templates
* **UI changes**: Modify Streamlit components
* **Additional models**: Update model selection lists

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** to `app.py`
4. **Test thoroughly**
5. **Submit a pull request**

### Development Tips

* Test with multiple LLaVA models
* Validate with various image formats and sizes
* Ensure mobile responsiveness
* Check accessibility compliance

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* [Ollama](https://ollama.com) – Local AI model platform
* [LLaVA](https://llava-vl.github.io/) – Vision-language model
* [Streamlit](https://streamlit.io) – Web app framework
* [Pillow](https://pillow.readthedocs.io) – Image processing library

---

**🎨 Happy analyzing! Made with ❤️ for designers and developers**
