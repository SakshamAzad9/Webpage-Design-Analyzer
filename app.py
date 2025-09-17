import streamlit as st
from PIL import Image
import json
import logging
from datetime import datetime
import ollama
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Webpage Design Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .feature-card {
        background-color: #111924;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #ffe6e6;
        color: #d63384;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #d63384;
    }
    .success-message {
        background-color: #e6f7e6;
        color: #28a745;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def pil_image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """Convert PIL Image to bytes"""
    with io.BytesIO() as buffer:
        if format.upper() == 'JPEG' and image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            rgb_image.save(buffer, format=format, optimize=True)
        else:
            image.save(buffer, format=format, optimize=True)
        return buffer.getvalue()

def validate_image(image: Image.Image) -> dict:
    """Validate uploaded image"""
    validation_result = {
        'valid': True,
        'error': None,
        'warnings': []
    }
    
    try:
        width, height = image.size
        
        if width < 200 or height < 200:
            validation_result['warnings'].append("Image is quite small, analysis quality may be reduced")
        
        if width > 4000 or height > 4000:
            validation_result['warnings'].append("Large image detected, processing may take longer")
        
        if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
            validation_result['error'] = f"Unsupported image mode: {image.mode}"
            validation_result['valid'] = False
            return validation_result
            
    except Exception as e:
        validation_result['error'] = f"Validation failed: {str(e)}"
        validation_result['valid'] = False
        
    return validation_result

def resize_image_if_needed(image: Image.Image, max_size=(1920, 1080)) -> Image.Image:
    """Resize image if it's too large"""
    original_size = image.size
    max_width, max_height = max_size
    
    if original_size[0] <= max_width and original_size[1] <= max_height:
        return image
    
    ratio = min(max_width / original_size[0], max_height / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)

def get_image_info(image: Image.Image) -> dict:
    """Get image information"""
    try:
        image_bytes = pil_image_to_bytes(image)
        size_mb = len(image_bytes) / (1024 * 1024)
        
        return {
            'width': image.size[0],
            'height': image.size[1],
            'format': getattr(image, 'format', 'Unknown'),
            'size': f"{size_mb:.1f} MB" if size_mb >= 1 else f"{len(image_bytes) / 1024:.1f} KB"
        }
    except Exception as e:
        return {'error': str(e)}

def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        client = ollama.Client()
        models = client.list()
        return True, models.get('models', [])
    except Exception as e:
        return False, str(e)

def get_available_models():
    """Get available LLaVA models"""
    try:
        client = ollama.Client()
        models = client.list()
        llava_models = []
        
        for model in models.get('models', []):
            model_name = model.get('name', '')
            if 'llava' in model_name.lower():
                llava_models.append(model_name)
        
        return llava_models
    except Exception:
        return []

def analyze_webpage_image(image: Image.Image, prompt: str, model: str = "llava:13b") -> str:
    """Analyze webpage image using Ollama"""
    try:
        # Prepare image
        processed_image = resize_image_if_needed(image)
        image_bytes = pil_image_to_bytes(processed_image)
        
        # Generate response
        client = ollama.Client()
        full_response = ""
        
        for response in client.generate(
            model=model,
            prompt=prompt,
            images=[image_bytes],
            stream=True
        ):
            part = response.get("response", "")
            full_response += part
            
        return full_response.strip()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise e

def get_prompt_templates():
    """Get predefined prompt templates"""
    return {
        "Comprehensive Analysis": """
Analyze this webpage screenshot comprehensively and return results in JSON format:

{
  "overall_score": "score out of 10",
  "style": "design style description (e.g., minimalistic, professional, cyberpunk)",
  "color_scheme": {
    "primary_colors": ["color1", "color2"],
    "contrast_rating": "rating out of 10",
    "accessibility": "color accessibility assessment"
  },
  "typography": {
    "font_hierarchy": "assessment of font hierarchy",
    "readability": "readability score and issues",
    "consistency": "font consistency evaluation"
  },
  "layout": {
    "structure": "layout structure assessment",
    "spacing": "white space and padding evaluation",
    "alignment": "element alignment review"
  },
  "errors": ["list of design issues and problems"],
  "improvements": ["detailed improvement suggestions"],
  "accessibility_issues": ["accessibility concerns"]
}

Provide detailed, actionable feedback in each category.
""",
        
        "Quick Scan": """
Provide a quick analysis of this webpage screenshot in JSON format:

{
  "quick_score": "overall score out of 10",
  "main_strengths": ["top 3 design strengths"],
  "main_issues": ["top 3 design issues"],
  "priority_fixes": ["most important improvements needed"],
  "style_category": "design style classification"
}

Focus on the most important aspects only.
""",
        
        "Accessibility Focus": """
Analyze this webpage for accessibility and inclusive design in JSON format:

{
  "accessibility_score": "score out of 10",
  "color_contrast": "contrast ratio assessment",
  "text_readability": "text size and readability",
  "navigation_clarity": "navigation accessibility",
  "accessibility_violations": ["WCAG guideline violations"],
  "inclusive_design_suggestions": ["suggestions for better accessibility"]
}

Focus primarily on accessibility and inclusive design principles.
"""
    }

def initialize_session_state():
    """Initialize session state"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def display_header():
    """Display header"""
    st.markdown('<h1 class="main-header">🎨 Webpage Design Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1em;">AI-powered webpage screenshot analysis using LLaVA + Ollama</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎨 Design Analysis</h4>
            <p>Style, colors, and visual appeal assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📐 Layout Review</h4>
            <p>Typography, spacing, and hierarchy evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Smart Suggestions</h4>
            <p>AI-powered improvement recommendations</p>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar with settings"""
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Check Ollama connection
    connected, models_or_error = check_ollama_connection()
    
    if not connected:
        st.sidebar.error(f"❌ Ollama not connected: {models_or_error}")
        st.sidebar.info("Please start Ollama with: `ollama serve`")
        return None
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.sidebar.warning("⚠️ No LLaVA models found")
        st.sidebar.info("Pull a model with: `ollama pull llava:13b`")
        available_models = ["llava:13b"]  # Default fallback
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "🤖 Model Selection",
        available_models,
        help="Choose the LLaVA model for analysis"
    )
    
    # Analysis type
    analysis_types = {
        "Comprehensive Analysis": "Complete design analysis",
        "Quick Scan": "Fast overview of main issues", 
        "Accessibility Focus": "Focus on accessibility concerns"
    }
    
    analysis_type = st.sidebar.selectbox(
        "📊 Analysis Type",
        list(analysis_types.keys()),
        help="Select the type of analysis to perform"
    )
    
    # Get prompt templates
    prompt_templates = get_prompt_templates()
    custom_prompt = st.sidebar.text_area(
        "✏️ Custom Prompt",
        value=prompt_templates[analysis_type],
        height=200,
        help="Customize the analysis prompt"
    )
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.info("Temperature and max tokens are handled by Ollama defaults")
        show_raw_response = st.checkbox("Show Raw Response", False)
    
    return {
        'model': selected_model,
        'analysis_type': analysis_type,
        'prompt': custom_prompt,
        'show_raw_response': show_raw_response
    }

def display_analysis_results(results: dict, settings: dict):
    """Display analysis results"""
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("📊 Analysis Results")
    
    # Try to parse JSON
    try:
        json_data = json.loads(results['response'])
        
        # Display structured results
        if 'overall_score' in json_data:
            st.metric("Overall Design Score", json_data['overall_score'])
        
        # Create tabs for different sections
        if len(json_data) > 3:
            tabs = st.tabs(["📝 Summary", "🎨 Design", "⚠️ Issues", "💡 Suggestions"])
            
            with tabs[0]:
                summary_keys = ['style', 'overall_score', 'quick_score', 'style_category']
                for key in summary_keys:
                    if key in json_data:
                        st.write(f"**{key.replace('_', ' ').title()}:** {json_data[key]}")
            
            with tabs[1]:
                design_keys = ['color_scheme', 'typography', 'layout']
                for key in design_keys:
                    if key in json_data:
                        st.subheader(key.replace('_', ' ').title())
                        if isinstance(json_data[key], dict):
                            for subkey, subvalue in json_data[key].items():
                                st.write(f"**{subkey.replace('_', ' ').title()}:** {subvalue}")
                        else:
                            st.write(json_data[key])
            
            with tabs[2]:
                issue_keys = ['errors', 'accessibility_issues', 'main_issues', 'accessibility_violations']
                for key in issue_keys:
                    if key in json_data and json_data[key]:
                        st.subheader(key.replace('_', ' ').title())
                        if isinstance(json_data[key], list):
                            for issue in json_data[key]:
                                st.error(f"• {issue}")
                        else:
                            st.error(json_data[key])
            
            with tabs[3]:
                suggestion_keys = ['improvements', 'priority_fixes', 'inclusive_design_suggestions']
                for key in suggestion_keys:
                    if key in json_data and json_data[key]:
                        st.subheader(key.replace('_', ' ').title())
                        if isinstance(json_data[key], list):
                            for suggestion in json_data[key]:
                                st.success(f"• {suggestion}")
                        else:
                            st.success(json_data[key])
        else:
            # Simple display for smaller responses
            for key, value in json_data.items():
                st.subheader(key.replace('_', ' ').title())
                if isinstance(value, list):
                    for item in value:
                        st.write(f"• {item}")
                elif isinstance(value, dict):
                    st.json(value)
                else:
                    st.write(value)
                    
    except json.JSONDecodeError:
        # Display as text if not JSON
        st.markdown("**Analysis Response:**")
        st.markdown(results['response'])
    
    # Show raw response if requested
    if settings.get('show_raw_response', False):
        with st.expander("🔍 Raw Response"):
            st.text(results['response'])
    
    # Download options
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download as TXT",
            data=results['response'],
            file_name=f"analysis_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        try:
            json_data = json.loads(results['response'])
            st.download_button(
                label="📋 Download as JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"analysis_{results['timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except json.JSONDecodeError:
            st.button("📋 JSON (Not Available)", disabled=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    initialize_session_state()
    display_header()
    
    # Create sidebar and get settings
    settings = create_sidebar()
    
    if settings is None:
        st.error("❌ Please start Ollama and ensure LLaVA models are available")
        st.info("Setup instructions:")
        st.code("""
# Start Ollama
ollama serve

# Pull a LLaVA model
ollama pull llava:13b
        """)
        return
    
    # File upload section
    st.subheader("📤 Upload Webpage Screenshot")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a screenshot of the webpage you want to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Load and validate image
            image = Image.open(uploaded_file)
            validation_result = validate_image(image)
            
            if not validation_result['valid']:
                st.error(f"❌ {validation_result['error']}")
                return
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            # Display image and info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(image, caption="Uploaded Screenshot", use_column_width=True)
            with col2:
                st.subheader("📊 Image Info")
                info = get_image_info(image)
                for key, value in info.items():
                    if key != 'error':
                        st.metric(key.title(), value)
            
            # Analysis button
            if st.button("🔍 Analyze Design", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔍 Analyzing webpage design..."):
                        response = analyze_webpage_image(
                            image=image,
                            prompt=settings['prompt'],
                            model=settings['model']
                        )
                        
                        results = {
                            'response': response,
                            'timestamp': datetime.now(),
                            'settings': settings
                        }
                        
                        # Display results
                        display_analysis_results(results, settings)
                        
                        # Add to history
                        st.session_state.analysis_history.append(results)
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("Make sure Ollama is running and the selected model is available")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    
    # Display history
    if st.session_state.analysis_history:
        st.subheader("📈 Recent Analysis")
        latest = st.session_state.analysis_history[-1]
        with st.expander(f"Latest - {latest['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Model:** {latest['settings']['model']}")
            st.write(f"**Type:** {latest['settings']['analysis_type']}")
            st.text_area("Response Preview", latest['response'][:500] + "..." if len(latest['response']) > 500 else latest['response'], height=100)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🖥️ Powered by <a href="https://ollama.com" target="_blank">Ollama</a> + LLaVA Models</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()