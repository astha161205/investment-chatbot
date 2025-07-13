import google.generativeai as genai
import sys

# Configure the API key
api_key = 'AIzaSyApehQ1TkiRU0_WMjlvavHUPyzKIxBoli8'

print(f"Python version: {sys.version}")
print(f"Testing API key: {api_key[:5]}...{api_key[-5:]}")
print(f"Google Generative AI version: {genai.__version__}")

try:
    # Configure the client library
    genai.configure(api_key=api_key)
    print("API key configured successfully")
    
    # List available models to see which ones we can use
    print("Available models:")
    for model in genai.list_models():
        print(f"- {model.name}")
    
    # Try with a model that should be available
    model = genai.GenerativeModel('gemini-1.5-pro')
    print("Model initialized successfully")
    
    # Generate content
    response = model.generate_content('Hello, please respond with "API is working" if you receive this message.')
    print("Response received:")
    print("-" * 40)
    print(response.text)
    print("-" * 40)
    
    if "API is working" in response.text:
        print("SUCCESS: API is working correctly!")
    else:
        print("WARNING: API responded but with unexpected content.")
        
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc() 