import streamlit as st
import requests
import json
import time
import base64
import os
import yfinance as yf
import fitz  # PyMuPDF for PDF processing
from gtts import gTTS
from io import BytesIO
import string # Import string module for punctuation

# --- Streamlit App Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Voice-Powered Financial Assistant",
    # Set initial sidebar state to 'expanded' or 'collapsed'
    # initial_sidebar_state="expanded"
)

st.title("🗣️ Voice-Powered Financial Assistant")
st.markdown("Ask about stock prices using your voice, or analyze PDFs!")

# --- API Keys Configuration ---
# IMPORTANT: Hardcoding API keys is generally NOT recommended for production.
# For secure deployment on Streamlit Community Cloud, use Streamlit secrets.
# For this specific request, the keys are hardcoded as asked.

ASSEMBLYAI_API_KEY = "a5c865ecb6cd4152ad9c91564a753cd2" # Your AssemblyAI API Key
FMP_API_KEY = "vLJtmE98fFBnb8zw65y0Sl9yJjmB2u9Q" # Your FMP API key

# Headers for AssemblyAI API calls
ASSEMBLYAI_HEADERS = {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}


# --- Custom Streamlit Component for Audio Recorder ---
# This line declares and caches the custom Streamlit audio recorder component.
# Ensure the 'st_audiorecorder_v2' directory is in the same folder as main.py.
@st.cache_resource
def get_audio_recorder_component():
    """
    Declares and caches the custom Streamlit audio recorder component.
    This ensures the component's JavaScript and associated files are loaded
    only once per app session, improving performance.
    The 'path' argument should point to the directory containing the component's
    __init__.py and frontend build files (e.g., index.html, bundle.js).
    """
    # Adjust this path if your st_audiorecorder_v2 folder is located elsewhere
    return st.components.v1.declare_component(
        "voice_query_recorder_component",
        path="./st_audiorecorder_v2"
    )

# Initialize the custom component. This function will be called once due to @st.cache_resource.
st_audiorecorder_v2 = get_audio_recorder_component()


# --- Helper Functions ---

def transcribe_audio_assemblyai(audio_base64):
    """
    Transcribes base64 encoded audio data using the AssemblyAI API.
    Handles uploading the audio, initiating a transcription job,
    and polling for results.

    Args:
        audio_base64 (str): Base64 encoded audio data (without 'data:audio/webm;base64,' prefix).
    Returns:
        str: The transcribed text, or None if transcription fails.
    """
    # Check if the AssemblyAI API key is set
    if not ASSEMBLYAI_API_KEY:
        st.error("AssemblyAI API Key not found. Please configure it in Streamlit secrets or directly in the code.")
        return None

    # Step 1: Upload the audio file to AssemblyAI's secure cloud storage
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    try:
        # Decode the base64 string to bytes before sending
        upload_response = requests.post(upload_endpoint, headers=ASSEMBLYAI_HEADERS, data=base64.b64decode(audio_base64))
        upload_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        upload_url = upload_response.json()["upload_url"] # Get the URL of the uploaded audio
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading audio to AssemblyAI: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"AssemblyAI Upload Response: {e.response.text}")
        return None

    # Step 2: Start the transcription process using the uploaded audio URL
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    json_data = {"audio_url": upload_url}
    try:
        transcript_response = requests.post(transcript_endpoint, headers=ASSEMBLYAI_HEADERS, json=json_data)
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()["id"] # Get the transcription job ID
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting transcription with AssemblyAI: {e}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"AssemblyAI Start Transcription Response: {e.response.text}")
        return None

    # Step 3: Poll for the transcription result until it's completed or an error occurs
    polling_endpoint = f"{transcript_endpoint}/{transcript_id}"
    with st.spinner("Transcribing audio..."):
        while True:
            try:
                polling_response = requests.get(polling_endpoint, headers=ASSEMBLYAI_HEADERS)
                polling_response.raise_for_status()
                transcription_result = polling_response.json()

                if transcription_result["status"] == "completed":
                    return transcription_result["text"] # Return the transcribed text
                elif transcription_result["status"] == "error":
                    st.error(f"AssemblyAI transcription error: {transcription_result['error']}")
                    return None
                else:
                    time.sleep(1) # Wait for 1 second before polling again to avoid excessive requests
            except requests.exceptions.RequestException as e:
                st.error(f"Error polling AssemblyAI transcription: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"AssemblyAI Polling Response: {e.response.text}")
                return None


@st.cache_data(ttl=3600) # Cache stock symbol searches for 1 hour
def search_stock_symbol_fmp(keyword):
    """
    Searches for a stock symbol using the Financial Modeling Prep (FMP) API.
    Caches the result. Handles FMP's rate limit.

    Args:
        keyword (str): A company name or keyword to search for.
    Returns:
        str: The stock symbol (e.g., "AAPL"), "RATE_LIMIT_EXCEEDED" if rate-limited, or None.
    """
    if not FMP_API_KEY:
        st.error("FMP API Key not found. Please configure it in Streamlit secrets or directly in the code.")
        return None

    url = f"https://financialmodelingprep.com/api/v3/search?query={keyword}&limit=1&apikey={FMP_API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status() # Raise an HTTPError for bad responses
        results = res.json()
        if results:
            return results[0]["symbol"]
        return None
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            st.warning("FMP API rate-limit exceeded for search. Please wait a moment and try again.")
            return "RATE_LIMIT_EXCEEDED"
        st.error(f"FMP search error for '{keyword}': {e}")
        return None

@st.cache_data(ttl=600) # Cache stock data for 10 minutes (600 seconds)
def get_stock_summary_fmp(symbol):
    """
    Fetches detailed stock information from Financial Modeling Prep (FMP) API.
    Caches the result to prevent redundant API calls.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").
    Returns:
        dict: A dictionary containing comprehensive stock information, or None if fetching fails.
    """
    if not FMP_API_KEY:
        st.error("FMP API Key not found. Please configure it in Streamlit secrets or directly in the code.")
        return None

    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
    quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    
    try:
        profile_response = requests.get(profile_url)
        profile_response.raise_for_status()
        profile_data = profile_response.json()

        quote_response = requests.get(quote_url)
        quote_response.raise_for_status()
        quote_data = quote_response.json()

        if profile_data and quote_data:
            profile = profile_data[0]
            quote = quote_data[0]

            # Calculate daily percentage change
            current_price = quote.get("price")
            previous_close = quote.get("previousClose") # FMP usually provides this in quote

            daily_pct_change = None
            if current_price is not None and previous_close is not None and previous_close != 0:
                daily_pct_change = ((current_price - previous_close) / previous_close) * 100

            return {
                "name": profile.get("companyName", symbol),
                "symbol": symbol,
                "price": current_price,
                "daily_pct": round(daily_pct_change, 2) if daily_pct_change is not None else None,
                "exchange": profile.get("exchangeShortName"),
                "description": profile.get("description", "No description available."),
                "industry": profile.get("industry"),
                "sector": profile.get("sector"),
            }
        else:
            return None
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            st.warning("FMP API rate-limit exceeded for stock data. Please wait a moment and try again.")
            time.sleep(2) # Add a small delay for rate limit
            return "RATE_LIMIT_EXCEEDED"
        st.error(f"Error fetching stock info from FMP for {symbol}: {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding FMP JSON for {symbol}. Check API key validity or response format.")
        return None


@st.cache_data(ttl=600) # Cache yfinance data for 10 minutes
def get_stock_summary_yfinance(symbol):
    """
    Fetches stock information using the yfinance library as a fallback.
    Caches the result.

    Args:
        symbol (str): The stock ticker symbol.
    Returns:
        dict: Stock information, or None if fetching fails.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info

        if not info or not info.get("regularMarketPrice"):
            return None

        # Fetch history for daily change calculation
        hist = t.history(period="2d") # Get last 2 days to calculate daily change
        
        current_price = info.get("currentPrice", info.get("regularMarketPrice"))
        
        # Calculate daily percentage change from historical data if available
        daily_pct = None
        if not hist.empty and len(hist) >= 2:
            prev_close = hist["Close"].iloc[-2]
            if prev_close != 0:
                daily_pct = ((current_price - prev_close) / prev_close) * 100
        elif info.get("previousClose") is not None and info.get("previousClose") != 0:
            # Fallback to info.get("previousClose") if historical data is insufficient
            prev_close = info.get("previousClose")
            daily_pct = ((current_price - prev_close) / prev_close) * 100


        return {
            "name": info.get("longName", info.get("shortName", symbol)),
            "symbol": symbol,
            "price": current_price,
            "daily_pct": round(daily_pct, 2) if daily_pct is not None else None,
            "exchange": info.get("exchange", "N/A"),
            "description": info.get("longBusinessSummary", "No description available."),
            "industry": info.get("industry", "N/A"),
            "sector": info.get("sector", "N/A"),
        }
    except Exception as e:
        st.error(f"Error fetching stock info with yfinance for {symbol}: {e}")
        return None

def get_stock_summary(symbol):
    """
    Attempts to fetch stock information using FMP first, with yfinance as a fallback.
    """
    # Try FMP first
    st.info(f"Attempting to fetch data for {symbol} using FMP...")
    stock_data = get_stock_summary_fmp(symbol)
    if stock_data and stock_data != "RATE_LIMIT_EXCEEDED":
        return stock_data
    elif stock_data == "RATE_LIMIT_EXCEEDED":
        st.warning(f"FMP rate limit hit for {symbol}. Trying yfinance as fallback.")
    else:
        st.warning(f"FMP data not found for {symbol}. Trying yfinance as fallback.")

    # If FMP fails or hits rate limit, try yfinance
    st.info(f"Attempting to fetch data for {symbol} using yfinance...")
    stock_data = get_stock_summary_yfinance(symbol)
    if stock_data:
        return stock_data
    else:
        st.error(f"Could not fetch data for {symbol} from either FMP or yfinance.")
        return {"error": f"No data found for {symbol}"}


@st.cache_data(ttl=3600) # Cache PDF text for 1 hour
def extract_text_from_pdf(pdf_file_content):
    """
    Extracts text from a PDF file's binary content using PyMuPDF (fitz).
    Caches the result.

    Args:
        pdf_file_content (bytes): The binary content of the PDF file.
    Returns:
        str: The extracted text from the PDF, or an empty string if an error occurs.
    """
    try:
        # Use fitz.open with 'stream' argument for in-memory bytes
        doc = fitz.open(stream=pdf_file_content, filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, size=300, overlap=50):
    """
    Chunks a given text into smaller pieces for easier processing.
    """
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks


@st.cache_data(ttl=86400) # Cache TTS audio for a day
def text_to_audio(text):
    """
    Generates speech from text using gTTS. Caches the audio bytes.
    """
    try:
        buf = BytesIO()
        gTTS(text=text, lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def extract_possible_company_names(text):
    """
    Extracts potential company names/keywords from transcribed text.
    Removes punctuation and common stopwords.
    """
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    # Expanded set of stopwords to filter out common query words
    stopwords = {
        "the","what","about","is","stock","market","price","update",
        "how","in","to","and","of","a","an","can","you","give","me",
        "tell","show","today","quote","value","exchange","it","doing",
        "for","report","news","brief","summary","daily","weekly","change",
        "percent","at","on","up","down","high","low","open","close","last",
        "morning","afternoon","evening","today's","yesterday's","current",
        "companies","company","shares","indices","index","group","holdings",
        "performance","analysis","latest","find","out","about","looking",
        "which","are","these","those","any","some","get","please",
        "its", "their", "them", "from", "with", "per", "by", "this", "that", "were",
        "has", "have", "had", "will", "would", "should", "could", "be", "been", "being",
        "was", "were", "here", "there", "where", "when", "why", "who", "whom", "whose",
        "do", "does", "did", "not", "no", "yes", "my", "your", "his", "her", "its", "our", "their",
        "so", "then", "than", "now", "just", "only", "very", "too", "also", "even", "much", "more",
        "most", "less", "least", "first", "second", "third", "other", "another", "new", "old",
        "big", "small", "good", "bad", "better", "best", "worse", "worst"
    }
    # Filter words longer than 2 characters and not in stopwords
    return [w for w in words if w.lower() not in stopwords and len(w) > 2]

# --- Layout: Two Columns ---
col1, col2 = st.columns(2)

# --- COLUMN 1: Voice Query Analysis ---
with col1:
    st.subheader("🎤 Voice Query Analysis")
    st.markdown("Record your voice query or upload an audio file to get stock information.")

    # Custom Audio Recorder Component
    # This component returns the base64 encoded audio data when recording stops.
    # The value is updated in Streamlit's session state via the custom component's JavaScript.
    recorded_audio_base64 = st_audiorecorder_v2(key="voice_query_recorder_component")

    # File Uploader for MP3 fallback/alternative
    audio_upload = st.file_uploader("Upload MP3 (or use recorder below)", type=["mp3", "wav"])

    use_audio_data = None
    if audio_upload:
        # For uploaded files, read the raw bytes
        use_audio_data = audio_upload.read()
        st.success("✅ Audio file uploaded.")
    elif isinstance(recorded_audio_base64, str) and recorded_audio_base64.startswith("data:audio"):
        # For recorded audio, decode the base64 part
        use_audio_data = base64.b64decode(recorded_audio_base64.split(",", 1)[1])
        st.success("✅ Audio recorded.")
    elif recorded_audio_base64 is not None:
        # Catch cases where recorder might return something unexpected
        st.warning("Audio recorder returned data in an unexpected format. Please try again.")

    # Session state to store last processed audio and transcription to prevent re-processing on reruns
    if 'last_processed_audio_hash' not in st.session_state:
        st.session_state.last_processed_audio_hash = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = None
    if 'last_run_symbols' not in st.session_state:
        st.session_state.last_run_symbols = {} # Stores fetched stock info for re-display


    # Only show the button if there's audio data available
    if use_audio_data:
        # Create a hash of the audio data to detect if it's new audio
        current_audio_hash = hash(use_audio_data)

        if st.button("Transcribe & Fetch Stock Info", key="process_audio_button"):
            if current_audio_hash != st.session_state.last_processed_audio_hash:
                # Process new audio
                st.info("Processing new audio...")
                transcribed_text = transcribe_audio_assemblyai(base64.b64encode(use_audio_data).decode('utf-8'))
                st.session_state.transcribed_text = transcribed_text
                st.session_state.last_processed_audio_hash = current_audio_hash
                st.session_state.last_run_symbols = {} # Clear previous results

                if transcribed_text:
                    st.success("Transcription Complete!")
                    st.write(f"Transcribed Query: **{transcribed_text}**")

                    keywords = extract_possible_company_names(transcribed_text)
                    st.write(f"Detected Keywords: {', '.join(keywords)}")

                    symbols = set()
                    with st.spinner("Searching for stock symbols..."):
                        for kw in keywords:
                            s = search_stock_symbol_fmp(kw)
                            if s == "RATE_LIMIT_EXCEEDED":
                                st.error("FMP API rate limit hit during symbol search. Please try again later.")
                                break
                            if s:
                                symbols.add(s)
                    
                    if symbols:
                        st.info(f"Found potential symbols: {', '.join(symbols)}")
                        st.subheader("Stock Information:")
                        fetched_symbols_info = {}
                        for sym in symbols:
                            with st.spinner(f"Fetching data for {sym}..."):
                                info = get_stock_summary(sym)
                                if "error" in info:
                                    st.warning(f"Could not retrieve info for {sym}: {info['error']}")
                                else:
                                    st.markdown(f"**{info['name']} ({sym})**")
                                    st.metric(label="Current Price", value=f"${info['price']:.2f}")
                                    
                                    if info['daily_pct'] is not None:
                                        delta_value = f"{info['daily_pct']:.2f}%"
                                        delta_color = "green" if info['daily_pct'] >= 0 else "red"
                                        st.metric(label="Daily Change", value=delta_value, delta=delta_value, delta_color=delta_color)
                                    else:
                                        st.write("Daily Change: N/A")
                                        
                                    st.write(f"Exchange: {info['exchange']}")
                                    st.write(f"Sector: {info['sector']} | Industry: {info['industry']}")
                                    with st.expander(f"More about {info['name']}"):
                                        st.write(info['description'])

                                    tts_text = f"The current price for {info['name']} is {info['price']:.2f} dollars."
                                    if info['daily_pct'] is not None:
                                        change_type = "up" if info['daily_pct'] >= 0 else "down"
                                        tts_text += f" It is {change_type} {abs(info['daily_pct']):.2f} percent today."
                                    
                                    audio_bytes = text_to_audio(tts_text)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format="audio/mp3")
                                    
                                    fetched_symbols_info[sym] = info # Store for re-display
                                    st.markdown("---")
                        st.session_state.last_run_symbols = fetched_symbols_info # Update session state
                    else:
                        st.warning("No valid stock symbols detected in your query or could not fetch data.")
                else:
                    st.error("Could not transcribe audio.")
            else:
                # Re-display results from session state if audio is the same
                st.info("Using cached transcription and results from previous processing of this audio.")
                st.write(f"Transcribed Query: **{st.session_state.transcribed_text}**")
                if st.session_state.last_run_symbols:
                    st.subheader("Stock Information (Cached):")
                    for sym, info in st.session_state.last_run_symbols.items():
                           if "error" in info:
                               st.warning(f"Could not retrieve info for {sym}: {info['error']}")
                           else:
                               st.markdown(f"**{info['name']} ({sym})**")
                               st.metric(label="Current Price", value=f"${info['price']:.2f}")
                               
                               if info['daily_pct'] is not None:
                                   delta_value = f"{info['daily_pct']:.2f}%"
                                   delta_color = "green" if info['daily_pct'] >= 0 else "red"
                                   st.metric(label="Daily Change", value=delta_value, delta=delta_value, delta_color=delta_color)
                               else:
                                   st.write("Daily Change: N/A")
                                   
                               st.write(f"Exchange: {info['exchange']}")
                               st.write(f"Sector: {info['sector']} | Industry: {info['industry']}")
                               with st.expander(f"More about {info['name']}"):
                                   st.write(info['description'])
                               
                               tts_text = f"The current price for {info['name']} is {info['price']:.2f} dollars."
                               if info['daily_pct'] is not None:
                                   change_type = "up" if info['daily_pct'] >= 0 else "down"
                                   tts_text += f" It is {change_type} {abs(info['daily_pct']):.2f} percent today."
                               
                               audio_bytes = text_to_audio(tts_text)
                               if audio_bytes:
                                   st.audio(audio_bytes, format="audio/mp3")
                               st.markdown("---")
                else:
                    st.warning("No valid stock symbols found or previous fetching failed.")
    else:
        st.info("Upload an audio file or record your voice to enable transcription and stock fetching.")


# --- COLUMN 2: PDF Document Analysis ---
with col2:
    st.subheader("📄 PDF Document Analysis")
    st.markdown("Upload a PDF to extract its text content and see how it can be chunked for further analysis.")

    pdf_file = st.file_uploader("Upload a financial PDF document", type=["pdf"])

    if pdf_file:
        with st.spinner("Extracting text from PDF..."):
            pdf_content_bytes = pdf_file.read() # Read the content as bytes
            extracted_text = extract_text_from_pdf(pdf_content_bytes) # Pass bytes directly
            
            if extracted_text:
                st.success("Text extracted successfully!")
                st.subheader("Extracted Text (First 1000 characters):")
                st.text_area("PDF Content Preview", extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""), height=250)
                
                chunks = chunk_text(extracted_text)
                st.info(f"The PDF text has been divided into {len(chunks)} chunks (approx. {len(extracted_text.split())} words total).")
                
                if st.checkbox("Show first 5 text chunks"):
                    for i, c in enumerate(chunks[:5]):
                        st.markdown(f"**Chunk {i+1} (Length: {len(c.split())} words):**")
                        st.text_area(f"Chunk {i+1} Content", c, height=150, key=f"chunk_{i}")
                        st.markdown("---")
            else:
                st.error("Failed to extract text from the PDF. The file might be scanned, image-based, or corrupted.")

