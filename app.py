import numpy as np
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import os
import urllib.request  # Replace wget with urllib.request
import zipfile
from scipy.spatial.distance import cosine
import sys  # Import sys to check Python version or exit gracefully
import json # Import json for manual JSON formatting

# This Flask application provides an API endpoint that, given a list of words,
# finds other words that are semantically similar to the average meaning
# of the input words. It achieves this by using pre-trained word vectors
# (specifically GloVe vectors), which are numerical representations of words
# where words with similar meanings or contexts are represented by vectors
# that are numerically "close" to each other in a high-dimensional space.

# Initialize the Flask application.
# __name__ is a special Python variable that gets the name of the current module.
# Flask uses this to determine the root path for the application.
app = Flask(__name__)

# --- Configuration for JSON Output Formatting ---
# We configure how JSON responses are formatted based on the environment
# (development or production). This is useful for debugging (pretty-print in dev)
# and efficiency (minified in production).

# Get the FLASK_ENV environment variable. Default to 'production' if not set.
# Convert to lowercase for case-insensitive comparison.
FLASK_ENV = os.environ.get('FLASK_ENV', 'production').lower()
# Determine if the application is running in development mode.
IS_DEVELOPMENT = FLASK_ENV == 'development'

# Configure Flask's built-in JSON settings.
if IS_DEVELOPMENT:
    # In development, disable Flask's default compact JSON output.
    # This allows us to use our custom pretty-printing later.
    app.json.compact = False
    # In development, sort JSON keys alphabetically for consistent output.
    app.json.sort_keys = True
    # Enable pretty-printing for jsonify responses in debug mode.
    # This is an alternative or complementary setting to the custom_jsonify approach below.
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    # Set the indentation level for pretty-printed JSON (2 spaces).
    app.config['JSONIFY_PRETTYPRINT_REGULAR_INDENT'] = 2
else:
    # In production, enable compact JSON output (no extra whitespace).
    # This reduces response size, which is good for performance.
    app.json.compact = True
    # Disable pretty-printing for regular jsonify responses in production.
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Global Variables for Word Vectors ---
# We'll store the loaded word vectors in a dictionary where keys are words
# and values are their corresponding numpy arrays (the vectors).
word_vectors = {}
# This variable will store the expected dimension (length) of each word vector.
# It's initialized but will be set precisely when the vector file is loaded.
vector_dim = 300  # Default dimension, will be updated when loading

# --- GloVe Data Configuration ---
# Define the URL for the GloVe 6B (6 billion words) dataset zip file.
GLOVE_ZIP_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
# Define the directory where we will store the downloaded and extracted GloVe files.
GLOVE_DIR = 'glove'
# Define the path to the specific GloVe file we want to use.
# We've chosen the 300-dimensional vectors (glove.6B.300d.txt).
GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt') # Target 300d file

# --- Data Download and Loading Functions ---

def download_glove_vectors():
    """
    Downloads the GloVe word vectors zip file if it doesn't exist
    and extracts the required vector file from it.

    This function checks if the target GloVe file (glove.6B.300d.txt)
    already exists. If not, it checks for the zip file. If the zip
    file doesn't exist, it downloads it. Finally, if the target
    file still doesn't exist (either because the zip was just
    downloaded or already existed but wasn't extracted), it extracts
    the specific file from the zip.
    """
    # Create the directory for GloVe files if it doesn't exist.
    if not os.path.exists(GLOVE_DIR):
        print(f"Creating directory: {GLOVE_DIR}")
        os.makedirs(GLOVE_DIR)

    # Define the full path for the downloaded zip file.
    glove_zip_path = os.path.join(GLOVE_DIR, 'glove.6B.zip')

    # Check if the specific 300d file we need already exists.
    if not os.path.exists(GLOVE_FILE):
         print(f"GloVe file not found: {GLOVE_FILE}")
         # If the target file doesn't exist, check if the zip file exists.
         if not os.path.exists(glove_zip_path):
            # If the zip file also doesn't exist, download it.
            print(f"Downloading GloVe vectors from {GLOVE_ZIP_URL}...")
            try:
                # Use urllib.request to download the file
                # This is Python's built-in solution for downloading files
                with urllib.request.urlopen(GLOVE_ZIP_URL) as response, open(glove_zip_path, 'wb') as out_file:
                    # Get the total file size for progress reporting
                    file_size = int(response.info().get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 8192  # 8KB chunks
                    
                    # Download in chunks and show progress
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        out_file.write(chunk)
                        
                        # Print download progress
                        if file_size > 0:
                            percent = int(100 * downloaded / file_size)
                            # Print progress on the same line
                            print(f"\rDownload progress: {percent}% ({downloaded}/{file_size} bytes)", end='')
                    
                print("\nDownload complete")
            except Exception as e:
                 # Handle potential errors during download.
                 print(f"\nError downloading GloVe vectors: {e}")
                 print("Please ensure you have internet access.")
                 # Exit the application if the download fails, as we cannot proceed.
                 sys.exit(1)
         else:
             # If the zip file exists but the target file doesn't, skip downloading the zip again.
             print(f"{glove_zip_path} already exists. Skipping download.")


    # After potentially downloading the zip, check again if the target file exists.
    # This covers the case where the zip existed but the target file wasn't extracted.
    if not os.path.exists(GLOVE_FILE):
            print(f"Extracting {GLOVE_FILE} from {glove_zip_path}...")
            try:
                # Open the zip file in read mode.
                with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
                    # Get a list of all files inside the zip archive.
                    zip_contents = zip_ref.namelist()
                    # Get just the filename of our target file.
                    target_file_in_zip = os.path.basename(GLOVE_FILE)

                    # Check if the specific target file exists within the zip.
                    if target_file_in_zip in zip_contents:
                        # Extract only the required file to the GloVe directory.
                        print(f"Found '{target_file_in_zip}' in zip. Extracting...")
                        zip_ref.extract(target_file_in_zip, GLOVE_DIR)
                        print("Extraction complete")
                        # Optional: Remove the zip file after successful extraction to save space.
                        # print(f"Removing zip file: {glove_zip_path}")
                        # os.remove(glove_zip_path)
                    else:
                        # If the target file isn't in the zip, something is wrong.
                        print(f"Error: Target file '{target_file_in_zip}' not found in the zip archive.")
                        print(f"Available files in zip: {zip_contents}")
                        # Exit as we cannot find the necessary data.
                        sys.exit(1)
            except Exception as e:
                # Handle potential errors during zip extraction.
                print(f"Error extracting zip file: {e}")
                # Exit if extraction fails.
                sys.exit(1)

    else:
        # If the target GloVe file already exists, skip the download and extraction steps.
        print(f"{GLOVE_FILE} already exists. Skipping download and extraction.")


def load_glove_vectors(file_path=GLOVE_FILE):
    """
    Loads pre-trained GloVe vectors from a text file into a dictionary.

    Each line in the file is expected to start with a word, followed by
    a space-separated list of numbers representing its vector.

    Args:
        file_path (str): The path to the GloVe vector file.
    """
    # Declare global variables so we can modify them within this function.
    global word_vectors, vector_dim

    # Check if the vector file exists before attempting to load.
    if not os.path.exists(file_path):
         print(f"Error: GloVe vector file not found at {file_path}.")
         print("Please ensure it's downloaded and extracted correctly.")
         # Exit if the data file is missing.
         sys.exit(1)

    print(f"Loading GloVe vectors from {file_path}...")
    # Initialize an empty dictionary to store the word vectors.
    word_vectors = {}
    count = 0 # Counter for tracking loaded vectors.
    try:
        # Open the GloVe file for reading with UTF-8 encoding.
        with open(file_path, 'r', encoding='utf-8') as f:
            # Iterate through each line in the file.
            for line in f:
                # Split the line into parts based on whitespace.
                values = line.split()
                # A valid line should have at least the word and one vector component.
                if len(values) < 2:
                    # Skip lines that don't seem to contain a word and a vector.
                    continue

                # The first element is the word.
                word = values[0]
                try:
                    # The rest of the elements should be the floating-point numbers
                    # representing the vector. Convert them to a NumPy array.
                    vector = np.array([float(val) for val in values[1:]])
                except ValueError:
                    # If any value in the vector part cannot be converted to a float,
                    # skip this line and print a warning (commented out by default
                    # to avoid excessive output for minor data issues).
                    # print(f"Skipping line with parsing error for word '{word}': {line.strip()[:50]}...")
                    continue # Skip this line

                # Store the word and its vector in the dictionary.
                word_vectors[word] = vector

                # Set the vector dimension based on the first successfully loaded vector.
                if count == 0:
                    vector_dim = len(vector)
                # Optional: Check if subsequent vectors have the same dimension.
                # This helps catch potential issues in the data file format.
                elif vector_dim != len(vector):
                    # Print a warning if a vector has an unexpected dimension and skip it.
                    # print(f"Warning: Inconsistent vector dimension for word '{word}'. Expected {vector_dim}, got {len(vector)}. Skipping word.")
                    continue # Skip words with wrong dimensions

                count += 1
                # Print a progress update periodically.
                if count % 500000 == 0:
                    print(f"Loaded {count} vectors...")

    except Exception as e:
        # Handle any errors that occur during file reading or processing.
        print(f"Error loading GloVe vectors from {file_path}: {e}")
        # Exit the application if loading fails, as the core data is missing.
        sys.exit(1)

    # Report the total number of vectors loaded and their dimension.
    print(f"Finished loading {len(word_vectors)} word vectors with dimension {vector_dim}")
    # Perform a basic check to ensure that at least some vectors were loaded.
    if not word_vectors:
         print("Error: No word vectors were loaded. Check the GloVe file format or path.")
         # Exit if no data was loaded.
         sys.exit(1)


def find_centroid(word_list):
    """
    Calculates the centroid (average vector) for a list of words.

    The centroid represents a summary vector for the group of words.
    Only words found in the loaded vocabulary (word_vectors) are included
    in the calculation.

    Args:
        word_list (list): A list of strings (words).

    Returns:
        np.ndarray or None: A NumPy array representing the centroid vector,
                            or None if none of the words in the list were
                            found in the vocabulary.
    """
    vectors = [] # List to hold vectors for the input words.

    # Iterate through each word provided in the input list.
    for word in word_list:
        # Convert the word to lowercase to match the vocabulary format (GloVe is typically lowercase).
        word = word.lower()
        # Check if the lowercase word exists in our loaded word vectors dictionary.
        if word in word_vectors:
            # If the word is found, add its vector to our list.
            vectors.append(word_vectors[word])

    # Optional: Add a warning if some words from the input list were not found.
    # if len(vectors) < len(word_list):
    #    print(f"Warning: {len(word_list) - len(vectors)} input word(s) not found in vocabulary.")

    # If no vectors were found for the input words (meaning none of the words
    # were in the vocabulary), return None.
    if not vectors:
        return None

    # Calculate the mean (average) of the vectors along axis 0 (across the rows).
    # This computes the centroid vector.
    centroid = np.mean(vectors, axis=0)
    return centroid

# Modified function to return top N words
def find_closest_words(centroid, exclude_words=None, top_n=5):
    """
    Finds the word(s) in the vocabulary whose vectors are numerically closest
    to a given centroid vector, excluding a specified list of words.

    Closeness is measured using cosine similarity/distance. Words with higher
    cosine similarity (lower cosine distance) are considered more related
    in the vector space.

    Args:
        centroid (np.ndarray): The target vector (e.g., the centroid of input words).
        exclude_words (list, optional): A list of words to exclude from the results
                                       (typically the input words themselves). Defaults to None.
        top_n (int, optional): The number of closest words to return. Defaults to 5.

    Returns:
        list: A list of dictionaries, where each dictionary contains a 'word'
              and its 'similarity_score' to the centroid, sorted by similarity
              in descending order. Returns an empty list if no words are found
              after exclusion or if the centroid is invalid.
    """
    # Prepare a set of lowercase words to exclude for efficient lookup.
    if exclude_words is None:
        exclude_words_lower = set()
    else:
        # Convert excluded words to lowercase and store in a set.
        exclude_words_lower = {word.lower() for word in exclude_words}

    # List to store the calculated distances for each word in the vocabulary.
    # We store (distance, word) tuples.
    distances = []

    # Iterate through every word and its vector in our loaded vocabulary.
    for word, vector in word_vectors.items():
        # Skip this word if it's in our list of words to exclude.
        if word in exclude_words_lower:
            continue

        # Calculate the cosine distance between the centroid and the current word's vector.
        # Cosine distance is a measure of similarity between two vectors in an inner product space.
        # It measures the cosine of the angle between them. A distance of 0 means they are
        # identical (point in the same direction), and a distance of 1 means they are
        # orthogonal (at a 90-degree angle, no similarity).
        # scipy.spatial.distance.cosine is a robust implementation that handles
        # potential edge cases like zero vectors.
        try:
             distance = cosine(centroid, vector)
             # scipy.spatial.distance.cosine returns 1.0 if one vector is zero,
             # and 0.0 if both are zero. This is usually the desired behavior.
             # We add a check for NaN just in case, although unlikely with scipy.
             if np.isnan(distance):
                 distance = 1.0 # Treat NaN as maximum distance (no similarity)
        except ValueError:
             # A ValueError might occur if one of the vectors (centroid or word vector)
             # has a zero norm (all elements are zero), which can happen in specific cases
             # although rare with standard pre-trained vectors.
             # Treat this case as maximum distance as well.
             distance = 1.0
             # Optionally, you could skip such words entirely instead of assigning a distance.
             # continue

        # Append the calculated distance and the word to our list.
        distances.append((distance, word))

    # Sort the list of (distance, word) tuples based on the distance.
    # We sort in ascending order because we want the *smallest* distances first
    # (indicating highest similarity).
    # The 'key' argument specifies that we sort based on the first element of each tuple (the distance).
    distances.sort(key=lambda item: item[0])

    # Prepare the list of results to return.
    top_results = []
    # Determine the actual number of results to return, which is the minimum
    # of the requested 'top_n' and the total number of valid distances calculated.
    num_results_to_return = min(top_n, len(distances))

    # Iterate through the top results based on the sorted distances.
    for i in range(num_results_to_return):
        # Get the distance and word from the sorted list.
        dist, word = distances[i]
        # Cosine similarity is calculated as 1 - cosine distance.
        # A similarity of 1 means identical, 0 means orthogonal, -1 means completely opposite.
        # With standard word vectors, distances are usually between 0 and 2, so similarity
        # is between -1 and 1. 0.0 distance means 1.0 similarity.
        similarity = 1 - dist
        # Append a dictionary containing the word and its similarity score to the results list.
        # Convert the numpy float to a standard Python float to ensure compatibility
        # with JSON serialization in case jsonify has stricter type checks.
        top_results.append({'word': word, 'similarity_score': float(similarity)})

    # Return the list of closest words and their similarity scores.
    return top_results

# --- Custom JSONIFY Function ---
# We create a custom function to handle JSON responses. This gives us more
# fine-grained control over the output format (pretty-print vs. minified)
# compared to just relying on Flask's default jsonify behavior, especially
# when interacting with environment variables.

def custom_jsonify(*args, **kwargs):
    """
    Custom jsonify function that formats JSON based on the application environment.
    Uses indent=2 and sort_keys=True for development mode, and compact separators
    for production mode.
    """
    # First, use Flask's built-in jsonify to create a Response object.
    # This handles content type headers and other HTTP response details.
    response = jsonify(*args, **kwargs)

    # Determine the data payload from the arguments passed to custom_jsonify.
    # It handles both positional arguments (like a single dictionary) and keyword arguments.
    if args and len(args) == 1 and not kwargs:
        data = args[0]
    else:
        data = dict(*args, **kwargs)

    # Based on whether we are in development mode:
    if IS_DEVELOPMENT:
        # In development, serialize the data with pretty-printing (indentation and sorted keys).
        # Encode the resulting string to bytes using UTF-8, as response data needs to be bytes.
        response.data = json.dumps(data, indent=2, sort_keys=True).encode('utf-8')
    else:
        # In production, serialize the data in a compact format with no extra whitespace.
        # Using separators=(',', ':') removes spaces after commas and colons.
        # Encode the resulting string to bytes using UTF-8.
        response.data = json.dumps(data, separators=(',', ':')).encode('utf-8')

    # Return the modified Flask Response object.
    return response


# --- API Endpoint Definition ---

# This is a Flask route decorator. It tells Flask that the function
# directly below it should handle requests for the '/find-common-word' URL path.
# It only accepts POST requests, which is standard practice for sending data
# to a server (like the list of words).
@app.route('/find-common-word', methods=['POST'])
def find_common_word_api():
    """
    API endpoint that receives a list of words via a POST request,
    calculates the centroid of their word vectors, finds the closest
    words in the vocabulary to that centroid (excluding the input words),
    and returns the results as JSON.

    Expects a JSON payload in the request body with a 'words' key
    containing a list of strings, e.g., {"words": ["king", "man", "woman"]}.
    Optionally accepts a 'top_n' key for the number of results to return.
    """
    # Get the JSON data from the request body.
    # Flask's request object automatically parses JSON if the request
    # has the 'Content-Type: application/json' header.
    data = request.get_json()

    # --- Input Validation ---
    # Check if the request contained valid JSON data and if the 'words' key is present.
    if not data or 'words' not in data:
        # If validation fails, return an error message with a 400 Bad Request status code.
        # We use our custom_jsonify for consistent output formatting.
        return custom_jsonify({'error': 'Input must be a JSON object containing a list of words under the key "words".'}), 400

    # Extract the list of words and the optional 'top_n' value from the JSON data.
    # Use .get() for 'top_n' to provide a default value (5) if the key is missing.
    words = data['words']
    top_n = data.get('top_n', 5)

    # Perform more detailed validation on the 'words' list.
    # Check if 'words' is indeed a list, is not empty, and all its elements are strings.
    if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
        return custom_jsonify({'error': 'The value for the "words" key must be a non-empty list of strings.'}), 400

    # Validate that 'top_n' is a positive integer.
    if not isinstance(top_n, int) or top_n <= 0:
         return custom_jsonify({'error': 'The value for the "top_n" key must be a positive integer.'}), 400

    # --- NLP Processing ---
    # Calculate the centroid (average vector) for the input words.
    centroid = find_centroid(words)

    # Check if the centroid calculation was successful. It returns None if none
    # of the input words were found in the vocabulary.
    if centroid is None:
        # If no words were found, identify which ones were missing for a helpful error.
        missing_words = [word for word in words if word.lower() not in word_vectors]
        error_msg = f"None of the provided words were found in the vocabulary used for the word vectors."
        if missing_words:
             # Display the first few missing words if there are many.
             missing_display = missing_words[:10] + (['...'] if len(missing_words) > 10 else [])
             error_msg += f" Missing words ({len(missing_words)} total): {', '.join(missing_display)}"
        # Return an error with a 400 status code.
        return custom_jsonify({'error': error_msg}), 400

    # Find the words in the vocabulary that are closest to the calculated centroid.
    # We pass the original input 'words' list so these words are excluded from the results.
    # We also pass the requested number of results, 'top_n'.
    common_words_results = find_closest_words(centroid, exclude_words=words, top_n=top_n)

    # Check if any related words were found after calculating distances and excluding input words.
    if not common_words_results:
         # If no results are found, return an error message.
         return custom_jsonify({'error': 'Could not find any related words in the vocabulary (after excluding the input words). This might happen if the input words are very specific or rare, cover a concept poorly represented in the vocabulary, or if the vocabulary itself is limited.'}), 400

    # --- Prepare and Return Response ---
    # Construct the final result dictionary.
    result = {
        'input_words': words, # Echo the input words.
        'top_n_requested': top_n, # Indicate how many results were asked for.
        'common_words': common_words_results # The list of closest words and their scores.
    }

    # Return the result dictionary as a JSON response with a 200 OK status code (default).
    # Use our custom_jsonify function for formatting.
    return custom_jsonify(result)

# --- Web Interface Route ---
@app.route('/', methods=['GET', 'POST'])
def web_interface():
    """
    Provides a web interface for the word similarity functionality.
    GET: Shows a form for entering words.
    POST: Processes the form, finds similar words, and shows results.
    """
    results = None
    input_words = []
    error_message = None
    
    if request.method == 'POST':
        # Get form data
        words_input = request.form.get('words', '').strip()
        top_n = request.form.get('top_n', '5')
        
        # Validate input
        if not words_input:
            error_message = "Please enter at least one word."
        else:
            # Parse input into a list of words
            input_words = [word.strip() for word in words_input.split(',') if word.strip()]
            
            if not input_words:
                error_message = "Please enter valid words separated by commas."
            else:
                try:
                    top_n = int(top_n)
                    if top_n <= 0:
                        error_message = "Number of results must be a positive number."
                    else:
                        # Calculate centroid
                        centroid = find_centroid(input_words)
                        
                        if centroid is None:
                            missing_words = [word for word in input_words if word.lower() not in word_vectors]
                            error_message = "None of the provided words were found in the vocabulary."
                            if missing_words:
                                missing_display = missing_words[:10] + (['...'] if len(missing_words) > 10 else [])
                                error_message += f" Missing words: {', '.join(missing_display)}"
                        else:
                            # Find similar words
                            results = find_closest_words(centroid, exclude_words=input_words, top_n=top_n)
                            
                            if not results:
                                error_message = "Could not find any related words after excluding the input words."
                except ValueError:
                    error_message = "Number of results must be a valid number."
    
    # HTML template for the web interface
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Word Similarity Finder</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="text"], input[type="number"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .results {
                margin-top: 30px;
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 4px;
            }
            .error {
                color: #e74c3c;
                background-color: #fadbd8;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .info {
                margin-top: 30px;
                font-size: 14px;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <h1>Word Similarity Finder</h1>
        
        <form method="POST">
            <div class="form-group">
                <label for="words">Enter words (separated by commas):</label>
                <input type="text" id="words" name="words" value="{{ ','.join(input_words) if input_words else '' }}" required>
            </div>
            
            <div class="form-group">
                <label for="top_n">Number of results to show:</label>
                <input type="number" id="top_n" name="top_n" value="{{ request.form.get('top_n', '5') }}" min="1" max="50" required>
            </div>
            
            <button type="submit">Find Similar Words</button>
        </form>
        
        {% if error_message %}
        <div class="error">
            {{ error_message }}
        </div>
        {% endif %}
        
        {% if results %}
        <div class="results">
            <h2>Results for: {{ ', '.join(input_words) }}</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Word</th>
                        <th>Similarity Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for result in results %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ result.word }}</td>
                        <td>{{ "%.4f"|format(result.similarity_score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <div class="info">
            <p>This tool finds words that are semantically similar to the average meaning of your input words using pre-trained GloVe word vectors.</p>
            <p>The similarity score ranges from -1 to 1, where 1 represents the highest similarity.</p>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                 request=request, 
                                 input_words=input_words,
                                 results=results,
                                 error_message=error_message)

# --- Application Startup ---

# Print the environment mode the application is starting in.
print(f"Starting application in {FLASK_ENV.upper()} mode")
# Print how the JSON responses will be formatted.
print(f"JSON formatting: {'Pretty-print' if IS_DEVELOPMENT else 'Minified'}")

# --- Data Loading on Import ---
# This section runs *when the Python module is imported*.
# In a production WSGI server environment (like Gunicorn), the application module
# is imported once when the server starts. This is the correct place to load
# large resources like the word vectors, so it only happens once.

# Download the GloVe vectors if they are not already present.
download_glove_vectors()
# Load the downloaded/existing GloVe vectors into memory.
load_glove_vectors()

# ------------------------------------------------------------


# --- Development Server Execution ---
# This block only runs when the script is executed directly (e.g., `python app.py`).
# It is *not* typically used when deploying with a production WSGI server like Gunicorn,
# which handles starting the application instance internally after importing the module.
if __name__ == '__main__':
    # Inform the user that this is for development and recommend Gunicorn for production.
    print("Running in development mode.")
    print("For production deployments, use a WSGI server like Gunicorn (e.g., 'gunicorn -w 4 app:app -b 0.0.0.0:4001').")
    print("Web interface available at http://localhost:4001/")

    # Run the Flask development server if in development mode.
    if IS_DEVELOPMENT:
        # debug=True enables the debugger and auto-reloader, useful during development.
        # host='0.0.0.0' makes the server accessible externally (useful in Docker).
        # port=4001 sets the port the server listens on.
        # threaded=True allows the development server to handle multiple requests concurrently.
        app.run(debug=True, host='0.0.0.0', port=4001, threaded=True)

    # If not in development mode and running directly (unlikely for production),
    # we could add a different run configuration or just let the script finish
    # after loading data. The 'pass' statement does nothing but is needed for
    # syntactical correctness if the 'if' block is the only thing in the 'if __name__ == "__main__":' block.
    pass