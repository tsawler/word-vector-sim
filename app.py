import numpy as np
from flask import Flask, request, jsonify
import os
import wget
import zipfile
from scipy.spatial.distance import cosine
import sys # Import sys to check Python version

app = Flask(__name__)

# Global variables
word_vectors = {}
vector_dim = 300  # Default dimension, will be updated when loading

# Define the GloVe 6B zip file URL and the specific file we want to load
GLOVE_ZIP_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_DIR = 'glove'
GLOVE_FILE = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt') # Target 300d file

def download_glove_vectors():
    """Download GloVe vectors if not already downloaded"""
    if not os.path.exists(GLOVE_DIR):
        os.makedirs(GLOVE_DIR)

    if not os.path.exists(GLOVE_FILE):
        glove_zip_path = os.path.join(GLOVE_DIR, 'glove.6B.zip')

        # Only download the zip if the specific file isn't already extracted
        if not os.path.exists(GLOVE_FILE):
             if not os.path.exists(glove_zip_path):
                print(f"Downloading GloVe vectors from {GLOVE_ZIP_URL}...")
                try:
                    # Use '--no-verbose' to reduce output clutter
                    # Need to handle potential issues if wget is not found in the container/env
                    # Or switch to Python's urllib/requests
                    wget.download(GLOVE_ZIP_URL, glove_zip_path)
                    print("\nDownload complete") # wget prints progress on the same line
                except Exception as e:
                     print(f"\nError downloading GloVe vectors: {e}")
                     print("Please ensure you have 'wget' installed and internet access.")
                     # In a Docker build context, this would ideally be handled outside this script
                     # or ensure wget is available in the build image. For runtime, exit is fine.
                     sys.exit(1) # Exit if download fails
             else:
                 print(f"{glove_zip_path} already exists. Skipping download.")


        # Check if the specific file exists after potentially downloading the zip
        if not os.path.exists(GLOVE_FILE):
             print(f"Extracting {GLOVE_FILE} from {glove_zip_path}...")
             try:
                 with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
                     # Check if the target file is actually in the zip
                     target_file_in_zip = os.path.basename(GLOVE_FILE)
                     if target_file_in_zip in zip_ref.namelist():
                         # Extract only the file we need into the GLOVE_DIR
                         zip_ref.extract(target_file_in_zip, GLOVE_DIR)
                         print("Extraction complete")
                         # Optionally remove the zip file after extraction
                         # print(f"Removing zip file: {glove_zip_path}")
                         # os.remove(glove_zip_path)
                     else:
                         print(f"Error: {target_file_in_zip} not found in the zip archive.")
                         print(f"Available files: {zip_ref.namelist()}")
                         sys.exit(1) # Exit if the target file is not in the zip
             except Exception as e:
                  print(f"Error extracting zip file: {e}")
                  sys.exit(1) # Exit if extraction fails

    else:
        print(f"{GLOVE_FILE} already exists. Skipping download.")


def load_glove_vectors(file_path=GLOVE_FILE):
    """Load pre-trained GloVe vectors"""
    global word_vectors, vector_dim

    if not os.path.exists(file_path):
         print(f"Error: GloVe vector file not found at {file_path}. Please ensure it's downloaded and extracted.")
         sys.exit(1) # Exit if the file doesn't exist

    print(f"Loading GloVe vectors from {file_path}...")
    word_vectors = {}
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                # Skip empty lines or lines with just a word and no vector
                if len(values) < 2:
                    continue

                word = values[0]
                try:
                    vector = np.array([float(val) for val in values[1:]])
                except ValueError:
                    # Handle lines where vector values aren't valid floats
                    # print(f"Skipping line with parsing error for word '{word}': {line.strip()[:50]}...")
                    continue # Skip this line

                word_vectors[word] = vector

                # Set the vector dimension based on the first successfully parsed vector
                if count == 0:
                    vector_dim = len(vector)
                elif vector_dim != len(vector):
                    # Warn if vectors have inconsistent dimensions (shouldn't happen in standard GloVe files, but good check)
                    # print(f"Warning: Inconsistent vector dimension for word '{word}'. Expected {vector_dim}, got {len(vector)}. Skipping word.")
                    continue # Skip words with wrong dimensions

                count += 1
                if count % 500000 == 0: # Increased update frequency for smaller output
                    print(f"Loaded {count} vectors...")

    except Exception as e:
        print(f"Error loading GloVe vectors from {file_path}: {e}")
        sys.exit(1) # Exit if loading fails

    print(f"Finished loading {len(word_vectors)} word vectors with dimension {vector_dim}")
    # Basic check to ensure some vectors were loaded
    if not word_vectors:
         print("Error: No word vectors were loaded. Check the GloVe file.")
         sys.exit(1)


def find_centroid(word_list):
    """Calculate the centroid of the word vectors for the given words"""
    vectors = []

    for word in word_list:
        word = word.lower()
        if word in word_vectors:
            vectors.append(word_vectors[word])
    # Consider adding a check if all words were OOV
    # if len(vectors) < len(word_list):
    #    print(f"Warning: {len(word_list) - len(vectors)} input word(s) not found in vocabulary.")

    if not vectors:
        return None # Return None if no words were found in the vocabulary

    # Calculate the centroid
    centroid = np.mean(vectors, axis=0)
    return centroid

# Modified function to return top N words
def find_closest_words(centroid, exclude_words=None, top_n=5):
    """Find the word(s) with the closest vector(s) to the centroid"""
    if exclude_words is None:
        exclude_words_lower = set()
    else:
        # Ensure excluded words are also lowercased for consistent comparison
        exclude_words_lower = {word.lower() for word in exclude_words}

    # Store results as a list of (distance, word) tuples
    distances = []

    # Pre-calculate centroid norm if using cosine similarity manual calculation
    # (scipy.spatial.distance.cosine handles this internally)
    # centroid_norm = np.linalg.norm(centroid)
    # if centroid_norm == 0:
    #     return [] # Cannot calculate distance if centroid is zero vector

    for word, vector in word_vectors.items():
        # Skip words in the exclude list
        if word in exclude_words_lower:
            continue

        # Calculate cosine distance
        # Handle potential division by zero if a vector is all zeros (shouldn't happen with GloVe)
        # Or if the centroid is all zeros (only happens if all input words are OOV and find_centroid returns None, which is handled)
        try:
             distance = cosine(centroid, vector)
             # Cosine distance returns 0.0 for identical non-zero vectors, 1.0 for orthogonal,
             # and is undefined or handled by throwing ValueError for zero vectors.
             # If centroid or vector is zero, cosine similarity is often considered 0, distance 1.
             # scipy.spatial.distance.cosine handles zero vectors by returning 1.0 if one is zero,
             # and 0.0 if both are zero (which shouldn't happen here with centroid check).
             if np.isnan(distance): # Check for potential NaNs, though unlikely with scipy
                 distance = 1.0 # Treat NaN distance as maximum distance
        except ValueError:
             # This can happen if centroid or vector has zero norm.
             # centroid check happens before this, so this catches zero-norm word vectors.
             # print(f"Warning: Skipping word '{word}' due to zero-norm vector in GloVe data.")
             distance = 1.0 # Treat zero-norm vector as maximum distance
             # continue # Or simply skip the word

        # We are looking for the *smallest* distance
        distances.append((distance, word))

    # Sort by distance (ascending)
    # Use a key for sorting for potentially better performance on large lists
    distances.sort(key=lambda item: item[0])

    # Get the top N results and convert distance to similarity
    top_results = []
    # Iterate only up to top_n or the available number of distances
    # Ensure we don't try to access indices beyond the list size
    num_results_to_return = min(top_n, len(distances))
    for i in range(num_results_to_return):
        dist, word = distances[i]
        # Cosine similarity is 1 - cosine distance
        similarity = 1 - dist
        # Convert numpy float to standard float for JSON serialization robustness
        top_results.append({'word': word, 'similarity_score': float(similarity)})

    return top_results

@app.route('/find_common_word', methods=['POST'])
def find_common_word_api(): # Renamed function to avoid conflict with app name
    """API endpoint to find common word(s) that describe a group of words"""
    data = request.get_json()

    if not data or 'words' not in data:
        return jsonify({'error': 'Input must contain a list of words'}), 400

    words = data['words']
    # Get top_n from the request, default to 5 if not provided
    top_n = data.get('top_n', 5)

    # Input validation
    if not words or not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        return jsonify({'error': 'Words must be provided as a non-empty list of strings'}), 400

    if not isinstance(top_n, int) or top_n <= 0:
         return jsonify({'error': 'top_n must be a positive integer'}), 400

    # Calculate centroid of word vectors
    centroid = find_centroid(words)

    # Check if any of the input words were found in the vocabulary
    # find_centroid returns None if no words were found
    if centroid is None:
        # Find which words were missing for a more helpful error message
        missing_words = [word for word in words if word.lower() not in word_vectors]
        error_msg = f"None of the provided words were found in the vocabulary."
        if missing_words:
             # Limit missing words shown to avoid huge messages
             missing_display = missing_words[:10] + (['...'] if len(missing_words) > 10 else [])
             error_msg += f" Missing words ({len(missing_words)} total): {', '.join(missing_display)}"
        return jsonify({'error': error_msg}), 400


    # Find the closest words to the centroid, excluding input words
    # Pass the input 'words' list to be excluded
    common_words_results = find_closest_words(centroid, exclude_words=words, top_n=top_n)

    # Check if any relevant words were found after exclusion
    if not common_words_results:
         return jsonify({'error': 'Could not find any related words in the vocabulary (excluding input words). This might happen if input words are very specific, cover a concept poorly represented, or if the vocabulary is small.'}), 400


    result = {
        'input_words': words,
        'top_n_requested': top_n, # Indicate how many were requested
        'common_words': common_words_results # This is now a list of results
    }

    return jsonify(result)

# --- GloVe Loading happens here when the module is imported ---
# This will run when Gunicorn starts the app
download_glove_vectors()
load_glove_vectors()
# ------------------------------------------------------------


if __name__ == '__main__':
    # This block is primarily for simple direct execution (e.g., python app.py)
    # For production, use Gunicorn as defined in the Dockerfile/docker-compose
    print("Running in development mode. Use Gunicorn for production deployments.")
    # app.run(debug=True, host='0.0.0.0', port=4001, threaded=True) # Removed for production setup
    # You could keep this for convenience if you sometimes want to run it without Docker/Gunicorn,
    # but be mindful it's the development server. If kept, adjust port or logic
    # to not conflict if running inside Docker compose which maps port 4001.
    # A common pattern is to check an environment variable:
    # if os.environ.get('FLASK_ENV') == 'development':
    #    app.run(debug=True, ...)
    pass # Or just add a message