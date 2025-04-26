# Word Vector Similarity Microservice

This microservice provides an API to find common words that semantically describe a group of input words using pre-trained GloVe word vectors. It uses vector similarity to identify words that are conceptually related to a given set of words.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Docker Installation](#docker-installation)
  - [Manual Installation](#manual-installation)
- [Usage](#usage)
  - [Development Mode](#development-mode)
  - [Production Mode](#production-mode)
- [API Documentation](#api-documentation)
  - [Endpoint: `/find_common_word`](#endpoint-find_common_word)
  - [Request Format](#request-format)
  - [Response Format](#response-format)
  - [Example API Calls](#example-api-calls)
- [Technical Details](#technical-details)
  - [GloVe Vectors](#glove-vectors)
  - [Algorithm](#algorithm)
  - [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This microservice leverages pre-trained GloVe word embeddings to find common semantic concepts among a set of words. It calculates the centroid (average) of the word vectors for the input words and then finds the closest words to this centroid in the vector space, excluding the input words themselves.

## Features

- Find semantically related words for a group of input words
- Uses pre-trained GloVe 6B word vectors (300-dimensional)
- REST API with JSON input/output
- Docker support for easy deployment
- Persistent volume for storing GloVe data
- Configurable number of results

## Requirements

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- 2GB+ RAM (GloVe vectors require significant memory)
- ~1GB disk space (for GloVe vectors)

## Installation

### Docker Installation

This is the recommended method for both development and production environments:

1. Clone this repository:
   ```bash
   git clone https://your-repository-url.git
   cd word-vector-similarity-service
   ```

2. Build and start the Docker container:
   ```bash
   docker compose up -d
   ```

   This will:
   - Build the Docker image
   - Start the container
   - Download the GloVe vectors (on first run)
   - Expose the API on port 4001

3. Check if the service is running:
   ```bash
   docker compose logs -f
   ```

   Look for the message "Finished loading X word vectors with dimension 300" to confirm the service is ready.

### Manual Installation

If you prefer to run the service without Docker:

1. Clone this repository:
   ```bash
   git clone https://your-repository-url.git
   cd word-vector-similarity-service
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the service:
   ```bash
   python -m gunicorn -w 4 -b 0.0.0.0:4001 app:app
   ```

4. The service will download GloVe vectors on first run (this may take a few minutes).

## Usage

### Development Mode

For development (with pretty-printed JSON responses), you can use either Docker or manual installation:

1. **Docker Development**:
   ```bash
   # Use the development configuration override
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up
   
   # Make changes to the code, then rebuild and restart
   docker compose down
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
   ```

   The development configuration sets `FLASK_ENV=development`, which enables:
   - Pretty-printed JSON responses with 2-space indentation
   - Flask development server with auto-reloading
   - Source code directory mounted for live changes

2. **Manual Development**:
   ```bash
   # Set environment variables for development mode
   export FLASK_ENV=development
   export FLASK_APP=app.py
   
   # Start Flask's development server
   flask run --host=0.0.0.0 --port=4001
   ```

### Production Mode

For production deployment (with minified JSON responses):

1. **Docker Production** (recommended):
   ```bash
   # Standard production deployment
  docker compose up -d
   ```

   The default Docker Compose setup sets `FLASK_ENV=production`, which:
   - Minifies JSON responses by removing whitespace
   - Uses Gunicorn with 4 workers (modify in Dockerfile's CMD line as needed)

2. **Manual Production**:
   ```bash
   # Set environment variable for production mode
   export FLASK_ENV=production
   
   # Start Gunicorn server
   gunicorn -w 4 -b 0.0.0.0:4001 app:app
   ```

   Consider using a process manager like supervisord to manage the Gunicorn process.

## API Documentation

### Endpoint: `/find_common_word`

This endpoint takes a list of words and returns the top N most semantically similar words that conceptually connect them.

### Request Format

**Method**: POST

**Content-Type**: application/json

**Body**:
```json
{
  "words": ["word1", "word2", "word3", ...],
  "top_n": 5
}
```

Parameters:
- `words`: (Required) Array of strings. The input words to find common concepts for.
- `top_n`: (Optional) Integer. The number of results to return. Default is 5.

### Response Format

**Success Response (200 OK)**:
```json
{
  "input_words": ["word1", "word2", "word3", ...],
  "top_n_requested": 5,
  "common_words": [
    {
      "word": "result1",
      "similarity_score": 0.85
    },
    {
      "word": "result2",
      "similarity_score": 0.82
    },
    ...
  ]
}
```

**Error Response (400 Bad Request)**:
```json
{
  "error": "Error message explaining what went wrong"
}
```

Common error messages:
- "Input must contain a list of words"
- "Words must be provided as a non-empty list of strings"
- "None of the provided words were found in the vocabulary"

### Example API Calls

**Example 1**: Find common words for fruits

```bash
curl -X POST http://localhost:4001/find_common_word \
  -H "Content-Type: application/json" \
  -d '{"words": ["apple", "banana", "orange", "grape"], "top_n": 3}'
```

Example response:
```json
{
  "input_words": ["apple", "banana", "orange", "grape"],
  "top_n_requested": 3,
  "common_words": [
    {
      "word": "fruit",
      "similarity_score": 0.7823
    },
    {
      "word": "fruits",
      "similarity_score": 0.7156
    },
    {
      "word": "berry",
      "similarity_score": 0.6893
    }
  ]
}
```

**Example 2**: Find common words for programming languages

```bash
curl -X POST http://localhost:4001/find_common_word \
  -H "Content-Type: application/json" \
  -d '{"words": ["python", "javascript", "java", "c++"], "top_n": 5}'
```

Example response:
```json
{
  "input_words": ["python", "javascript", "java", "c++"],
  "top_n_requested": 5,
  "common_words": [
    {
      "word": "programming",
      "similarity_score": 0.8342
    },
    {
      "word": "languages",
      "similarity_score": 0.7965
    },
    {
      "word": "programmers",
      "similarity_score": 0.7621
    },
    {
      "word": "coding",
      "similarity_score": 0.7509
    },
    {
      "word": "software",
      "similarity_score": 0.7490
    }
  ]
}
```

## Technical Details

### GloVe Vectors

This service uses Stanford's GloVe (Global Vectors for Word Representation) pre-trained word vectors. Specifically, it uses the 300-dimensional vectors trained on 6 billion tokens from Wikipedia 2014 + Gigaword 5.

Key facts about the vectors:
- 400,000 word vocabulary
- 300 dimensions per word
- Trained using window-based co-occurrence statistics
- More information: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)

### Algorithm

1. **Centroid Calculation**: The service calculates the centroid (average vector) of all input words found in the vocabulary.
2. **Similarity Calculation**: Using cosine similarity, it finds the words with vectors closest to the centroid.
3. **Filtering**: The original input words are excluded from the results.
4. **Ranking**: Results are sorted by similarity score in descending order.

### Environment-based JSON Formatting

The service dynamically formats JSON responses based on the environment:

- **Development Mode**: JSON responses are pretty-printed with 2-space indentation and sorted keys
  ```json
  {
    "input_words": [
      "apple",
      "banana"
    ],
    "top_n_requested": 3,
    "common_words": [
      {
        "similarity_score": 0.7823,
        "word": "fruit"
      }
    ]
  }
  ```

- **Production Mode**: JSON responses are minified to reduce bandwidth
  ```json
  {"input_words":["apple","banana"],"top_n_requested":3,"common_words":[{"similarity_score":0.7823,"word":"fruit"}]}
  ```

The formatting is controlled by the `FLASK_ENV` environment variable:
- Set to `development` for pretty-printed JSON
- Set to `production` (or anything else) for minified JSON

### Performance Considerations

- **Memory Usage**: The GloVe vectors require approximately 1-2GB of RAM when loaded.
- **Startup Time**: First startup may take several minutes to download and load vectors.
- **Persistence**: The Docker setup uses a named volume to persist GloVe data across container restarts.
- **Scaling**: The service uses Gunicorn with multiple workers for better performance under load.

## Troubleshooting

**Problem**: Service fails to start with memory errors.
**Solution**: Ensure your system has at least 2GB of free RAM. Reduce the number of Gunicorn workers if needed.

**Problem**: GloVe vectors fail to download.
**Solution**: Check your internet connection. You can also manually download and extract the vectors from the Stanford website and place them in the `glove` directory.

**Problem**: API returns "None of the provided words were found in the vocabulary."
**Solution**: Ensure your input words are common English words. GloVe has a large but limited vocabulary. Try using the base form of words (e.g., "cat" instead of "cats").

**Problem**: Docker container exits unexpectedly.
**Solution**: Check logs with `docker compose logs`. Ensure you have enough disk space and RAM available.

## License

This project uses the GloVe vectors, which are licensed under the Apache License, Version 2.0. Your use of the GloVe vectors should comply with this license.