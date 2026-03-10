# Cheat Sheet: Web App Deployment Using Flask

## Quick Reference Guide for Flask Concepts

This cheat sheet provides quick reference for common Flask packages, decorators, methods, and HTTP status codes used in web application deployment.

---

## Flask Application Setup

### Flask Class Instantiation

**Purpose:** Used to instantiate an object of the Flask class named app.

```python
from flask import Flask
app = Flask(__name__)
```

This creates the main Flask application object that you'll use throughout your application.

---

## URL Routing

### @app.route Decorator

**Purpose:** A decorator in Flask used to map URLs to specific functions in a Flask application.

```python
@app.route('/')
def hello_world():
    return "My first Flask application in action!"
```

This decorator tells Flask to trigger the `hello_world()` function when a user visits the root URL (`/`) of your application.

---

## HTTP Status Codes

### 200 OK (Success)

**Purpose:** Flask servers automatically return a 200 OK status when you return from the `@app.route` method. 200 is also returned by default when you use the `jsonify()` method to respond to a request. A successful response with a status code of 200 will be sent back when the given code executes.

```python
@app.route('/')
def hello_world():
    return ("My first Flask application in action!", 200)
```

This explicitly returns a 200 status code along with the response message.

---

### Client Error Status Codes (4xx)

**400 Bad Request**
- Indicates an invalid request
- The parameters may be missing, improper, or the request is invalid in another way

**401 Unauthorized**
- Indicates the credentials are missing or invalid
- The client must provide valid authentication

**403 Forbidden**
- Implies that the client credentials are not sufficient to fulfill the request
- The user is authenticated but lacks permission

**404 Not Found**
- If the server is unable to find the resource, it returns a 404 status
- The requested resource does not exist on the server

**405 Method Not Allowed**
- Indicates that the requested operation is not supported
- The HTTP method used is not allowed for that resource

#### Example: Handling 404 Errors

```python
@app.route('/')
def search_response():
    query = request.args.get("q")
    if not query:
        return {"error_message": "Input parameter missing"}, 422
    # fetch the resource from the database
    resource = fetch_from_database(query)
    if resource:
        return {"message": resource}
    else:
        return {"error_message": "Resource not found"}, 404
```

This example demonstrates:
- Retrieving query parameters from the request
- Returning a 422 status for missing input
- Returning a 404 status when a resource is not found

---

### Server Error Status Codes (5xx)

**500 Internal Server Error**
- Used when there is an error on the server
- Indicates an unexpected condition that prevented the server from fulfilling the request

#### Example: Handling 500 Errors

```python
@app.errorhandler(500)
def server_error(error):
    return {"message": "Something went wrong on the server"}, 500
```

This example demonstrates:
- Using the `@app.errorhandler()` decorator to catch server errors
- Returning a custom error response with a 500 status code
- Providing a user-friendly error message

---

## Status Code Summary Table

| Status Code | Category | Meaning | When to Use |
|-------------|----------|---------|------------|
| **200** | Success | OK - Request succeeded | Successful API response |
| **400** | Client Error | Bad Request | Invalid or missing parameters |
| **401** | Client Error | Unauthorized | Missing or invalid credentials |
| **403** | Client Error | Forbidden | Insufficient permissions |
| **404** | Client Error | Not Found | Resource does not exist |
| **405** | Client Error | Method Not Allowed | Unsupported HTTP method |
| **422** | Client Error | Unprocessable Entity | Validation failed |
| **500** | Server Error | Internal Server Error | Server-side error occurred |

---

## Project Setup and Environment

### Create and Navigate Project Directory

**Purpose:** Create and navigate into a new project directory for your Flask and AI application.

```bash
mkdir genai_flask_app
cd genai_flask_app
```

### Virtual Environment Setup

**Purpose:** Set up a Python virtual environment for isolated package management.

```bash
python3.11 -m venv venv
source venv/bin/activate
```

On Windows, use: `venv\Scripts\activate`

---

## IBM Watsonx AI Integration

### Install IBM Watsonx AI Library

**Purpose:** Install the IBM watsonx AI library to interact with Large Language Models.

```bash
pip install ibm-watsonx-ai
```

### Credentials Setup

**Purpose:** Authenticate with IBM watsonx AI using credentials.

```python
from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # api_key="<YOUR_API_KEY>"
)
```

Replace `<YOUR_API_KEY>` with your actual IBM Cloud API key.

### Model Parameters Configuration

**Purpose:** Define parameters for model inference and control output behavior.

```python
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}
```

**Common Parameters:**
- `DECODING_METHOD`: "greedy" for deterministic output or "sample" for varied outputs
- `MAX_NEW_TOKENS`: Maximum number of tokens to generate (100-500 typical)
- `TEMPERATURE`: Controls randomness (0.0-1.0)

### Model Inference Initialization

**Purpose:** Initialize an AI model for text generation using IBM Watsonx.

```python
from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    params=params,
    credentials=credentials,
    project_id="skills-network"
)
```

### Generating AI Responses

**Purpose:** Use an AI model to generate text based on a prompt.

```python
text = """
Only reply with the answer. What is the capital of Canada?
"""

print(model.generate(text)['results'][0]['generated_text'])
```

---

## LangChain Integration

### LangChain Prompt Templates

**Purpose:** Define reusable prompt templates for different models (e.g., Llama 3).

```python
from langchain.prompts import PromptTemplate

llama3_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"]
)
```

### LangChain Chaining

**Purpose:** Pipe a prompt template into an AI model to generate structured output.

```python
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    })
```

The pipe operator (`|`) chains components together for a streamlined data flow.

### Tokenization and Prompt Formatting

**Purpose:** Use specialized token formatting for different AI models (e.g., Llama 3).

```python
# Llama 3 formatted prompt
text = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert assistant who provides concise and accurate answers.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of Canada?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
```

Each model has specific formatting requirements—always check the model's documentation.

### JSON Output Parser

**Purpose:** Parse and structure AI-generated responses using LangChain.

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 to 100")
    response: str = Field(description="Generated AI response")

json_parser = JsonOutputParser(pydantic_object=AIResponse)
```

### Enhancing AI Outputs with Structured Responses

**Purpose:** Modify LangChain chaining to ensure structured JSON output.

```python
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_prompt': json_parser.get_format_instructions()
    })
```

The output will be automatically parsed into the defined JSON structure.

---

## Flask and AI Integration

### Flask API Endpoint for AI Interactions

**Purpose:** Create an API endpoint that integrates Flask with AI model responses.

```python
from flask import Flask, request, jsonify
from model import get_model_response

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model_name = data.get('model')
    user_message = data.get('message')

    if not user_message or not model_name:
        return jsonify({"error": "Missing message or model selection"}), 400

    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a concise response."

    try:
        response = get_model_response(model_name, system_prompt, user_message)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

**Endpoint Usage:**
- **URL:** `POST /generate`
- **Request Body:** `{"model": "granite", "message": "Your question here"}`
- **Response:** JSON object with generated AI response
- **Error Handling:** Returns 400 for missing parameters, 500 for server errors

---

## Key Takeaways

- Always return appropriate HTTP status codes to help clients understand the result of their requests
- Use `@app.route()` to map URLs to functions
- Use `@app.errorhandler()` to handle errors gracefully
- Test your error handling to ensure users receive helpful error messages
- Return status codes with your responses to provide clear feedback
