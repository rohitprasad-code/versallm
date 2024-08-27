# VersaLLM

VersaLLM is a versatile language model client that supports multiple backend models including OpenAI, Anthropic, and Groq. It provides a unified interface for interacting with these models and includes tools for executing functions based on user prompts.

## Features

- **Versatile Multi-Backend Compatibility**: Effortlessly transition between leading language models like OpenAI, Anthropic, and Groq, using a unified interface that simplifies integration and operation.
- **Robust Conversational Memory**: Automatically track and manage conversation history, ensuring seamless dialogue flow and context retention without the need for manual oversight.
- **Consistent Tool Integration**: Simplify the incorporation of external tools with a uniform interface across different language models, allowing for smooth and consistent functionality regardless of the backend.
- **Error Handling and Logging**: Robust error handling and logging for better debugging and monitoring.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Mohwit/versallm
    cd versallm
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Simple Usage

You can initialize a client for any supported model. For example, to initialize an 'claude-3-opus-20240229' of anthropic, you can use the following code:

```python
## import the VersaLLM class
from versallm import VersaLLM

## create a llm client
client = VersaLLM(model="claude-3-opus-20240229", api_key="your_api_key")

## set your system prompt
client.system_message("your system message")

## use completion method to get the response
response = client.completion("your prompt")
print(response)
```

### Using Function calling

### Defining Tools
Simply define your function and define the tools schema as shown below:

```python
def get_customer_info(customer_id):
    # Your implementation here
    pass

def get_order_details(order_id):
    # Your implementation here
    pass

def cancel_order(order_id):
    # Your implementation here
    pass

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_info",
            "description": "your description",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "your description"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "your description",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "your description"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "your description",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "your description"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
```

#### Initialize the client, and pass the list of functions to the during client initialization and pass tools schema to the completion method:

```python
## import the VersaLLM class
from versallm import VersaLLM

## create the llm client
client = VersaLLM(model="claude-3-opus-20240229", functions=[get_customer_info, get_order_details, cancel_order])

## set your system prompt
client.system_message("your system message")

## use completion method to get the response by passing the tools schema
response = client.completion("your prompt", tools=tools)

## printing the client response
for message in response:
    print(message)
```

## Project Structure

- `versallm/llms/`: Contains the client implementations for different models.
- `versallm/utils/`: Utility modules including response handling and conversational memory.

## Contributing

Contributions are welcome! Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
```