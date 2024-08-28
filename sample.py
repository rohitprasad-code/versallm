import os
from dotenv import load_dotenv
load_dotenv()

from versallm import VersaLLM

# Retrieve the API key from environment variables
api_key = os.getenv("API_KEY_ANTHROPIC")

# Debug: Print the API key to ensure it's loaded correctly
if api_key is None:
    raise ValueError("API key not found. Please check your .env file.")
else:
    print("API key loaded successfully.")

# Define some example functions to be used as tools
def get_customer_info(customer_id):
    # Simulated function to retrieve customer info
    return {"customer_id": customer_id, "name": "John Doe", "status": "Active"}

def get_order_details(order_id):
    # Simulated function to retrieve order details
    return {"order_id": order_id, "item": "Laptop", "quantity": 1}

def cancel_order(order_id):
    # Simulated function to cancel an order
    return {"order_id": order_id, "status": "Cancelled"}

# Define tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_info",
            "description": "Retrieve customer information by customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The unique ID of the customer."
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
            "description": "Retrieve details of an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique ID of the order."
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
            "description": "Cancel an order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique ID of the order."
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]

# Initialize the VersaLLM client with an Anthropic model
client = VersaLLM(
    model="claude-3-opus-20240229",  # specify the model
    api_key=api_key,
    functions=[get_customer_info, get_order_details, cancel_order]
)

# Set the system message (context) for the conversation
client.system_message("You are a helpful assistant.")

# User prompt
user_prompt = "Can you tell me the details of the order with ID 12345?"



# Get the response and print the messages as they come in
try:
    response = client.completion(user_prompt, tools=tools)
    for res in response:
        print(res.message)
except Exception as e:
    print(f"An error occurred: {e}")
