from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from .utils.logger import logger
import json
import os

from google.generativeai import configure, GenerativeModel


def clean_schema_recursively(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively clean a JSON schema to make it compatible with Google's API."""
    if not isinstance(schema, dict):
        return schema
    
    # Create a new dict to avoid modifying the original
    cleaned = {}
    
    for key, value in schema.items():
        # Skip 'title' fields
        if key == 'title':
            continue
        
        # Recursively clean nested dictionaries
        if isinstance(value, dict):
            cleaned[key] = clean_schema_recursively(value)
        # Recursively clean items in lists
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_recursively(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    
    return cleaned


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Get API key from environment variables
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            self.logger.warning("GOOGLE_API_KEY not found in environment variables")
        else:
            # Configure the Google API with the key
            configure(api_key=api_key)
        
        # Initialize Google Generative AI model
        self.llm = GenerativeModel(model_name="gemini-2.0-flash")
        self.tools = []
        self.messages = []
        self.logger = logger
        
    def add_message(self, role: str, content: Any):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user', 'assistant', or 'tool')
            content: The content of the message (can be string, dict, or list)
        """
        try:
            message = {"role": role, "content": content}
            self.messages.append(message)
            self.logger.debug(f"Added message: {message}")
        except Exception as e:
            self.logger.error(f"Error adding message: {e}", exc_info=True)
            raise

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            self.logger.info("Connected to MCP server")

            mcp_tools = await self.get_mcp_tools()
            # Convert MCP tools to Google's function calling format
            self.tools = []
            for tool in mcp_tools:
                # Create a Google-compatible schema by recursively cleaning it
                schema = tool.inputSchema
                clean_schema = clean_schema_recursively(schema)
                
                # Ensure we have the minimum required fields for a valid schema
                if isinstance(clean_schema, dict):
                    if 'type' not in clean_schema:
                        clean_schema['type'] = 'object'
                else:
                    # If schema is not a dict, create a minimal valid schema
                    clean_schema = {'type': 'object', 'properties': {}}
                
                self.tools.append({
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": clean_schema,
                        }
                    ]
                })
                
                # Log the cleaned schema for debugging
                self.logger.info(f"Cleaned schema for {tool.name}: {clean_schema}")

            # Log available tools
            tool_names = []
            if self.tools and 'function_declarations' in self.tools[0]:
                tool_names = [tool['name'] for tool in self.tools[0]['function_declarations']]
            self.logger.info(f"Available tools: {tool_names}")

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            self.add_message("user", query)
            
            # Initial LLM call
            response = await self.call_llm()
            
            # Process the response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Handle text response
                            if hasattr(part, 'text') and part.text:
                                self.add_message("assistant", part.text)
                                return part.text
                                
                            # Handle function calls
                            elif hasattr(part, 'function_call'):
                                func_name = part.function_call.name
                                
                                # Parse arguments safely
                                tool_args = {}
                                try:
                                    if hasattr(part.function_call, 'args'):
                                        if isinstance(part.function_call.args, str):
                                            tool_args = json.loads(part.function_call.args)
                                        elif hasattr(part.function_call.args, 'items'):
                                            # Convert to a regular dict
                                            tool_args = {k: v for k, v in part.function_call.args.items()}
                                except Exception as e:
                                    self.logger.error(f"Error parsing function args: {e}")
                                    continue
                                
                                self.logger.info(f"Calling tool {func_name} with args {tool_args}")
                                try:
                                    # Call the tool with the parsed arguments
                                    result = await self.session.call_tool(func_name, tool_args)
                                    result_content = result.content if hasattr(result, 'content') else str(result)
                                    
                                    # Add tool result to conversation
                                    self.add_message("tool", {
                                        "tool_use_id": f"call_{func_name}",
                                        "content": str(result_content)
                                    })
                                    
                                    # Call LLM again with tool result
                                    response = await self.call_llm()
                                    print(response,"aaaaaaaaaaaaaaaaaa")
                                    if hasattr(response, 'candidates') and response.candidates:
                                        for candidate in response.candidates:
                                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                                for part in candidate.content.parts:
                                                    # Handle text response
                                                    if hasattr(part, 'text') and part.text:
                                                        self.add_message("assistant", part.text)
                                                        return part.text
                                                    # Handle function call
                                                    elif hasattr(part, 'function_call'):
                                                        func_name = part.function_call.name
                                                        # Convert MapComposite to a regular dict
                                                        func_args = dict(part.function_call.args.items()) if hasattr(part.function_call.args, 'items') else {}
                                                        self.logger.info(f"Calling function: {func_name} with args: {func_args}")
                                                        print("gokul",[msg.get('role') for msg in self.messages[-5:]])
                                                        # Check if we're already in a function call to prevent infinite loops
                                                       
                                                        # Find and call the tool function
                                                        tool_found = False
                                                        for tool in self.tools:
                                                            for func_decl in tool.get('function_declarations', []):
                                                                if func_decl['name'] == func_name:
                                                                    tool_found = True
                                                                    try:
                                                                        # Call the tool function with the provided arguments
                                                                        result = await self.session.call_tool(func_name, func_args)
                                                                        
                                                                        # If we get a result, format it and return directly
                                                                        if result:
                                                                            return f"Here's what I found about {func_args.get('query', 'your query')} in {func_args.get('library', 'the library')}:\n\n{result}"
                                                                        else:
                                                                            return "I couldn't find any relevant information. Could you try a different query?"
                                                                    except Exception as e:
                                                                        self.logger.error(f"Error executing tool {func_name}: {e}")
                                                                        return f"I encountered an error while processing your request: {str(e)}"
                                                        
                                                        if not tool_found:
                                                            error_msg = f"I don't have access to the '{func_name}' tool right now."
                                                            self.logger.error(error_msg)
                                                            return error_msg
                                    
                                except Exception as e:
                                    self.logger.error(f"Error calling tool {func_name}: {e}")
                                    continue
            
            # If we get here, we didn't find any text response
            error_msg = "Sorry, I couldn't generate a response. Please try again."
            self.logger.error(f"No valid response found in: {response}")
            return error_msg
            
        except Exception as e:
            self.logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}"

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            # Convert messages to Google's format if needed
            google_messages = []
            for msg in self.messages:
                role = msg["role"]
                content = msg["content"]
                
                # Handle string content
                if isinstance(content, str):
                    google_messages.append({"role": role, "parts": [{"text": content}]})
                # Handle list content (for tool results)
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            # Format tool result for Google API
                            parts.append({
                                "function_response": {
                                    "name": item.get("tool_use_id", "").split("_")[-1],  # Extract tool name
                                    "response": {"content": item.get("content", "")}
                                }
                            })
                        else:
                            # Default to text for other types
                            parts.append({"text": str(item)})
                    google_messages.append({"role": role, "parts": parts})
            
            self.logger.info(f"Google messages: {google_messages}")
            self.logger.info(f"Tools: {self.tools}")
            
            # Call the Google Generative AI model
            response = self.llm.generate_content(
                google_messages,
                generation_config={"max_output_tokens": 1000},
                tools=self.tools if self.tools else None
            )
            return response
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                # Create a basic serializable message structure
                serializable_message = {"role": message["role"]}

                # Handle different content types
                if isinstance(message["content"], str):
                    # String content can be used directly
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    # For list content, we need to process each item
                    serializable_content = []
                    for content_item in message["content"]:
                        # Convert various object types to dictionaries
                        if hasattr(content_item, "to_dict"):
                            serializable_content.append(content_item.to_dict())
                        elif hasattr(content_item, "dict"):
                            serializable_content.append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_content.append(content_item.model_dump())
                        elif isinstance(content_item, dict):
                            # Make sure all dict values are serializable
                            safe_dict = {}
                            for k, v in content_item.items():
                                # Convert any non-serializable values to strings
                                if isinstance(v, (str, int, float, bool, type(None))):
                                    safe_dict[k] = v
                                else:
                                    safe_dict[k] = str(v)
                            serializable_content.append(safe_dict)
                        else:
                            # For any other type, convert to string
                            serializable_content.append(str(content_item))
                    
                    serializable_message["content"] = serializable_content
                else:
                    # For any other type, convert to string
                    serializable_message["content"] = str(message["content"])

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message for logging: {str(e)}")
                self.logger.debug(f"Problematic message: {str(message)[:200]}...")
                # Don't raise the exception, just log it and continue
                continue

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            # Log a truncated version of the conversation to avoid overwhelming logs
            self.logger.debug(f"Serializable conversation sample: {str(serializable_conversation)[:500]}...")
            # Don't raise the exception, just log it