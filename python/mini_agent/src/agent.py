import openai
import subprocess
import re
import io
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Any

class AgentTool(ABC):
    """Abstract base class for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the description of the tool."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the parameters schema for the tool."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with the provided parameters."""
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the tool to a dictionary format for the OpenAI API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class PythonTool(AgentTool):
    """Tool for executing Python code."""
    
    @property
    def name(self) -> str:
        return "run_python"
    
    @property
    def description(self) -> str:
        return "Execute Python code. Returns the output or error message."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute"
                }
            },
            "required": ["code"]
        }
    
    def execute(self, code: str) -> str:
        """Execute Python code using subprocess and return the result."""
        try:
            # Create a temporary file to hold the Python code
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(code)
            
            # Run the Python code as a separate process
            result = subprocess.run(
                [sys.executable, temp_filename],
                text=True,
                capture_output=True,
                timeout=30  # Timeout after 30 seconds
            )
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
            # Process the results
            if result.returncode != 0:
                return f"Error (exit code {result.returncode}):\n{result.stderr}"
            if result.stdout:
                return f"Output:\n{result.stdout}"
            return "Code executed successfully with no output."
        except subprocess.TimeoutExpired:
            # Clean up temp file if timeout occurs
            try:
                os.unlink(temp_filename)
            except:
                pass
            return "Error: Execution timed out after 30 seconds"
        except Exception as e:
            # Clean up temp file if any other exception occurs
            try:
                os.unlink(temp_filename)
            except:
                pass
            return f"Error: {str(e)}"


class BashTool(AgentTool):
    """Tool for executing bash commands."""
    
    @property
    def name(self) -> str:
        return "run_bash"
    
    @property
    def description(self) -> str:
        return "Execute bash commands and return the result"
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }
    
    def execute(self, command: str) -> str:
        """Execute bash command and return the result."""
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            if result.returncode != 0:
                return f"Error (exit code {result.returncode}):\n{result.stderr}"
            if result.stdout:
                return f"Output:\n{result.stdout}"
            return "Command executed successfully with no output."
        except Exception as e:
            return f"Error: {str(e)}"


class SubmitTool(AgentTool):
    """Tool for submitting final answers."""
    
    def __init__(self, agent):
        self.agent = agent
    
    @property
    def name(self) -> str:
        return "submit"
    
    @property
    def description(self) -> str:
        return "Submit final answers or results"
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to submit"
                }
            },
            "required": ["content"]
        }
    
    def execute(self, content: str) -> str:
        """Process a submission and store it in the agent."""
        self.agent.submission = content.strip()
        return "Submission recorded successfully."


class Agent:
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the agent with configuration parameters.
        
        Args:
            config: Dictionary containing:
                - api_key: OpenAI API key
                - prompt: System prompt for the agent
                - max_iterations: Maximum number of times to prompt the model
                - token_limit: Maximum tokens for model responses
        """
        self.api_key = config.get('api_key')
        self.prompt = config.get('prompt', "You are a helpful AI assistant.")
        self.max_iterations = config.get('max_iterations', 10)
        self.token_limit = config.get('token_limit', 4000)
        self.messages = []
        self.submission = None
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize tools
        self.available_tools = {
            "run_python": PythonTool(),
            "run_bash": BashTool(),
            "submit": SubmitTool(self)
        }
        
        # Convert tools for OpenAI API
        self.tools = [tool.to_dict() for tool in self.available_tools.values()]
        
        # Add system prompt
        system_prompt = f"{self.prompt}\n\n"
        system_prompt += "You have access to tools for executing code and submitting results."
        system_prompt += " Use them whenever you need to run code or submit a final answer."
        
        self.messages.append({"role": "system", "content": system_prompt})
    
    def loop(self, user_input: str, verbose: bool = False):
        """
        Process user input and loop through model invocations until submission or max messages.
        
        Args:
            user_input: The initial user query or problem
            verbose: Whether to print debugging information
        """
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        
        # Manage conversation history size
        if len(self.messages) > self.max_iterations + 1:  # +1 for system message
            self.messages = [self.messages[0]] + self.messages[-(self.max_iterations):]
        
        if verbose:
            print(f"User input: {user_input}")
            self._print_conversation_history()
        
        iteration = 0
        
        while iteration < self.max_iterations and self.submission is None:
            iteration += 1
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            # Get response from model with tool calling capability
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=self.token_limit
            )
            
            response_message = response.choices[0].message

            # Add the full message object to the conversation history
            self.messages.append(response_message)
            
            if verbose:
                print("\nModel response:")
                if response_message.content:
                    print(response_message.content)
                else:
                    print("[No content]")
            
            # If no tool calls, we're done with this iteration
            if not response_message.tool_calls:
                if verbose:
                    print("No tool calls in this response. Continuing to next iteration.")
                continue
            
            # Process each tool call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = eval(tool_call.function.arguments)
                
                if verbose:
                    print(f"\nExecuting tool: {function_name}")
                    print(f"Arguments: {function_args}")
                
                # Execute the appropriate tool
                if function_name in self.available_tools:
                    tool = self.available_tools[function_name]
                    result = tool.execute(**function_args)
                    tool_result = f"{tool.name} execution result:\n{result}"
                else:
                    tool_result = f"Unknown tool {function_name}"
                
                if verbose:
                    print(f"Tool result: {tool_result}")
                
                # Add the tool response to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
                
            if verbose:
                self._print_conversation_history()
        
        if verbose:
            print("\n=== Loop completed ===")
            if self.submission:
                print(f"Final submission: {self.submission}")
            else:
                print("No submission was made after maximum iterations.")
    
    def _print_conversation_history(self):
        """Print the conversation history in a readable format."""
        print("\nConversation history:")
        for i, message in enumerate(self.messages):
            # Check if message is a dict (our manually created messages) or a ChatCompletionMessage
            if isinstance(message, dict):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                if role == "tool":
                    print(f"{i}. {role} ({message.get('name', 'unknown')}): {content[:100]}...")
                else:
                    print(f"{i}. {role}: {content[:100]}...")
            else:
                # Handle ChatCompletionMessage objects
                role = message.role
                content = message.content if message.content else "[No content]"
                print(f"{i}. {role}: {content[:100]}...")
                
                # If it has tool calls, print them too
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"   - Tool call: {tool_call.function.name}")


if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        exit(1)
    
    # Configure the agent
    config = {
        "api_key": api_key,
        "prompt": "You are a helpful AI assistant that solves problems step-by-step.",
        "max_iterations": 5,
        "token_limit": 4000
    }
    
    # Create the agent
    agent = Agent(config)
    
    # Example task that requires using Python tool
    user_query = "Calculate the sum of all prime numbers below 50 and submit the result."
    
    print(f"User query: {user_query}")
    print("\nRunning agent loop...\n")
    
    # Run the agent loop
    agent.loop(user_query, verbose=True)
    
    # Display the submission if available
    if agent.submission:
        print(f"\nFinal submission: {agent.submission}")
    else:
        print("\nNo submission was made.")
