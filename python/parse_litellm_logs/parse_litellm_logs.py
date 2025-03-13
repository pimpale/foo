import argparse
import json
import ast

# Global variable for the token counter message prefix
TOKEN_COUNTER_MESSAGE_PREFIX = "token_counter messages received: "

def main():
    parser = argparse.ArgumentParser(description="Parse a JSONL file and print its contents.")
    parser.add_argument('file', type=str, help="Path to the JSONL file")
    args = parser.parse_args()

    parsed_lines = []

    with open(args.file, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "message" in data and data["message"].startswith(TOKEN_COUNTER_MESSAGE_PREFIX):
                    # Strip the prefix from the message
                    message_content = data["message"][len(TOKEN_COUNTER_MESSAGE_PREFIX):]

                    try:
                        # Attempt to parse the remaining string as JSON
                        parsed_json = ast.literal_eval(message_content)
                        parsed_lines.append((message_content, parsed_json))
                    except SyntaxError:
                        # If parsing fails, print the original line that caused the error
                        print(f"Failed to parse JSON: {message_content}")
            except json.decode.JSONDecodeError:
                print("Error on line {i}")
    
    # now we need to dedupe the parsed lines
    # convert list into hashmap and then look at the values
    d = {}
    for k, v in parsed_lines:
        if k not in d:
            d[k] = []
        d[k] = v

    deduped_parsed_lines = list(d.values())

    print(json.dumps(deduped_parsed_lines, indent=2))

if __name__ == "__main__":
    main()
