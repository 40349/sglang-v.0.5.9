import json
import requests
from typing import Dict, Any

# def load_and_split_request(file_path: str):
def load_and_split_request(file_path: str, endpoint: str = "http://localhost:8000/generate"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    context = data.get("context", {})
    
    sys_prompt = context.get("systemPrompt", "")
    if sys_prompt:
        # print_subcontext(sys_prompt, "system_prompt_key")
        send_subcontext(endpoint, sys_prompt, "system_prompt_key")

    tools = context.get("tools", [])
    if tools:
        tools_str = json.dumps(tools)
        # print_subcontext(tools_str, "tools_key") 
        send_subcontext(endpoint, tools_str, "tools_key")

    messages = context.get("messages", [])
    if messages:
        msg_str = json.dumps(messages)
        # print_subcontext(msg_str, "messages_key")
        send_subcontext(endpoint, msg_str, "messages_key")

def print_subcontext(text: str, extra_key: str):
    payload = {
        "text": text,
        "extra_key": extra_key,
        "max_new_tokens": 0,
    }
    print(f"--- Subcontext Payload for: {extra_key} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n")

def send_subcontext(endpoint: str, text: str, extra_key: str):
    payload = {
        "text": text,
        "extra_key": extra_key,
        "max_new_tokens": 5000,
    }
    
    try:
        print(f"Sending {extra_key} to server...")
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        print(f"Successfully sent subcontext: {extra_key}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send subcontext {extra_key}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")


if __name__ == "__main__":
    load_and_split_request("/home/t2503-3090/Desktop/MiaoChen/sglang/2026-05-04T14-52-26-258Z_127_0001_chat_google_gemma-4-31b-it.json")