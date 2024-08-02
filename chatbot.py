import re

def get_response(user_input):
    rules = {
        r"hello": "Hello! How can I help you today?",
        r"bye": "Goodbye! Have a nice day!",
        r"how are you": "I'm just a bot, but I'm doing great! How about you?",
        "default": "I'm sorry, I don't understand that."
    }
    
    user_input = user_input.lower()

    for pattern, response in rules.items():
        if re.search(pattern, user_input):
            return response

    return rules["default"]

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")
