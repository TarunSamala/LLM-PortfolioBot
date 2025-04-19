import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login("TOKEN")

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_portfolio(prompt):
    # System message to guide the LLM
    system_message = """
    You are an expert web developer tasked with creating a professional portfolio website. 
    Based on the user's requirements, generate the complete HTML, CSS, and JavaScript code for a single-page application. 
    The website should be mobile-friendly, SEO-optimized, and use modern design principles. 
    Include sections for Home, About, Projects, Blog, and Contact. 
    Use Tailwind CSS for styling and ensure the code is ready for deployment on platforms like Netlify or Vercel.
    """
    
    # Combine the system message and user prompt
    full_prompt = f"{system_message}\n\nUser requirements: {prompt}"
    
    # Generate the response
    response = chatbot(full_prompt, max_length=4000, num_return_sequences=1, do_sample=True, temperature=0.7)
    
    # Extract the generated code
    generated_code = response[0]["generated_text"]
    
    return generated_code

if __name__ == "__main__":
    # Get user input and generate the portfolio
    user_prompt = input("Enter your portfolio requirements: ")
    portfolio_code = generate_portfolio(user_prompt)
    
    # Save the generated code to a file
    with open("examples/index.html", "w") as f:
        f.write(portfolio_code)
    print("Portfolio generated at examples/index.html")