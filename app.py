import yaml
from smolagents import CodeAgent, HfApiModel
from final_answer import FinalAnswerTool

# Initialize FinalAnswerTool and custom model
final_answer_tool = FinalAnswerTool()
model = HfApiModel(model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud', temperature=0.3)

# Load the prompt templates
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

# Initialize the CodeAgent with the custom model
agent = CodeAgent(
    tools=[final_answer_tool],  
    model=model,  # Use the custom model callable
    max_steps=1,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="Memory Reconstruction Assistant",
    description="An AI assistant helping dementia patients recall verified memories.",
    prompt_templates=prompt_templates  # Use the prompt templates loaded from the file
)

# Initialize conversation history
conversation_history = []

# Function to run the agent with a conversation loop
def maintain_conversation():
    global conversation_history
    while True:
        # Get the user's query (simulate user input for testing purposes)
        user_input = input("Patient: ")

        # Append the user input to the conversation history
        conversation_history.append(f"Patient: {user_input}")
        
        # Create a new prompt using the entire conversation history
        prompt_with_context = "\n".join(conversation_history)
        
        # Run the agent to generate a response with the current context
        response = agent.run(prompt_with_context)
        
        # Display the assistant's response
        print("Assistant: ", response)

        # Append the assistant's response to the conversation history
        conversation_history.append(f"Assistant: {response}")

# Start the conversation
maintain_conversation()
