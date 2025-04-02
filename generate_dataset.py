import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because PyTorch wasn't built with MPS enabled.")
        device = "cpu"
    else:
        print("MPS not available because macOS version is not 12.3+ or you don't have an MPS-enabled device.")
        device = "cpu"
else:
    device = "mps"
    print(f"Using MPS device for GPU acceleration on Apple Silicon")

model_id = "microsoft/Phi-3-mini-4k-instruct"
# model_id = "microsoft/phi-2"

print(f"Loading model: {model_id} on device: {device}")

generator = pipeline(
    "text-generation",
    model=model_id,
    device_map=device,
    torch_dtype=torch.float16,
    model_kwargs={
        "low_cpu_mem_usage": True,
    }
)


def generate(prompt, max_length=128, temperature=0.7):
    response = generator(
        prompt,
        max_new_tokens=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.eos_token_id,
        use_cache=True,
    )[0]['generated_text']

    if prompt in response:
        return response[len(prompt):].strip()
    return response.strip()


print(f"Model initialized successfully on {device}")

confusion_types = [
    "Confuse Past and Present Events",
    "Misremember Scheduled Activities",
    "Reference Non-Existent Information",
    "Confusion about Current Dates",
    "Reference Incorrect Location/Context",
    "Repeat Questions",
    "Confusion About Life Stage",
    "Incomplete or Vague Statements",
]


def generate_life_events(num_events):
    events = []

    event_categories = [
        "Family Gatherings", "Vacations", "Career Milestones",
        "Celebrations", "Daily Activities", "Medical Appointments",
        "Hobbies", "Social Events", "Educational Experiences",
        "Object Locations", "Medications",
    ]

    prompt = """
    Generate a detailed description of a significant life event for an elderly person. 
    Include specific details about:
    - When and where it happened (specific date and location)
    - Who was present (names and relationships)
    - What activities occurred
    - Notable conversations or moments
    - Any emotional significance

    The event should be in the category: {category}
    Make the event between 150-200 words, detailed enough to contain multiple facts
    that could be remembered or forgotten. Write from a third-person perspective.
    """

    for _ in tqdm(range(num_events), desc="Generating events"):
        category = np.random.choice(event_categories)
        event = generate(prompt.format(category=category))
        events.append(event.strip())

    return events


def generate_incomplete_memories(events):
    memories = []

    memory_prompt = """
    Original Event: {event}

    Create an incomplete and somewhat confused memory of this event from the 
    perspective of an elderly person with mild dementia. The memory should:
    - Include some accurate details from the event
    - Contain some inaccuracies or gaps
    - Reflect the confusion type: {confusion_type}
    - Be written in first-person perspective ("I remember...")
    - Be 50-100 words in length
    """

    for event in tqdm(events, desc="Generating memories"):
        confusion = np.random.choice(confusion_types)
        memory = generate(memory_prompt.format(
            event=event,
            confusion_type=confusion
        ))
        memories.append(memory.strip())

    return memories


def generate_queries(events, memories, in_event_count=500):
    queries = []
    is_in_event = []  # Track whether query info is in event

    # For queries about information IN the event
    in_event_prompt = """
    Event: {event}
    Patient's Memory: {memory}

    Generate a natural question that the patient might ask about this event, where 
    the answer IS clearly stated in the original event description but might be
    missing or confused in their memory. The question should sound natural 
    as if asked by an elderly person with dementia.
    """

    # For queries about information NOT in the event
    not_in_event_prompt = """
    Event: {event}
    Patient's Memory: {memory}

    Generate a natural question that the patient might ask about this event, where 
    the answer is NOT clearly stated in the original event description. The question 
    should be related to the event but ask for details that weren't documented. 
    The question should sound natural as if asked by an elderly person with dementia.
    """

    # Generate in-event queries
    for i in tqdm(range(in_event_count), desc="Generating in-event queries"):
        query = generate(in_event_prompt.format(
            event=events[i],
            memory=memories[i]
        ))
        queries.append(query.strip())
        is_in_event.append(True)

    # Generate not-in-event queries
    for i in tqdm(range(in_event_count, len(events)), desc="Generating not-in-event queries"):
        query = generate(not_in_event_prompt.format(
            event=events[i],
            memory=memories[i]
        ))
        queries.append(query.strip())
        is_in_event.append(False)

    return queries, is_in_event


def generate_responses(events, memories, queries, is_in_event):
    responses = []

    # For queries with information in the event
    in_event_response_prompt = """
    Event: {event}
    Patient's Memory: {memory}
    Patient's Query: {query}

    Generate a compassionate and helpful response from a memory assistant. The response should:
    - Directly answer the patient's question with information from the original event
    - Be gentle and supportive in tone
    - Provide the specific facts requested
    - Acknowledge any confusion in a respectful way
    - Be 2-3 sentences long
    """

    # For queries without information in the event
    not_in_event_response_prompt = """
    Event: {event}
    Patient's Memory: {memory}
    Patient's Query: {query}

    Generate a compassionate response from a memory assistant for a question about 
    information NOT contained in the original event description. The response should:
    - NOT make up information that wasn't in the original event
    - Gently ask 2-3 prompting questions to help the patient recall more details
    - Be supportive and encouraging
    - Acknowledge the difficulty of remembering
    - Be 3-4 sentences long
    """

    for i in tqdm(range(len(queries)), desc="Generating responses"):
        if is_in_event[i]:
            prompt = in_event_response_prompt
        else:
            prompt = not_in_event_response_prompt

        response = generate(prompt.format(
            event=events[i],
            memory=memories[i],
            query=queries[i]
        ))
        responses.append(response.strip())

    return responses


def create_dataset(events, memories, queries, responses, is_in_event):
    df = pd.DataFrame({
        "event": events,
        "memory": memories,
        "query": queries,
        "response": responses,
        "info_in_event": is_in_event
    })
    return df


def validate_dataset(df):
    # Check balance of in_event vs not_in_event queries
    in_event_count = df['info_in_event'].sum()
    not_in_event_count = len(df) - in_event_count

    print(f"Queries with info in event: {in_event_count}")
    print(f"Queries with info not in event: {not_in_event_count}")

    # Check for appropriate response types
    question_in_event = 0
    no_question_not_in_event = 0

    for i, row in df.iterrows():
        if row['info_in_event'] and "?" in row['response']:
            question_in_event += 1
        elif not row['info_in_event'] and "?" not in row['response']:
            no_question_not_in_event += 1

    print(f"Warning: {question_in_event} rows have info in event but response contains questions")
    print(f"Warning: {no_question_not_in_event} rows have info not in event but response doesn't contain questions")

    return question_in_event == 0 and no_question_not_in_event == 0


def clear_gpu_memory():
    if torch.backends.mps.is_available():
        # Force garbage collection
        import gc
        gc.collect()
        # Clear CUDA cache if using CUDA
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def main():
    clear_gpu_memory()

    total_examples = 100
    in_event_count = 50

    batch_size = 5  # Smaller batches use less memory

    all_events = []
    all_memories = []
    all_queries = []
    all_responses = []
    all_is_in_event = []

    # Process in batches
    for batch_idx in range(0, total_examples, batch_size):
        batch_end = min(batch_idx + batch_size, total_examples)
        current_batch_size = batch_end - batch_idx
        print(f"Processing batch {batch_idx // batch_size + 1}/{(total_examples - 1) // batch_size + 1}")

        # Generate batch of events
        events = generate_life_events(current_batch_size)
        all_events.extend(events)

        # Clear memory before next intensive operation
        clear_gpu_memory()

        # Generate batch of memories
        memories = generate_incomplete_memories(events)
        all_memories.extend(memories)

        clear_gpu_memory()

        # Generate batch of queries
        in_event_for_batch = min(in_event_count - sum(all_is_in_event), current_batch_size)
        queries, is_in_event = generate_queries(events, memories, in_event_for_batch)
        all_queries.extend(queries)
        all_is_in_event.extend(is_in_event)

        clear_gpu_memory()

        # Generate batch of responses
        responses = generate_responses(events, memories, queries, is_in_event)
        all_responses.extend(responses)

        # Save checkpoint
        temp_df = create_dataset(
            all_events, all_memories, all_queries, all_responses, all_is_in_event
        )
        temp_df.to_pickle(f"checkpoint_batch_{batch_idx // batch_size + 1}.pkl")

        # Memory cleanup
        clear_gpu_memory()

        print(f"Completed {len(all_events)}/{total_examples} examples")

    # Create final dataset
    df = create_dataset(all_events, all_memories, all_queries, all_responses, all_is_in_event)
    validate_dataset(df)
    df.to_csv("dataset.csv", index=False)
    print("Dataset generation completed successfully!")

    return df


if __name__ == "__main__":
    main()
