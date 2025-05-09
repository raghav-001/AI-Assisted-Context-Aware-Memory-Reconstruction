- For all of this to work first install ollama (https://ollama.com/download)
- After installing run these command in your cmd
ollama pull llama3:latest
ollama pull mxbai-embed-large 

Now you have the model(llama3:latest) and embeddings locally and you are good to go.
Ensure you have smolagents==1.12.0

1. app.py
This is the main file. When you run this, it starts the assistant and lets you type messages to talk to it.
NOTE: If you want to change the prompts file used, you need to make the change in this file. Line 78 has the code to load the prompts file. Use the file you want to use.

--What it does:

1) Takes your message
2) Checks if you said something similar in the past
3) Sends everything to the AI model
4) Prints a supportive response
5) Saves what you said for future reference

2. retriever_tool.py
This file helps the assistant find old things you said. If you ask something like “Did I mention going swimming?”, this tool searches through db to check.

3. final_answer.py
This file controls how the assistant ends a response. When it’s done thinking, it uses this to output a clear, warm message back to you.

4. prompts3.yaml (Dynamic) (OR) prompts0.yaml (calls retriever_tool() everytime)
This is like the assistant’s brain and personality. This file is super important to make the assistant behave in a helpful and respectful way.

--It contains:

1) Rules for how to think and respond
2) Examples to teach the LLM what to do and how to do
3) Templates that guide how responses are formed

5. db/ folder (Chroma db)
This is where your past conversations are saved. The assistant uses these to remember things you’ve said before. It's handled using ChromaDB.
Right now it already has few conversation. 
NOTE: If you want to start it fresh, just delete the db folder.

6. eval.py is the evaluation script
Computes cosine similarity, ROGUE and Bert scores comparing LLM responses against dataset(Mentalchat16K) outputs for every input

6. rag_eval.py
This is used to evaluate the RAG using DeepEval's LLM-as-a-judge framework for the following metrics:
- Answer Relevancy
- Contextual Relevancy
- Faithfulness
- Hallucination

Steps to run rag_eval.py:
i. pip install deepeval
ii. deepeval set-ollama llama3 --base-url="http://localhost:11434"
iii. Set the ip and op in rag_eval as follows:
    ip: 10 contextual statements given by a patient, followed by 1 query from the patient
    op: 2 responses to the queries from using Dynamic RAG followed by Static RAG
iv. python rag_eval.py
