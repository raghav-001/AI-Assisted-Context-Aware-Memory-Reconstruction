from smolagents.tools import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Call this tool when you are ready to give the final answer to the user."

    inputs = {
        "answer": {
            "type": "string",
            "description": "The final, human-readable message to the user. It should be empathetic, gentle, and context-aware.",
        }
    }

    output_type = "string"

    def forward(self, answer: str) -> str:
        return answer
