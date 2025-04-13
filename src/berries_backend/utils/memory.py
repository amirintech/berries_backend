"""Memory management for the financial assistant."""

from typing import Optional, List # Added List
import json
import os


class ConversationMemory:
    """
    Manages conversation history for the financial assistant.
    Stores and retrieves conversation history for maintaining context across queries.
    """

    def __init__(self, max_history: int = 15, memory_file: Optional[str] = None):
        """
        Initialize the conversation memory.

        Args:
            max_history: Maximum number of conversation turns to store (each turn is query + response)
            memory_file: Optional path to save conversation history to disk
        """
        self.max_history = max_history
        self.memory_file = memory_file
        # Store as {'role': 'user'/'assistant', 'content': message} for clarity
        self.conversation_history: List[Dict[str, str]] = []

        # Load from file if it exists
        if memory_file and os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    self.conversation_history = json.load(f)
                # Ensure loaded history respects max_history
                if len(self.conversation_history) > self.max_history * 2:
                     self.conversation_history = self.conversation_history[-(self.max_history * 2):]
                print(f"Loaded conversation history from {memory_file} ({len(self.conversation_history)//2} turns)")
            except Exception as e:
                print(f"Error loading conversation history: {e}")


    def add_interaction(self, query: str, response: str) -> None:
        """
        Add a new query-response pair to the conversation history.

        Args:
            query: The user's query
            response: The assistant's response
        """
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Limit history (max_history turns = max_history * 2 messages)
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]

        # Save to file if specified
        if self.memory_file:
            try:
                with open(self.memory_file, 'w') as f:
                    json.dump(self.conversation_history, f, indent=2)
            except Exception as e:
                print(f"Error saving conversation history: {e}")


    def get_history_as_list(self) -> List[str]:
        """
        Get conversation history formatted as a list of strings suitable for QueryProcessor.

        Returns:
            List of alternating "User: ..." and "Assistant: ..." strings.
        """
        result = []
        for interaction in self.conversation_history:
            role = interaction.get("role", "unknown").capitalize()
            content = interaction.get("content", "")
            result.append(f"{role}: {content}")
        return result


    def get_history_as_text(self) -> str:
        """
        Get conversation history formatted as a single text block.

        Returns:
            Formatted conversation history text.
        """
        return "\n".join(self.get_history_as_list())


    def clear(self) -> None:
        """Clear the conversation history in memory and delete the file if it exists."""
        self.conversation_history = []
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
                print(f"Removed conversation history file {self.memory_file}")
            except Exception as e:
                print(f"Error removing conversation history file: {e}")