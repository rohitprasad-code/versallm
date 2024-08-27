from typing import List, Any


class ConversationalMemory(
    List[Any]
):  # Specify the type of elements in the list if needed
    def __init__(self) -> None:
        """
        A class to represent the conversational memory of VersaLLM.
        """
        super().__init__()  # Properly initialize the List
        self.chat_history = []

    def __repr__(self) -> str:
        return f"ConversationalMemory({super().__repr__()})"  # Use the List's repr

    # To be implemented
    def save_memory(self, location: str = "./"):
        pass
