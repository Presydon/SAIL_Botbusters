import os
import json
import re

class SmallTalkManager:
    def __init__(self, json_path="src/conversation/small_talks.json"):
        """ Initializes the small talk manager by loading responses from a JSON file.
        :param json_path: Path to the small talks JSON file.
         """
        
        self.json_path = json_path
        
    def load_small_talks(self):
        """ Loads small talk response from the JSON file. 
            :return: Dictionary of small talk responses,
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"File not Found: {os.path.abspath(self.json_path)}")
        
        with open(self.json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        

    @staticmethod
    def clean_input(user_input):
        """ Cleans user input by removing punctuation and converting to lowercase.
            :param user_input: User's text input.
            :return: Cleaned input string.
        """
        return re.sub(r'[^\w\s]', '', user_input).strip().lower()