import aiml
from requests.exceptions import HTTPError

from scryfall_api import Scryfall

class Chatbot:
    def __init__(self):
        self.set_kernel()
        self.scryfall_api = Scryfall()

    def is_command(self, user_input):
        return user_input[0] == "#"

    def get_command(self, user_input):
        command_dict = {}
        command = user_input[1:].split("$")[0]
        parameter = user_input[1:].split("$")[1]
        command_dict["command"] = command
        command_dict["parameter"] = parameter
        return command_dict

    def set_kernel(self):
        self.kernel = aiml.Kernel()
        self.kernel.setTextEncoding(None)
        self.kernel.bootstrap(learnFiles="aiml-mtg.xml")

    def run(self):
        print("Welcome to the Magic: The Gathering chatbot! Ask me questions about the game of Magic, as well as cards in the game! I can describe and show you cards.")

        while True:
            try:
                user_input = input("> ")
            except (KeyboardInterrupt, EOFError):
                print("Bye!")
                break

            if not user_input:
                continue

            agent = "aiml"
            if agent == "aiml":
                answer = self.kernel.respond(user_input)
            if not answer:
                print("I don't understand")
            if self.is_command(answer):
                command = self.get_command(answer)
                if command["command"] == "quit":
                    print(command["parameter"])
                    break

                if command["command"] == "describe":
                    try:
                        card = self.scryfall_api.get_card(command["parameter"])
                        print(f'{card["name"]}:')
                        print(card["mana_cost"])
                        print(card["type_line"])
                        print(card["oracle_text"])
                    except RuntimeError as e:
                        print(f'{e}')

                if command["command"] == "default":
                    print(f"No match, what is {command['parameter']}?")
            else:
                print(answer)


Chatbot().run()


# Describe X card?
# Show me X card?
# What colour is X?
# How much does X cost?
# What is your favourite card?
# Show me a random card


