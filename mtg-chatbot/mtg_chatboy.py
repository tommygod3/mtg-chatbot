import aiml, requests
from PIL import Image
from io import BytesIO
from requests.exceptions import HTTPError

from scryfall import Scryfall

class Chatbot:
    def __init__(self):
        self.set_kernel()
        self.scryfall_api = Scryfall()

    def is_command(self, user_input):
        return user_input[0] == "#"

    def get_command_and_parameter(self, user_input):
        command = user_input[1:].split("$")[0]
        parameter = user_input[1:].split("$")[1]
        return command, parameter

    def set_kernel(self):
        self.kernel = aiml.Kernel()
        self.kernel.setTextEncoding(None)
        self.kernel.bootstrap(learnFiles="aiml-mtg.xml")

    def print_description(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(f'{card["name"]}:')
            print(card["mana_cost"])
            print(card["type_line"])
            print(card["oracle_text"])
        except RuntimeError as e:
            print(f'{e}')

    def print_description_random(self):
        try:
            card = self.scryfall_api.random_card()
            print(f'{card["name"]}:')
            print(card["mana_cost"])
            print(card["type_line"])
            print(card["oracle_text"])
        except RuntimeError as e:
            print(f'{e}')

    def print_colour(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if not card["colors"]:
                print(f"{card['name']} is colourless!")
            else:
                print(card["colors"])
        except RuntimeError as e:
            print(f'{e}')

    def print_cost(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(card["mana_cost"])
        except RuntimeError as e:
            print(f'{e}')

    def print_type(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(card["type_line"])
        except RuntimeError as e:
            print(f'{e}')

    def print_text(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(card["oracle_text"])
        except RuntimeError as e:
            print(f'{e}')

    def print_favourite(self, name):
        try:
            card = self.scryfall_api.get_card(name)
        except RuntimeError as e:
            print(f'{e}')

        print(f"{card['name']}? Cool")
    
    def show_card(self, name):
        try:
            card = self.scryfall_api.get_card(name)
        except RuntimeError as e:
            print(f'{e}')
        response = requests.get(card["image_uris"]["large"])
        img = Image.open(BytesIO(response.content))
        img.show()

    def show_card_random(self):
        try:
            card = self.scryfall_api.random_card()
        except RuntimeError as e:
            print(f'{e}')
        response = requests.get(card["image_uris"]["large"])
        img = Image.open(BytesIO(response.content))
        img.show()

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
                print("I don't understand :(")
            if self.is_command(answer):
                command, parameter = self.get_command_and_parameter(answer)
                if command == "quit":
                    print(parameter)
                    break

                if command == "describe":
                    self.print_description(parameter)
                if command == "describe_random":
                    self.print_description_random()
                if command == "colour":
                    self.print_colour(parameter)
                if command == "cost":
                    self.print_cost(parameter)
                if command == "type":
                    self.print_type(parameter)
                if command == "text":
                    self.print_text(parameter)
                if command == "favourite":
                    self.print_favourite(parameter)
                if command == "show":
                    self.show_card(parameter)
                if command == "show_random":
                    self.show_card_random()

                if command == "default":
                    print(f"No match, what is {parameter}?")
            else:
                print(answer)

Chatbot().run()
