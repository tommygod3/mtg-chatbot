import aiml, requests, sys, os
import xml.etree.ElementTree as ET
from PIL import Image
from io import BytesIO
from requests.exceptions import HTTPError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scryfall import Scryfall

class Chatbot:
    def __init__(self, aiml_filename):
        self.aiml_filename = self.get_full_filename_path(aiml_filename)
        self.set_kernel()
        self.scryfall_api = Scryfall()
        self.set_patterns()

    def get_full_filename_path(self, filename):
        return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}/{filename}"

    def is_command(self, user_input):
        return user_input[0] == "#"

    def get_command_and_parameter(self, user_input):
        command = user_input[1:].split("$")[0]
        parameter = user_input[1:].split("$")[1]
        return command, parameter

    def set_kernel(self):
        self.kernel = aiml.Kernel()
        self.kernel.setTextEncoding(None)
        self.kernel.bootstrap(learnFiles=self.aiml_filename)

    def set_patterns(self):
        self.patterns = []
        tree = ET.parse(self.aiml_filename)
        all_patterns = tree.findall("*/pattern")
        for pattern in all_patterns:
            if "*" not in pattern.text:
                self.patterns.append(pattern.text)

    def print_description(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(f'{card["name"]}:')
            if "card_faces" in card:
                for face in card["card_faces"]:
                    print(face["mana_cost"])
                    print(face["type_line"])
                    print(face["oracle_text"])
            else:
                print(card["mana_cost"])
                print(card["type_line"])
                print(card["oracle_text"])
        except RuntimeError as e:
            print(f'{e}')

    def print_description_random(self):
        card = self.scryfall_api.random_card()
        print(f'{card["name"]}:')
        if "card_faces" in card:
            for face in card["card_faces"]:
                print(face["mana_cost"])
                print(face["type_line"])
                print(face["oracle_text"])
        else:
            print(card["mana_cost"])
            print(card["type_line"])
            print(card["oracle_text"])

    def print_colour(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if "card_faces" in card:
                for face in card["card_faces"]:
                    if not face["colors"]:
                        print(f"{face['name']} is colourless!")
                    else:
                        print(face["colors"])
            else:
                if not card["colors"]:
                    print(f"{card['name']} is colourless!")
                else:
                    print(card["colors"])
        except RuntimeError as e:
            print(f'{e}')

    def print_cost(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if "card_faces" in card:
                for face in card["card_faces"]:
                    print(face["mana_cost"])
            else:
                print(card["mana_cost"])
        except RuntimeError as e:
            print(f'{e}')

    def print_type(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if "card_faces" in card:
                for face in card["card_faces"]:
                    print(face["type_line"])
            else:
                print(card["type_line"])
        except RuntimeError as e:
            print(f'{e}')

    def print_text(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if "card_faces" in card:
                for face in card["card_faces"]:
                    print(face["oracle_text"])
            else:
                print(card["oracle_text"])
        except RuntimeError as e:
            print(f'{e}')

    def print_price(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            price = card['prices']['eur']
            if price:
                print(f"{price} EUR")
            else:
                print(f"I have no price data for {name}")
        except RuntimeError as e:
            print(f'{e}')

    def print_favourite(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            print(f"{card['name']}? Cool")
        except RuntimeError as e:
            print(f'{e}')

    def show_card(self, name):
        try:
            card = self.scryfall_api.get_card(name)
            if "card_faces" in card:
                for face in card["card_faces"]:
                    response = requests.get(face["image_uris"]["large"])
                    img = Image.open(BytesIO(response.content))
                    img.show()
            else:
                response = requests.get(card["image_uris"]["large"])
                img = Image.open(BytesIO(response.content))
                img.show()
        except RuntimeError as e:
            print(f'{e}')

    def show_card_random(self):
        card = self.scryfall_api.random_card()
        if "card_faces" in card:
            for face in card["card_faces"]:
                response = requests.get(face["image_uris"]["large"])
                img = Image.open(BytesIO(response.content))
                img.show()
        else:
            response = requests.get(card["image_uris"]["large"])
            img = Image.open(BytesIO(response.content))
            img.show()

    def similarity(self, phrase):
        corpus = self.patterns[:]
        corpus.append(phrase)
        tfidf_matrix = TfidfVectorizer().fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix, tfidf_matrix[-1])[:-1]
        similarities = list(similarities.flatten())
        if max(similarities) < 0.6:
            print("I am sorry, I do not understand :(")
        else:
            most_similar_index = similarities.index(max(similarities))
            self.handle_answer(self.kernel.respond(self.patterns[most_similar_index]))

    def handle_answer(self, answer):
        if not answer:
            print("I don't understand :(")
        if self.is_command(answer):
            command, parameter = self.get_command_and_parameter(answer)
            if command == "quit":
                print(parameter)
                return True

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
            if command == "price":
                self.print_price(parameter)
            if command == "favourite":
                self.print_favourite(parameter)
            if command == "show":
                self.show_card(parameter)
            if command == "show_random":
                self.show_card_random()
            if command == "default":
                self.similarity(parameter)
        else:
            print(answer)

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
            
            if self.handle_answer(answer):
                break


Chatbot("aiml-mtg.xml").run()
