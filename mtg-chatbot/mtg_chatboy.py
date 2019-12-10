import aiml, requests, sys, os, math
import xml.etree.ElementTree as ET
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import tensorflow as tf
from tensorflow.keras import models, backend, layers
import urllib.request
import numpy

import scrython
from scrython.foundation import ScryfallError

class Chatbot:
    def __init__(self, aiml_filename, model):
        self.aiml_filename = Chatbot.get_absolute_path(aiml_filename)
        self.set_kernel()
        self.set_patterns()
        self.load_image_models(model)
        self.set_classnames()

    def get_absolute_path(filename):
        return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}{os.path.sep}{filename}"

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

    def load_image_models(self, model):
        model_filename = Chatbot.get_absolute_path(model)
        self.model = models.load_model(model_filename)

    def set_classnames(self):
        self.classnames = ['Artifact', 'Black', 'Blue', 'Colorless', 'Creature', 'Enchantment', 'Green', 'InstantSorcery', 'Land', 'Planeswalker', 'Red', 'White']

    def print_image_model_result(self, name):
        print(*self.run_image_model(name))

    def run_image_model(self, name):
        parsed_name = None
        for word in name.split(" "):
            if "jpg" in word or "png" in word or "jpeg" in word:
                parsed_name = word
        if not parsed_name:
            return ["Input is not a jpg or a png file"]
        if os.path.exists(parsed_name):
            raw_image = cv2.imread(parsed_name)
        else:
            try:
                file_to_delete = parsed_name.split("/")[-1]
                urllib.request.urlretrieve(parsed_name, file_to_delete)
                raw_image = cv2.imread(file_to_delete)
                os.remove(file_to_delete)
            except Exception as e:
                return ["Input is neither a valid file nor url"]
        img_rows = 150
        img_cols = 150
        color_channels = 3
        input_shape = (img_rows, img_cols, color_channels)
        scaled_image = cv2.resize(raw_image, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
        scaled_image = scaled_image/255
        image_array = scaled_image.reshape(1, img_rows, img_cols, color_channels)

        predictions = self.model.predict(image_array)
        
        colors, types = self.get_class_predictions(predictions[0])
        
        probable_colors = self.trim_predictions(colors)
        probable_types = self.trim_predictions(types)
        
        predicted_classes = [classname for classname in probable_colors]
        for card_type in probable_types:
            predicted_classes.append(card_type)

        if not predicted_classes:
            return ["I can't tell what this is :("]
        
        return predicted_classes

    def get_class_predictions(self, predictions):
        colors = {
            "Black": predictions[1],
            "Blue": predictions[2],
            "Colorless": predictions[3],
            "Green": predictions[6],
            "Red": predictions[10],
            "White": predictions[11]
        }
        types = {
            "Artifact": predictions[0],
            "Creature": predictions[4],
            "Enchantment": predictions[5],
            "Instant/Sorcery": predictions[7],
            "Land": predictions[8],
            "Planeswalker": predictions[9]
        }
        return colors, types

    def trim_predictions(self, predictions):
        likely_predictions = {}
        probability_threshold = 0.7
        found = False
        while not found:
            if likely_predictions:
                found = True
            probability_threshold -= 0.1
            if math.isclose(probability_threshold, 0.1):
                return likely_predictions
            for category, probability in predictions.items():
                if probability >= probability_threshold:
                    likely_predictions[category] = probability
        return likely_predictions

    def print_description(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            print(f'{card.name()}:')
            print(card.mana_cost())
            print(card.type_line())
            print(card.oracle_text())
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                print(face["mana_cost"])
                print(face["type_line"])
                print(face["oracle_text"])

    def print_description_random(self):
        try:
            card = scrython.cards.Random()
            print(f'{card.name()}:')
            print(card.mana_cost())
            print(card.type_line())
            print(card.oracle_text())
        except KeyError:
            for face in card.card_faces():
                print(face["mana_cost"])
                print(face["type_line"])
                print(face["oracle_text"])

    def print_colour(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            if not card.colors():
                print(f"{card.name()} is colourless!")
            else:
                print(card.colors())
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                if not face["colors"]:
                    print(f"{face['name']} is colourless!")
                else:
                    print(face["colors"])

    def print_cost(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            print(card.mana_cost())
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                print(face["mana_cost"])

    def print_type(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            print(card.type_line())
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                print(face["type_line"])

    def print_text(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            print(card.oracle_text())
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                print(face["oracle_text"])

    def print_price(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            price = card.prices("eur")
            if price:
                print(f"{price} EUR")
            else:
                print(f"I have no price data for {name}")
        except ScryfallError as e:
            print(f'{e}')

    def print_favourite(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            print(f"{card.name()}? Cool")
        except ScryfallError as e:
            print(f'{e}')

    def show_card(self, name):
        try:
            card = scrython.cards.Named(exact=name)
            response = requests.get(card.image_uris()["large"])
            img = Image.open(BytesIO(response.content))
            img.show()
        except ScryfallError as e:
            print(f'{e}')
        except KeyError:
            for face in card.card_faces():
                response = requests.get(face["image_uris"]["large"])
                img = Image.open(BytesIO(response.content))
                img.show()

    def show_card_random(self):
        try:
            card = scrython.cards.Random()
            response = requests.get(card.image_uris()["large"])
            img = Image.open(BytesIO(response.content))
            img.show()
        except KeyError:
            for face in card.card_faces():
                response = requests.get(face["image_uris"]["large"])
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

            if command == "image_classification":
                self.print_image_model_result(parameter)
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
                user_input = user_input.split("?")[0]
            except (KeyboardInterrupt, EOFError):
                print("Bye!")
                break

            if not user_input:
                continue

            agent = "aiml"
            if agent == "aiml":
                answer = self.kernel.respond(user_input)
                answer = answer.replace("  #default$", ".")
            
            if self.handle_answer(answer):
                break


Chatbot(aiml_filename="aiml-mtg.xml",
        model="../../mtg-image-classify/classify/model.h5").run()
