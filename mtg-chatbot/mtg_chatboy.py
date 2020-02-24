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
import nltk
import pickle

import scrython
from scrython.foundation import ScryfallError

class Chatbot:
    def __init__(self, aiml_filename, model, grammar_file):
        self.aiml_filename = Chatbot.get_absolute_path(aiml_filename)
        self.set_kernel()
        self.set_patterns()
        self.load_image_models(model)
        self.set_classnames()
        self.load_nltk(grammar_file)

    def load_nltk(self, grammar_file):
        valuation_string = """
        card => {}
        my_hand => h1
        opp_hand => h2
        my_graveyard => g1
        opp_graveyard => g2
        my_exile => e1
        opp_exile => e2
        my_battlefield => b1
        opp_battlefield => b2
        be_in => {}
        """
        self.valuation = nltk.Valuation.fromstring(valuation_string)
        #self.valuation = pickle.load(open("save.p", "rb"))
        #pickle.dump(self.valuation, open("save.p", "wb"))
        self.grammar_file = Chatbot.get_absolute_path(grammar_file)
        self.object_counter = 0

    def get_absolute_path(filename):
        return f"{os.path.dirname(os.path.realpath(sys.argv[0]))}{os.path.sep}{filename}"

    def is_command(self, user_input):
        return user_input[0] == "#"

    def get_command_and_parameters(self, user_input):
        command = user_input[1:].split("$")[0]
        parameters = user_input[1:].split("$")[1:]
        return command, parameters

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

    def translate_ownership(self, input):
        vocabulary = {
            "i": "my",
            "opponent": "opp",
            "my": "my"
        }
        for term, ownership in vocabulary.items():
            if term in input:
                return ownership

    def translate_zone(self, input):
        vocabulary = {
            "hand": "hand",
            "graveyard": "graveyard",
            "exile": "exile",
            "battlefield": "battlefield"
        }
        for term, zone in vocabulary.items():
            if term in input:
                return zone
    
    def encode_card_name(self, card_name):
        self.object_counter += 1
        card = scrython.cards.Named(exact=card_name)
        return f"{card.name()}${str(self.object_counter)}"

    def decode_card_name(self, object_name):
        return object_name.split("$")[0]

    def is_a_permanent(self, card_name):
        card = scrython.cards.Named(exact=card_name)
        if "Sorcery" in card.type_line() or "Instant" in card.type_line():
            return False
        return True
    
    def add_to_zone(self, player, zone, card_name):
        ownership = self.translate_ownership(player)
        location = f"{ownership}_{zone}"
        try:
            object_id = self.encode_card_name(card_name)
            # insert constant
            self.valuation[object_id] = object_id 
            # clean up default model value
            if len(self.valuation["card"]) == 1:
                if ('',) in self.valuation["card"]:
                    self.valuation["card"].clear()

            # insert object_id into card list
            self.valuation["card"].add((object_id,))

            # clean up default model value
            if len(self.valuation["be_in"]) == 1:
                if ('',) in self.valuation["be_in"]:
                    self.valuation["be_in"].clear()
            # insert object_id and zone into ownership
            self.valuation["be_in"].add((object_id, self.valuation[location]))
        except ScryfallError as e:
            print(e)
    
    def draw_card(self, player, card_name):
        self.add_to_zone(player, "hand", card_name)

    def cast_card(self, player, card_name):
        remove_from_hand_result = self.remove(card_name, f"{player} hand")
        if remove_from_hand_result:
            print(f"{remove_from_hand_result} in hand")
            return

        if self.is_a_permanent(card_name):
            self.add_to_zone(player, "battlefield", card_name)
        else:
            self.add_to_zone(player, "graveyard", card_name)
    
    def is_card_in_zone(self, location_input):
        g = nltk.Assignment(self.valuation.domain)
        m = nltk.Model(self.valuation.domain, self.valuation)
        ownership = self.translate_ownership(location_input)
        zone = self.translate_zone(location_input)
        location = f"{ownership}_{zone}"
        sent = "some card are_in " + location
        results = nltk.evaluate_sents([sent], self.grammar_file, m, g)[0][0]
        if results[2] == True:
            print("Yes.")
        else:
            print("No.")

    def is_all_cards_in_zone(self, location_input):
        g = nltk.Assignment(self.valuation.domain)
        m = nltk.Model(self.valuation.domain, self.valuation)
        ownership = self.translate_ownership(location_input)
        zone = self.translate_zone(location_input)
        location = f"{ownership}_{zone}"
        sent = "all card are_in " + location
        results = nltk.evaluate_sents([sent], self.grammar_file, m, g)[0][0]
        if results[2] == True:
            print("Yes.")
        else:
            print("No.")

    def get_cards_in_zone(self, location_input):
        g = nltk.Assignment(self.valuation.domain)
        m = nltk.Model(self.valuation.domain, self.valuation)
        ownership = self.translate_ownership(location_input)
        zone = self.translate_zone(location_input)
        location = f"{ownership}_{zone}"
        e = nltk.Expression.fromstring("be_in(x," + location + ")")
        sat = m.satisfiers(e, "x", g)
        if len(sat) == 0:
            print("None.")
        else:
            # find satisfying objects in the valuation dictionary,
            # and print their decoded object_id names
            sol = self.valuation.values()
            for so in sat:
                print(self.decode_card_name(so))
    
    def remove(self, card_name, location_input):
        g = nltk.Assignment(self.valuation.domain)
        m = nltk.Model(self.valuation.domain, self.valuation)
        ownership = self.translate_ownership(location_input)
        zone = self.translate_zone(location_input)
        location = f"{ownership}_{zone}"
        e = nltk.Expression.fromstring("be_in(x," + location + ")")
        to_remove = m.satisfiers(e, "x", g)
        if len(to_remove) == 0:
            return f"No cards named {card_name}"
        else:
            for relationship in self.valuation["be_in"]:
                if relationship[0] in to_remove:
                    self.valuation["be_in"].remove(relationship)
                    return False

    def handle_answer(self, answer):
        if not answer:
            print("I don't understand :(")
        if self.is_command(answer):
            command, parameters = self.get_command_and_parameters(answer)
            if command == "quit":
                print(parameters[0])
                return True

            if command == "image_classification":
                self.print_image_model_result(parameters[0])
            if command == "describe":
                self.print_description(parameters[0])
            if command == "describe_random":
                self.print_description_random()
            if command == "colour":
                self.print_colour(parameters[0])
            if command == "cost":
                self.print_cost(parameters[0])
            if command == "type":
                self.print_type(parameters[0])
            if command == "text":
                self.print_text(parameters[0])
            if command == "price":
                self.print_price(parameters[0])
            if command == "favourite":
                self.print_favourite(parameters[0])
            if command == "show":
                self.show_card(parameters[0])
            if command == "show_random":
                self.show_card_random()
            if command == "draw":
                self.draw_card(parameters[0], parameters[1])
            if command == "cast":
                self.cast_card(parameters[0], parameters[1])
            if command == "cards_in_zone":
                self.is_card_in_zone(parameters[0])
            if command == "all_cards_in":
                self.is_all_cards_in_zone(parameters[0])
            if command == "which_cards_in":
                self.get_cards_in_zone(parameters[0])
            if command == "remove":
                self.remove(parameters[0], parameters[1])






            if command == "default":
                self.similarity(parameters[0])
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
        model="../../mtg-image-classify/classify/model.h5",
        grammar_file="simple-sem.fcfg").run()
