import aiml

from scryfall_api import Scryfall

class Chatbot:
    def __init__(self):
        self.set_kernel()

    def is_command(self, user_input):
        return user_input[0] == "#"

    def get_command_dict(self, user_input):
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
        print("Welcome , welcome, to the Magic: The Gathering chatbot! Ask me questions about the game of Magic, as well as cards in the game! I can describe and show you cards.")

        while True:
            try:
                user_input = input("> ")
            except (KeyboardInterrupt, EOFError) as e:
                print("Bye!")
                break

            if not user_input:
                continue

            agent = "aiml"
            if agent == "aiml":
                answer = self.kernel.respond(user_input)

            if self.is_command(answer):
                command_dict = self.get_command_dict(answer)
                if command_dict["command"] == "example":
                    # do api thing
                    print(command_dict["parameter"])
            else:
                print(answer)

Scryfall().get_card("divination")

Chatbot().run()


# Describe X card?
# Show me X card?
# What colour is X?
# How much does X cost?
# What is your favourite card?
# Show me a random card


