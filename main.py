import discord
import os
import pickle
import io

import numpy as np

# from ai import TwoLayerNet
from common.network import MultiLayerNet
from PIL import Image
from dataset.mnist import load_mnist


with open("network2.0.pkl", "rb") as f:
    network: MultiLayerNet = pickle.load(f)

# =====TEST=====
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# test_index = 5

# Show sample image
# sample_img, sample_label = x_train[test_index].reshape(28, 28), t_train[test_index]
# print(f"Label of the sample image is {sample_label}")
# pil_sample = Image.fromarray(sample_img * 255)
#
# print("Predict sample: ")
# print(np.argmax(network.predict(sample_img.flatten())))
# pil_sample.show()


def main():
    os.chdir(os.path.dirname(__file__))
    token = input("Please enter bot token: ")
    intents = discord.Intents.default()
    intents.members = True
    intents.presences = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print('Bot successfully connected to Discord.')

    @client.event
    async def on_message(message: discord.Message):
        if client.user.mentioned_in(message):
            print("Message")
            for image_raw in message.attachments:
                image = Image.open(io.BytesIO(await image_raw.read()))
                image = image.resize((28, 28))
                image = image.convert("L")
                img_array = np.array(image)
                img_array = 1.0 - img_array / 255
                new_img = Image.fromarray(np.resize(img_array * 255, (28, 28)))
                await message.channel.send(str(np.argmax(network.predict(img_array.flatten()))))
    client.run(token)


main()
