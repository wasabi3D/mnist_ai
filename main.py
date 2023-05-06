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


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def convert_to_input(x: io.BytesIO, contrast_scale=1.0):
    def contrast(c):
        return 128 + contrast_scale * (c - 128)
    image = remove_transparency(Image.open(x).resize((28, 28))).convert("L").point(contrast)
    img_array = np.array(image)
    img_array = 1.0 - img_array / 255
    preview = Image.fromarray(np.resize(img_array * 255, (28, 28)))
    preview.show()
    return img_array.flatten()


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
            for image_raw in message.attachments:
                img_array = convert_to_input(io.BytesIO(await image_raw.read()), contrast_scale=4.5)
                await message.channel.send(str(np.argmax(network.predict(img_array.flatten()))))
    client.run(token)


main()
