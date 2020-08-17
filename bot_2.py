# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:45:14 2020

@author: Thomas DeWitt
"""

# ArmOfState2

import discord
#from google.colab import files
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

def read_token():
    with open("token.txt", "r") as file:
        lines = file.readlines()
        return lines[0].strip()


token = read_token()
prefix = "?"

master_users = ["das.lionfish#9316"]

client = discord.Client()

@client.event
async def on_message(message):
    # we do not want the bot to reply to itself
    if message.author == client.user:
        return

    if message.content == '?hello':
        reply = 'Hello there!'
        await message.channel.send(reply)
    
    if message.content.startswith('!#killall'):
        if str(message.author) in master_users:
            await client.logout()
            
    if message.content.startswith('!#init_network'):
        if str(message.author) in master_users:
            await message.channel.send('Unpacking model...')
            model = tf.keras.models.load_model('saved_model/chestX_V2')
            await message.channel.send('Model loaded.  Displaying parmeters...')
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            await message.channel.send(short_model_summary)
    
            

print('Connected to client.')

client.run(token)