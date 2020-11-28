# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:45:14 2020

@author: Thomas DeWitt

Contributor: Andrew Lu
"""

# ArmOfState2

import discord
import csv
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
print(tf.__version__)
print(keras.__version__)

# TO-DO:
# DONE - Create and import ExeNet tokenizer and ExeNet model alongside ToxiNet resources
# Dynamically handle rankings
# Create user list
# DONE - Handle new users coming in (See manage_toxiscores)

def read_token():
    with open("token.txt", "r") as file:
        lines = file.readlines()
        return lines[0].strip()

# Format: <string uid> <float 2 score>\n
def load_scores():
    scores = {}
    try:
        with open("scores.csv", "r", newline="") as csvfile:
            lines = csv.reader(csvfile)
            for line in lines:
                scores[str(line[0])] = float(line[1])
        return scores
    except:
        with open("scores.csv", "w"):
            return scores
    
def save_scores():  # have it update every 10 minutes or something eventually
    with open("scores.csv", "w", newline="") as csvfile:
        save = csv.writer(csvfile)
        for pairs in toxiscores.items():
            save.writerow([str(pairs[0]), str(pairs[1])])

def load_preprocessing():
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)  # load network tokenizer
    max_length = 200  # cut/pad all sentences to 400 tokens (words)
    trunc_type = 'post' #leave as-is
    padding_type = 'post' #leave as-is
    embedding_dimension = 100
    return tokenizer, max_length, trunc_type, padding_type, embedding_dimension

def score_text(toxic_preds):
    if max(toxic_preds) <= 60:
        score = 0
    else:
        scale = max(toxic_preds)
        rank = toxic_preds.index(scale)
        score = scale * rank
    return score

def manage_toxiscores(uid, message_score):
    if uid not in toxiscores.keys():
        toxiscores[uid] = 0.00
    
    user_score = toxiscores[uid]
    
    if message_score <= 0.00:  # Losing toxicity
        user_score -= modifier(user_score) * loss
        user_score = 0.00 if toxiscores[uid] <= 0.00 else user_score
    else:  # Gaining toxicity
        user_score += modifier(user_score) * message_score * gain

    toxiscores[uid] = round(user_score, 2)

"""
Notes:
For handling users, instead of using User.name maybe use User.id
-Done (ATD 11/26/20 19:50)

"""
# Tokens
token = read_token()

# Master users
master_users = ["641816848865689611"]

# Scraping info (Eventually write to file instead)
messages = []
authors = []

# Model statuses
scrape_messages = False
prep_to_analyze = False
model_ready = False

# Load toxicity scores
toxiscores = load_scores()

# Toxicity score modifier & factor modifier should be multiplied by for gaining/losing toxicity, 1 gain/loss default
modifier = lambda x : round(pow(1.075, -3/8 * x), 2)
gain = 1
loss = 1

client = discord.Client()

@client.event
async def on_message(message):
    # Don't talk to bots (including self)
    if message.author.bot:
        return
    
    if message.content.startswin("!#scrape"):
        if str(message.author.id) in master_users:
            scrape_messages = True
            await message.channel.send("WARNING: all messages following this message will be filed for training purpose.")
    
    if message.content.startswith("!#init_network"):
        if str(message.author.id) in master_users:
            await message.channel.send("Unpacking model...")
            model = tf.keras.models.load_model('classifier/AoS_GPnet') #leave as-is
            tokenizer, max_length, trunc_type, padding_type, embedding_dimension = load_preprocessing()
            model_ready = True
            await message.channel.send("Models loaded.  Displaying parameters...")
            stringlist = []
            toxic_model.summary(print_fn=lambda x: stringlist.append(x))
            short_model_summary = "\n".join(stringlist)
            await message.channel.send(short_model_summary)
            
    
    if message.content.startswith("!#process"):
            if str(message.author.id) in master_users:
                if model_ready == True:
                    prep_to_analyze = True
                    await message.channel.send("Analysis framework active.")
                else:
                    await message.channel.send("Models not loaded.  Please load models.")

    if message.content.startswith("!#killall"):
        if str(message.author.id) in master_users:
            await message.channel.send("Saving data...")
            data = pd.DataFrame(data={"user": authors, "message": messages})
            data.to_csv("C:/Users/Thomas DeWitt/Downloads/discord_messages.csv", sep=",", index=False)  # Change this eventually
            await client.logout()
    
    if scrape_messages == True:  # Eventually directly write to a buffer then file, also maybe only from one channel (specific channel id)
        messages.append(str(message.content))
        authors.append(str(message.author.id))
        
    if prep_to_analyze == True:
        words = str(message.content)
        toxic_tokens = tokenizer.texts_to_sequences(words)
        toxic_ready = pad_sequences(toxic_tokens, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        toxic_preds = model.predict(toxic_ready)
        toxic_results = np.argmax(toxic_preds, axis=1)
        toxic_score = score_text(toxic_results)
        offender = str(message.author.id)
        toxiscores = manage_toxiscores(offender, toxic_score)

    if message.content == "?hello":
        reply = "Greetings, citizen."
        await message.channel.send(reply)

    if message.content == "?watchlist":  # Untested
        tmp = dict(sorted(toxiscores.items(), key=lambda x: x[1], reverse=True))
        tmp_keys = tmp.keys()
        reply = "{: <40}".format("```Toxicity Watchlist:" + ": Score\n")
        place = 0
        while place < 20 and place < len(tmp):
            reply +=
                    "{: <40}".format(f"{str(i+1)}. {str(client.get_user(tmp_keys[i]))}") + ": {:2.2f}\n".format(score)
            place += 1
        await message.channel.send(reply + "```")
    
    



client.run(token)

print('Connected to client.')
