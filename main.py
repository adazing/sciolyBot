import discord
from discord.ext import commands
import os
import json
import random
from keep_alive import keep_alive
from ciphers import atbash_cipher, caesar_cipher, aristocrat_encoder, generate_key, xenocrypt_encoder, generate_spanish_key, encrypt_hill, generate_hill_key, decrypt_hill, encrypt_baconian, encrypt_pollux, encrypt_morbit, generate_cryptarithm, encode_railfence
import numpy as np
from table2ascii import table2ascii as t2a, PresetStyle
import string
from PIL import Image, ImageDraw, ImageFont
import io


# returns frequency table
def frequency_table(text, english):
  frequencies = {x: 0
                 for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"} if english else {
                   x: 0
                   for x in "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
                 }
  for c in text:
    if c != " ":
      frequencies[c] += 1
  header = [x for x in frequencies]
  body = [frequencies[x] for x in frequencies]
  output = t2a(
    header=header,
    body=[body],
    style=PresetStyle.thin_box,
    cell_padding=0,
  )
  return output


#constants
QUOTES = json.load(open("quotes.json"))
FORENSICS = json.load(open("forensics.json"))
SPANISH_QUOTES = json.load(open("spanish_quotes.json"))
KEY = os.environ['SciolyBot_KEY']
OFFICERS = os.environ['officers_id']
VERIFICATION = os.environ['verification_channel_id']

#discord preparation
intents = discord.Intents.all()
intents.message_content = True
client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='', intents=intents)


@client.event
async def on_ready():
  print("We have logged in as {0.user}".format(client))


#sends alert to member on join
@client.event
async def on_member_join(member):
  print(member.id)
  await client.get_channel(int(VERIFICATION)).send(
    "<@" + str(member.id) + "> " +
    "Welcome to the Scioly Discord! \nIf you want to be verified, please reply to this message with:\n `$verify FIRSTNAME LASTNAME`"
  )


@bot.command()
@client.event
async def on_message(message):
  if message.author == client.user:
    return

  # HELP
  if message.content.startswith('$help'):
    await message.channel.send(
      "Current commands: $help, $caesar, $verify, $atbash, $aristocrat, $xenocrypt, $patristocrat, $hill, $forensics"
    )

  #forensics questions
  if message.content.startswith('$forensics'):
    quote = random.choice(FORENSICS)
    await message.channel.send(quote["Question"])
    await message.channel.send("||" + quote["Answer"] + "||")

  # CAESAR CIPHER
  if message.content.startswith("$caesar"):
    quote = random.choice(QUOTES)
    author = quote["author"] or "Anonymous"
    shift = random.randint(0, 26)
    encrypted = caesar_cipher(quote["text"].upper(), shift)
    await message.channel.send('"' + encrypted + '" - ' + author)
    await message.channel.send('Answer: ||"' + quote["text"].upper() + '" - ' +
                               author + " (shift was " + str(shift) + ")||")

  #atbash
  if message.content.startswith("$atbash"):
    quote = random.choice(QUOTES)
    author = quote["author"] or "Anonymous"
    encrypted = atbash_cipher(quote["text"].upper())
    await message.channel.send('"' + encrypted + '" - ' + author)
    await message.channel.send('Answer: ||"' + quote["text"].upper() + '" - ' +
                               author + "||")

    #aristocrat
  if message.content.startswith("$aristocrat"):
    quote = random.choice(QUOTES)  # choose quote
    author = quote["author"] or "Anonymous"
    key = generate_key()  # generate key

    encrypted = aristocrat_encoder(quote["text"].upper(), key)  #encrypt
    await message.channel.send('"' + encrypted + '" - ' + author
                               )  #send encrypted message
    output = frequency_table(encrypted, True)
    await message.channel.send(f"```\n{output}\n```")

    enc_list = quote["text"].split()  #split quote into words
    hint = random.randint(1, len(enc_list))  #choose random word
    hinttxt = ""
    if hint == 1:
      hinttxt = "1st"
    elif hint == 2:
      hinttxt = "2nd"
    elif hint == 3:
      hinttxt = "3rd"
    else:
      hinttxt = str(hint) + "th"

    await message.channel.send('Hints: ||There exists the word ' +
                               enc_list[hint - 1].upper() + '|| || The ' +
                               hinttxt.upper() + " word is " +
                               enc_list[hint - 1].upper() + "||")
    await message.channel.send('Answer: ||"' + quote["text"].upper() + '" - ' +
                               author + " (key: " + key + ")||")

  if message.content.startswith("$xenocrypt"):
    quote = random.choice(SPANISH_QUOTES)  # choose quote
    key = generate_spanish_key()  # generate key
    encrypted = xenocrypt_encoder(quote.upper(), key)  #encrypt
    await message.channel.send('"' + encrypted + '"')  #send encrypted message
    output = frequency_table(encrypted, False)
    await message.channel.send(f"```\n{output}\n```")
    enc_list = quote.split()  #split quote into words
    hint = random.randint(1, len(enc_list))  #choose random word
    hinttxt = ""
    if hint == 1:
      hinttxt = "1st"
    elif hint == 2:
      hinttxt = "2nd"
    elif hint == 3:
      hinttxt = "3rd"
    else:
      hinttxt = str(hint) + "th"

    await message.channel.send('Hints: ||There exists the word ' +
                               enc_list[hint - 1].upper() + '|| || The ' +
                               hinttxt.upper() + " word is " +
                               enc_list[hint - 1].upper() + "||")
    await message.channel.send('Answer: ||"' + quote.upper() + '" (key: ' +
                               key + ')||')

  if message.content.startswith("$patristocrat"):
    quote = random.choice(QUOTES)  # choose random quote
    author = quote["author"] or "Anonymous"
    key = generate_key()  # make key
    encrypted = aristocrat_encoder(quote["text"].upper(), key).replace(
      " ", "")  # encode like aristocrat, but take away all the spaces.
    enc_list = quote["text"].split()  # split quote into words
    hint_info = bool(random.getrandbits(1))  # get random boolean
    hint = random.randint(1, len(enc_list))  #get random word from quote
    #grammer
    hinttxt = ""
    if str(hint)[-1] == 1:
      hinttxt = str(hint) + "st"
    elif str(hint)[-1] == 2:
      hinttxt = str(hint) + "nd"
    elif str(hint)[-1] == 3:
      hinttxt = str(hint) + "rd"
    else:
      hinttxt = str(hint) + "th"
    await message.channel.send('"' + encrypted + '" - ' + author
                               )  # send encrypted message
    output = frequency_table(encrypted, True)
    await message.channel.send(f"```\n{output}\n```")
    if hint_info:
      await message.channel.send('Hints: ||The first word is: ' +
                                 enc_list[0].upper() + '|| ||The ' + hinttxt +
                                 " word is " +
                                 str(len(enc_list[hint - 1].upper())) +
                                 " characters long.||")
    else:
      await message.channel.send('Hints: ||The last word is: ' +
                                 enc_list[-1].upper() + '|| ||The ' + hinttxt +
                                 " word is " +
                                 str(len(enc_list[hint - 1].upper())) +
                                 " characters long.||")
    await message.channel.send('Answer: ||"' + quote["text"].upper() + '" - ' +
                               quote["author"] + " (key: " + key + ")||")

  if message.content.startswith("$hill"):
    quote = random.choice(QUOTES)  # choose quote
    key = generate_hill_key(2)  # generate key
    enc = encrypt_hill(quote["text"], key)
    # dec = decrypt_hill(enc, key)
    author = quote["author"] or "Anonymous"
    choice = random.randrange(0, 2)
    if choice == 0:
      await message.channel.send('Encrypt the plaintext given the key:')
      await message.channel.send('Plaintext: "' + quote["text"] + '" -' +
                                 author)
      await message.channel.send('Key: "' + key + '"')
      await message.channel.send('Encrypted Text: ||"' + enc + '" -' + author +
                                 '||')
    else:
      await message.channel.send(
        'Find the plaintext given the key and encrypted text:')
      await message.channel.send('Key: "' + key + '"')
      await message.channel.send('Encrypted Text: "' + enc + '" -' + author)
      await message.channel.send('Plaintext: ||"' + quote["text"] + '" -' +
                                 author + '||')
    # else:
    #   await message.channel.send('Find the key given the plaintext and encrypted text:')
    #   await message.channel.send('Encrypted Text: "'+enc+'" -'+author)
    #   await message.channel.send('Plaintext: "'+quote["text"]+'" -'+author)
    #   await message.channel.send('Key: ||"'+key+'"||')

  if message.content.startswith("$baconian"):
    quote = random.choice(QUOTES)  # choose quote
    author = quote["author"] or "Anonymous"
    hint = ""
    encrypted, choice = encrypt_baconian(quote["text"])
    if choice == 0:
      hint = "just 0=0 and 1=1!"
    if choice == 1:
      hint = "one letter means 0 and another means 1."
    if choice == 2:
      hint = "A-N is 0 and O-Z is 1."
    if choice == 3:
      hint = "A-N is 1 and O-Z is 0."
    if choice == 4:
      hint = "even letters are 0, odd letters are 1 (A=0, B=1...)."
    if choice == 5:
      hint = "even letters are 1, odd letters are 0 (A=0, B=1...)."
    if choice == 6:
      hint = "prime letters are 1, composite letters are 0 (B=1, C=2...)."
    if choice == 7:
      hint = "prime letters are 0, composite letters are 1 (B=1, C=2...)."
    if choice == 8:
      hint = "some letters mean 0 and some letters mean 1."
    if choice == 9:
      hint = "some symbols mean 0 and some symbols mean 1."
    if choice == 10:
      hint = "one symbol means 0 and the other means 1."
    if choice == 11:
      hint = "even numbers are 1, odd numbers are 0."
    if choice == 12:
      hint = "even numbers are 0, odd numbers are 1."
    if choice == 13:
      hint = "prime numbers are 1, composite numbers are 0."
    if choice == 14:
      hint = "prime numbers are 0, composite numbers are 1."
    if choice == 15:
      hint = "0 and 1 are switched!"
    await message.channel.send("Decrypt the encrypted text:")
    await message.channel.send('Encrypted text: "' + encrypted + '"')
    await message.channel.send("Hint: ||" + hint + "||")
    await message.channel.send('Answer: ||"' + quote["text"] + '" - ' +
                               author + '||')

  if message.content.startswith("$pollux"):
    quote = random.choice(QUOTES)
    author = quote['author'] or "Anonymous"
    encrypted, key = encrypt_pollux(quote["text"])
    choice = random.randrange(0, 6)
    hint = ""
    if choice == 0:
      sample = random.sample(list(key), 4)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 1:
      sample = random.sample(list(key), 5)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 2:
      sample = random.sample(list(key), 6)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 3:
      sample = random.sample(list(key), 7)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 4:
      first_word = ("".join([
        x.upper() for x in quote["text"] if x.isalpha() or x == " "
      ])).split()[0]
      hint = "The first word is " + first_word + "."
    if choice == 5:
      last_word = ("".join([
        x.upper() for x in quote["text"] if x.isalpha() or x == " "
      ])).split()[-1]
      hint = "The last word is " + last_word + "."
    await message.channel.send("Decrypt the encrypted text:")
    await message.channel.send("Hint: ||" + hint + "||")
    await message.channel.send('Encrypted text: "' + encrypted + '"')
    await message.channel.send('Answer: ||"' + quote['text'] + '" - ' +
                               author + ' (key: ' + str(key) + ')||')

  if message.content.startswith("$morbit"):
    quote = random.choice(QUOTES)
    author = quote['author'] or "Anonymous"
    encrypted, key = encrypt_morbit(quote["text"])
    choice = random.randrange(0, 4)
    hint = ""
    if choice == 0:
      sample = random.sample(list(key), 5)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 1:
      sample = random.sample(list(key), 6)
      for x in sample:
        hint += x + ' is "' + key[x] + '", '
      hint = hint[:-2] + "."
    if choice == 2:
      first_word = ("".join([
        x.upper() for x in quote["text"] if x.isalpha() or x == " "
      ])).split()[0]
      hint = "The first word is " + first_word + "."
    if choice == 3:
      last_word = ("".join([
        x.upper() for x in quote["text"] if x.isalpha() or x == " "
      ])).split()[-1]
      hint = "The last word is " + last_word + "."
    await message.channel.send("Decrypt the encrypted text:")
    await message.channel.send("Hint: ||" + hint + "||")
    await message.channel.send('Encrypted text: "' + encrypted + '"')
    await message.channel.send('Answer: ||"' + quote['text'] + '" - ' +
                               author + ' (key: ' + str(key) + ')||')

  if message.content.startswith("$cryptarithm"):
    await message.channel.send(
      "This might take a while (up to 3ish min). Please be pacient.")
    operator = random.choice(["+", "-", "*"])
    if operator == "+":
      word_lengths = range(1, 7)
      # Randomly choose the number of words on the left side of the equation
      num_words = random.randint(2, 7)
    elif operator == "-":
      word_lengths = range(1, 7)
      # Randomly choose the number of words on the left side of the equation
      num_words = random.randint(2, 7)

    elif operator == "*":
      word_lengths = range(1, 3)
      # Randomly choose the number of words on the left side of the equation
      num_words = random.randint(2, 3)

    equation, solution = generate_cryptarithm(operator, word_lengths,
                                              num_words)
    await message.channel.send("Formula: \n```" + equation + "```")
    await message.channel.send("Solutions: ||```" + str(solution) + "```||")

  if message.content.startswith("$railfence"):
    quote = random.choice(QUOTES)
    author = quote['author'] or "Anonymous"
    choice = random.randint(2, 7)
    if choice in range(2, 7):
      encoded = encode_railfence(quote["text"], choice,
                                 random.randint(0, choice - 1))
      await message.channel.send(
        "Decode the text below, which was encoded with " + str(choice) +
        " rails. There is an unknown offset.")
      await message.channel.send('"' + encoded + '" - ' + author)
    else:
      choice = random.randint(2, 6)
      encoded = encode_railfence(quote, choice, random.randint(0, choice - 1))
      await message.channel.send(
        "Decode the text below, which was encoded with an unknown number of rails and unkown offset."
      )
      await message.channel.send('"' + encoded + '" - ' + author)
    await message.channel.send('Answer: ||"' + quote["text"] + '" - ' +
                               author + '||')

  # VERIFY

  if message.content.startswith("$verify"):
    user = message.author
    role = discord.utils.get(message.guild.roles, name="Verified")
    if role not in user.roles:
      await message.channel.send("<@&" + str(OFFICERS) + "> " + str(user) +
                                 " would like to be verified as " +
                                 message.content.split()[-2] + " " +
                                 message.content.split()[-1])


keep_alive()

client.run(KEY)
