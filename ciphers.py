import string
import random
import numpy as np
from math import gcd
from sympy import Matrix
from table2ascii import table2ascii as t2a, PresetStyle, Alignment
import itertools

def encode_railfence(text, rail_num, offset):
  text = "".join([x.upper() for x in text if x.isalpha()])
  # Create an empty list to hold the rail fence pattern
  rails = [[] for _ in range(rail_num)]
  
  # Compute the rail fence pattern
  pattern = list(range(rail_num)) + list(range(rail_num-2, 0, -1))
  
  # Apply the offset to the pattern
  pattern = pattern[offset:] + pattern[:offset]
  
  # Iterate over each character in the text
  for i, char in enumerate(text):
      # Calculate the rail index based on the pattern
      rail_index = pattern[i % len(pattern)]
      
      # Append the character to the corresponding rail
      rails[rail_index].append(char)
  
  # Join the characters in each rail to form the encoded message
  encoded_text = ''.join(''.join(rail) for rail in rails)
  
  return encoded_text
  

def generate_cryptarithm(operator, word_lengths, num_words):
  # Generate random letters for the cryptarithm
  letters = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 10)
  #make left side of equation
  left = ""
  starts = set()
  for x in range(num_words):
    w_length = random.choice(word_lengths)
    word = "".join([
      str(random.randrange(1, 10)) if d == 0 else str(random.randrange(0, 10))
      for d in range(w_length)
    ])
    starts.add(letters[int(word[0])])
    left += word + operator
  #remove last operator
  left = left[:-1]
  #find right side of equation
  right = str(eval(left))
  if right[0] == "-":
    starts.add(letters[int(right[1])])
  else:
    starts.add(letters[int(right[0])])
  left_words = left.translate(
    str.maketrans({str(x): letters[x]
                   for x in range(10)}))
  right_words = right.translate(
    str.maketrans({str(x): letters[x]
                   for x in range(10)}))
  left_words_list = left_words.split(operator)
  formatted = t2a(body=[["", left_words_list[x]]
                        if x == 0 else [operator, left_words_list[x]]
                        for x in range(len(left_words_list))] +
                  [["", "---------"]] + [["", right_words]],
                  style=PresetStyle.plain,
                  cell_padding=0,
                  alignments=Alignment.RIGHT)
  equation = left_words + "==" + right_words

  unique_letters = set(''.join(equation.split())) - {"=", operator}
  digit_permutations = itertools.permutations('0123456789',
                                              len(unique_letters))
  solutions = []
  for digits in digit_permutations:
    digit_map = {
      letter: digit
      for letter, digit in zip(unique_letters, digits)
    }
    if list(digit_map.keys())[list(
        digit_map.values()).index("0")] not in starts if '0' in list(
          digit_map.values()) else True:
      num_equation = equation.translate(str.maketrans(digit_map))
      if eval(num_equation):
        if len(solutions) > 0:
          return generate_cryptarithm(operator, word_lengths, num_words)
        else:
          solutions.append(num_equation)
  return formatted, solutions


def cryptarithm():
  nums = []
  letters = [chr(ord('A') + i) for i in range(26)]
  random.shuffle(letters)
  key = {str(n): letters[n] for n in range(len(letters))}
  choice = 0
  if choice == 0:  #addition
    amt = random.randrange(2, 5)
    for x in range(amt):
      digits = random.randrange(1, 6)
      if digits == 1:
        n = random.randrange(0, 9)
      elif digits == 2:
        n = random.randrange(10, 99)
      elif digits == 3:
        n = random.randrange(100, 999)
      elif digits == 4:
        n = random.randrange(1000, 9999)
      else:
        n = random.randrange(10000, 99999)
      nums.append(n)
    nums.append(sum(nums))
    formula = ""
    body = []
    quick_formula = ""
    for x in range(len(nums) - 2):
      body.append([" ", "".join([key[i] for i in str(nums[x])])])
      quick_formula += "".join([key[i] for i in str(nums[x])]) + "+"
    body.append(["+", "".join([key[i] for i in str(nums[-2])])])
    quick_formula += "".join([key[i] for i in str(nums[x])]) + "=="
    body.append(["", "---------"])
    body.append([" ", "".join([key[i] for i in str(nums[-1])])])
    quick_formula += "".join([key[i] for i in str(nums[-1])])
    print(quick_formula.split("=="))
    formula = "```" + t2a(
      header="",
      body=body,
      style=PresetStyle.plain,
      cell_padding=0,
      alignments=Alignment.RIGHT,
    ) + "```"
    solutions = find_cryptarithm_solutions(quick_formula)
  return formula, solutions


def find_cryptarithm_solutions(puzzle):
  # Extract unique letters from the puzzle
  letters = set(puzzle.replace(' ', ''))
  # Generate all possible digit permutations
  digits = [str(x) for x in range(10)]
  permutations = itertools.permutations(digits, len(letters))
  # print(str(list(permutations)))
  # Iterate over each permutation and check if it satisfies the puzzle
  solutions = []
  for perm in permutations:
    mapping = dict(zip(letters, perm))
    del mapping["="]
    del mapping["+"]
    left, right = puzzle.split("==")
    left = left.split("+")
    good_equation = ""
    for x in left:
      x = x.translate(str.maketrans(mapping))
      good_equation += "+" + str(int(x))
    good_equation = good_equation[1:]
    good_equation += "==" + str(int(right.translate(str.maketrans(mapping))))
    # right=right.split("+")
    # print(left)
    # print(right)
    # print(mapping)
    # print(str.maketrans(mapping))
    # mapping={l:t for }
    # equation = puzzle.translate(str.maketrans(mapping))
    # print(good_equation)
    # print(equation)
    if eval(good_equation):
      solutions.append(mapping)
  print(solutions)
  return solutions


def encrypt_morbit(text):
  #prep
  text = "".join([x.upper() for x in text if x.isalpha() or x == "   "])
  #convert to morse code
  morse_code = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    ' ': 'x'
  }
  encrypted = ""
  for x in text:
    encrypted += morse_code[x] + 'x'
  if len(encrypted) % 2 == 1:
    encrypted = encrypted[:-1]
  print(encrypted)
  left = ["..", ".-", ".x", "-.", "--", "-x", "x.", "x-", "xx"]
  random.shuffle(left)
  key = dict()
  for x in range(9):
    key[str(x)] = left[x]
  print(key)
  output = ""
  for c in range(0, len(encrypted), 2):
    good_nums = [
      x for x in key.keys() if key[x] == encrypted[c] + encrypted[c + 1]
    ]
    output += random.choice(good_nums)
  return output, key


def encrypt_pollux(text):
  #prep
  text = "".join([x.upper() for x in text if x.isalpha() or x == " "])

  #convert to morse code
  morse_code = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    ' ': 'x'
  }
  encrypted = ""
  for x in text:
    encrypted += morse_code[x] + 'x'
  encrypted = encrypted[:-1]
  print(encrypted)
  key = {}
  left = range(9)
  #designate out one X
  c = random.randrange(0, 9)
  key[str(c)] = "x"
  left = list(set(left) - {c})
  #designate out one .
  c = random.choice(left)
  key[str(c)] = "."
  left = list(set(left) - {c})
  #designate out one -
  c = random.choice(left)
  key[str(c)] = "-"
  left = list(set(left) - {c})
  print(left)
  for n in left:
    choice = random.randrange(0, 3)
    if choice == 0:
      key[str(n)] = 'x'
    elif choice == 1:
      key[str(n)] = '.'
    else:
      key[str(n)] = '-'

  output = ""
  for c in encrypted:
    good_nums = [x for x in key.keys() if key[x] == c]
    output += random.choice(good_nums)
  return output, key


def encrypt_baconian(text):
  text = "".join([x.upper() for x in text if x.isalpha()])
  choice = random.randrange(0, 16)  # choose type of encryption
  if choice == 0:  # 0 and 1
    zeroes = [0]
    ones = [1]
  if choice == 1:  #random 2 letters
    letters = [
      x for x in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]
    zeroes = [random.choice(letters)]
    print(letters)
    print(zeroes)
    print("HAHAH")
    letters = list(set(letters) - set(zeroes))
    print(letters)
    ones = [random.choice(letters)]
    print(ones)
  if choice == 2:  #A-N is 0 and O-Z is 1
    zeroes = [x for x in "ABCDEFGHIJKLMN"]
    ones = [x for x in "OPQRSTUVWXYZ"]
  if choice == 3:  #A-N is 1 and O-Z is 0
    ones = [x for x in "ABCDEFGHIJKLMN"]
    zeroes = [x for x in "OPQRSTUVWXYZ"]
  if choice == 4:  #even letters are 0, odd letters are 1
    zeroes = [
      x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (ord(x) - 65) % 2 == 0
    ]
    ones = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (ord(x) - 65) % 2 == 1]
  if choice == 5:  #even letters are 1, odd letters are 0
    ones = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (ord(x) - 65) % 2 == 0]
    zeroes = [
      x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (ord(x) - 65) % 2 == 1
    ]
  if choice == 6:  #prime letters are 1, composite letters are 0
    ones = [
      x for x in "BCDEFGHIJKLMNOPQRSTUVWXYZ"
      if ord(x) - 65 in [2, 3, 5, 7, 11, 13, 17, 19, 23]
    ]
    zeroes = [
      x for x in "BCDEFGHIJKLMNOPQRSTUVWXYZ"
      if ord(x) - 65 not in [2, 3, 5, 7, 11, 13, 17, 19, 23]
    ]
  if choice == 7:  #prime letters are 0, composite letters are 1
    ones = [
      x for x in "BCDEFGHIJKLMNOPQRSTUVWXYZ"
      if ord(x) - 65 not in [2, 3, 5, 7, 11, 13, 17, 19, 23]
    ]
    zeroes = [
      x for x in "BCDEFGHIJKLMNOPQRSTUVWXYZ"
      if ord(x) - 65 in [2, 3, 5, 7, 11, 13, 17, 19, 23]
    ]
  if choice == 8:  #multiple different letters
    distribution = random.randrange(2, 6)
    letters_list = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    zeroes = random.sample(letters_list, distribution)
    letters_list = list(set(letters_list) - set(zeroes))
    distribution = random.randrange(2, 6)
    ones = random.sample(letters_list, distribution)
  if choice == 9:  #multiple symbols
    distribution = random.randrange(2, 6)
    letters_list = [x for x in "{}[]!#$%^&()_-+=;:?>.<,0123456789"]
    zeroes = random.sample(letters_list, distribution)
    letters_list = list(set(letters_list) - set(zeroes))
    distribution = random.randrange(2, 6)
    ones = random.sample(letters_list, distribution)
  if choice == 10:  # random 2 symbols
    letters = [x for x in "{}[]!#$%^&()_-+=;:?>.<,0123456789"]
    zeroes = [random.choice(letters)]
    letters = list(set(letters) - set(zeroes))
    ones = [random.choice(letters)]
  if choice == 11:  #even numbers are 1, odd numbers are 0
    ones = [0, 2, 4, 6, 8]
    zeroes = [1, 3, 5, 7, 9]
  if choice == 12:  #even numbers are 0, odd numbers are 1
    zeroes = [0, 2, 4, 6, 8]
    ones = [1, 3, 5, 7, 9]
  if choice == 13:  #prime numbers are 1, composite numbers are 0
    ones = [2, 3, 5, 7]
    zeroes = [1, 4, 6, 8, 9]
  if choice == 14:  #prime numbers are 0, composite numbers are 1
    zeroes = [2, 3, 5, 7]
    ones = [1, 4, 6, 8, 9]
  if choice == 15:  #0 and 1 are switched
    zeroes = [1]
    ones = [0]
  encrypted = ""
  for x in text:
    if ord(x) - 65 <= 8:
      letter = list(format(ord(x) - 65, '#07b'))[-5:]
    elif 8 < ord(x) - 65 <= 20:
      letter = list(format(ord(x) - 66, '#07b'))[-5:]
    else:
      letter = list(format(ord(x) - 67, '#07b'))[-5:]
    for c in range(len(letter)):
      if letter[c] == "0":
        letter[c] = str(random.choice(zeroes))
      else:
        letter[c] = str(random.choice(ones))
    encrypted += "".join(letter) + " "
  return encrypted, choice


def generate_key():
  alphabet = list(string.ascii_uppercase)
  random.shuffle(alphabet)
  key = ''.join(alphabet)
  return key


def generate_spanish_key():
  alphabet = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
  random.shuffle(alphabet)
  key = ''.join(alphabet)
  return key


def atbash_cipher(text):
  # Create translation tables for uppercase and lowercase letters
  uppercase_table = str.maketrans(string.ascii_uppercase,
                                  string.ascii_uppercase[::-1])
  # lowercase_table = str.maketrans(string.ascii_lowercase, string.ascii_lowercase[::-1])

  # Apply the translation tables to the input text
  translated_text = text.translate(uppercase_table)
  return translated_text


def caesar_cipher(plaintext, n):
  ans = ""
  # iterate over the given text
  for i in range(len(plaintext)):
    ch = plaintext[i]

    if ch.isalpha():
      ans += chr((ord(ch) + n - 65) % 26 + 65)

  return ans


def aristocrat_encoder(plaintext, key):
  alphabet = string.ascii_uppercase
  encoded_text = ""

  for char in plaintext:
    if char in alphabet:
      index = alphabet.index(char.upper())
      encoded_char = key[index]
      encoded_text += encoded_char
    if char == " ":
      encoded_text += " "

  return encoded_text


def xenocrypt_encoder(plaintext, key):
  alphabet = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
  encoded_text = ""

  for char in plaintext:
    if char in alphabet:
      index = alphabet.index(char.upper())
      encoded_char = key[index]
      encoded_text += encoded_char
    elif char in "ÁÉÍÓÚ":
      index = "ÁÉÍÓÚ".index(char.upper())
      new_char = "AEIOU"[index]
      new_index = alphabet.index(new_char)
      encoded_char = key[index]
      encoded_text += encoded_char
    elif char == " ":
      encoded_text += " "

  return encoded_text


def generate_hill_key(key_size):
  # Generate a random key matrix
  while True:
    key = np.random.randint(0, 26, size=(key_size, key_size))
    key = key.astype(np.int32)

    # Calculate the determinant of the key matrix
    determinant = int(round(np.linalg.det(key)))

    # Check if the key is invertible (determinant must be coprime with 26)
    if np.gcd(determinant, 26) == 1:
      break

  # Convert the key matrix to string representation
  key = "".join([chr(c + ord('A')) for row in key for c in row])

  return key


def encrypt_hill(input_text, key):
  # Convert input text to uppercase and remove any spaces
  input_text = input_text.upper()
  new_input = ""
  for x in input_text:
    if x.isalpha():
      new_input += x
  input_text = new_input
  # Pad the input text with 'X' if its length is not divisible by 2
  if len(input_text) % 2 != 0:
    input_text += 'X'

  # Convert the input text into pairs of numbers
  input_nums = [ord(c) - ord('A') for c in input_text]
  input_pairs = [(input_nums[i], input_nums[i + 1])
                 for i in range(0, len(input_nums), 2)]

  # Create the key matrix
  key = key.upper()
  key_nums = [ord(c) - ord('A') for c in key]
  key_matrix = np.array(key_nums).reshape(2, 2)

  # Encrypt the input pairs using the key matrix
  encrypted_pairs = []
  for pair in input_pairs:
    pair_matrix = np.array(pair).reshape(2, 1)
    encrypted_matrix = np.dot(key_matrix, pair_matrix) % 26
    encrypted_pairs.append((encrypted_matrix[0][0], encrypted_matrix[1][0]))

  # Convert the encrypted pairs back to characters
  encrypted_text = ""
  for pair in encrypted_pairs:
    encrypted_text += chr(pair[0] + ord('A')) + chr(pair[1] + ord('A'))

  return encrypted_text


import numpy as np


def decrypt_hill(encrypted_text, key):
  # Convert the encrypted text to uppercase and remove any spaces
  encrypted_text = encrypted_text.replace(" ", "").upper()

  # Convert the encrypted text into pairs of numbers
  encrypted_nums = [ord(c) - ord('A') for c in encrypted_text]
  encrypted_pairs = [(encrypted_nums[i], encrypted_nums[i + 1])
                     for i in range(0, len(encrypted_nums), 2)]

  # Create the key matrix
  key = key.upper()
  key_nums = [ord(c) - ord('A') for c in key]
  key_matrix = np.array(key_nums).reshape(2, 2)

  # Calculate the modular inverse of the key matrix
  det = (key_matrix[0][0] * key_matrix[1][1] -
         key_matrix[0][1] * key_matrix[1][0]) % 26
  det_inv = -1
  for i in range(26):
    if (det * i) % 26 == 1:
      det_inv = i
      break

  # Check if the modular inverse exists
  if det_inv == -1:
    raise ValueError("The key matrix is not invertible.")

  # Calculate the adjugate matrix
  adjugate_matrix = np.array([[key_matrix[1][1], -key_matrix[0][1]],
                              [-key_matrix[1][0], key_matrix[0][0]]])

  # Calculate the inverse matrix
  inverse_matrix = (det_inv * adjugate_matrix) % 26

  # Decrypt the encrypted pairs using the inverse matrix
  decrypted_pairs = []
  for pair in encrypted_pairs:
    pair_matrix = np.array(pair).reshape(2, 1)
    decrypted_matrix = np.dot(inverse_matrix, pair_matrix) % 26
    decrypted_pairs.append((decrypted_matrix[0][0], decrypted_matrix[1][0]))

  # Convert the decrypted pairs back to characters
  decrypted_text = ""
  for pair in decrypted_pairs:
    decrypted_text += chr(pair[0] + ord('A')) + chr(pair[1] + ord('A'))

  return decrypted_text
