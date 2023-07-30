import tensorflow as td
from tensorflow import keras
import numpy

# We have a magic box called "data" that contains lots of movie reviews, and our friends TensorFlow and Keras will help us understand these reviews.
data = keras.datasets.imdb

# We are opening the magic box to take out two sets of movie reviews - one set for training and one set for testing. Each review has some words, and we also have special labels to know if the reviews are good or bad.
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# We are looking at the 7th movie review in our training set (starting from 0). This review is just a bunch of numbers that represent the words in the review.
print(train_data[6])

# We have a dictionary, which is like a special book that tells us the code for each word in the reviews. But the code starts from 3, not from 0.
word_index = data.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}

# We have some special words that need their own codes, so we give them numbers: 0, 1, 2, and 3.
word_index["<PAD>"] = 0
word_index ["<START>"] = 1
word_index["<UNK>"] = 2
word_index ["<UNUSED>"] = 3

# We made another special book called "reverse_word_index" that tells us the words for each code. So now we know the codes and the words for all the reviews.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# All the movie reviews have different lengths, but our friends TensorFlow and Keras like reviews of the same length. So we make all reviews 250 words long by adding special words (code 0) at the end.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# We made a magic spell called "decode_review" that turns the numbers in a movie review back into words, so we can read the review.
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text.tolist()])

# We use the magic spell to read the first movie review in the training set. And we also check the lengths of the first two reviews in the test set.
print(decode_review(train_data[0]))
print(len(test_data[0]), len(test_data[1]))
      
# model down here
# We are building a special machine called a "model" that will understand the movie reviews. It will have different layers like a sandwich.
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# We tell our special machine how to learn by giving it a recipe. The recipe says, "Use 'adam' magic and 'binary_crossentropy' magic to learn from the reviews. Also, remember to use 'accuracy' magic to see how well you're learning."
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# We split the training set into two parts: one for training the machine and the other for checking if the machine is learning wel
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Now, our machine is learning from the movie reviews! It will read the reviews 40 times (epochs) and learn in groups of 512 reviews at a time (batch_size). It will also use some reviews from the validation set to check how well it's learning.
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# The machine finished learning, and now we are checking how well it understands the reviews. We can see the results to know how good it did.
results = model.evaluate(x_val, y_val)
print(results)

# Now, our machine is reading a new movie review from the test set and telling us if it's good or bad. We can also see the review, the machine's guess, and the actual label to know if the machine was right.
test_review = test_data[:1]
predict = model.predict(test_review)
print("Review: ")
print(decode_review(test_review[0]))
print("Prediction: " + str(predict[0][0]))
print("Actual: " + str(test_labels[0]))

# Save Model
model.save("model.h5")

# Load Model
# model = keras.models.load_model("model.h5")

# def review_encode(s):
#     encoded = [1]

#     for word in s:
#         if word.lower() in word_index:
#             encoded.append(word_index[word.lower()])
#         else:
#             encoded.append(2)

#     return encoded

# with open("test.txt", "r") as f:
#     for line in f.readlines():
#         nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
#         encode = [word_index[word] for word in nline]
#         encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
#         predict = model.predict(encode)
#         print(line)
#         print(encode)
#         print(predict[0])