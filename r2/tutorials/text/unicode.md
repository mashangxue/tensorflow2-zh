---
title: Unicode strings
categories: tensorflow2官方教程
tags: tensorflow2.0
top: 199
abbrlink: tensorflow/tf2-tutorials-text-unicode
---

##### Copyright 2018 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Unicode strings

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/alpha/tutorials/text/unicode"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/unicode.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/unicode.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

## Introduction

Models that process natural language often handle different languages with different character sets.  *Unicode* is a standard encoding system that is used to represent character from almost all languages.  Each character is encoded using a unique integer [code point](https://en.wikipedia.org/wiki/Code_point) between `0` and `0x10FFFF`. A *Unicode string* is a sequence of zero or more code points.

This tutorial shows how to represent Unicode strings in TensorFlow and manipulate them using Unicode equivalents of standard string ops. It separates Unicode strings into tokens based on script dedection.


```
from __future__ import absolute_import, division, print_function, unicode_literals

!pip install tensorflow==2.0.0-alpha0
import tensorflow as tf
```

## The `tf.string` data type

The basic TensorFlow `tf.string` `dtype` allows you to build tensors of byte strings.
Unicode strings are utf-8 encoded by default.


```
tf.constant(u"Thanks 😊")
```

A `tf.string` tensor can hold byte strings of varying lengths because the byte strings are treated as atomic units. The string length is not included in the tensor dimensions.



```
tf.constant([u"You're", u"welcome!"]).shape
```

## Representing Unicode

There are two standard ways to represent a Unicode string in TensorFlow:

* `string` scalar — where the sequence of code points is encoded using a known [character encoding](https://en.wikipedia.org/wiki/Character_encoding).
* `int32` vector — where each position contains a single code point.

For example, the following three values all represent the Unicode string `"语言处理"` (which means "language processing" in Chinese):


```
# Unicode string, represented as a UTF-8 encoded string scalar.
text_utf8 = tf.constant(u"语言处理")
text_utf8
```


```
# Unicode string, represented as a UTF-16-BE encoded string scalar.
text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
text_utf16be
```


```
# Unicode string, represented as a vector of Unicode code points.
text_chars = tf.constant([ord(char) for char in u"语言处理"])
text_chars
```

### Converting between representations

TensorFlow provides operations to convert between these different representations:

* `tf.strings.unicode_decode`: Converts an encoded string scalar to a vector of code points.
* `tf.strings.unicode_encode`: Converts a vector of code points to an encoded string scalar.
* `tf.strings.unicode_transcode`: Converts an encoded string scalar to a different encoding.


```
tf.strings.unicode_decode(text_utf8,
                          input_encoding='UTF-8')
```


```
tf.strings.unicode_encode(text_chars,
                          output_encoding='UTF-8')
```


```
tf.strings.unicode_transcode(text_utf8,
                             input_encoding='UTF8',
                             output_encoding='UTF-16-BE')
```

### Batch dimensions

When decoding multiple strings, the number of characters in each string may not be equal.  The return result is a [`tf.RaggedTensor`](../../guide/ragged_tensors.ipynb), where the length of the innermost dimension varies depending on the number of characters in each string:


```
# A batch of Unicode strings, each represented as a UTF8-encoded string.
batch_utf8 = [s.encode('UTF-8') for s in
              [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']]
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8,
                                               input_encoding='UTF-8')
for sentence_chars in batch_chars_ragged.to_list():
  print(sentence_chars)
```

You can use this `tf.RaggedTensor` directly, or convert it to a dense `tf.Tensor` with padding or a `tf.SparseTensor` using the methods `tf.RaggedTensor.to_tensor` and `tf.RaggedTensor.to_sparse`.


```
batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
print(batch_chars_padded.numpy())
```


```
batch_chars_sparse = batch_chars_ragged.to_sparse()
```

When encoding multiple strings with the same lengths, a `tf.Tensor` may be used as input:


```
tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [ 99, 111, 119]],
                          output_encoding='UTF-8')
```

When encoding multiple strings with varyling length, a `tf.RaggedTensor` should be used as input:


```
tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8')
```

If you have a tensor with multiple strings in padded or sparse format, then convert it to a `tf.RaggedTensor` before calling `unicode_encode`:


```
 tf.strings.unicode_encode(
     tf.RaggedTensor.from_sparse(batch_chars_sparse),
     output_encoding='UTF-8')
```


```
 tf.strings.unicode_encode(
     tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1),
     output_encoding='UTF-8')
```

## Unicode operations

### Character length

The `tf.strings.length` operation has a parameter `unit`, which indicates how lengths should be computed.  `unit` defaults to `"BYTE"`, but it can be set to other values, such as `"UTF8_CHAR"` or `"UTF16_CHAR"`, to determine the number of Unicode codepoints in each encoded `string`.


```
# Note that the final character takes up 4 bytes in UTF8.
thanks = u'Thanks 😊'.encode('UTF-8')
num_bytes = tf.strings.length(thanks).numpy()
num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))
```

### Character substrings

Similarly, the `tf.strings.substr` operation accepts the "`unit`" parameter, and uses it to determine what kind of offsets the "`pos`" and "`len`" paremeters contain.


```
# default: unit='BYTE'. With len=1, we return a single byte.
tf.strings.substr(thanks, pos=7, len=1).numpy()
```


```
# Specifying unit='UTF8_CHAR', we return a single character, which in this case
# is 4 bytes.
print(tf.strings.substr(thanks, pos=7, len=1, unit='UTF8_CHAR').numpy())
```

### Split Unicode strings

The `tf.strings.unicode_split` operation splits unicode strings into substrings of individual characters:


```
tf.strings.unicode_split(thanks, 'UTF-8').numpy()
```

### Byte offsets for characters

To align the character tensor generated by `tf.strings.unicode_decode` with the original string, it's useful to know the offset for where each character begins.  The method `tf.strings.unicode_decode_with_offsets` is similar to `unicode_decode`, except that it returns a second tensor containing the start offset of each character.


```
codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')

for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
  print("At byte offset {}: codepoint {}".format(offset, codepoint))
```

## Unicode scripts

Each Unicode code point belongs to a single collection of codepoints known as a [script](https://en.wikipedia.org/wiki/Script_%28Unicode%29) .  A character's script is helpful in determining which language the character might be in. For example, knowing that 'Б' is in Cyrillic script indicates that modern text containing that character is likely from a Slavic language such as Russian or Ukrainian.

TensorFlow provides the `tf.strings.unicode_script` operation to determine which script a given codepoint uses. The script codes are `int32` values corresponding to [International Components for
Unicode](http://site.icu-project.org/home) (ICU) [`UScriptCode`](http://icu-project.org/apiref/icu4c/uscript_8h.html) values.



```
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']

print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]
```

The `tf.strings.unicode_script` operation can also be applied to multidimensional `tf.Tensor`s or `tf.RaggedTensor`s of codepoints:


```
print(tf.strings.unicode_script(batch_chars_ragged))
```

## Example: Simple segmentation

Segmentation is the task of splitting text into word-like units. This is often easy when space characters are used to separate words, but some languages (like Chinese and Japanese) do not use spaces, and some languages (like German) contain long compounds that must be split in order to analyze their meaning. In web text, different languages and scripts are frequently mixed together, as in "NY株価" (New York Stock Exchange).

We can perform very rough segmentation (without implementing any ML models) by using changes in script to approximate word boundaries. This will work for strings like the "NY株価" example above. It will also work for most languages that use spaces, as the space characters of various scripts are all classified as USCRIPT_COMMON, a special script code that differs from that of any actual text.


```
# dtype: string; shape: [num_sentences]
#
# The sentences to process.  Edit this line to try out different inputs!
sentence_texts = [u'Hello, world.', u'世界こんにちは']
```

First, we decode the sentences into character codepoints, and find the script identifeir for each character.


```
# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_codepoint[i, j] is the codepoint for the j'th character in
# the i'th sentence.
sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
print(sentence_char_codepoint)

# dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_scripts[i, j] is the unicode script of the j'th character in
# the i'th sentence.
sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
print(sentence_char_script)
```

Next, we use those script identifiers to determine where word boundaries should be added.  We add a word boundary at the beginning of each sentence, and for each character whose script differs from the previous character:


```
# dtype: bool; shape: [num_sentences, (num_chars_per_sentence)]
#
# sentence_char_starts_word[i, j] is True if the j'th character in the i'th
# sentence is the start of a word.
sentence_char_starts_word = tf.concat(
    [tf.fill([sentence_char_script.nrows(), 1], True),
     tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])],
    axis=1)

# dtype: int64; shape: [num_words]
#
# word_starts[i] is the index of the character that starts the i'th word (in
# the flattened list of characters from all sentences).
word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
print(word_starts)
```

We can then use those start offsets to build a `RaggedTensor` containing the list of words from all batches:


```
# dtype: int32; shape: [num_words, (num_chars_per_word)]
#
# word_char_codepoint[i, j] is the codepoint for the j'th character in the
# i'th word.
word_char_codepoint = tf.RaggedTensor.from_row_starts(
    values=sentence_char_codepoint.values,
    row_starts=word_starts)
print(word_char_codepoint)
```

And finally, we can segment the word codepoints `RaggedTensor` back into sentences:


```
# dtype: int64; shape: [num_sentences]
#
# sentence_num_words[i] is the number of words in the i'th sentence.
sentence_num_words = tf.reduce_sum(
    tf.cast(sentence_char_starts_word, tf.int64),
    axis=1)

# dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
#
# sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
# in the j'th word in the i'th sentence.
sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
    values=word_char_codepoint,
    row_lengths=sentence_num_words)
print(sentence_word_char_codepoint)
```

To make the final result easier to read, we can encode it back into UTF-8 strings:


```
tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()
```
