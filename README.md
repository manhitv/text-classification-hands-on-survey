# Text Classification Hands-on Survey

## Introduction

In this repository, I have surveyed approaches to a typical Text Classification problem. Those are some opinions on text processing methods, text classfication algorithms and some utilities when working with text data.

## Table of Contents

### Text and Document Feature Extraction
**1. Introduction to data preprocessing**

Data goes through a series of steps during preprocessing:
    
- *Data cleaning*: is the process of
    + Detecting and correcting (removing) corrupt or inaccurate records from a record set, table or database
    + Identifying incomplete, incorrect, inaccurate or irrelevant parts of data
    + Replacing, modifying or deleting the dirty or coarse data.

    A common data cleansing practice is data enhancement, where data is made more complete by adding related information and data editing that is the process involving the review and adjustment of collected survey data, to control the quality of the collected data.

- *Data wrangling*: Sometimes referred to as data munging, is the process of transforming and mapping data from one “raw” data form into another format with the intent of making it more appropriate and valuable for a variety of downstream purposes such as analytics. This may include further munging, data visualization, data aggregation, training a statistical model...

- *Data reduction or instance/feature selection and extraction*: Approaches for instance selection can be applied for reducing the original dataset to a manageable volume, leading to a reduction of the computational resources that are necessary for performing the learning process and might improve the accuracy in classification problems.

**2. Text Cleaning and Pre-processing**

Text preprocessing method is used to preprocess your text simply means to bring your text into a form that is predictable and analyzable for your task. Text preprocessing is task-specific.

- *Must Do*:
    + Noise removal
    + Lowercasing (can be task dependent in some cases)
- *Should Do*:
    + Simple normalization – (e.g. standardize near identical words)
- *Task Dependent*:
    + Advanced normalization (e.g. addressing out-of-vocabulary words)
    + Stop-word removal
    + Stemming / Lemmatization
    + Text enrichment / Augmentation
- *Additional considerations*:
    + Handling large documents and large collections of text documents that do not fit into memory.
    + Extracting text from markup like HTML, PDF or other structured document formats.
    + Trans-literation of characters from other languages into English.
    + Handling of domain specific words, phrases and acronyms.

Below are examples of some techniques.
* *Tokenization*

    Tokenization is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence.
        + You could use spaCy (or gensim, nltk). Recommend to use spaCy because it's one of the most versatile and widely used libraries in NLP.
        + You could also use tf.text with BERTTokenizer for this tasks (and more - full pipeline in tensorflow)
* *Stop words*

    In spaCy, you could get the complete list via: ```from spacy.lang.en.stop_words import STOP_WORDS```
* *Capitalization*

    To reduce the problem space, the most common approach is to reduce everything to lower case. This brings all words in a document in same space, but it often changes the meaning of some words, such as "US" to "us" where first one represents the United States of America and second one is a pronoun. To solve this, slang and abbreviation converters can be applied.
* *Slangs and Abbreviations*

    An abbreviation is a shortened form of a word, such as SVM stand for Support Vector Machine. Slang is a version of language that depicts informal conversation or text that has different meaning, such as "lost the plot", it essentially means that 'they've gone mad'. Common method to deal with these words is converting them to formal language.
* *Noise Removal*

    Remove punctuations or special characters - this is one of the most essential text preprocessing steps and also highly domain dependent. There are various ways to remove noise, it all depends on which domain you are working in and what entails noise for your task. This includes:
    * punctuation removal
    * special character removal
    * numbers removal
    * html formatting removal
    * domain specific keyword removal (e.g. ‘RT’ for retweet)
    * source code removal
    * header removal and more.
* *Spelling Correction*
    
    Some keywords about algorithms such as hashing-based and context-sensitive spelling correction techniques, or spelling correction using trie and damerau-levenshtein distance bigram.
* *Stemming*

    Modifying a word to obtain its variants using different linguistic processeses like affixation (addition of affixes). E.g. changed, changes, changing, changer --> chang but not change as in Lemmatization, so it might not be useful as Lemmatization (but you also find it via nltk library)
* *Lemmatization*

    Lemmatization is the process of eliminating redundant prefix or suffix of a word and extract the base word (lemma). More linguistic features (such as POS, is_alpha...) about NLP could be found at spaCy website: https://spacy.io/usage/linguistic-features
* *Some other examples with Web data*
* *Others: Extracting Text from PDF, MSWord and other Binary Formats*

    ASCII text and HTML text are human readable formats. Text often comes in binary formats — like PDF and MSWord — that can only be opened using specialized software. Third-party libraries such as pypdf and pywin32 provide access to these formats. Extracting text from multi-column documents is particularly challenging. For once-off conversion of a few documents, it is simpler to open the document with a suitable application, then save it as text to your local drive, and access it as described below. If the document is already on the web, you can enter its URL in Google's search box. The search result often includes a link to an HTML version of the document, which you can save as text.

**3. Feature Extraction 1 - Word Embedding**

Different word embedding procedures have been proposed to translate these unigrams into consummable input for machine learning algorithms. A very simple way to perform such embedding is term-frequency~(TF) where each word will be mapped to a number corresponding to the number of occurrence of that word in the whole corpora. The other term frequency functions have been also used that represent word-frequency as Boolean or logarithmically scaled number. Here, each document will be converted to a vector of same length containing the frequency of the words in that document. Although such approach may seem very intuitive but it suffers from the fact that particular words that are used very commonly in language literature might dominate this sort of word representations.

  * *Word2Vec*

    Original & reference from [https://code.google.com/p/word2vec/]. Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous Bag-of-Words or the Skip-Gram neural network architectures. The user should specify the following: desired vector dimensionality (size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model), training algorithm (hierarchical softmax and / or negative sampling) or threshold for downsampling the frequent words, number of threads to use, format of the output word vector file (text or binary).
    
   * *Global Vectors for Word Representation (GloVe)*
    
    An implementation of the GloVe model for learning word representations is provided, and describe how to download web-dataset vectors or train your own. See the project page at [https://nlp.stanford.edu/projects/glove/] for more information on glove vectors.
   * *Contextualized Word Representations*
    
    ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis.

    ELMo representations are:

        - Contextual: The representation for each word depends on the entire context in which it is used.
        - Deep: The word representations combine all layers of a deep pre-trained neural network.
        - Character based: ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.
    
    You could use ELMo via two ways:
        - Easy version at [https://tfhub.dev/google/elmo/2]
        - Some other pretrained models: [https://allennlp.org/elmo] - including a JSON formatted "options" file with hyperparameters and a hdf5 formatted file with the model weights. There are three ways to integrate ELMo representations into a downstream task, depending on your use case.
            + Compute representations on the fly from raw text using character input. This is the most general method and will handle any input text. It is also the most computationally expensive. This method is necessary for evaluating at test time on unseen data (e.g. public SQuAD leaderboard)
            + Precompute and cache the context independent token representations, then compute context dependent representations using the biLSTMs for input data. This method is less computationally expensive then #1, but is only applicable with a fixed, prescribed vocabulary. This approach is a good compromise for large datasets where the size of the file in is unfeasible (SNLI, SQuAD)
            + Precompute the representations for your entire dataset and save to a file. It is a good choice for smaller datasets or in cases where you'd like to use ELMo in other frameworks.

          In all cases, the process roughly follows the same steps. First, create a Batcher (or TokenBatcher for #2) to translate tokenized strings to numpy arrays of character (or token) ids. Then, load the pretrained ELMo model (class BidirectionalLanguageModel). Finally, for steps #1 and #2 use weight_layers to compute the final ELMo representations. For #3, use BidirectionalLanguageModel to write all the intermediate layers to a file.
   * *FastText*
    
    FastText is a library for efficient learning of word representations and sentence classification. Reference at https://github.com/facebookresearch/fastText and [https://fasttext.cc/docs/en/english-vectors.html]
    
    Features:
        - Recent state-of-the-art English word vectors.
        - Word vectors for 157 languages trained on Wikipedia and Crawl.
        - Models for language identification and various supervised tasks.
**4. Feature Extraction 2 - Weighted Words**
**5. Comparison of Feature Extraction Techniques**
----
### Text Classification Techniques 
1. Non-neural network
    * Linear models (RidgeClassifier, SGDClassifier)
    * Naive Bayes Classifier
    * Support Vector Machine (LinearSVC)
    * Random Forest (Bagging)
    * GradientBoosting (Boosting)
2. Deep Learning
    * Deep Neural Networks
    * Recurrent Neural Networks (RNN): Gated Recurrent Unit (GRU) / Long Short-Term Memory (LSTM)
    * Convolutional Neural Networks (CNN)
    * Recurrent Convolutional Neural Networks (RCNN)
    * Transformers
    * Others
3. Comparison Text Classification Algorithms
----
### Others
1. End-to-End model
2. Utilize getting/loading data
3. Some applications

