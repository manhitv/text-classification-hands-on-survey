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

    Original & reference from https://code.google.com/p/word2vec/. Given a text corpus, the word2vec tool learns a vector for every word in the vocabulary using the Continuous Bag-of-Words or the Skip-Gram neural network architectures. The user should specify the following: desired vector dimensionality (size of the context window for either the Skip-Gram or the Continuous Bag-of-Words model), training algorithm (hierarchical softmax and / or negative sampling) or threshold for downsampling the frequent words, number of threads to use, format of the output word vector file (text or binary).
   
  * *Global Vectors for Word Representation (GloVe)*
    
    An implementation of the GloVe model for learning word representations is provided, and describe how to download web-dataset vectors or train your own. See the project page at https://nlp.stanford.edu/projects/glove/ for more information on glove vectors.
  * *Contextualized Word Representations*
    
    ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis.

    ELMo representations are:
    
    * Contextual: The representation for each word depends on the entire context in which it is used.
    * Deep: The word representations combine all layers of a deep pre-trained neural network.
    * Character based: ELMo representations are purely character based, allowing the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.
    
    You could use ELMo via two ways:
    
    * Easy version at https://tfhub.dev/google/elmo/2
    * Some other pretrained models: https://allennlp.org/elmo - including a JSON formatted "options" file with hyperparameters and a hdf5 formatted file with the model weights. There are three ways to integrate ELMo representations into a downstream task, depending on your use case.
        + Compute representations on the fly from raw text using character input. This is the most general method and will handle any input text. It is also the most computationally expensive. This method is necessary for evaluating at test time on unseen data (e.g. public SQuAD leaderboard)
        + Precompute and cache the context independent token representations, then compute context dependent representations using the biLSTMs for input data. This method is less computationally expensive then #1, but is only applicable with a fixed, prescribed vocabulary. This approach is a good compromise for large datasets where the size of the file in is unfeasible (SNLI, SQuAD)
        + Precompute the representations for your entire dataset and save to a file. It is a good choice for smaller datasets or in cases where you'd like to use ELMo in other frameworks.
        
        In all cases, the process roughly follows the same steps. First, create a Batcher (or TokenBatcher for #2) to translate tokenized strings to numpy arrays of character (or token) ids. Then, load the pretrained ELMo model (class BidirectionalLanguageModel). Finally, for steps #1 and #2 use weight_layers to compute the final ELMo representations. For #3, use BidirectionalLanguageModel to write all the intermediate layers to a file.
  * *FastText*
    
    FastText is a library for efficient learning of word representations and sentence classification. Reference at https://github.com/facebookresearch/fastText and https://fasttext.cc/docs/en/english-vectors.html
    
    Features:
    * Recent state-of-the-art English word vectors.
    * Word vectors for 157 languages trained on Wikipedia and Crawl.
    * Models for language identification and various supervised tasks.

**4. Feature Extraction 2 - Weighted Words**

Term frequency is Bag of words that is one of the simplest techniques of text feature extraction. This method is based on counting number of the words in each document and assign it to feature space.
Term Frequency-Inverse Document Frequency Although tf-idf tries to overcome the problem of common terms in document (through lowering weight by divide the number of documents contain these terms), it still suffers from some other descriptive limitations. Namely, tf-idf cannot account for the similarity between words in the document since each word is presented as an index. In the recent years, with development of more complex models, such as neural nets, new methods has been presented that can incorporate concepts, such as similarity of words and part of speech tagging. This work uses, word2vec and Glove, two of the most common methods that have been successfully used for deep learning techniques.

**5. Comparison of Feature Extraction Techniques**

There are 4 main performance metrics with these techniques:
 - Syntactic (capture the position in the text): TF-IDF could not solve this
 - Semantics (capture meaning in the text): TF-IDF could not solve this
 - Polysemy (meaning of the word from the text - context): only ELMo (Contextualized Word Representations) could solve this
 - Out-of-vocabulary words from corpus: FastText and TF-IDF support this via n-gram hyperparameters

Reference detail version at https://github.com/kk7nc/Text_Classification#comparison-of-feature-extraction-techniques

----
### Text Classification Techniques 
**1. Non-neural network**
  * *Linear models (RidgeClassifier, SGDClassifier)*
  * *Naive Bayes Classifier*
  * *Support Vector Machine (LinearSVC)*
  * *Random Forest (Bagging)*
  * *GradientBoosting (Boosting)*

**2. Neural network (Deep Learning)**

  Deep Neural Networks architectures are designed to learn through multiple connection of layers where each single layer only receives connection from previous and provides connections only to the next layer in hidden part.
    
  The input is a connection of feature space (as feature_extraction with first hidden layer). For Neural Networks, input layer could be tf-idf, word embedding or etc. The output layer houses neurons equal to the number of classes for multi-class classification and only one neuron for binary classification.

   * *Deep Neural Networks*
   * *Recurrent Neural Networks (RNN): Gated Recurrent Unit (GRU) / Long Short-Term Memory (LSTM)*
   * *Convolutional Neural Networks (CNN)*
   * *Recurrent Convolutional Neural Networks (RCNN)*
   * *Transformers*
   * *Some pretrained models*

**3. Comparison Text Classification Algorithms**

   * *Non-neural network*:
     * Pros:
        - Work well with small dataset
        - Computational complexity is cheap & faster
        - Implementation is much easier
        -  Interpretability: available, but might loss with some complex algorithms such as bagging, boosting
     * Cons:
        - Scalability: small-scale
   * *Deep Learning*:
     * Pros:
        - Flexible with features design (reduces the need for feature engineering, one of the most time-consuming parts of machine learning practice)
        - Architecture that can be adapted to new problems
        - Can deal with complex input-output mappings: End-to-End architecture
        - Can easily handle online learning (It makes it very easy to re-train the model when newer data becomes available)
        - Parallel processing capability (It can perform more than one job at the same time)
     * Cons:
        - Requires a large amount of data (if you only have small sample text data, deep learning is unlikely to outperform other approaches)
        - Problem: complex input-output mappings
        - Extremely computationally expensive to train
        - Model interpretability is hard
        - Finding an efficient architecture and structure is still the main challenge of this technique
----
### Others
**1. Utilize getting/loading data**

**2. Some applications**

* *Information Retrieval*
    
    Information retrieval is finding documents of an unstructured data that meet an information need from within large collections of documents. With the rapid growth of online information, particularly in text format, text classification has become a significant technique for managing this type of data. Some of the important methods used in this area are Naive Bayes, SVM, decision tree, J48, k-NN and IBK. One of the most challenging applications for document and text dataset processing is applying document categorization methods for information retrieval.

* *Information Filtering*

    Information filtering refers to selection of relevant information or rejection of irrelevant information from a stream of incoming data. Information filtering systems are typically used to measure and forecast users' long-term interests. Probabilistic models, such as Bayesian inference network, are commonly used in information filtering systems. Bayesian inference networks employ recursive inference to propagate values through the inference network and return documents with the highest ranking. Chris used vector space model with iterative refinement for filtering task.

* *Sentiment Analysis*
    
    Sentiment analysis is a computational approach toward identifying opinion, sentiment, and subjectivity in text. Sentiment classification methods classify a document associated with an opinion to be positive or negative. The assumption is that document d is expressing an opinion on a single entity e and opinions are formed via a single opinion holder h. Naive Bayesian classification and SVM are some of the most popular supervised learning methods that have been used for sentiment classification. Features such as terms and their respective frequency, part of speech, opinion words and phrases, negations and syntactic dependency have been used in sentiment classification techniques.

* *Recommender Systems*
    
    Content-based recommender systems suggest items to users based on the description of an item and a profile of the user's interests. A user's profile can be learned from user feedback (history of the search queries or self reports) on items as well as self-explained features~(filter or conditions on the queries) in one's profile. In this way, input to such recommender systems can be semi-structured such that some attributes are extracted from free-text field while others are directly specified. Many different types of text classification methods, such as decision trees, nearest neighbor methods, Rocchio's algorithm, linear classifiers, probabilistic methods, and Naive Bayes, have been used to model user's preference.

* *Knowledge Management*
    
    Textual databases are significant sources of information and knowledge. A large percentage of corporate information (nearly 80 %) exists in textual data formats (unstructured). In knowledge distillation, patterns or knowledge are inferred from immediate forms that can be semi-structured ( e.g.conceptual graph representation) or structured/relational data representation). A given intermediate form can be document-based such that each entity represents an object or concept of interest in a particular domain. Document categorization is one of the most common methods for mining document-based intermediate forms. In the other work, text classification has been used to find the relationship between railroad accidents' causes and their correspondent descriptions in reports.

* *Document Summarization*
    
    Text classification used for document summarizing which summary of a document may employ words or phrases which do not appear in the original document. Multi-document summarization also is necessitated due to increasing online information rapidly. So, many researchers focus on this task using text classification to extract important feature out of a document.    
----

Thanks to [Kamran Kowsari](https://github.com/kk7nc) for very helpful repository.

