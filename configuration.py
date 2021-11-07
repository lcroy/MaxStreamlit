import os


class Config:
    def __init__(self, mode='conv'):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        # image path
        self.image_path = os.path.join(self.project_path, 'image')
        ## face path
        self.botx_face_path = os.path.join(self.image_path, 'JS.jpg')

        ## Text_Generation_image path
        self.TG_image_path = os.path.join(self.image_path, 'TG.jpg')
        ## Text_Generation_architecture_image path
        self.TG_architecture_image_path = os.path.join(self.image_path, 'gpt.png')

        ## Summarization_image path
        self.SU_image_path = os.path.join(self.image_path, 'su.jpg')
        ## Summarization_image_architecture_image path
        self.SU_architecture_image_path = os.path.join(self.image_path, 'bart.png')

        ## Question Answering image path
        self.QA_image_path = os.path.join(self.image_path, 'qa_process.jpg')
        ## Bert model Question Answering image path
        self.QA_bert_model_image_path = os.path.join(self.image_path, 'bert_model.jpg')

        ## Sentiment_Analysis_image path
        self.SA_image_path = os.path.join(self.image_path, 'SA.png')

        # models path
        self.model_path = os.path.join(self.project_path, 'models')
        self.QA_model_path = os.path.join(self.project_path, 'main', 'QA', 'model')
        # document path
        self.docs_path = os.path.join(self.project_path, 'docs')

        # home page
        self.home_title = 'Applied NLP to Robotics'
        self.home_des = "This is a demonstration platform for research work of " \
                        "**Natural Language Processing** of Robotics and Automation group in Aalborg University. " \
                        "The purpose of the platform is to share knowledge on how to apply Natural Language Processing to " \
                        "**manufacturing** domain. The platform shares the same backend of a Virtual Assistant, BotX.  \n\r" \
                        "This platform provides \n\r" \
                        "* **Task-oriented dialogue system** for industrial robots manipulation. \n\r" \
                        "* **Question and Answering** for answering the question based on the user provided info.\n\r" \
                        "* **Speech Recognition** for audio processing. \n\r" \
                        "* **Sentiment Analysis** for understanding customer/employee's feeling.\n\r" \
                        "* **Text Generation** for response auto-generation."

        # Text Generation page
        self.TG_title = "Text Generation - Play with Transformer"
        self.TG_des = "One of the cool functions of BotX is to complete your sentences. The function is designed based " \
                      "on the **OpenAI GPT-2** model which is a causal **transformer** " \
                      "pre-trained language modelling on 40GB of text model. You can type a snippet in the following " \
                      "box and see how Botx helps to generate the sentence."
        self.TG_model_overview = "The OpenAI GPT-2 language model is a direct successor to GPT (**Generative " \
                                 "Pre-training Transformer**). It expands the unsupervised language model to a much " \
                                 "larger scale by training on a giant collection of free text corpora GPT-2 has **1.5B** " \
                                 "parameters, **10x** more than the original GPT, and it achieves SOTA results on 7 out " \
                                 "of 8 tested language modeling datasets in a zero-shot transfer setting without any " \
                                 "task-specific fine-tuning. The pre-training data contains 8 million Web pages " \
                                 "collected by crawling qualified outbound links from Reddit. Large improvements by " \
                                 "OpenAI GPT-2 are specially noticeable on small datasets and datasets used for " \
                                 "measuring long-term dependency."
        self.TG_about = "OpenAI GPT-2 model was proposed in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by " \
                        "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever."

        # Summarization page
        self.SU_title = "Text Summarization - Extract key information from the given text"
        self.SU_des = "Text summarization in NLP is the process of summarizing the information in large texts for quicker consumption." \
                      "In our work, we leverage the **BART** model. It is an encoder-decoder model. It converts all language problems " \
                      "into a text-to-text format. You can type a snippet in the following " \
                      "box and see how Botx helps to generate summarization."
        self.SU_model_overview = "According to the [paper](https://arxiv.org/abs/1910.13461), BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) " \
                                 "corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original " \
                                 "text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, " \
                                 "can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), " \
                                 "and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best " \
                                 "performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, " \
                                 "where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for " \
                                 "text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable " \
                                 "training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question " \
                                 "answering, and summarization tasks, with gains of up to 6 ROUGE."
        self.SU_about = "BART model was proposed in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, " \
                        "Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, " \
                        "Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019."


        # Question Answering page
        self.QA_title = "Question Answering - Understanding conversations"
        self.QA_des = "The quest for knowledge is deeply human, and so it is not surprising that practically as soon " \
                      "as there were computers we were asking them questions. Most question answering systems focus on" \
                      " **factoid questions**, questions that can be answered with simple facts expressed in short texts."
        self.QA_process = "The approach used in BotX is information-retrieval-based factoid " \
                          "question-answering system. The goal of information retrieval based question answering is " \
                          "to answer a user’s question by finding short text segments on the web or some other " \
                          "collection of documentsThe key processes are question processing, passage retrieval and " \
                          "ranking, and answer extraction. "
        self.QA_model_overview = "BotX uses **BERT-based Question Answering**. The power of contextual embeddings allow " \
                                 "question answering models based on BERT contextual embeddings and the transformer " \
                                 "architecture to achieve even higher accuracy."
        self.QA_about = "Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). [BERT: Pretraining of deep " \
                        "bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805). In " \
                        "NAACL HLT 2019, 4171–4186. "

        # Sentiment Analysis page
        self.SA_title = "Sentiment Analysis - Understand your feeling"
        self.SA_des = "What does people feel about a certain topic? How can we know the operator's feeling when " \
                      "he/her works with a robot? Can we tell the emotion of the operator? \n\r We are looking into " \
                      "those questions and try to leverage the modern neural network to help us understand your feeeling."

        # sidebar
        self.nav_menu = ['Home', 'Natural Language Processing (NLP)', 'NLP + Computer Vision', 'NLP + Reinforcement Learning']
        self.NLP_menu = ['Task-Oriented Dialogue System', 'Question & Answering',
                     'Speech Recognition', 'Sentiment Analysis', 'Text Generation', 'Summarization', 'Other']
        self.CV_menu = ['Facial Recognition', 'Object Detection', 'Other']
        self.RL_menu = ['Dialogue Policy Planning', 'Other']
        self.contr_info = "This an open source project and you are very welcome to contribute your awesome comments, " \
                          "questions and pull requests to the source code (TBC). "
        self.abt_info = 'This app is maintained by Chen Li. You can learn more about me at [VBN](https://vbn.aau.dk/en/persons/142294) of Aalborg University. '

        # NER
        self.ner_model_path = self.project_path + '/spacy_NER/models'
        self.ner_dataset_path = self.project_path + '/spacy_NER/data/Odoo.json'

        # Intents
        self.intent_model_path = self.project_path + '/model/cnn_model.json'
        self.intent_weight_path = self.project_path + '/model/cnn_weights.h5'
        self.intent_RNN_model_path = self.project_path + '/model/rnn_model.json'
        self.intent_RNN_weight_path = self.project_path + '/model/rnn_weights.h5'
        self.intent_LSTM_model_path = self.project_path + '/model/lstm_model.json'
        self.intent_LSTM_weight_path = self.project_path + '/model/lstm_weights.h5'
        self.intent_BERT_model_path = self.project_path + '/model/bert_model.json'
        self.intent_BERT_weight_path = self.project_path + '/model/bert_weights.h5'

        self.bert_model_name = "wwm_uncased_L-24_H-1024_A-16"
        self.bert_ckpt_dir = self.project_path + '/model/' + self.bert_model_name
        self.bert_ckpt_file = self.project_path + '/model/' + self.bert_model_name + '/bert_model.ckpt'
        self.bert_config_file = self.project_path + '/model/' + self.bert_model_name + '/bert_config.json'
        self.vocab_file = self.project_path + '/model/' + self.bert_model_name + '/vocab.txt'
        self.log_dir = self.project_path + '/model/log/intent_detection/'

        # sound classification
        self.mode = mode
        self.nfilt = 26
        self.nfeat = 13
        self.nfft = 512
        self.rate = 16000
        self.step = int(self.rate / 10)
        self.sound_model_path = os.path.join(self.project_path + '/sound_classification/models',
                                             self.mode + '.model')
        self.p_path = os.path.join(self.project_path + '/sound_classification/pickles',
                                   self.mode + '.p')
        self.voice_list_path = os.path.join(self.project_path + '/sound_classification/voicelist.csv')
        self.speaker_voice_path = os.path.join(self.project_path + '/sound_classification/speakervoice')

        # Production line
        self.lego_production_line_path = os.path.join(
            self.project_path + '/Odoo/LEGO_counting_machine_two/legoCountingTwo.py')

        # googlenewsvector
        self.google_news_path = 'C:/Users/Admin/Downloads/googlenews-vectors-negative300.bin.gz'

        # cnn/rnn parameters
        self.maxlen = 400
        self.steps_per_epoch = 10
        self.validation_steps = 4
        self.embedding_dims = 300
        self.filters = 250
        self.kernel_size = 3
        self.hidden_dims = 250
        self.epochs = 10
        self.num_class = 10
