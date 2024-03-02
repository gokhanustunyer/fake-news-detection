from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Conv2DTranspose, LSTM, Embedding, SpatialDropout1D
from keras.models import load_model, model_from_json
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.image import DirectoryIterator
from keras.preprocessing import image
import os, re
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

class DLModel:
    # Egitim sonrasi modelin kaydedilecegi klasor yolu
    TrainedModelPath = './trained_model'
    TrainedModelName = 'cnn_model'
    # Verilerin oldugu klasor yolu
    DataDirectory = './data/patato'

    def __init__(self) -> None:
        self.model = Sequential()
        self.layer_count = 0
        self.class_count = 2

    def initialize_model(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        absdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fake-news')
        fake_news = pd.read_csv(os.path.join(absdir, 'fake.csv'))
        true_news = pd.read_csv(os.path.join(absdir, 'true.csv'))
        
        fake_news['true'] = 0
        true_news['true'] = 1
       
        df = pd.concat([fake_news, true_news])
        df['text'] = df['text'].apply(DLModel.clean_text)
        
        x = df[['title', 'text', 'subject', 'date']]
        y = df[['true']]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train['true'].values
        self.y_test = y_test['true'].values
        
        
    def tokenize(self):
        tokenizer = Tokenizer(num_words=5000, split=' ')
        tokenizer.fit_on_texts(self.x_train['text'].values)
        X_train_lstm = tokenizer.texts_to_sequences(self.x_train['text'].values)
        X_train_lstm = pad_sequences(X_train_lstm, maxlen=500)
        
        X_test_lstm = tokenizer.texts_to_sequences(self.x_test['text'].values)
        X_test_lstm = pad_sequences(X_test_lstm, maxlen=500)
        self.x_test = X_test_lstm
        self.x_train = X_train_lstm

    def fit_lstm_model(self):
        # LSTM Model
        self.model.add(Embedding(5000, 128, input_length=500))
        self.model.add(SpatialDropout1D(0.4))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(self.x_test, self.y_test, epochs=2)
        score = self.model.evaluate(self.x_test, self.y_test)
        print(score)
    
    
    def compile_and_fit(self, optimizer: str = 'Adam', loss: str = 'binary_crossentropy', metrics: list = ['accuracy'], epochs: int = 25, save_model: bool = False) -> None:
        # Olusturulan modelin derlenmesi
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # Compile edilen modelin train datalariyla fit edilmesi
        self.model.fit(self.x_train_scaled, self.y_train.values.ravel(), epochs=epochs)
        # Eger isteniyorsa modeli kaydet
        if save_model:
            DLModel.save_model(self.model)

    def fit(self, generator: DirectoryIterator, epochs: int):
        self.model.fit(generator, epochs=epochs)

    def add_conv_layer(self, activation: str, dropout: float = None, input_shape: tuple = None):
        # Katman icin filtre sayisinin hesaplanmasi
        filter_count = 2 ** (5+self.layer_count)

        # Katmanin eklenmesi
        if input_shape != None: 
            self.model.add(Conv2D(filter_count, (3, 3), activation=activation, input_shape = input_shape))
        else:
            self.model.add(Conv2D(filter_count, (3, 3), activation=activation))

        # MaxPooling'in eklenmesi
        self.model.add(MaxPooling2D((2, 2)))

        # Varsa dropout degerinin eklenmesi
        if dropout != None:
            self.model.add(Dropout(dropout))
        
        # Layer sayisinin guncellenmesi
        self.layer_count += 1

    def add_flatten_layer(self, activation1: str = 'relu', activation2: str = 'softmax'):
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation1))
        self.model.add(Dense(self.class_count, activation=activation2))

    def evaluate(self, print_acc: bool = False) -> tuple:
        score = self.model.evaluate(self.x_test)
        if print_acc:
            print("Test Loss:", score[0])
            print("Test Accuracy:", score[1])
        return score[0], score[1]

    def train_w_tranformer(self):
        self.texts = []
        self.x_test['text'].apply(self.tokenize_by_bert_tokenizer)
        self.texts = torch.cat(self.texts, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        self.x_test['text'] = TensorDataset(self.texts, attention_masks, labels)

        self.texts = []
        self.x_train['text'].apply(self.tokenize_by_bert_tokenizer)
        self.texts = torch.cat(self.texts, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        self.x_train['text'] = TensorDataset(self.texts, attention_masks, labels)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        batch_size = 32

        train_dataloader = DataLoader(self.x_train, sampler=RandomSampler(self.x_train), batch_size=batch_size)
        val_dataloader = DataLoader(self.x_test, sampler=SequentialSampler(self.x_test), batch_size=batch_size)

        # Optimizer ve scheduler'ı tanımlayın
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))



        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        epochs = 4

        for epoch in range(epochs):
            model.train()
            
            for batch in train_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                model.zero_grad()
                
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
            
            # Val dataloader ile modeli değerlendirme kodu buraya eklenir
            
            model.save_pretrained('fake_news_transformer')
            

    def evaluate_transformer(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained('sahte_haber_model')
        model.to(device)

        test_dataloader = DataLoader(self.x_test, sampler=SequentialSampler(self.x_test), batch_size=32)

        model.eval()

        y_true = []
        y_pred = []

        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            y_true.extend(label_ids)
            y_pred.extend(np.argmax(logits, axis=1))

        print(classification_report(y_true, y_pred))
        
    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize_by_bert_tokenizer(self, input_texts):
        for text in input_texts:
            encoded_dict = self.tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=64,
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt',
                        )
            self.texts.append(encoded_dict['input_ids'])

    @staticmethod
    def save_model(model: Sequential):
        model.save(os.path.join(DLModel.TrainedModelPath, DLModel.TrainedModelName + 'h5'))
    
    @staticmethod
    def load_created_model(model_file) -> Sequential:
        return load_model(os.path.join(DLModel.TrainedModelPath, model_file))

def main():
    dlModel = DLModel()
    dlModel.initialize_model()
    dlModel.tokenize()
    dlModel.fit_lstm_model()
    dlModel.add_flatten_layer()
    dlModel.compile_and_fit(save_model=True)
    dlModel.evaluate(True)
    

if __name__ == '__main__':
    main() 


