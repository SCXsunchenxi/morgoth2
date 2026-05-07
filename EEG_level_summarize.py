import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import pandas as pd
import argparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics as sklearn_metrics
from utils import BinaryFocalLoss,FocalLoss
from torch.nn.utils.rnn import pad_sequence
import sys
import numpy as np
import random

class Logger:
    """ Makes print() output to both terminal and log file, and provides `print_only()` for terminal-only output """
    def __init__(self, file_path):
        self.terminal = sys.stdout  # save default terminal output
        self.log = open(file_path, "a")  # open log file (append mode)

    def write(self, message):
        """ Makes `print()` write to both terminal and log file """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """ Compatible with Python's requirement for flush() """
        self.terminal.flush()
        self.log.flush()

    def log_only(self, message):
        """ **Only** writes to log file, does not print to terminal """
        self.log.write(message + "\n")
        self.log.flush()  # flush immediately

    def print_only(self, message):
        """ **Only** prints to terminal, does not write to log file """
        self.terminal.write(message + "\n")
        self.terminal.flush()

def setup_logger(output_dir, filename="training_log.txt"):
    """ Initialize log file and make `print()` automatically write to it """
    log_path = os.path.join(output_dir, filename)
    sys.stdout = Logger(log_path)  # make `print()` auto-save to log
    return sys.stdout  # return `Logger` instance


class ClipAndExtend:
    def __init__(self):
        self.transform_idx = np.random.choice([0, 1], size=3).tolist()

    def __call__(self,X):

        if self.transform_idx[0]==1 and len(X)>10:
            X=self.clip(X)

        if self.transform_idx[1]==1 and len(X)>60 and len(X)<50000:
            X=self.pad(X)

        if self.transform_idx[2]==1 and len(X)<20000:
            X=self.repeat(X)

        return X

    def clip(self, X):
        center = len(X) // 2
        random_int1 = random.randint(0, center-5)
        random_int2 = random.randint(0, center-5)
        X = X[len(X) // 2 -5 - random_int1: len(X) // 2 + 5+ random_int2, :]
        return X

    def pad(self,X):
        # padding with others
        max_length = (50000 - len(X)) // 2
        random_int1 = random.randint(0, max_length)
        random_int2 = random.randint(0, max_length)
        m = X.shape[1]
        X1 = self.get_random_matrix(random_int1, m)
        X2 = self.get_random_matrix(random_int2, m)
        X = np.vstack((X1, X, X2))
        return X

    def repeat(self, X):
        n = np.random.randint(2, 6)
        X = np.tile(X, (n, 1))
        return X

    def get_random_matrix(self, n, m):
        # initialize an n x m matrix with dtype float32
        matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            if m == 1:
                # if m=1, generate numbers in [0, 0.5)
                row = np.random.uniform(0, 0.5, size=(1,)).astype(np.float32)

            else:
                # generate the first number in range [0.5, 1), dtype float32
                a_i1 = np.random.uniform(0.5, 1)

                # generate remaining m-1 numbers in range [0, 1 - a_i1), dtype float32
                rest = np.random.uniform(0, 1 - a_i1, m - 1)

                # normalize so that each row sums to 1
                row = np.concatenate(([a_i1], rest))
                row = row / np.sum(row)

             # store row data into matrix
            matrix[i] = row.astype(np.float32)

        return matrix



class CSVDataset(Dataset):
    def __init__(self, csv_dirs, class_idx, file_list_path='',  transform=None, is_predict_dataset=False):
        if is_predict_dataset:
            self.csv_files=get_predicting_files(csv_dirs)
        else:
            self.file_list=pd.read_csv(file_list_path)
            self.csv_files=get_training_files(csv_dirs)
            self.class_idx=class_idx
        self.predicting_dataset=is_predict_dataset
        self.transform = transform

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        file_name = os.path.basename(csv_file).split('.')[0]

        df = pd.read_csv(csv_file)
        if 'pred_class' in df.columns:
            # continuous_class=df['pred_class'].tolist()
            df = df.drop(columns=['pred_class'])

        if df.isna().any().any():
            # num_nan_rows = df.isna().any(axis=1).sum()
            # print(f"{csv_file} contains {num_nan_rows}/{len(df)} rows with NaN, filling with 1")
            ###########this is due to a previous continuous event-level issue, may be changed later###########
            df = df.fillna(1)
            ###########this is due to a previous continuous event-level issue, may be changed later###########

        X =df.values.astype('float32')

        if len(X) == 0:  # if X is empty, return None
            print(f'{file_name} is none')
            return None
        if X.shape[1] == 1 and np.all(X < 0.1):
            y = 0

            length = len(X)
            if length < 30:
                mean_values = np.mean(X, axis=0)
                padding = np.tile(mean_values, (30 - length, 1))
                X = np.vstack([X, padding])

            if self.predicting_dataset:
                return X, None, file_name, length

        else:
            if self.transform is not None:
                X = self.transform(X)

            length = len(X)
            # minimum length 30, otherwise model pooling will error
            if length < 30:
                mean_values = np.mean(X, axis=0)
                padding = np.tile(mean_values, (30 - length, 1))
                X = np.vstack([X, padding])

            # maximum supported: 10000(pe)*30(pooling)=300000 seconds
            elif length>300000:
                X = X[10:300000]

            if self.predicting_dataset:
                return X, None, file_name, length

            matched = self.file_list[self.file_list['file_name'] == file_name]
            if matched.empty:
                #print(f'{file_name} can not localize label, not match file')
                return None

            else:
                if self.class_idx in matched['label'].to_list():
                    y=1
                elif 0 in matched['label'].to_list():
                    y=0
                else:
                    #print(f'{file_name} can not localize label, no label')
                    return None

        return X, y, file_name, length


def get_training_files(training_dirs):
    files=[]
    for training_dir in training_dirs:
        for file in os.listdir(training_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(training_dir, file)
                if os.path.isfile(file_path):
                    files.append(file_path)
    return files


def get_predicting_files(test_dirs):
    files = []
    for test_dir in test_dirs:
        for file in os.listdir(test_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(test_dir, file)
                if os.path.isfile(file_path):
                    files.append(file_path)

    return files


def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # filter out None
    if not batch:  # if batch is empty, return empty values
        return None, None, None

    X, y, file_names, lengths = zip(*batch)

    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=0)

    y = torch.tensor(y) if isinstance(y[0], int) else None  # Handle empty y (predicting mode)
    lengths = torch.tensor(lengths)

    return X_padded, y, file_names, lengths


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=15000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(float(max_len))) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # simplified slice operation


class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim, cnn_channels=16, transformer_layers=2, transformer_heads=4,
                 transformer_hidden_dim=64, output_dim=1, dropout=0.1, pe_max_length=10000):
        super(CNNTransformerClassifier, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=10), # combine 10s
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=3) # combine 30s
        )

        self.seq_len_factor = 30  # CNN reduces sequence length by a factor of 30

        # Transformer layers
        self.transformer_input_dim = cnn_channels * 2
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_input_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.pe_max_length=pe_max_length
        # Normalization layers
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_input_dim)
        self.post_transformer_norm = nn.LayerNorm(self.transformer_input_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.transformer_input_dim, max_len=self.pe_max_length)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.transformer_input_dim, transformer_hidden_dim),
            nn.BatchNorm1d(transformer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(transformer_hidden_dim, output_dim)
        )

    def forward(self, x, lengths=None):
        batch_size = x.size(0)

        # CNN processing
        x = x.transpose(1, 2)  # (batch, seq, feature) -> (batch, feature, seq)
        if lengths is not None:
            lengths = (lengths // self.seq_len_factor).clamp(min=1).long()

        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, feature, seq) -> (batch, seq, feature)

        # Pre-transformer normalization
        x = self.pre_transformer_norm(x)

        # Position encoding
        x = self.positional_encoding(x)

        # Transformer processing
        if lengths is not None:
            padding_mask = self.create_padding_mask(lengths, x.size(1))
            padding_mask = padding_mask.to(x.device)

            # print("x shape:", x.shape)
            # print("padding_mask shape:", padding_mask.shape)

            # if isinstance(x, torch.nested.nested_tensor):
            #     x = x.to_padded_tensor(padding_value=0.0)
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        # Post-transformer normalization
        x = self.post_transformer_norm(x)

        # Sequence pooling
        if lengths is not None:
            indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))
            x = x.gather(1, indices).squeeze(1)
        else:
            x = x[:, -1, :]

        # Dimension verification
        if x.size(0) != batch_size:
            raise ValueError(f"Expected batch size {batch_size}, got {x.size(0)}")
        if x.size(1) != self.transformer_input_dim:
            raise ValueError(f"Expected feature dim {self.transformer_input_dim}, got {x.size(1)}")

        return self.fc(x)

    def create_padding_mask(self, lengths, max_len):
        device = lengths.device
        mask = (torch.arange(max_len, device=device, dtype=torch.long)[None, :] >= lengths[:, None]).to(torch.bool)
        return mask  # return bool type directly


def train(args, model, device, optimizer, criterion, num_epochs,train_loader_raw, train_loader_transform=None,
          test_loader=None, save_freq=5, resume_training=False):
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger(args.output_dir)

    best_accuracy = 0.0
    best_model_path=os.path.join(args.output_dir, 'checkpoint-best.pth')
    # If resuming training, load the last checkpoint
    if resume_training:
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_accuracy = checkpoint.get('accuracy', 0.0)
            logger.print_only(f"Resuming training from previous epoch {start_epoch}")
        else:
            logger.print_only("No checkpoint found. Starting training from scratch.")


    model.train()
    for epoch in range(num_epochs):
        both_epoch_loss=0
        both_accuracy = 0
        both_balanced_accuracy=0
        for train_loader in [train_loader_transform,train_loader_raw]:
            if train_loader is None:
                continue
            epoch_loss = 0
            y_true = []
            y_pred = []
            for batch_idx, (inputs, labels, _, lengths) in enumerate(train_loader):
                if inputs is None:
                    continue

                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, lengths=lengths)

                # Adjust labels based on task type
                if args.n_classes == 1:  # Binary classification
                    labels = labels.view(-1, 1).float()
                else:  # Multi-class classification
                    labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate predictions
                if args.n_classes == 1:  # Binary classification
                    probabilities = torch.sigmoid(outputs)
                    predicted = torch.round(probabilities).squeeze().detach()

                else:  # Multi-class classification
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = torch.argmax(probabilities, dim=1)

                y_true_batch=labels.squeeze().cpu().numpy()
                y_pred_batch= predicted.cpu().numpy()
                loss_batch=loss.item()

                batch_accuracy = sklearn_metrics.accuracy_score(y_true_batch, y_pred_batch)
                batch_balanced_accuracy = balanced_accuracy_score(y_true_batch, y_pred_batch)

                logger.print_only(
                    f'Epoch [{epoch + 1}/{num_epochs}] Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss_batch:.4f}; Accuracy: {batch_accuracy:.4f}; Balanced Accuracy: {batch_balanced_accuracy:.4f}')

                y_true.extend(y_true_batch)
                y_pred.extend(y_pred_batch)
                epoch_loss+=loss_batch

            train_accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            epoch_loss=epoch_loss/len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}] Train [1] Loss: {epoch_loss:.4f}; [2] Accuracy: {train_accuracy:.4f}; [3] Balanced Accuracy: {balanced_accuracy:.4f}')

            both_epoch_loss += epoch_loss
            both_accuracy+=train_accuracy
            both_balanced_accuracy += balanced_accuracy

        both_epoch_loss=both_epoch_loss/2
        both_accuracy=both_accuracy/2
        both_balanced_accuracy=both_balanced_accuracy/2

        print(
            f'[*] Epoch [{epoch + 1}/{num_epochs}] Avg loss:{both_epoch_loss:.4f}; Avg Accuracy: {both_accuracy:.4f}; AvgBalanced Accuracy: {both_balanced_accuracy:.4f}')

        # Save model at specified frequency
        if (epoch + 1) % save_freq == 0:
            model_save_path = os.path.join(args.output_dir,f'checkpoint-{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': train_accuracy
            }, model_save_path)
            logger.print_only(f"Model saved at epoch {epoch + 1}")

            # Evaluate on test set if available
            if test_loader:
                logger.print_only('test')
                test_accuracy, balanced_accuracy = test(args.n_classes, model, device, test_loader)
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}')

                # Save best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_accuracy
                    }, best_model_path)
                    logger.print_only(f"[*] New best model saved with accuracy {best_accuracy:.4f}")
            else:
                if both_accuracy>best_accuracy:
                    best_accuracy = both_accuracy
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_accuracy
                    }, best_model_path)
                    logger.print_only(f"[*] New best model saved with accuracy {best_accuracy:.4f}")

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


def test(n_classes, model, device, test_loader, result_dir, type, n_files, save_result=True):
    os.makedirs(result_dir, exist_ok=True)
    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, (inputs, y, csv_file,lengths) in enumerate(test_loader):
            if inputs is None:  # if batch is empty, skip
                print('None inputs, skip')
                continue

            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs,lengths=lengths)

            if n_classes == 1:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities).squeeze()
                labels = y.view(-1, 1).float()

            else:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                labels = y.long()

            accuracy = sklearn_metrics.accuracy_score(labels, predicted.cpu().numpy())
            balanced_accuracy = balanced_accuracy_score(labels, predicted.cpu().numpy())

            print(
                f'Batch [{batch_idx + 1}/{len(test_loader)}]: Accuracy: {accuracy:.4f}; Balanced Accuracy: {balanced_accuracy:.4f}')


            if save_result:
                for i in range(len(csv_file)):
                    results.append({
                        'file_name': csv_file[i],
                        'probability' :probabilities[i].item(),
                        'pred_class': predicted[i].item(),
                        'true': y[i].item()
                    })

    if save_result:
        results_df = pd.DataFrame(results)
        file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
        results_df.to_csv(file_path, index=False)
        os.chmod(file_path, 0o777)

def predict(n_classes, model, device, test_loader, result_dir, type, n_files):
    model.eval()
    results = []
    with torch.no_grad():
        progress_bar = tqdm(total=n_files, desc=f"{type} EEG level results")
        for inputs, _, csv_file, lengths in test_loader:
            if inputs is None:  # if batch is empty, skip
                print('None inputs, skip')
                continue

            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs,lengths=lengths)

            if n_classes == 1:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities).squeeze()
            else:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)

            for i in range(len(csv_file)):
                results.append({
                    'file_name': csv_file[i],
                    'probability':probabilities[i].item(),
                    'pred_class':predicted[i].item() if predicted.dim() > 0 else predicted.item()
                })
            progress_bar.update(len(csv_file))

    results_df = pd.DataFrame(results)
    file_path=os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
    results_df.to_csv(file_path, index=False)
    os.chmod(file_path, 0o777)


def load_model(args):
    model = CNNTransformerClassifier(
        input_dim=args.input_dim,
        output_dim=args.n_classes,
        pe_max_length=args.pe_max_length,
    ).to(args.device)

    return model

def load_model_parameters(model,model_parameters_path):
    model.load_state_dict(torch.load(model_parameters_path,weights_only=True)['model_state_dict'])
    return model




def summarize_sleep_eeg_level_results(train_csv_dirs,result_dir,event_step=1):
    def check_all_consecutive_labels(series):
        consecutive_counts = {0:0, 1: 0, 2: 0, 3: 0, 4: 0}
        # initialize results
        has_consecutive = {0:False, 1: False, 2: False, 3: False, 4: False}
        # define thresholds
        thresholds = {0: int(30 - 10 / event_step), 1: int(30 - 10 / event_step), 2: int(30 - 10 / event_step), 3: int(30 - 10 / event_step),
                      4: int(30 - 10 / event_step)}

        # Iterate through the sequence only once
        for value in series:
            if value == 0:
                consecutive_counts[0] += 1
                if not has_consecutive[0] and consecutive_counts[0] >= thresholds[0]:
                    has_consecutive[0] = True
            else:
                consecutive_counts[0] = 0

            # handle consecutiveness for value 1
            if value == 1:
                consecutive_counts[1] += 1
                if not has_consecutive[1] and consecutive_counts[1] >= thresholds[1]:
                    has_consecutive[1] = True
            else:
                consecutive_counts[1] = 0

            # handle consecutiveness for value 2 - increment when value is 2, keep unchanged when value is 1 or 3
            if value == 2:
                consecutive_counts[2] += 1
                if not has_consecutive[2] and consecutive_counts[2] >= thresholds[2]:
                    has_consecutive[2] = True
            elif value == 1 or value == 3:
                # for 1 or 3, keep the value-2 counter unchanged
                pass
            else:
                consecutive_counts[2] = 0

            # handle consecutiveness for value 3
            if value == 3:
                consecutive_counts[3] += 1
                if not has_consecutive[3] and consecutive_counts[3] >= thresholds[3]:
                    has_consecutive[3] = True
            else:
                consecutive_counts[3] = 0

            # handle consecutiveness for value 4
            if value == 4:
                consecutive_counts[4] += 1
                if not has_consecutive[4] and consecutive_counts[4] >= thresholds[4]:
                    has_consecutive[4] = True
            else:
                consecutive_counts[4] = 0

        # return five boolean values
        return has_consecutive[0], has_consecutive[1], has_consecutive[2], has_consecutive[3], has_consecutive[4]

    result_list_df = pd.DataFrame(columns=['file_name'] + [f'pred_{i}_class' for i in range(5)])
    for dir in tqdm(train_csv_dirs):
        for file in tqdm(os.listdir(dir),desc=f'{dir}'):
            file_name=file.split('.')[0]
            event_level_results_df=pd.read_csv(os.path.join(dir,file))

            #less than 1 min, no need to check consecutiveness
            if len(event_level_results_df) <= 10*60/event_step:
                new_row = pd.DataFrame({
                    'file_name': [file_name],
                    'pred_0_class': [event_level_results_df['class_0_prob'].max()],
                    'pred_1_class': [event_level_results_df['class_1_prob'].max()],
                    'pred_2_class': [event_level_results_df['class_2_prob'].max()],
                    'pred_3_class': [event_level_results_df['class_3_prob'].max()],
                    'pred_4_class': [event_level_results_df['class_4_prob'].max()]
                })

            else:
                continuous_labels = event_level_results_df['pred_class']
                if (event_level_results_df['pred_class'] == 0).mean()>= 0.95:
                    new_row = pd.DataFrame({
                        'file_name': [file_name],
                        'pred_0_class': [1],
                        'pred_1_class': [0],
                        'pred_2_class': [0],
                        'pred_3_class': [0],
                        'pred_4_class': [0]
                    })
                else:


                    has_0, has_1, has_2, has_3, has_4= check_all_consecutive_labels(continuous_labels)

                    if has_0:
                        pred_0_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] == 0, 'class_0_prob'].mean()
                        pred_0_class = pred_0_class * 0.5 + 0.5
                    else:
                        pred_0_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] != 0, 'class_0_prob'].mean()

                    if has_4:
                        pred_4_class = event_level_results_df.loc[event_level_results_df['pred_class'] == 4, 'class_4_prob'].mean()
                        pred_4_class = pred_4_class * 0.5 + 0.5
                    else:
                        pred_4_class=event_level_results_df.loc[event_level_results_df['pred_class'] != 4, 'class_4_prob'].mean()

                    if has_3:
                        pred_3_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] == 3, 'class_3_prob'].mean()
                        if has_4:
                            pred_3_class = pred_3_class * 0.5 + 0.5
                        else:
                            pred_3_class = pred_3_class * 0.5 + 0.4
                    else:
                        pred_3_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] != 3, 'class_3_prob'].mean()

                    if has_2:
                        pred_2_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] == 2, 'class_2_prob'].mean()
                        pred_2_class = pred_2_class * 0.5 + 0.5
                    else:
                        pred_2_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] != 2, 'class_2_prob'].mean()
                        if has_3:
                            pred_2_class = pred_2_class * 0.5 + 0.5

                    if has_1:
                        pred_1_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] == 1, 'class_1_prob'].mean()
                        if has_2 or has_3 or has_4:
                            pred_1_class = pred_1_class * 0.5 + 0.5
                    else:
                        pred_1_class = event_level_results_df.loc[
                            event_level_results_df['pred_class'] != 1, 'class_1_prob'].mean()
                        if has_2 or has_3 or has_4:
                            pred_1_class = pred_1_class * 0.5 + 0.5
                    new_row = pd.DataFrame({
                        'file_name': [file_name],
                        'pred_0_class': [pred_0_class],
                        'pred_1_class': [pred_1_class],
                        'pred_2_class': [pred_2_class],
                        'pred_3_class': [pred_3_class],
                        'pred_4_class': [pred_4_class]
                    })

            result_list_df = pd.concat([result_list_df, new_row], ignore_index=True)

    result_list_df.to_csv(os.path.join(result_dir, 'pred_EEG_level_SLEEP.csv'),index=False)


def get_args():
    parser = argparse.ArgumentParser(description='CNN + Transformer Classifier')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'predict'],
                        help='Mode: train, test, or predict')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--pe_max_length', type=int, default=10000,
                        help='the maximum length of positional encoding')

    parser.add_argument('--train_csv_dirs', type=str, help='Directories containing training CSV files')
    parser.add_argument('--file_list_path', default='', type=str, help='Data file contains file_name and labels')

    parser.add_argument('--test_csv_dir', type=str, help='Directory containing test CSV files')
    parser.add_argument('--result_dir', type=str, help='Directory containing test CSV files')
    parser.add_argument('--dataset', type=str, required=True, help='SEIZURE/LPD/GPD/LRDA/GRDA | SPIKES | FOC/GEN_SPIKES | FOC/GEN_SLOWING | BS | NORMAL')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')

    parser.add_argument('--focal_alpha', type=str, default="", help='Focal Loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2, help='Gamma parameter for Focal Loss')
    parser.add_argument('--task_model', type=str, default='cnn_transformer_classifier_model.pth',
                        help='Path to the model file')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency of saving model checkpoints (every n epochs)')
    parser.add_argument('--resume_training', action='store_true',
                        help='Continue training from the last checkpoint')
    return parser.parse_args()


def get_dataset_info(dataset_name):
    n_classes=1
    if dataset_name=='SEIZURE':
        class_idx=1
        input_dim=6
    elif dataset_name=='LPD':
        class_idx = 2
        input_dim = 6
    elif dataset_name=='GPD':
        class_idx = 3
        input_dim = 6
    elif dataset_name=='LRDA':
        class_idx = 4
        input_dim = 6
    elif dataset_name=='GRDA':
        class_idx = 5
        input_dim = 6
    elif dataset_name == 'FOC_SLOWING':
        class_idx = 1
        input_dim = 3
    elif dataset_name == 'GEN_SLOWING':
        class_idx = 2
        input_dim = 3
    elif dataset_name == 'FOC_SPIKES':
        class_idx = 1
        input_dim = 3
    elif dataset_name == 'GEN_SPIKES':
        class_idx = 2
        input_dim = 3
    elif dataset_name == 'BS':
        class_idx = 1
        input_dim = 1
    elif dataset_name == 'NORMAL':
        class_idx = 1
        input_dim = 1
    elif dataset_name == 'SPIKES':
        class_idx = 1
        input_dim = 1
    elif dataset_name == 'SLEEP':
        class_idx=None
        input_dim = 5

    else:
        print('wrong dataset name')
        exit(0)

    return class_idx, n_classes,input_dim


def main():
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args.class_idx, args.n_classes, args.input_dim = get_dataset_info(args.dataset)



    if args.mode == 'train':

        args.train_csv_dirs=list(map(str, args.train_csv_dirs.split()))

        train_dataset_raw = CSVDataset(csv_dirs=args.train_csv_dirs, file_list_path=args.file_list_path,
                                   class_idx=args.class_idx, transform=None, is_predict_dataset=False)

        train_dataset_transform = CSVDataset(csv_dirs= args.train_csv_dirs, file_list_path=args.file_list_path, class_idx=args.class_idx, transform=ClipAndExtend(), is_predict_dataset=False)


        model = load_model(args)

        if args.n_classes==1:
            if args.focal_alpha!="":
                alpha = list(map(float, args.focal_alpha.split()))[0]
                gamma = args.focal_gamma
                criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        else:
            if args.focal_alpha!="":
                alpha = list(map(float, args.focal_alpha.split()))
                alpha = torch.tensor(alpha).to(args.device, non_blocking=True)
                gamma = args.focal_gamma
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
            else:
                criterion = torch.nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_loader_raw = DataLoader(train_dataset_raw, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        train_loader_transform = DataLoader(train_dataset_transform, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn)

        if args.test_csv_dir:
            test_dataset = CSVDataset(csv_dirs= args.test_csv_dir, file_list_path=args.file_list_path, class_idx=args.class_idx, transform=None, is_predict_dataset=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            test_loader = None

        train(args,
              model=model,
              device=args.device,
              train_loader_raw=train_loader_raw,
              train_loader_transform=train_loader_transform,
              optimizer=optimizer,
              criterion=criterion,
              num_epochs=args.num_epochs,
              test_loader=test_loader,
              save_freq=args.save_freq,
              resume_training=args.resume_training)

    elif args.mode == 'test':
        args.test_csv_dir = list(map(str, args.test_csv_dir.split()))

        test_dataset = CSVDataset(csv_dirs=args.test_csv_dir, file_list_path=args.file_list_path,
                                   class_idx=args.class_idx, transform=None, is_predict_dataset=False)


        model = load_model(args)
        model = load_model_parameters(model, model_parameters_path=args.task_model)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        test(n_classes=args.n_classes,
                model=model,
                device=args.device,
                test_loader=test_loader,
                result_dir=args.result_dir,
                type=args.dataset,
                n_files=len(test_dataset))

    elif args.mode == 'predict':
        os.makedirs(args.result_dir, exist_ok=True)
        args.test_csv_dir = list(map(str, args.test_csv_dir.split()))

        if args.dataset == 'SLEEP':
            summarize_sleep_eeg_level_results(train_csv_dirs=args.test_csv_dir,result_dir=args.result_dir)


        else:
            predict_dataset = CSVDataset(args.test_csv_dir,class_idx=args.class_idx,is_predict_dataset=True)

            model = load_model(args)

            model=load_model_parameters(model,model_parameters_path=args.task_model)

            test_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

            predict(n_classes=args.n_classes,
                    model=model,
                    device=args.device,
                    test_loader=test_loader,
                    result_dir=args.result_dir,
                    type=args.dataset,
                    n_files=len(predict_dataset))

    else:
        print('mode input error')
        return


if __name__ == "__main__":
    main()