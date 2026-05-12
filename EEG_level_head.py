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

import warnings
warnings.filterwarnings("ignore")

class Logger:
    """ Make print() output to both terminal and log, and provide `print_only()` for terminal-only printing """
    def __init__(self, file_path):
        self.terminal = sys.stdout  # Save default terminal output
        self.log = open(file_path, "a")  # Open log file (append mode)

    def write(self, message):
        """ Make `print()` write to both terminal and log """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        """ Compatible with Python's flush() requirement """
        self.terminal.flush()
        self.log.flush()

    def log_only(self, message):
        """ **Only** write to log file, do not print to terminal """
        self.log.write(message + "\n")
        self.log.flush()  # Flush immediately

    def print_only(self, message):
        """ **Only** print to terminal, do not write to log file """
        self.terminal.write(message + "\n")
        self.terminal.flush()

def setup_logger(output_dir, filename="training_log.txt"):
    """ Initialize log file and make `print()` automatically write to log """
    log_path = os.path.join(output_dir, filename)
    sys.stdout = Logger(log_path)  # Make `print()` automatically save to log
    return sys.stdout  # Return `Logger` instance


class ClipAndExtend:
    def __init__(self):
        self.transform_idx = np.random.choice([0, 1], size=3).tolist()

    def __call__(self,X):

        if self.transform_idx[0]==1 and len(X)>10:
            X=self.clip(X)

        if self.transform_idx[1]==1 and len(X)>60 and len(X)<10000:
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
        max_length = (10000 - len(X)) // 2
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
        # Initialize an n x m matrix with dtype float32
        matrix = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            if m == 1:
                # If m=1, generate numbers in [0, 0.5) # other class
                row = np.random.uniform(0, 0.5, size=(1,)).astype(np.float32)

            else:
                # Generate the first number in range [0.5, 1), dtype float32
                a_i1 = np.random.uniform(0.5, 1)

                # Generate the remaining m-1 numbers in range [0, 1 - a_i1), dtype float32
                rest = np.random.uniform(0, 1 - a_i1, m - 1)

                # Normalize so that each row sums to 1
                row = np.concatenate(([a_i1], rest))
                row = row / np.sum(row)

             # Store row data into matrix
            matrix[i] = row.astype(np.float32)

        return matrix

class CSVDataset(Dataset):
    def __init__(self, csv_dirs, class_idx, file_list_path='',  transform=None, is_predict_dataset=False, id_distinguished=False):
        if is_predict_dataset:
            self.csv_files=get_predicting_files(csv_dirs)
        else:
            self.file_list=pd.read_csv(file_list_path)
            self.csv_files=get_training_files(csv_dirs)

        self.predicting_dataset=is_predict_dataset
        self.transform = transform
        self.class_idx = class_idx
        self.id_distinguished = id_distinguished

    def __len__(self):
        return len(self.csv_files)

    @staticmethod
    def extract_match_id(file_name):
        first_part = str(file_name).split("_")[0]

        # Case 1: first segment starts with "sub"
        if first_part.startswith("sub"):
            return first_part[-9:]

        # Case 2: first segment is exactly 9 digits
        elif len(first_part) == 9:
            return first_part

        # Case 3: does not match rules, do not split by _
        else:
            return str(file_name)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        file_name = os.path.basename(csv_file).split('.')[0]

        df = pd.read_csv(csv_file)
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])

        if 'pred_class' in df.columns:
            # continuous_class=df['pred_class'].tolist()
            df = df.drop(columns=['pred_class'])

        if df.isna().any().any():
            # num_nan_rows = df.isna().any(axis=1).sum()
            # print(f"{csv_file} number of rows with NaN: {num_nan_rows}/{len(df)}, filled with 1")
            ###########This is due to a previous issue with continuous event-level, may be changed later###########
            df = df.fillna(1)
            ###########This is due to a previous issue with continuous event-level, may be changed later###########

        X = df.values.astype('float32')

        if len(X) == 0:  # If X is empty, return None
            print(f'{file_name} is none')
            return None

        # Check for low signal condition
        is_low_signal = False

        # Single column data, all values less than 0.4
        if X.shape[1] == 1 and np.all(X < 0.4):
            is_low_signal = True
        # Multi-column data, check specific column or all columns
        elif X.shape[1] > 1:
            # If this is a prediction dataset and has self.class_idx
            if self.predicting_dataset and hasattr(self, 'class_idx') and self.class_idx is not None:
                # If class_idx is within valid range and is an integer
                if isinstance(self.class_idx, int) and 0 <= self.class_idx < X.shape[1]:
                    # Check if the max value of that column is below the threshold (1/num_columns)
                    threshold = 1.0 / X.shape[1]
                    column_max = np.max(X[:, self.class_idx]) if self.class_idx > 0 else 0
                    if column_max < threshold:
                        is_low_signal = True

        # Record data length
        length = len(X)

        # If length is less than 30, pad to at least 30
        if length < 30:
            mean_values = np.mean(X, axis=0)
            padding = np.tile(mean_values, (30 - length, 1))
            X = np.vstack([X, padding])
        # If length exceeds 300000, truncate
        elif length > 300000:
            X = X[10:300000]

        # Apply transform to normal data
        if not is_low_signal and self.transform is not None:
            X = self.transform(X)

        # If this is a prediction dataset
        if self.predicting_dataset:
            # If low signal, set prediction directly to 0
            if is_low_signal:
                return X, 0, file_name, length, True  # Add flag indicating this is low signal
            else:
                return X, None, file_name, length, False  # Normal data, fifth parameter is False

        # If this is a training dataset
        if is_low_signal:
            y = 0  # Low signal, set label to 0
        else:
            # Find label by exact filename match (positive samples always use exact match)
            matched_exact = self.file_list[self.file_list['file_name'] == file_name]

            if matched_exact.empty:
                return None

            exact_labels = matched_exact['label'].to_list()

            # First check positive sample: must match exactly
            if self.class_idx in exact_labels:
                y = 1
            else:
                # If not a positive sample, decide how to find negative samples
                if self.id_distinguished:
                    current_id = self.extract_match_id(file_name)
                    matched_id = self.file_list[
                        self.file_list['file_name'].apply(self.extract_match_id) == current_id
                        ]
                    id_labels = matched_id['label'].to_list()

                    # Under this ID there may be other positive classes, so it cannot be counted as negative
                    if 0 in id_labels and self.class_idx not in id_labels:
                        y = 0
                    else:
                        return None
                else:
                    if 0 in exact_labels:
                        y = 0
                    else:
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
    batch = [item for item in batch if item is not None]  # Filter out None items
    if not batch:  # If batch is empty, return empty values
        return None, None, None, None

    # Check if batch contains low signal flag (5th element)
    if len(batch[0]) == 5:
        X, y, file_names, lengths, is_low_signals = zip(*batch)
        is_low_signals = torch.tensor(is_low_signals)
    else:
        X, y, file_names, lengths = zip(*batch)
        is_low_signals = None

    X_padded = pad_sequence([torch.tensor(x) for x in X], batch_first=True, padding_value=0)

    if y[0] is not None and all(isinstance(item, (int, float)) for item in y):
        y = torch.tensor(y)
    else:
        y = None

    lengths = torch.tensor(lengths)

    if is_low_signals is not None:
        return X_padded, y, file_names, lengths, is_low_signals
    else:
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
        return x + self.pe[:, :x.size(1)]  # Simplified slicing


class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_dim, cnn_channels=16, transformer_layers=2, transformer_heads=4,
                 transformer_hidden_dim=64, output_dim=1, dropout=0.1, pe_max_length=15000):
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
        # In the forward method
        if lengths is not None:
            padding_mask = self.create_padding_mask(lengths, x.size(1))
            padding_mask = padding_mask.to(x.device)

            # Try the most basic call style
            try:
                x = self.transformer(x, src_key_padding_mask=padding_mask)
            except TypeError:
                # If the above call fails, try another common parameter form
                x = self.transformer(x, mask=None, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)

        # Post-transformer normalization
        x = self.post_transformer_norm(x)

        # Sequence pooling
        if lengths is not None:
            indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(-1))

            # check the valid of index
            if indices.max() >= x.size(1) or indices.min() < 0:
                # print(f"Warning: index out of bound! indices range [{indices.min()}, {indices.max()}], but x.size(1)={x.size(1)}")
                indices = torch.clamp(indices, 0, x.size(1) - 1)

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
        # In newer versions of PyTorch, positions where src_key_padding_mask is True are ignored
        # (i.e., padding positions), so we need to ensure True in the mask represents padding positions
        mask = (torch.arange(max_len, device=device, dtype=torch.long)[None, :] >= lengths[:, None])
        return mask


def train(args, model, device, optimizer, criterion, num_epochs,train_loader_raw, train_loader_transform=None,
          test_loader=None, save_freq=5, resume_training=False):
    os.makedirs(args.output_dir, exist_ok=True)

    logger = setup_logger(args.output_dir)

    best_accuracy = 0.0
    start_epoch = 0
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
    for epoch in range(start_epoch, start_epoch + num_epochs):
        loader_names = ['transform', 'raw']
        for loader_idx, train_loader in enumerate([train_loader_transform, train_loader_raw]):
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
            print(f'Epoch [{epoch + 1}/{num_epochs}] [{loader_names[loader_idx]}] Loss: {epoch_loss:.4f}; Accuracy: {train_accuracy:.4f}; Balanced Accuracy: {balanced_accuracy:.4f}')

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
                if train_accuracy>best_accuracy:
                    best_accuracy = train_accuracy
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
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_data is None:  # If batch is empty, skip
                print('None inputs, skip')
                continue

            # Check if batch contains low signal flag (5th element)
            if len(batch_data) == 5:
                inputs, y, csv_file, lengths, is_low_signals = batch_data
                # Handle low signal samples
                low_signal_indices = torch.where(is_low_signals)[0]
                if len(low_signal_indices) > 0:
                    for idx in low_signal_indices:
                        i = idx.item()
                        if save_result:
                            results.append({
                                'file_name': csv_file[i],
                                'probability': 0,  # Low signal, probability set to 0
                                'pred_class': 0,  # Low signal, class set to 0
                                'true': y[i].item()
                            })

                # If all samples are low signal, skip model evaluation
                if torch.all(is_low_signals):
                    print(
                        f'Batch [{batch_idx + 1}/{len(test_loader)}]: All samples are low signal, skipping model evaluation')
                    continue

                # Filter out non-low-signal samples
                non_low_indices = torch.where(~is_low_signals)[0]
                inputs = inputs[non_low_indices]
                y = y[non_low_indices]
                lengths = lengths[non_low_indices]
                csv_file = [csv_file[i.item()] for i in non_low_indices]
            else:
                # No low signal flag, process normally
                inputs, y, csv_file, lengths = batch_data

            inputs, lengths = inputs.to(device), lengths.to(device)
            outputs = model(inputs, lengths=lengths)

            if n_classes == 1:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities).squeeze()
                # Ensure prediction and label dimensions match
                if predicted.dim() == 0 and len(y) == 1:
                    predicted = predicted.unsqueeze(0)
                labels = y.float().view(-1, 1) if y.dim() == 1 else y.float()

            else:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                labels = y.long()

            # Evaluate model performance
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()

            # Handle dimension issues
            if y_true.ndim > 1 and y_true.shape[1] == 1:
                y_true = y_true.ravel()
            if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()

            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            print(
                f'Batch [{batch_idx + 1}/{len(test_loader)}]: Accuracy: {accuracy:.4f}; Balanced Accuracy: {balanced_accuracy:.4f}')

            # Save results
            if save_result:
                for i in range(len(csv_file)):
                    prob_val = probabilities[i].item() if isinstance(probabilities[i], torch.Tensor) else probabilities[
                        i]
                    pred_val = predicted[i].item() if isinstance(predicted[i], torch.Tensor) else predicted[i]
                    true_val = y[i].item() if isinstance(y[i], torch.Tensor) else y[i]

                    results.append({
                        'file_name': csv_file[i],
                        'probability': prob_val,
                        'pred_class': pred_val,
                        'true': true_val
                    })

    if save_result:
        results_df = pd.DataFrame(results)
        file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
        results_df.to_csv(file_path, index=False)
        os.chmod(file_path, 0o777)

    return accuracy, balanced_accuracy  # Return the last batch's accuracy and balanced accuracy


def predict(args, input_dim,n_classes, class_idx, model, device, test_loader, result_dir, type, n_files, event_precision=1, check_signal=True):
    model.eval()
    results = []
    with torch.no_grad():
        progress_bar = tqdm(total=n_files, desc=f"{type} EEG level results")
        for batch_data in test_loader:
            if batch_data is None:  # If batch is empty, skip
                print('None inputs, skip')
                continue

            # Check if batch contains low signal flag (5th element)
            if check_signal and len(batch_data) == 5:
                inputs, y, csv_file, lengths, is_low_signals = batch_data
                # Handle low signal samples
                low_signal_indices = torch.where(is_low_signals)[0]
                if len(low_signal_indices) > 0:
                    for idx in low_signal_indices:
                        i = idx.item()

                        results.append({
                            'file_name': csv_file[i],
                            'probability': 0,  # Low signal, probability set to 0
                            'pred_class_p': 0,  # Low signal, class set to 0
                            #'positive_count': 0,
                            #'positive_proportion':0,
                            'confidence': 0,
                            'pred_class': 0,  # Low signal, class set to 0
                        })

                # Only process non-low-signal samples
                if check_signal and torch.all(is_low_signals):
                    progress_bar.update(len(csv_file))
                    continue

                # Filter out non-low-signal samples
                non_low_indices = torch.where(~is_low_signals)[0]
                inputs = inputs[non_low_indices]
                if y is not None:
                    y = y[non_low_indices]
                lengths = lengths[non_low_indices]
                csv_file = [csv_file[i.item()] for i in non_low_indices]
            else:
                # No low signal flag, process normally
                inputs, y, csv_file, lengths = batch_data

            if inputs.size(0) == 0:  # If no samples to predict, skip
                continue

            # Send data to device
            inputs, lengths = inputs.to(device), lengths.to(device)

            # Run model prediction
            outputs = model(inputs, lengths=lengths)

            if n_classes == 1:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                predicted = torch.round(probabilities).squeeze()
                # Ensure predictions are iterable even with a single sample
                if not isinstance(predicted, torch.Tensor) or predicted.dim() == 0:
                    predicted = predicted.view(1)
                    probabilities = probabilities.view(1)
            else:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)

            # Collect prediction results for non-low-signal samples
            for i in range(len(csv_file)):
                prob_val = probabilities[i].item() if isinstance(probabilities[i], torch.Tensor) else probabilities[i]
                pred_val = predicted[i].item() if isinstance(predicted[i], torch.Tensor) else predicted[i]

                if input_dim == 1:
                    mask = inputs[i] > 0.5
                    count = int(mask.sum().item() * event_precision)
                    total = inputs[i].numel()

                else:
                    total = inputs[i].shape[0]  # Number of rows, i.e., number of samples
                    if type in ['LRDA', 'GRDA']:
                        # Use probability threshold instead of argmax to capture intermittent patterns
                        # argmax requires LRDA/GRDA to beat all 5 other classes — too strict for intermittent rhythms
                        mask = (inputs[i][:, class_idx] > 1.0 / input_dim)
                    else:
                        max_indices = torch.argmax(inputs[i], dim=1)  # Get index of max value per row
                        mask = (max_indices == class_idx)  # Check if max value is in current class
                    count = int(mask.sum().item() * event_precision)

                proportion = count / total
                if type in ['LRDA', 'GRDA']:
                    # Boost-only: small additive boost from count signal, never penalizes low count
                    boost = min(0.1, proportion * 0.5)
                    confidence = min(1, prob_val + boost)
                else:
                    confidence = min(1, prob_val + proportion)
                if type=='SPIKES':
                    if count >= 5:
                        confidence=min(1,confidence/2+0.5)
                elif type not in ['LRDA', 'GRDA']:
                    if count < 10 and confidence>0.5:
                        confidence = 0.5

                if n_classes == 1:
                    pred_class_p = 1 if prob_val > 0.5 else 0
                    pred_class = 1 if confidence > 0.5 else 0

                else:
                    pred_class_p = pred_val
                    pred_class = pred_val

                if type == 'NORMAL':
                    results.append({
                        'file_name': csv_file[i],
                        'probability': prob_val,
                        'pred_class_p': pred_class_p,
                        #'positive_count': count,
                        # 'positive_proportion': proportion,
                        'confidence': confidence,
                        'pred_class': pred_class,
                        'revised_confidence': confidence,
                        'revised_pred_class': pred_class,

                    })

                else:
                    results.append({
                        'file_name': csv_file[i],
                        'probability': prob_val,
                        'pred_class_p': pred_class_p,
                        #'positive_count': count,
                        #'positive_proportion': proportion,
                        'confidence': confidence,
                        'pred_class': pred_class,
                    })

            progress_bar.update(len(csv_file))

        progress_bar.n = n_files
        progress_bar.refresh()
        progress_bar.close()

    results_df = pd.DataFrame(results)

    normal_file_path=os.path.join(result_dir, f'pred_EEG_level_NORMAL.csv')
    if os.path.exists(normal_file_path) and type!='NORMAL':
        normal_df = pd.read_csv(normal_file_path)
        normal_df=normal_results_align(results_df, normal_df)
        if normal_df is not False:
            normal_df.to_csv(normal_file_path, index=False)


    if type=='SPIKES' and args.align_spike_detection_and_location:
        foc_spike_path=os.path.join(result_dir, f'pred_EEG_level_FOC_SPIKES.csv')
        gen_spike_path=os.path.join(result_dir, f'pred_EEG_level_GEN_SPIKES.csv')
        if os.path.exists(foc_spike_path):
            foc_spike_df=pd.read_csv(foc_spike_path)
            results_df_new,foc_spike_df=spikes_results_align(results_df,foc_spike_df)
            if results_df_new is not False:
                results_df=results_df_new
                foc_spike_df.to_csv(foc_spike_path, index=False)

        if os.path.exists(gen_spike_path):
            gen_spike_df = pd.read_csv(gen_spike_path)
            results_df_new,gen_spike_df = spikes_results_align(results_df, gen_spike_df)
            if results_df_new is not False:
                results_df=results_df_new
                gen_spike_df.to_csv(gen_spike_path, index=False)

    # if (type == 'FOC_SPIKES' or type == 'GEN_SPIKES') and args.align_spike_detection_and_location:
    #     spike_path=os.path.join(result_dir, f'pred_EEG_level_SPIKES.csv')
    #     if os.path.exists(spike_path):
    #         spike_df=pd.read_csv(spike_path)
    #         spike_df,results_df=spikes_results_align(spike_df, results_df)
    #         if spike_df is not False:
    #             spike_df.to_csv(spike_path, index=False)

    file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
    results_df.to_csv(file_path, index=False)
    os.chmod(file_path, 0o777)
    print(f'EEG level results are saved to {file_path}')


def spikes_results_align(df_a, df_b):
    # Check required columns
    required_columns = ['file_name', 'confidence', 'pred_class']
    for df in [df_a, df_b]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False,False

    # Ensure file_name column types are consistent
    df_a['file_name'] = df_a['file_name'].astype(str)
    df_b['file_name'] = df_b['file_name'].astype(str)

    # Create index on df_b for fast lookup
    b_index = df_b.set_index('file_name')

    # Get file_names common to both DataFrames
    common_files = set(df_a['file_name']).intersection(set(df_b['file_name']))

    # Check if df_a has a count column
    has_count_column = 'positive_count' in df_a.columns

    # Iterate over common file_names
    for file_name in common_files:
        # Get confidence values from a and b
        a_idx = df_a[df_a['file_name'] == file_name].index
        b_idx = df_b[df_b['file_name'] == file_name].index

        a_confidence = df_a.loc[a_idx, 'confidence'].values[0]
        b_confidence = b_index.loc[file_name, 'confidence']

        # Apply condition: if b confidence > 0.5 and a confidence < 0.5
        if b_confidence > 0.5 and a_confidence < 0.5:
            # Compute mean
            mean_confidence = (a_confidence + b_confidence) / 2

            # If mean > 0.5 and df_a count = 0, set to 0.5
            if mean_confidence > 0.5 and has_count_column:
                a_count = df_a.loc[a_idx, 'positive_count'].values[0]
                if a_count == 0:
                    mean_confidence = 0.5

            # Update values in df_a
            df_a.loc[a_idx, 'confidence'] = mean_confidence
            df_a.loc[a_idx, 'pred_class'] = 1 if mean_confidence > 0.5 else 0

            # Also update values in df_b
            df_b.loc[b_idx, 'confidence'] = mean_confidence
            df_b.loc[b_idx, 'pred_class'] = 1 if mean_confidence > 0.5 else 0

    return df_a, df_b


def normal_results_align(df_a, df_b):
    # df_b is normal file
    required_columns = ['file_name', 'confidence', 'pred_class']
    for df in [df_a, df_b]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    df_a['file_name'] = df_a['file_name'].astype(str)
    df_b['file_name'] = df_b['file_name'].astype(str)

    b_index = df_b.set_index('file_name')

    common_files = set(df_a['file_name']).intersection(set(df_b['file_name']))

    # Iterate over common file_names
    for file_name in common_files:
        # Get confidence values from a and b
        a_idx = df_a[df_a['file_name'] == file_name].index
        b_idx = df_b[df_b['file_name'] == file_name].index

        a_confidence = df_a.loc[a_idx, 'confidence'].values[0]
        b_confidence = b_index.loc[file_name, 'confidence']

        if b_confidence < 0.5 and a_confidence > 0.5:
            df_b.loc[b_idx, 'revised_confidence'] = 0.5
            df_b.loc[b_idx, 'revised_pred_class'] = 1

    return df_b

def predict_based_10min(args,input_dim, n_classes, class_idx, model, device, test_loader, result_dir, type, n_files,event_precision=1,check_signal=True):
    model.eval()
    results = []
    sub_sample_length = 591  # Target sub-sample length (rows) 10min
    min_sample_length = 30  # Minimum sub-sample length (rows)
    batch_size = 64  # Batch size, adjustable based on GPU memory

    if type == 'FOC_SPIKES' or type == 'GEN_SPIKES' or type=='FOC_SLOWING' or type=='GEN_SLOWING':
        predict(args=args, input_dim=input_dim,n_classes= n_classes, class_idx=class_idx, model=model, device=device, test_loader=test_loader, result_dir=result_dir, type=type, n_files=n_files, event_precision=event_precision, check_signal=True)

    with torch.no_grad():
        progress_bar = tqdm(total=n_files, desc=f"{type} EEG level results")
        for batch_data in test_loader:
            if batch_data is None:  # If batch is empty, skip
                print('None inputs, skip')
                continue

            # Check if batch contains low signal flag (5th element)
            if check_signal and len(batch_data) == 5:
                inputs, y, csv_file, lengths, is_low_signals = batch_data
                # Handle low signal samples
                low_signal_indices = torch.where(is_low_signals)[0]
                if len(low_signal_indices) > 0:
                    for idx in low_signal_indices:
                        i = idx.item()
                        results.append({
                            'file_name': csv_file[i],
                            'probability': 0,
                            'pred_class_p': 0,  # Low signal, class set to 0
                            # 'positive_count': 0,
                            #'positive_proportion': 0,
                            #'high_positive_proportion':0,
                            #'positive_10min_count': 0,
                            'confidence': 0,
                            'pred_class': 0,  # Low signal, class set to 0
                        })

                # Only process non-low-signal samples
                if check_signal and torch.all(is_low_signals):
                    progress_bar.update(len(csv_file))
                    continue

                # Filter out non-low-signal samples
                non_low_indices = torch.where(~is_low_signals)[0]
                inputs = inputs[non_low_indices]
                if y is not None:
                    y = y[non_low_indices]
                lengths = lengths[non_low_indices]
                csv_file = [csv_file[i.item()] for i in non_low_indices]
            else:
                # No low signal flag, process normally
                inputs, y, csv_file, lengths = batch_data

            if inputs.size(0) == 0:  # If no samples to predict, skip
                continue

            # Initialize per-file result dictionary
            file_results = {}
            for name in csv_file:
                if n_classes == 1:
                    file_results[name] = {
                        'max_prob': 0.0,  # Initial max probability is 0
                        'max_confidence': 0.0,
                        'total_segments': 0,
                        'positive_segments': 0,
                        'positive_count':0,
                        'positive_proportion': 0,
                        'high_positive_proportion':0,
                        'sequence_length':0,
                    }
                else:
                    file_results[name] = {
                        'max_prob': [0.0] * n_classes,  # Initialize max probability per class
                        'max_confidence':[0.0] * n_classes,
                        'total_segments': 0,
                        'class_counts': [0] * n_classes,
                        'positive_count': 0,
                        'positive_proportion': 0,
                        'high_positive_proportion': 0,
                        'sequence_length': 0,
                    }

            # Collect all sub-samples for batch processing
            all_sub_samples = []
            sub_sample_lengths = []  # Record original lengths for model input
            sub_sample_file_indices = []  # Track which file each sub-sample belongs to
            sub_sample_file_names = []  # Store corresponding file names for debugging
            sub_positive_proportion = []

            # Split each sample (by row)
            for i in range(len(csv_file)):

                file_name = csv_file[i]
                input_sample = inputs[i]
                num_rows = input_sample.shape[0]
                file_results[file_name]['sequence_length'] = num_rows

                if input_dim == 1:
                    total = input_sample.numel()
                    mask = input_sample > 0.5
                    count = int(mask.sum().item() * event_precision)
                    #mask_2 = input_sample > 0.9
                    #count_2 = int(mask_2.sum().item() * event_precision)

                else:
                    total = num_rows  # Number of rows, i.e., number of samples
                    if type in ['LRDA', 'GRDA']:
                        # Use probability threshold instead of argmax to capture intermittent patterns
                        # argmax requires LRDA/GRDA to beat all 5 other classes — too strict for intermittent rhythms
                        mask = (input_sample[:, class_idx] > 1.0 / input_dim)
                    else:
                        max_indices = torch.argmax(input_sample, dim=1)  # Get index of max value per row
                        mask = (max_indices == class_idx)  # Check if max value is in current class
                    count = int(mask.sum().item() * event_precision)
                    #mask_2 = input_sample[:,class_idx] > 0.8
                    #count_2 = int(mask_2.sum().item() * event_precision)

                file_results[file_name]['positive_count'] = count

                if count==0:
                    file_results[file_name]['positive_proportion'] = 0
                    #file_results[file_name]['high_positive_proportion'] =  0
                else:
                    file_results[file_name]['positive_proportion'] = count / total
                    #file_results[file_name]['high_positive_proportion'] = count_2 / count


                # Check input shape
                if input_sample.dim() == 1:
                    # If 1D tensor, add feature dimension
                    input_sample = input_sample.unsqueeze(1)
                    num_rows = input_sample.shape[0]

                # Compute number of sub-samples based on rows
                num_full_segments = num_rows // sub_sample_length
                last_segment_length = num_rows % sub_sample_length
                total_segments = num_full_segments + (1 if last_segment_length >= min_sample_length else 0)
                file_results[file_name]['total_segments'] += total_segments

                # Collect complete sub-samples (each 600 rows)
                for j in range(num_full_segments):
                    start_row = j * sub_sample_length
                    end_row = (j + 1) * sub_sample_length
                    sub_input = input_sample[start_row:end_row]
                    sub_mask = mask[start_row:end_row]
                    sub_count = int(sub_mask.sum().item() * event_precision)

                    # Ensure sub-sample shape is valid
                    if 0 not in sub_input.shape:
                        all_sub_samples.append(sub_input)
                        sub_sample_lengths.append(sub_sample_length)
                        sub_sample_file_indices.append(i)
                        sub_sample_file_names.append(file_name)
                        sub_positive_proportion.append(sub_count/sub_sample_length)
                    else:
                        print(f"Warning: skipping sub-sample with shape {sub_input.shape}")

                # Handle the last incomplete sub-sample (if enough rows)
                if last_segment_length >= min_sample_length:
                    start_row = num_full_segments * sub_sample_length
                    end_row = num_rows
                    sub_input = input_sample[start_row:end_row]
                    sub_mask = mask[start_row:end_row]
                    sub_count = int(sub_mask.sum().item() * event_precision)

                    # Ensure sub-sample shape is valid
                    if 0 not in sub_input.shape:
                        # Compute mean for padding
                        mean_values = sub_input.mean(dim=0, keepdim=True)

                        # Create padded sub-sample
                        padded_sub_input = torch.zeros((sub_sample_length, sub_input.shape[1]),
                                                       dtype=sub_input.dtype,
                                                       device=sub_input.device)
                        padded_sub_input[:last_segment_length] = sub_input

                        # Fill remaining rows with mean values
                        for row_idx in range(last_segment_length, sub_sample_length):
                            padded_sub_input[row_idx] = mean_values

                        all_sub_samples.append(padded_sub_input)
                        sub_sample_lengths.append(last_segment_length)  # Record original length
                        sub_sample_file_indices.append(i)
                        sub_sample_file_names.append(file_name)
                        sub_positive_proportion.append(sub_count/sub_sample_length)
                    else:
                        print(f"Warning: skipping sub-sample with shape {sub_input.shape}")

            # If no valid sub-samples, continue to next batch
            if len(all_sub_samples) == 0:
                print("Warning: no valid sub-samples in this batch")
                progress_bar.update(len(csv_file))
                continue

            # Batch process sub-samples
            for batch_start in range(0, len(all_sub_samples), batch_size):
                batch_end = min(batch_start + batch_size, len(all_sub_samples))
                batch_samples = all_sub_samples[batch_start:batch_end]
                batch_lengths = sub_sample_lengths[batch_start:batch_end]
                batch_positive_proportion = sub_positive_proportion[batch_start:batch_end]
                batch_file_indices = sub_sample_file_indices[batch_start:batch_end]
                batch_file_names = sub_sample_file_names[batch_start:batch_end]

                try:
                    # Create batch tensor
                    batch_tensor = torch.stack(batch_samples).to(device)
                    batch_lengths_tensor = torch.tensor(batch_lengths, device=device)

                    # Model prediction
                    batch_outputs = model(batch_tensor, lengths=batch_lengths_tensor)

                    # Process prediction results
                    for k in range(len(batch_samples)):
                        file_name = batch_file_names[k]
                        positive_proportion_=batch_positive_proportion[k]

                        if n_classes == 1:  # Binary classification
                            sub_prob = torch.sigmoid(batch_outputs[k]).item()
                            # [MODIFIED] SEIZURE/LPD/GPD/LRDA/GRDA only: replace additive sub_confidence with multiplicative softening
                            # Before: sub_confidence = min(1, proportion + prob)
                            #   Additive formula clips many samples to 1.0, collapsing rank in the high-score region (hurts PR)
                            #   Also causes rank inversions since proportion and prob have incompatible scales
                            # After: sub_confidence = min(1, prob * (1 + 0.5 * proportion))
                            #   proportion acts as a scaling factor on prob (up to +50%), avoiding rank collapse
                            # Other dataset types keep the original additive formula
                            if type in ['SEIZURE', 'LPD', 'GPD', 'LRDA', 'GRDA']:
                                sub_confidence = min(1, sub_prob * (1 + 0.5 * positive_proportion_))
                            else:
                                sub_confidence = min(1, positive_proportion_ + sub_prob)
                            # Update max probability
                            if sub_confidence > file_results[file_name]['max_confidence']:
                                file_results[file_name]['max_confidence'] = sub_confidence
                                file_results[file_name]['max_prob'] = sub_prob
                            # Count positive segments
                            # if sub_prob > 0.5:
                            #     file_results[file_name]['positive_segments'] += 1

                        else:  # Multi-class classification
                            sub_prob = torch.softmax(batch_outputs[k], dim=0)
                            pred_class = torch.argmax(sub_prob).item()

                            # Update max probability per class
                            for c in range(n_classes):
                                if sub_prob[c].item() > file_results[file_name]['max_prob'][c]:
                                    file_results[file_name]['max_prob'][c] = sub_prob[c].item()
                                    file_results[file_name]['max_confidence'][c] = sub_prob[c].item()

                            # Count predicted class
                            file_results[file_name]['class_counts'][pred_class] += 1

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Print shape info for debugging
                    shapes = [sample.shape for sample in batch_samples]
                    print(f"Shapes in batch: {shapes}")
                    continue

            # Aggregate results per file
            for file_name, res in file_results.items():
                if res['total_segments'] == 0:  # If no sub-samples were processed, skip
                    continue

                if n_classes == 1:
                    # Use max probability as confidence
                    max_prob = res['max_prob']
                    max_confidence= res['max_confidence']

                    # [MODIFIED] SEIZURE/LPD/GPD/LRDA/GRDA only: remove double-counting of global_proportion
                    # Before (long EEG): confidence = min(1, max_confidence + global_proportion)
                    # proportion was already incorporated in each sub-segment's sub_confidence,
                    # adding it again at the global level causes double-counting
                    # Other dataset types keep the original logic
                    if type in ['LRDA', 'GRDA']:
                        # Boost-only: small additive boost from positive_proportion, never penalizes
                        boost = min(0.1, res['positive_proportion'] * 0.5)
                        confidence = min(1, max_prob + boost)
                    elif type in ['SEIZURE', 'LPD', 'GPD']:
                        confidence = max_confidence
                    else:
                        if file_results[file_name]['sequence_length'] > sub_sample_length * 2:
                            confidence = min(1, max_confidence + res['positive_proportion'])
                        else:
                            confidence = max_confidence

                    pred_class_p = 1 if max_prob > 0.5 else 0
                    pred_class = 1 if confidence > 0.5 else 0

                else:
                    # For multi-class, use the class with the highest probability
                    max_prob_class_idx = res['max_confidence'].index(max(res['max_confidence']))
                    pred_class_p=max_prob_class_idx
                    pred_class = max_prob_class_idx
                    max_confidence = res['max_confidence'][max_prob_class_idx]
                    max_prob = res['max_prob'][max_prob_class_idx]

                    # [MODIFIED] SEIZURE/LPD/GPD/LRDA/GRDA only: remove double-counting of global_proportion in multi-class path
                    if type in ['SEIZURE', 'LPD', 'GPD', 'LRDA', 'GRDA']:
                        confidence = max_confidence
                    else:
                        if file_results[file_name]['sequence_length'] > sub_sample_length * 2:
                            confidence = min(1, max_confidence + res['positive_proportion'])
                        else:
                            confidence = max_confidence

                count=res['positive_count']

                if type=='SPIKES':
                    if count >= 5:
                        confidence=min(1,confidence/2+0.5)

                    elif count==0 and confidence>0.5:
                        confidence=0.5
                    pred_class = 1 if confidence > 0.5 else 0

                # [MODIFIED] SEIZURE/LPD/GPD/LRDA/GRDA only: replace hard zero-gate with sigmoid soft-gate
                # LRDA/GRDA use center=5 (intermittent rhythmic patterns have fewer positive segments)
                # SEIZURE/LPD/GPD use center=10
                # Other types keep the original hard zero-gate
                if type in ['SEIZURE', 'LPD', 'GPD']:
                    gate = 1.0 / (1.0 + np.exp(-(count - 10) / 3.0))
                    confidence = confidence * gate
                    pred_class = 1 if confidence > 0.5 else 0
                    results.append({
                        'file_name': file_name,
                        'probability': max_prob,
                        'pred_class_p': pred_class_p,
                        # 'positive_count': count,
                        #'positive_proportion': res['positive_proportion'],
                        # 'high_positive_proportion': res['high_positive_proportion'],
                        # 'positive_10min_count': res['positive_segments'] if n_classes == 1 else res['class_counts'][class_idx],
                        'confidence': confidence,
                        'pred_class': pred_class
                    })
                elif count < 10 and type != 'SPIKES':
                    results.append({
                        'file_name': file_name,
                        'probability': 0,
                        'pred_class_p': 0,
                        # 'positive_count': 0,
                        #'positive_proportion': 0,
                        # 'high_positive_proportion': 0,
                        #'positive_10min_count': 0,
                        'confidence': 0,
                        'pred_class': 0,
                    })
                else:
                    results.append({
                        'file_name': file_name,
                        'probability': max_prob,
                        'pred_class_p': pred_class_p,
                        # 'positive_count': count,
                        #'positive_proportion': res['positive_proportion'],
                        # 'high_positive_proportion': res['high_positive_proportion'],
                        # 'positive_10min_count': res['positive_segments'] if n_classes == 1 else res['class_counts'][class_idx],
                        'confidence': confidence,
                        'pred_class': pred_class
                    })

            progress_bar.update(len(csv_file))

        progress_bar.n = n_files
        progress_bar.refresh()
        progress_bar.close()

    results_df = pd.DataFrame(results)

    normal_file_path=os.path.join(result_dir, f'pred_EEG_level_NORMAL.csv')
    if os.path.exists(normal_file_path) and type!='NORMAL':
        normal_df = pd.read_csv(normal_file_path)
        normal_df=normal_results_align(results_df, normal_df)
        if normal_df is not False:
            normal_df.to_csv(normal_file_path, index=False)

    if type == 'SPIKES' and args.align_spike_detection_and_location:
        foc_spike_path = os.path.join(result_dir, f'pred_EEG_level_FOC_SPIKES.csv')
        gen_spike_path = os.path.join(result_dir, f'pred_EEG_level_GEN_SPIKES.csv')
        if os.path.exists(foc_spike_path):
            foc_spike_df = pd.read_csv(foc_spike_path)
            results_df_new,foc_spike_df = spikes_results_align(results_df, foc_spike_df)
            if results_df_new is not False:
                results_df = results_df_new
                foc_spike_df.to_csv(foc_spike_path, index=False)

        if os.path.exists(gen_spike_path):
            gen_spike_df = pd.read_csv(gen_spike_path)
            results_df_new ,gen_spike_df = spikes_results_align(results_df, gen_spike_df)
            if results_df_new is not False:
                results_df = results_df_new
                gen_spike_df.to_csv(gen_spike_path, index=False)

    if type == 'FOC_SPIKES' or type == 'GEN_SPIKES':
        spike_path = os.path.join(result_dir, f'pred_EEG_level_SPIKES.csv')
        if os.path.exists(spike_path) and args.align_spike_detection_and_location:
            spike_df = pd.read_csv(spike_path)
            spike_df, results_df= spikes_results_align(spike_df, results_df)
            if spike_df is not False:
                spike_df.to_csv(spike_path, index=False)

        file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
        if os.path.exists(file_path):
            original_df = pd.read_csv(file_path)
            required_columns = ['file_name', 'confidence','pred_class']
            missing_columns = [col for col in required_columns if col not in original_df.columns]
            if not missing_columns:
                original_df=original_df[required_columns]
                original_df.rename(
                    columns={'confidence':'probability',
                             'pred_class':'pred_class_p'}, inplace=True)
                results_df=results_df[['file_name','confidence','pred_class']]
                results_df=pd.merge(original_df, results_df, on='file_name', how='right')
                results_df['confidence'] = results_df[['confidence', 'probability']].max(axis=1)
                results_df['pred_class'] = results_df[['pred_class', 'pred_class_p']].max(axis=1)

    if type=='FOC_SLOWING' or type=='GEN_SLOWING':
        file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
        if os.path.exists(file_path):
            original_df = pd.read_csv(file_path)
            required_columns = ['file_name', 'probability','pred_class_p']
            missing_columns = [col for col in required_columns if col not in original_df.columns]
            if not missing_columns:
                original_df = original_df[required_columns]

                results_df = results_df[['file_name', 'confidence', 'pred_class']]
                results_df = pd.merge(original_df, results_df, on='file_name', how='right')
                results_df['confidence'] = results_df[['confidence', 'probability']].max(axis=1)
                results_df['pred_class'] = results_df[['pred_class', 'pred_class_p']].max(axis=1)

    file_path = os.path.join(result_dir, f'pred_EEG_level_{type}.csv')
    results_df.to_csv(file_path, index=False)
    os.chmod(file_path, 0o777)
    print(f'EEG level results are saved to {file_path}')


def load_model(args):
    model = CNNTransformerClassifier(
        input_dim=args.input_dim,
        output_dim=args.n_classes,
        pe_max_length=args.pe_max_length,
    ).to(args.device)

    return model




# def load_model_parameters(model,model_parameters_path):
#     model.load_state_dict(torch.load(model_parameters_path,weights_only=True)['model_state_dict'])
#     return model

def load_model_parameters(model, model_parameters_path, device=None, strict=False):
    # 1) Select device: cuda > mps > cpu
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # 2) Map to CPU first for stability (especially on Mac/MPS); CUDA machines can map directly to cuda
    map_loc = device if device.type == "cuda" else "cpu"

    # 3) Safe loading (newer torch supports weights_only; fallback if not supported)
    try:
        state = torch.load(model_parameters_path, map_location=map_loc, weights_only=True)
    except TypeError:
        state = torch.load(model_parameters_path, map_location=map_loc)

    # 4) Extract state_dict (compatible with multiple save formats)
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        raise ValueError("Unexpected checkpoint format.")

    # 5) Strip DDP 'module.' prefix
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # 6) Load parameters and print any mismatched keys for debugging
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[load_model_parameters] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:    print("  missing(sample):", missing[:10])
        if unexpected: print("  unexpected(sample):", unexpected[:10])

    # 7) Move model to target device
    model.to(device)
    return model

def summarize_sleep_eeg_level_results(dataset_type,train_csv_dirs,result_dir,event_step=1):
    def check_all_consecutive_labels(series):
        consecutive_counts = {0:0, 1: 0, 2: 0, 3: 0, 4: 0}
        # Initialize results
        has_consecutive = {0:False, 1: False, 2: False, 3: False, 4: False}
        # Define thresholds
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

            # Handle consecutiveness of value 1
            if value == 1:
                consecutive_counts[1] += 1
                if not has_consecutive[1] and consecutive_counts[1] >= thresholds[1]:
                    has_consecutive[1] = True
            else:
                consecutive_counts[1] = 0

            # Handle consecutiveness of value 2 - increment when value is 2, keep count unchanged when value is 1 or 3
            if value == 2:
                consecutive_counts[2] += 1
                if not has_consecutive[2] and consecutive_counts[2] >= thresholds[2]:
                    has_consecutive[2] = True
            elif value == 1 or value == 3 or value == 4:
                # For 1 or 3, keep the value-2 counter unchanged
                pass
            else:
                consecutive_counts[2] = 0

            # Handle consecutiveness of value 3
            if value == 3:
                consecutive_counts[3] += 1
                if not has_consecutive[3] and consecutive_counts[3] >= thresholds[3]:
                    has_consecutive[3] = True
            else:
                consecutive_counts[3] = 0

            # Handle consecutiveness of value 4
            if value == 4:
                consecutive_counts[4] += 1
                if not has_consecutive[4] and consecutive_counts[4] >= thresholds[4]:
                    has_consecutive[4] = True
            else:
                consecutive_counts[4] = 0

        # Return five boolean values
        return has_consecutive[0], has_consecutive[1], has_consecutive[2], has_consecutive[3], has_consecutive[4]

    result_list_df = pd.DataFrame(columns=['file_name'] + [f'pred_{i}_class' for i in range(5)])
    for dir in tqdm(train_csv_dirs):
        for file in tqdm(os.listdir(dir),desc=f'{dir}'):
            file_name=file.split('.')[0]
            event_level_results_df=pd.read_csv(os.path.join(dir,file))

            # Skip consecutive check for recordings shorter than 1 min
            if len(event_level_results_df) <= 10*60/event_step:
                if 'class_3_prob' in  event_level_results_df.columns and  'pred_4_class' in  event_level_results_df.columns:
                    new_row = pd.DataFrame({
                        'file_name': [file_name],
                        'pred_0_class': [event_level_results_df['class_0_prob'].max()],
                        'pred_1_class': [event_level_results_df['class_1_prob'].max()],
                        'pred_2_class': [event_level_results_df['class_2_prob'].max()],
                        'pred_3_class': [event_level_results_df['class_3_prob'].max()],
                        'pred_4_class': [event_level_results_df['class_4_prob'].max()]
                    })
                else:
                    new_row = pd.DataFrame({
                        'file_name': [file_name],
                        'pred_0_class': [event_level_results_df['class_0_prob'].max()],
                        'pred_1_class': [event_level_results_df['class_1_prob'].max()],
                        'pred_2_class': [event_level_results_df['class_2_prob'].max()],
                        'pred_3_class': 0,
                        'pred_4_class': 0
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
                        if dataset_type=='SLEEP3stages':
                            pred_4_class=0
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
                        if dataset_type == 'SLEEP3stages':
                            pred_3_class = 0
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
    if dataset_type=='SLEEPPSG':
        result_list_df.to_csv(os.path.join(result_dir, 'pred_EEG_level_SLEEP_5stage.csv'),index=False)

    elif dataset_type=='SLEEP3stages':
        result_list_df.to_csv(os.path.join(result_dir, 'pred_EEG_level_SLEEP_3stage.csv'), index=False)

def get_args():
    parser = argparse.ArgumentParser(description='CNN + Transformer Classifier')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'predict'],
                        help='Mode: train, test, or predict')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--pe_max_length', type=int, default=15000,
                        help='the maximum length of positional encoding')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--train_csv_dirs', type=str, help='Directories containing training CSV files')
    parser.add_argument('--file_list_path', default='', type=str, help='Data file contains file_name and labels')

    parser.add_argument('--test_csv_dir', type=str, help='Directory containing test CSV files')
    parser.add_argument('--result_dir', type=str, help='Directory containing test CSV files')

    parser.add_argument('--summary_window_minutes', type=int, default=0, help='Give EEG level results for each n-minute length EEG')

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

    parser.add_argument('--align_spike_detection_and_location', action='store_true',
                        help='check and align the results of spike, foc spike, gen spike')

    parser.add_argument('--add_data_transformation', action='store_true',
                        help='add data transformation')

    parser.add_argument('--id_distinguished', action='store_true',
                        help='make label by distinguished id')

    return parser.parse_args()


def get_dataset_info(dataset_name):
    n_classes=1
    if dataset_name=='SEIZURE':
        class_idx=1
        input_dim=6
        event_precision = 0.8
        method = 'predict_based_10min'

    elif dataset_name=='LPD':
        class_idx = 2
        input_dim = 6
        event_precision = 0.84
        method = 'predict_based_10min'

    elif dataset_name=='GPD':
        class_idx = 3
        input_dim = 6
        event_precision = 0.89
        method = 'predict_based_10min'

    elif dataset_name=='LRDA':
        class_idx = 4
        input_dim = 6
        event_precision = 0.8
        method = 'predict_based_10min'

    elif dataset_name=='GRDA':
        class_idx = 5
        input_dim = 6
        event_precision = 1
        method = 'predict_based_10min'

    elif dataset_name == 'FOC_SLOWING':
        class_idx = 1
        input_dim = 3
        event_precision = 0.95
        method = 'predict_based_10min'

    elif dataset_name == 'GEN_SLOWING':
        class_idx = 2
        input_dim = 3
        event_precision = 0.87
        method = 'predict_based_10min'

    elif dataset_name == 'FOC_SPIKES':
        class_idx = 1
        input_dim = 3
        event_precision = 1
        method='predict_based_10min'

    elif dataset_name == 'GEN_SPIKES':
        class_idx = 2
        input_dim = 3
        event_precision = 1
        method = 'predict_based_10min'

    elif dataset_name == 'BS':
        class_idx = 1
        input_dim = 1
        event_precision = 0.75
        method = 'predict_based_10min'

    elif dataset_name == 'NORMAL':
        class_idx = 1
        input_dim = 1
        event_precision = 0.9
        method = 'predict'

    elif dataset_name == 'SPIKES':
        class_idx = 1
        input_dim = 1
        event_precision=1
        method = 'predict_based_10min'

    elif dataset_name == 'SLEEPPSG':
        class_idx=None
        input_dim = 5
        event_precision=1
        method = 'predict'
    elif dataset_name == 'SLEEP3stages':
        class_idx=None
        input_dim = 3
        event_precision=1
        method = 'predict'

    else:
        print('wrong dataset name')
        exit(0)

    return class_idx, n_classes, input_dim,event_precision,method



def make_deterministic(seed=2025):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for some versions
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable TF32 to avoid numerical differences from matmul/cudnn
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Force deterministic algorithms (PyTorch will raise an error if unsupported ops are used)
    torch.use_deterministic_algorithms(True, warn_only=False)



def segment_continuous_result_by_min(file_path_list:list, output_dir: str, dataset_type:str, rows_per_unit_min: int = 10, padding=False, ):
    rows_per_unit= int(rows_per_unit_min * 60)

    for file_path in tqdm(file_path_list,desc=f'{rows_per_unit_min} continuous results segmented in {rows_per_unit}-minute units'):
        df = pd.read_csv(file_path)

        if padding:
            # Duplicate first row 5 times
            first_row = df.iloc[[0]].copy()
            first_rows_dup = pd.concat([first_row] * 5, ignore_index=True)

            # Duplicate last row 4 times
            last_row = df.iloc[[-1]].copy()
            last_rows_dup = pd.concat([last_row] * 4, ignore_index=True)

            # Concatenate to the front and back of df
            df_new = pd.concat([first_rows_dup, df, last_rows_dup], ignore_index=True)

            df =df_new

        total_rows = df.shape[0]
        num_parts = (total_rows + rows_per_unit - 1) // rows_per_unit

        base_name  = os.path.basename(file_path).split('.')[0]

        for i in range(num_parts):
            start = i * rows_per_unit
            end = min((i + 1) * rows_per_unit, total_rows)

            part_df = df.iloc[start:end]

            suffix = f"{i + 1}"
            new_filename = os.path.join(output_dir, f"{base_name}_{suffix}.csv")

            if part_df.isna().all().all():
                print(f"  -> skip {base_name}_{suffix}.csv (all NaN)")
                continue

            if part_df.isna().any().any():
                tmp = part_df.dropna()
                if tmp.empty:
                    print(f"  -> skip skip {base_name}_{suffix}.csv (empty after dropna)")
                    continue
                part_df = tmp

            part_df.to_csv(new_filename, index=False)






def main():
    args = get_args()
    # if args.device == 'cpu':
    #     args.device=torch.device("cpu")
    # else:
    #     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == 'cpu':
        args.device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")

    args.class_idx, args.n_classes, args.input_dim, event_precision,method = get_dataset_info(args.dataset)


    if args.mode == 'train':

        args.train_csv_dirs=list(map(str, args.train_csv_dirs.split()))

        train_dataset_raw = CSVDataset(csv_dirs=args.train_csv_dirs, file_list_path=args.file_list_path,
                                   class_idx=args.class_idx, transform=None, is_predict_dataset=False, id_distinguished=args.id_distinguished)

        if args.add_data_transformation:
            train_dataset_transform = CSVDataset(csv_dirs= args.train_csv_dirs, file_list_path=args.file_list_path, class_idx=args.class_idx, transform=ClipAndExtend(), is_predict_dataset=False,id_distinguished=args.id_distinguished)

        else:
            train_dataset_transform=None

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

        if args.add_data_transformation:
            train_loader_transform = DataLoader(train_dataset_transform, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=collate_fn)
        else:
            train_loader_transform=None

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
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = load_model(args)
        model = load_model_parameters(model, device=args.device,model_parameters_path=args.task_model)


        test(n_classes=args.n_classes,
                model=model,
                device=args.device,
                test_loader=test_loader,
                result_dir=args.result_dir,
                type=args.dataset,
                n_files=len(test_dataset))

    elif args.mode == 'predict':

        make_deterministic()

        os.makedirs(args.result_dir, exist_ok=True)
        args.test_csv_dir = list(map(str, args.test_csv_dir.split()))

        if args.summary_window_minutes !=0:

            original_files=get_predicting_files(args.test_csv_dir)
            new_dir=os.path.join(args.result_dir,f'{args.dataset}_continuous_result_segments_{args.summary_window_minutes}min')
            os.makedirs(new_dir,exist_ok=True)

            if args.dataset=='SPIKES':
                segment_continuous_result_by_min(file_path_list=original_files, output_dir=new_dir,dataset_type= args.dataset, rows_per_unit_min=args.summary_window_minutes, padding=False)

            else:
                segment_continuous_result_by_min(file_path_list=original_files, output_dir=new_dir, dataset_type= args.dataset, rows_per_unit_min=args.summary_window_minutes, padding=True)

            args.test_csv_dir=[new_dir]

        if args.dataset.startswith('SLEEP'):
            summarize_sleep_eeg_level_results(dataset_type=args.dataset,train_csv_dirs=args.test_csv_dir,result_dir=args.result_dir)


        else:
            predict_dataset = CSVDataset(args.test_csv_dir,class_idx=args.class_idx,is_predict_dataset=True)

            model = load_model(args)

            model=load_model_parameters(model,device=args.device,model_parameters_path=args.task_model)

            test_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)

            if method=='predict':
                predict(args=args,
                        input_dim=args.input_dim,
                        n_classes=args.n_classes,
                        class_idx=args.class_idx,
                        model=model,
                        device=args.device,
                        test_loader=test_loader,
                        result_dir=args.result_dir,
                        type=args.dataset,
                        n_files=len(predict_dataset),
                        event_precision=event_precision)

            elif method=='predict_based_10min':
                predict_based_10min(
                    args=args,
                    input_dim=args.input_dim,
                    n_classes=args.n_classes,
                    class_idx=args.class_idx,
                    model=model,
                    device=args.device,
                    test_loader=test_loader,
                    result_dir=args.result_dir,
                    type=args.dataset,
                    n_files=len(predict_dataset),
                    event_precision=event_precision)

    else:
        print('mode input error')
        return


if __name__ == "__main__":
    main()