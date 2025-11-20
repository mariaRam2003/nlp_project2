# ============================================================================
# LSTM para Análisis de Sentimientos - Proyecto NLP 2
# Ejecutar en Google Colab con GPU
# ============================================================================

# ==========================
# CELDA 1: Setup y verificación de GPU
# ==========================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("GPU no disponible. Verifica Runtime → Change runtime type → GPU")

# ==========================
# CELDA 2: Subir archivo CSV
# ==========================

from google.colab import files
print("Por favor sube tu archivo 'twitter_data_clean.csv'")
uploaded = files.upload()

# ==========================
# CELDA 3: Carga y exploración de datos
# ==========================

print("\n" + "="*80)
print("CARGANDO DATOS")
print("="*80)

df = pd.read_csv('twitter_data_clean.csv')
print(f"Dataset cargado: {len(df)} muestras")
print(f"\nDistribución de clases:")
print(df['category'].value_counts())
print(f"\nPrimeras filas:")
print(df.head())

# Mapear clases
class_mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
df['label'] = df['category'].map(class_mapping)

# ==========================
# CELDA 4: Tokenización y vocabulario
# ==========================

print("\n" + "="*80)
print("CREANDO VOCABULARIO")
print("="*80)

class Tokenizer:
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        
    def fit(self, texts):
        for text in texts:
            for word in text.split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_vocab_size-2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulario creado: {len(self.word2idx)} palabras")
        
    def encode(self, text, max_length=100):
        tokens = [self.word2idx.get(word, 1) for word in text.split()]
        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

tokenizer = Tokenizer(max_vocab_size=5000)
tokenizer.fit(df['clean_text'].values)

MAX_LENGTH = 100
X = np.array([tokenizer.encode(text, MAX_LENGTH) for text in df['clean_text']])
y = df['label'].values

print(f"Longitud de secuencia: {MAX_LENGTH}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ==========================
# CELDA 5: División de datos y DataLoader
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.LongTensor(texts)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"DataLoaders creados")
print(f"   Batches de entrenamiento: {len(train_loader)}")
print(f"   Batches de prueba: {len(test_loader)}")

# ==========================
# CELDA 6: Modelo LSTM
# ==========================

print("\n" + "="*80)
print("DEFINIENDO MODELO LSTM")
print("="*80)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Si es bidireccional, hidden_dim se duplica
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Si es bidireccional, concatenar último hidden de ambas direcciones
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
            
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Hiperparámetros
VOCAB_SIZE = len(tokenizer.word2idx)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
DROPOUT = 0.4
BIDIRECTIONAL = True

model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                       N_LAYERS, DROPOUT, BIDIRECTIONAL)
model = model.to(device)

print(f"Modelo LSTM creado")
print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
print(f"Bidireccional: {BIDIRECTIONAL}")

# ==========================
# CELDA 7: Funciones de entrenamiento y evaluación
# ==========================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in tqdm(loader, desc="Entrenando"):
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping para evitar explosión de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels

# ==========================
# CELDA 8: Entrenamiento
# ==========================

print("\n" + "="*80)
print("ENTRENANDO MODELO LSTM")
print("="*80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=2)

N_EPOCHS = 15
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_test_acc = 0

start_time = time.time()

for epoch in range(N_EPOCHS):
    print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"   Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
    
    # Guardar mejor modelo
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        print(f"   Mejor modelo guardado (acc: {best_test_acc*100:.2f}%)")

training_time = time.time() - start_time
print(f"\n⏱️  Tiempo de entrenamiento: {training_time/60:.2f} minutos")

# ==========================
# CELDA 9: Evaluación final
# ==========================

print("\n" + "="*80)
print("EVALUACIÓN FINAL")
print("="*80)

# Cargar mejor modelo
model.load_state_dict(torch.load('best_lstm_model.pth'))

_, final_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

# Mapeo inverso
reverse_mapping = {0: -1, 1: 0, 2: 1}
y_true_original = [reverse_mapping[label] for label in y_true]
y_pred_original = [reverse_mapping[label] for label in y_pred]

print(f"\nAccuracy Final: {final_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true_original, y_pred_original, 
                          target_names=['Negativo', 'Neutral', 'Positivo']))

# Calcular métricas adicionales
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_original, y_pred_original, average='weighted'
)

print(f"\nMétricas Generales:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-Score: {f1:.4f}")

# ==========================
# CELDA 10: Visualizaciones
# ==========================

# Gráfica de pérdida y accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(test_losses, label='Test Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('LSTM - Pérdida durante Entrenamiento')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([acc*100 for acc in train_accs], label='Train Accuracy', marker='o')
axes[1].plot([acc*100 for acc in test_accs], label='Test Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('LSTM - Accuracy durante Entrenamiento')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Matriz de confusión
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_original, y_pred_original)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negativo', 'Neutral', 'Positivo'],
            yticklabels=['Negativo', 'Neutral', 'Positivo'],
            cbar_kws={'label': 'Cantidad'})
plt.title('LSTM - Matriz de Confusión', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# CELDA 11: Análisis comparativo por longitud
# ==========================

print("\n" + "="*80)
print("ANÁLISIS POR LONGITUD DE SECUENCIA")
print("="*80)

# Crear DataFrame con resultados
test_indices = range(len(X_test))
df_test = pd.DataFrame({
    'length': [len(df['clean_text'].iloc[i].split()) for i in test_indices],
    'true': y_true_original,
    'pred': y_pred_original
})

df_test['correct'] = df_test['true'] == df_test['pred']

# Agrupar por longitud
bins = [0, 20, 50, 100, 300]
df_test['length_bin'] = pd.cut(df_test['length'], bins=bins, 
                                labels=['Corto (0-20)', 'Medio (21-50)', 
                                       'Largo (51-100)', 'Muy largo (100+)'])

accuracy_by_length = df_test.groupby('length_bin')['correct'].mean()

print("\nAccuracy por longitud de secuencia:")
for length_bin, acc in accuracy_by_length.items():
    print(f"   {length_bin}: {acc*100:.2f}%")

# Graficar
plt.figure(figsize=(10, 6))
accuracy_by_length.plot(kind='bar', color='lightcoral', edgecolor='black', alpha=0.8)
plt.title('LSTM - Accuracy según Longitud de Texto', fontsize=14, fontweight='bold')
plt.xlabel('Longitud del texto', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45)
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('lstm_length_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# CELDA 12: Guardar resultados
# ==========================

# Guardar modelo completo
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': tokenizer.word2idx,
    'hyperparameters': {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': OUTPUT_DIM,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'bidirectional': BIDIRECTIONAL,
        'max_length': MAX_LENGTH
    },
    'results': {
        'accuracy': final_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time
    },
    'history': {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }
}, 'lstm_model_complete.pth')

print("Modelo completo guardado: lstm_model_complete.pth")

# Descargar archivos
from google.colab import files
files.download('lstm_model_complete.pth')
files.download('lstm_training_history.png')
files.download('lstm_confusion_matrix.png')
files.download('lstm_length_analysis.png')

print("\nArchivos descargados exitosamente")

# ==========================
# CELDA 13: Resumen final
# ==========================

print("\n" + "="*80)
print("RESUMEN FINAL - LSTM")
print("="*80)
print(f"Accuracy: {final_acc*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Tiempo de entrenamiento: {training_time/60:.2f} minutos")
print(f"Dispositivo: {device}")
print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
print(f"Bidireccional: {BIDIRECTIONAL}")
print("="*80)