"""
RNN para Análisis de Sentimientos - Proyecto NLP 2
Ejecutar en ambiente local con GPU
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# ============================================================================
# 1. CONFIGURACIÓN Y VERIFICACIÓN DE GPU
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Dispositivo: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ============================================================================

print("\n" + "="*80)
print("📂 CARGANDO DATOS")
print("="*80)

# Cargar datos
df = pd.read_csv('twitter_data_clean.csv')
print(f"✅ Dataset cargado: {len(df)} muestras")
print(f"📊 Distribución de clases:\n{df['category'].value_counts()}")

# Mapear clases a índices
class_mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
df['label'] = df['category'].map(class_mapping)

# ============================================================================
# 3. TOKENIZACIÓN Y VOCABULARIO
# ============================================================================

print("\n" + "="*80)
print("🔤 CREANDO VOCABULARIO")
print("="*80)

class Tokenizer:
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        
    def fit(self, texts):
        # Contar frecuencias
        for text in texts:
            for word in text.split():
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Seleccionar palabras más frecuentes
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_vocab_size-2], start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"✅ Vocabulario creado: {len(self.word2idx)} palabras")
        
    def encode(self, text, max_length=100):
        tokens = [self.word2idx.get(word, 1) for word in text.split()]
        # Padding o truncamiento
        if len(tokens) < max_length:
            tokens = tokens + [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

# Crear tokenizer
tokenizer = Tokenizer(max_vocab_size=5000)
tokenizer.fit(df['clean_text'].values)

# Codificar textos
MAX_LENGTH = 100
X = np.array([tokenizer.encode(text, MAX_LENGTH) for text in df['clean_text']])
y = df['label'].values

print(f"📏 Longitud de secuencia: {MAX_LENGTH}")
print(f"📊 X shape: {X.shape}")
print(f"🎯 y shape: {y.shape}")

# ============================================================================
# 4. DIVISIÓN DE DATOS
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# 5. DATASET Y DATALOADER
# ============================================================================

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================================
# 6. MODELO RNN
# ============================================================================

print("\n" + "="*80)
print("🧠 DEFINIENDO MODELO RNN")
print("="*80)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3):
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.rnn(embedded)  # output: [batch_size, seq_len, hidden_dim]
        
        # Usar el último estado oculto
        hidden = hidden[-1]  # [batch_size, hidden_dim]
        hidden = self.dropout(hidden)
        
        return self.fc(hidden)  # [batch_size, output_dim]

# Hiperparámetros
VOCAB_SIZE = len(tokenizer.word2idx)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
DROPOUT = 0.3

model = RNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
model = model.to(device)

print(f"✅ Modelo creado y movido a {device}")
print(f"📊 Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 7. ENTRENAMIENTO
# ============================================================================

print("\n" + "="*80)
print("🚀 ENTRENANDO MODELO RNN")
print("="*80)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

# Entrenamiento
N_EPOCHS = 10
train_losses = []
train_accs = []
test_losses = []
test_accs = []

start_time = time.time()

for epoch in range(N_EPOCHS):
    print(f"\n📍 Epoch {epoch+1}/{N_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"   Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")

training_time = time.time() - start_time
print(f"\n⏱️  Tiempo de entrenamiento: {training_time/60:.2f} minutos")

# ============================================================================
# 8. EVALUACIÓN FINAL
# ============================================================================

print("\n" + "="*80)
print("📊 EVALUACIÓN FINAL")
print("="*80)

_, final_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

# Mapeo inverso de clases
reverse_mapping = {0: -1, 1: 0, 2: 1}
y_true_original = [reverse_mapping[label] for label in y_true]
y_pred_original = [reverse_mapping[label] for label in y_pred]

print(f"\n✅ Accuracy: {final_acc*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_true_original, y_pred_original, 
                          target_names=['Negativo', 'Neutral', 'Positivo']))

# ============================================================================
# 9. VISUALIZACIONES
# ============================================================================

# Gráfica de pérdida y accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(train_losses, label='Train Loss', marker='o')
axes[0].plot(test_losses, label='Test Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('RNN - Pérdida durante Entrenamiento')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot([acc*100 for acc in train_accs], label='Train Accuracy', marker='o')
axes[1].plot([acc*100 for acc in test_accs], label='Test Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('RNN - Accuracy durante Entrenamiento')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica guardada: rnn_training_history.png")

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_original, y_pred_original)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negativo', 'Neutral', 'Positivo'],
            yticklabels=['Negativo', 'Neutral', 'Positivo'])
plt.title('RNN - Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('rnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Matriz de confusión guardada: rnn_confusion_matrix.png")

# ============================================================================
# 10. GUARDAR MODELO
# ============================================================================

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
        'max_length': MAX_LENGTH
    },
    'results': {
        'accuracy': final_acc,
        'training_time': training_time
    }
}, 'rnn_model.pth')

print("\n💾 Modelo guardado: rnn_model.pth")

# ============================================================================
# 11. ANÁLISIS DE LIMITACIONES
# ============================================================================

print("\n" + "="*80)
print("🔍 ANÁLISIS DE LIMITACIONES DE RNN")
print("="*80)

# Comparar secuencias largas vs cortas
df_test = pd.DataFrame({
    'text': [df['clean_text'].iloc[i] for i in range(len(X_test))],
    'length': [len(text.split()) for text in [df['clean_text'].iloc[i] for i in range(len(X_test))]],
    'true': y_true_original,
    'pred': y_pred_original
})

df_test['correct'] = df_test['true'] == df_test['pred']

# Agrupar por longitud
bins = [0, 20, 50, 100, 300]
df_test['length_bin'] = pd.cut(df_test['length'], bins=bins, labels=['Corto (0-20)', 'Medio (21-50)', 'Largo (51-100)', 'Muy largo (100+)'])

accuracy_by_length = df_test.groupby('length_bin')['correct'].mean()

print("\n📏 Accuracy por longitud de secuencia:")
print(accuracy_by_length)

# Graficar
plt.figure(figsize=(10, 6))
accuracy_by_length.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('RNN - Accuracy según Longitud de Texto')
plt.xlabel('Longitud del texto')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('rnn_length_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Análisis de longitud guardado: rnn_length_analysis.png")

print("\n" + "="*80)
print("✅ ANÁLISIS RNN COMPLETADO")
print("="*80)
print(f"📊 Accuracy Final: {final_acc*100:.2f}%")
print(f"⏱️  Tiempo Total: {training_time/60:.2f} minutos")
print(f"🖥️  Dispositivo usado: {device}")