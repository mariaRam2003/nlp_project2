================================================================================
EVALUACIÓN FINAL LSTM
================================================================================

Accuracy Final: 96.02%

Classification Report:
              precision    recall  f1-score   support

    Negativo       0.94      0.92      0.93      7102
     Neutral       0.95      0.99      0.97     11042
    Positivo       0.97      0.96      0.97     14450

    accuracy                           0.96     32594
   macro avg       0.96      0.96      0.96     32594
weighted avg       0.96      0.96      0.96     32594


Métricas Generales:
   Precision: 0.9603
   Recall: 0.9602
   F1-Score: 0.9601


ANÁLISIS POR LONGITUD DE SECUENCIA
================================================================================
/tmp/ipython-input-2625941758.py:395: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  accuracy_by_length = df_test.groupby('length_bin')['correct'].mean()

Accuracy por longitud de secuencia:
   Corto (0-20): 96.08%
   Medio (21-50): 95.95%
   Largo (51-100): nan%
   Muy largo (100+): nan%


   Modelo completo guardado: lstm_model_complete.pth

Archivos descargados exitosamente

================================================================================
RESUMEN FINAL - LSTM
================================================================================
Accuracy: 96.02%
Precision: 0.9603
Recall: 0.9602
F1-Score: 0.9601
Tiempo de entrenamiento: 18.28 minutos
Dispositivo: cuda
Parámetros del modelo: 3,009,027
Bidireccional: True
================================================================================





================================================================================
EVALUACION FINAL RNN
================================================================================

Accuracy Final: 90.33%
Mejor Accuracy: 90.34%

Classification Report:
              precision    recall  f1-score   support

    Negativo       0.80      0.84      0.82      7102
     Neutral       0.93      0.98      0.95     11042
    Positivo       0.94      0.88      0.91     14450

    accuracy                           0.90     32594
   macro avg       0.89      0.90      0.89     32594
weighted avg       0.91      0.90      0.90     32594


Grafica: rnn_improved_training.png
Matriz: rnn_improved_confusion.png

================================================================================
COMPLETADO
================================================================================
Mejor Accuracy: 90.34%
Tiempo: 9.38 minutos