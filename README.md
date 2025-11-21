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


======================================================================
RESULTADOS DEL ENTRENAMIENTO CON ATTENTION
======================================================================

Mejor época: 1
Train Loss: 2.6744
Val Loss: 2.4644
Train Accuracy: 0.6385
Val Accuracy: 0.6502


================================================================================
EJEMPLOS DE TEXTOS MODIFICADOS COREFERENCIA
================================================================================

1. Original: one vote can make all the difference anil kapoor answers modis election 2019 clarion call extends support his vote kar campaign 
   Resuelto: one vote can make all the difference anil kapoor answers modis election 2019 clarion call extends support 2019 vote kar campaign 
   Cambios: [{'pronoun': 'his', 'entity': '2019'}]

2. Original: one vote can make all the difference anil kapoor answers modis election 2019 clarion call extends support his campaign 
   Resuelto: one vote can make all the difference anil kapoor answers modis election 2019 clarion call extends support 2019 campaign 
   Cambios: [{'pronoun': 'his', 'entity': '2019'}]

3. Original: was the one who recently said that people who vote against modi are anti national that put gen hooda all congress supporters and those jawans who not support modi anti national what great things did you hear about him
   Resuelto: was the one who recently said that people who vote against modi are anti national that put gen hooda all congress supporters and those jawans who not support modi anti national what great things did you hear about congress
   Cambios: [{'pronoun': 'him', 'entity': 'congress'}]

4. Original: india second most optimistic globally about executive job growth shows the survey indias senior executives said that they are optimistic about the growth the number job roles this year 
   Resuelto: india second most optimistic globally about executive job growth shows the survey indias senior executives said that second are optimistic about the growth the number job roles this year 
   Cambios: [{'pronoun': 'they', 'entity': 'second'}]

5. Original: people wish your vision india and least interested about your personal enmity with modi others its your personal problem handle this personally and dont expect nation will join your dirty fight with others tell why vote 
   Resuelto: people wish your vision india and least interested about your personal enmity with modi others india your personal problem handle this personally and dont expect nation will join your dirty fight with others tell why vote 
   Cambios: [{'pronoun': 'its', 'entity': 'india'}]

6. Original: this the new india modi trying build with these leaders his party why have live with these deplorable characters 
   Resuelto: tindia the new india modi trying build with these leaders his party why have live with these deplorable characters 
   Cambios: [{'pronoun': 'his', 'entity': 'india'}]

7. Original: entrepreneurs are rising india after modi govt created system for them took care their tax concerns and created infra for them incubate well never happened congress you guys just want power sit and shit its simple for you
   Resuelto: entrepreneurs are rising india after modi govt created system for india took care india tax concerns and created infra for india incubate well never happened congress you guys just want power sit and shit congress simple for you
   Cambios: [{'pronoun': 'its', 'entity': 'congress'}, {'pronoun': 'them', 'entity': 'india'}, {'pronoun': 'their', 'entity': 'india'}, {'pronoun': 'them', 'entity': 'india'}]

8. Original: where ever rgis going through out the length breadth the country such the reception for him masses india just love him modi all other leaders bjp are just match
   Resuelto: where ever rgis going through out the length breadth the country such the reception for india masses india just love him modi all other leaders bjp are just match
   Cambios: [{'pronoun': 'him', 'entity': 'india'}]

9. Original: congress fed biryani terrorists modi government fed them bullets and bombs yogi adityanath 
   Resuelto: congress fed biryani terrorists modi government fed biryani bullets and bombs yogi adityanath 
   Cambios: [{'pronoun': 'them', 'entity': 'biryani'}]

10. Original: this face doesn’ haunt you condemn the abduction girls but they’ alive and wel also recorded message still our personally ordered action unlike modi who treats muslims just vote bank
   Resuelto: this face doesn’ haunt you condemn the abduction girls but doesn’ alive and wel also recorded message still our personally ordered action unlike modi who treats muslims just vote bank
   Cambios: [{'pronoun': 'they', 'entity': 'doesn'}]

================================================================================
PREPARANDO DATOS PARA MODELOS
================================================================================

Datasets disponibles:
1. Sin co-referencia: df['clean_text']
2. Con co-referencia: df['text_with_coref']

Creando vocabularios...
Vocabulario sin co-ref: 5000 palabras
Vocabulario con co-ref: 5000 palabras

Codificando textos (max_length=100)...
Base: 100%
 162969/162969 [00:01<00:00, 84517.37it/s]
Coref: 100%
 162969/162969 [00:00<00:00, 173131.45it/s]

Datos codificados:
X_base:  (162969, 100)
X_coref: (162969, 100)
y:       (162969,)

================================================================================
DIVISION TRAIN/TEST
================================================================================
Train: 130375
Test:  32594

================================================================================
DEFINIENDO MODELO LSTM
================================================================================
Dispositivo: cuda
Modelo definido

================================================================================
COMPARACION: SIN vs CON CO-REFERENCIA
================================================================================

======================================================================
ENTRENANDO: LSTM SIN Co-referencia
======================================================================
Epoch 2/10 | Test Acc: 93.61%
Epoch 4/10 | Test Acc: 95.53%
Epoch 6/10 | Test Acc: 95.88%
Epoch 8/10 | Test Acc: 96.13%
Epoch 10/10 | Test Acc: 96.01%

RESULTADOS:
Accuracy:  96.01%
Precision: 0.9602
Recall:    0.9601
F1-Score:  0.9599

======================================================================
ENTRENANDO: LSTM CON Co-referencia
======================================================================
Epoch 2/10 | Test Acc: 93.99%
Epoch 4/10 | Test Acc: 95.55%
Epoch 6/10 | Test Acc: 95.84%
Epoch 8/10 | Test Acc: 95.91%
Epoch 10/10 | Test Acc: 96.00%

RESULTADOS:
Accuracy:  96.00%
Precision: 0.9601
Recall:    0.9600
F1-Score:  0.9598

================================================================================
ANALISIS DE MEJORA
================================================================================

Mejora porcentual con Co-referencia:

Accuracy    :  -0.01%  (DISMINUCION)
Precision   :  -0.01%  (DISMINUCION)
Recall      :  -0.01%  (DISMINUCION)
F1-Score    :  -0.01%  (DISMINUCION)

================================================================================
GUARDANDO RESULTADOS
================================================================================
Dataset guardado: data_with_coreference.csv
Resultados guardados: coreference_results.csv

================================================================================
DESCARGANDO ARCHIVOS
================================================================================

Proceso completado