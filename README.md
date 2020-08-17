# Änderungen, die für die Optimierung der WER vorgenommen wurden

## Änderungen am Model

Einfügen eines Batch Normalisierungs Layers
Einfügen eines Dropout Layers mit 10% Dropout
Änderung des Optimierers zu 'Nadam'
Änderung der Batchsize auf 150


## Änderungen an der HMM-Klasse (durch diese Änderungen funktioniert Übung 10 nicht mehr, da diese das Default HMM benötigt)
Anzahl der Zustände beim Wort 'oh' auf 9 geändert
Anzahl der Zustände beim Wort 'eight' auf 9 geändert
Initiale Wahrscheinlichkeit beim Wort 'oh' auf 40 geändert
Initialie Wahrscheinlichkeit beim Wort 'eight' auf 80 geändert
