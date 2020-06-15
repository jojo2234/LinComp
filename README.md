# LinComp
Computational linguistics project for university. The file named readme is written in Italian.

## Informazioni
Scopo del progetto portare a termine le istruzioni contenute nel file pdf. Realizzare due programmi in Python che facciano uso dei moduli presenti in Natural Language Toolkit.

## Struttura delle directory
```
corpus1.xml: Un corpus di recensioni in lingua inglese in formato XML codificato con standard TEI.
corpus2.xml: Un altro corpus di recensioni in lingua inglese in formato XML codificato con standard TEI.
DirettiveProgetto.pdf: Il documento che contiene le informazioni su cosa deve fare il programma.
output1.xml: Il primo risultato come richiesto dalle direttive.
output2.xml: Il secondo risultato come richiesto dalle direttive.
programma1.py: Il primo programma scritto in python
programma2.py: Il secondo programma scritto in python
tei_corpus.dtd: La document type definition in standard TEI
```

## Risoluzione dipendenze
Da terminale (o cmd su Windows) dare
```
pip install nltk
python
import nltk
nltk.download()
```
Nella finestra che si apre per evitare problemi consiglio di installare tutti i pacchetti così potrete utilizzarli anche con altri software.

## Note
```
Programmi avviati da cmd su Windows, con Python versione 3.8 (32bit)
python programma1.py corpus1.xml corpus2.xml
python programma2.py corpus1.xml corpus2.xml
Output generato da cmd cambiando la codifica in UTF-8 con "chcp 65001" in questo modo:
python programma1.py corpus1.xml corpus2.xml >> output1.txt
python programma2.py corpus1.xml corpus2.xml >> output2.txt
```
