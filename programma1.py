import sys
import os.path
import codecs
import re
import nltk
import xml.etree.ElementTree as ET

#Programma1: Confronta i 2 corpus sulla base delle seguenti informazioni statistiche:

# Nota: I corpus delle recensioni si trovano i 2 file xml, le recensioni comprendono:
# testo della recensione, titolo e valutazione che la persona assegna al prodotto o servizio.
# Questo programma usa nltk e i moduli averaged_perceptron_tagger e punktk
# python programma1.py nomeFile1.xml nomeFile2.xml

#---Apre il file
def rawTextFromFile(fileName):
    fileInput = codecs.open(fileName, "r", "utf-8") #Apro il file che viene passato da riga di comando
    raw = fileInput.read() #Carico tutto il file in memoria
    return raw #Ritorno il testo alla funzione chiamante

#---Part of Speech tagging
def POSTaggingAnalyze(nodes):
    bigList = list()
    for child in nodes: #Per ogni nodo prendi la recensione
        for sentences in nltk.sent_tokenize(child.text): #Per ogni recensione prendi le frasi
            words = nltk.word_tokenize(sentences) #Ogni frase la divido in token
            tagged = nltk.pos_tag(words) #Part of Speech tagging per ogni parola
            bigList += tagged #Inserisco ogni risultato dell'iterazione in una lista, quindi ad esempio ('buy',VB)
    nomi = 0
    verbi = 0
    aggettivi = 0
    #Sfoglio la lista con i tag per avere un'idea approssimativa del numero di certi elementi nelle frasi
    for elem in bigList:
        if(elem[1] == "NN" or elem[1] == "NNP" or elem[1] == "NNS" or elem[1] == "NNPS"):
            nomi += 1
        if(elem[1] == "VB" or elem[1] == "VBD" or elem[1] == "VBG" or elem[1] == "VBN" or elem[1] == "VBP" or elem[1] == "VBZ"):
            verbi += 1
        if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS"):
            aggettivi += 1
    print("Nomi: " + str(nomi) + " - Aggettivi: " + str(aggettivi) + " - Verbi: " + str(verbi))

#---Numero totale di frasi presenti nel testo
# ---Parametro nodes: Poichè quello che gli passo è una struttura a lista e quelli che sfoglio li ho chiamati nodi
def totFrasi(nodes):
    recens = 0
    for child in nodes: #Per ogni elemento prendo il testo (recensione)
        recens += len(nltk.sent_tokenize(child.text)) #Sommo il numero di frasi
    return recens

#---Numero totale di token presenti nel testo
def totToken(nodes):
    recens = 0
    for child in nodes: #Per ogni elemento prendo il testo (recensione)
        recens += len(nltk.word_tokenize(child.text)) #Sommo il numero di token
    return recens

#---Lunghezza media delle frasi in termini di token
def avgFrasi(nodes):
    totWord = 0
    numFrasi = 0
    for child in nodes: #Prendo una frase alla volta
        txtSentenced = nltk.sent_tokenize(child.text) #Divido la recensione in frasi e salvo in una variabile
        numFrasi += len(txtSentenced) #Salvo il numero di frasi per quella recensione e lo sommo al numero precedente
        for token in txtSentenced: #Per ogni frase effettuo la tokenizzazione
            totWord += len(nltk.word_tokenize(token)) #Quindi suddivido la frase in token e sommo la lunghezza della lista ottenuta al valore precedente
    return ("%.3f" %(totWord/numFrasi)) #Ritorno il numero totale di token diviso il numero totale di frasi presenti nel corpus

#---Lunghezza media dei token in termini di caratteri
def avgToken(nodes):
    totToken = 0
    totChar = 0
    for child in nodes: #Per ogni recensione presente nel corpus prendo il testo
        txtTokenized = nltk.word_tokenize(child.text)
        totToken += len(txtTokenized)
        for chars in txtTokenized: #Per ogni token ottenuto su quel testo..
            totChar += len(chars) #Sommo la lunghezza del token in termini di caratteri al valore precedente
    return (totChar/totToken)

#---Lunghezza media delle parole in termini di caratteri
def avgParole(nodes):
    totWord = 0
    totChar = 0
    for child in nodes: #Per ogni recensione presente nel corpus prendo il testo
        txtTokenized = re.findall(r'\w+',child.text) #Uso le espressioni regolari per prendere solo parole senza punteggiatura, anche se così non prendo le sigle U.S.A ecc..
        totWord += len(txtTokenized) #...però non sono parole sono sigle e nemmo il prezzo è una parola, anche se prende i numeri e non prende le abbreviazioni come parole 's, n't ecc..
        for chars in txtTokenized: #Per ogni token ottenuto su quel testo..
            totChar += len(chars) #Sommo la lunghezza del token in termini di caratteri al valore precedente
    return ("%.3f" % (totChar/totWord))

#---Dimensione del vocabolario
def Vt(nodes):
    corpus = "" #Questa funzione ritorna la dimensione del vocabolario sul corpus
    for child in nodes:
        corpus += child.text #Carico tutte le recensioni pulite senza xml in memoria
    return len(dict.fromkeys(nltk.word_tokenize(corpus))) #Il dizionario non ripete le parole quindi dala lista di parole ottengo un dizionario e poi prendo la dimensione

#---Type Token Ratio complessiva del testo
def ttr(nodes):
    corpus = ""
    for child in nodes:
        corpus += child.text
    listaWord = nltk.word_tokenize(corpus)
    vocaLen = len(dict.fromkeys(listaWord))#Stessa cosa di Vt con la differenza che salvo la lista di token del corpus dentro una variabile per prenderne la dimensione
    return ("%.6f" % (vocaLen/len(listaWord)))#Ritorno la ttr del testo approssimata a 6 unità dopo la virgola

#---Type Token Ratio incrementando di 1000 in 1000 token
def ttrIncr(nodes):
    corpus = ""
    i=1000
    vocaLen = 0
    for child in nodes:
        corpus += child.text
    listaWord = nltk.word_tokenize(corpus)
    lista = list()
    for i in range(i,len(listaWord),1000):
        vocaLen = len(dict.fromkeys(listaWord[0:i]))
        lista.append("%.6f" % (vocaLen/len(listaWord[0:i])))
    return lista

#Funzione per la grandezza delle classi di frequenza
def freqToken(nodes,numWord):
    corpus = ""
    for child in nodes:
        corpus += child.text
    listaWord = nltk.word_tokenize(corpus)
    count=0
    freq = 0
    for i in range(0,5000): #Per ogni token presente nel corpus
        for j in range((i+1),5000): #Lo confronto con gli altri fino a 5000
            if(listaWord[i] == listaWord[j]): #Se trovo una corrispondenza
                count+=1 #Aumento il conteggio
                if(count>numWord): #Se il conteggio è più alto di quello che sto cercando allora interrompo il ciclo
                    break #Ex. la parola si ripete 4 volte e io cercavo solo quelle che si ripetono 3
        if(count==numWord):
            freq+=1 #Se invece in tutto il testo trovo una parola che si ripete solamente 3 volte, incremento freq
        count=0
    return freq

#Funzione per il numero medio di sostantivi, aggettivi e verbi per frase
def avgNVA(nodes):
    bigList = list()
    count=0
    for child in nodes: #Per ogni nodo prendi la recensione
        for sentences in nltk.sent_tokenize(child.text): #Per ogni recensione prendi le frasi
            words = nltk.word_tokenize(sentences) #Ogni frase la divido in token
            tagged = nltk.pos_tag(words) #Part of Speech tagging per ogni parola
            bigList += tagged #Ogni frase la divido in token
            nomi = 0
            verbi = 0
            aggettivi = 0
            #Sfoglio la lista con i tag per avere un'idea approssimativa del numero di certi elementi nelle frasi
            for elem in bigList:
                if(elem[1] == "NN" or elem[1] == "NNP" or elem[1] == "NNS" or elem[1] == "NNPS"):
                    nomi += 1
                if(elem[1] == "VB" or elem[1] == "VBD" or elem[1] == "VBG" or elem[1] == "VBN" or elem[1] == "VBP" or elem[1] == "VBZ"):
                    verbi += 1
                if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS"):
                    aggettivi += 1
            count+=1
    return [nomi/count,aggettivi/count,verbi/count]

def densLess(nodes):
    bigList = list()
    count=0
    for child in nodes: #Per ogni nodo prendi la recensione
        for sentences in nltk.sent_tokenize(child.text): #Per ogni recensione prendi le frasi
            words = nltk.word_tokenize(sentences) #Ogni frase la divido in token
            tagged = nltk.pos_tag(words) #Part of Speech tagging per ogni parola
            bigList += tagged #Ogni frase la divido in token
            nomi = 0
            verbi = 0
            aggettivi = 0
            avverbi = 0
            punt = 0
            #Sfoglio la lista con i tag per decidere cosa contare
            for elem in bigList:
                if(elem[1] == "NN" or elem[1] == "NNP" or elem[1] == "NNS" or elem[1] == "NNPS"):
                    nomi += 1
                if(elem[1] == "VB" or elem[1] == "VBD" or elem[1] == "VBG" or elem[1] == "VBN" or elem[1] == "VBP" or elem[1] == "VBZ" or elem[1] == "TO"):
                    verbi += 1
                if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS"):
                    aggettivi += 1
                if(elem[1] == "RB" or elem[1] == "RBR"  or elem[1] == "RBS"):
                    avverbi += 1
                if(elem[0] == "." or elem[0] == "," or elem[0] == ";" or elem[0] == ":" or  elem[0] == "!" or  elem[0] == "?" or  elem[0] == " "):
                    punt+=1
            count+=len(tagged)
    return (nomi+verbi+aggettivi+avverbi)/(count-punt)

def coolPrint(stringa,dato1,dato2):
    totSpazi = 56-len(stringa) #Gli spazi che ci devono essere dopo la stringa Tipo Operazione
    for i in range(0,totSpazi):
        stringa += " "
    stringa +="|"
    totSpazi = (30 - len(dato1))
    for i in range(0,int(totSpazi/2)): #Metto gli spazi prima del dato1
        stringa += " "
    stringa += dato1
    totSpazi = (27 - len(dato1))
    for i in range(0,int(totSpazi/2)): #Metto gli spazi dopo il dato1 e infine metto il dato 2
        stringa += " "
    stringa += ("|         "+dato2)
    print(stringa)

def main(file1,file2):
    if(os.path.isfile(file1) and os.path.isfile(file2)):
        root = ET.fromstring(rawTextFromFile(file1)) #Passo il file in formato testo xml al parser xml in modo che possa lavorarci come in un albero
        x = root.findall(".//div/p/.") #Trovo tutti i <p> dentro un <div> usando xpath perchè è li che ho messo le recensioni
        #Quindi ottengo una variabile x di recensioni del file1 che posso sfogliare con un for perchè dentro struttura dati (lista o meglio albero)
        root = ET.fromstring(rawTextFromFile(file2))
        y = root.findall(".//div/p/.") #Faccio la stessa cosa che ho fatto sul file1 per il file2 e ottengo le rensioni del file2 dentro y      
        print("Informazioni sul file: " + file1)
        POSTaggingAnalyze(x)
        print("Informazioni sul file: " + file2)
        POSTaggingAnalyze(y)
        print("\n\n-------------------Tipo di operazione-------------------|--------"+file1+"---------|--------"+file2+"--------")
        coolPrint("Numero di recensioni in entrambi i file: ",str(len(x)),str(len(y)))
        coolPrint("Numero totale di frasi: ",str(totFrasi(x)),str(totFrasi(y)))
        coolPrint("Numero totale di parole: ",str(totToken(x)),str(totToken(y)))
        coolPrint("Lunghezza media delle frasi in termini di token: ",str(avgFrasi(x)),str(avgFrasi(y)))
        coolPrint("Lunghezza media dei token in termini di caratteri: ",str(avgParole(x)),str(avgParole(y)))
        coolPrint("Grandezza del vocabolario: ",str(Vt(x)),str(Vt(y)))
        coolPrint("Type Token Ratio complessiva: ",str(ttr(x)),str(ttr(y)))
        print("¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯")
        print("\n\nType Token Ratio di 1000 in 1000 token: ")
        print("Numero di parole-----|-----"+file1+"------|-----"+file2+"-----")
        ttrFile1 = ttrIncr(x)
        ttrFile2 = ttrIncr(y)
        lenttrmax = len(ttrFile1) if(len(ttrFile1)>len(ttrFile2)) else len(ttrFile2)
        for i in range(0,lenttrmax):
            print(((str(1000*(i+1))+" ")if((1000*(i+1))<10000) else str(1000*(i+1)))+"                |        "+(str(ttrFile1[i]) if(len(ttrFile1)>i) else " ")+"      |"+"       "+(str(ttrFile2[i]) if(len(ttrFile2)>i) else " "))
        print("\n\nGrandezza delle classi di frequenza V3, V6, V9 su 5000 token: ")
        print("-----"+file1+"-----|-----"+file2+"-----")
        print("V3:        " + str(freqToken(x,3))+"       |         "+ str(freqToken(y,3)))
        print("V6:        " + str(freqToken(x,6))+"       |         "+ str(freqToken(y,6)))
        print("V9:        " + str(freqToken(x,9))+"        |         "+ str(freqToken(y,9)))
        lisNVAX = avgNVA(x)
        lisNVAY = avgNVA(y)
        print("\n\n-------------------Tipo di operazione-------------------|--------"+file1+"---------|--------"+file2+"--------")
        coolPrint("Numero medio di sostantivi per frase: ",str("%.3f"%lisNVAX[0]),str("%.3f"%lisNVAY[0]))
        coolPrint("Numero medio di aggettivi per frase: ",str("%.3f"%lisNVAX[1]),str("%.3f"%lisNVAY[1]))
        coolPrint("Numero medio di verbi per frase: ",str("%.3f"%lisNVAX[2]),str("%.3f"%lisNVAY[2]))
        coolPrint("Densità lessicale: ",str("%.3f"%densLess(x)),str("%.3f"%densLess(y)))
    else:
        print("Inserire percorsi validi per i due file corpus!")
main(sys.argv[1],sys.argv[2])
#Alessandro Mazzeo