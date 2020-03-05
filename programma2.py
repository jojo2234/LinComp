import sys
import os.path
import codecs
import re
import nltk
import collections
import math
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer

#Apre i file di testo per la lettura
def rawTextFromFile(fileName):
    fileInput = codecs.open(fileName, "r", "utf-8") #Apro il file che viene passato da riga di comando
    raw = fileInput.read() #Carico tutto il file in memoria
    return raw #Ritorno il testo alla funzione chiamante

#Accede al file xml in modo da prendere solo le recensioni e inserisce tutto in una variabile che poi ritorna al chiamante
def loadFile(fileName):
    root = ET.fromstring(rawTextFromFile(fileName)) #Passo il file in formato testo xml al parser xml in modo che possa lavorarci come in un albero
    nodes = root.findall(".//div/p/.") #Trovo tutti i <p> dentro un <div> usando xpath perchè è li che ho messo le recensioni
    corpus = ""
    for child in nodes:
        corpus += child.text #Per ogni div/p prendo il testo che corrisponde alla recensione e lo salvo in una variabile insieme al resto delle recensioni
    return corpus

#Ritorna i 20 token più frequenti escludendo la punteggiatura
def freqToken(simTxt):
    #re.findall(r'\w+',simTxt) Devo prendere solo i token senza punteggiatura, quindi così no
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    punteggiatura = ["|","!","?","/","//",".",",",";",":","-","–","_","(",")","[","]","{","}","<",">","\"","'","^","’","...","``","''","“","”","‘","*"]
    clearTokens = list()
    for elem in tokens:
        if(elem not in punteggiatura):
            clearTokens.append(elem)#Per ogni elemento in tokens controllo che non sia parte della punteggiatura e poi lo inserisco dentro una lista
    freq = nltk.FreqDist(clearTokens)#FreqDist ritorna una forma di dizionario con frequenza da cui poi posso ottenere le frequenze degli elementi più comuni
    return freq.most_common(20)#Ritorno i 20 token che si ripetono di più

#Ritorna i 20 sostantivi più frequenti
def freqNomi(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione to, be, a, good, pop, corn...
    tagged = nltk.pos_tag(tokens)#Part of Speech tagging ex. (play,VB)
    sostantivi = list()
    for elem in tagged:
        if(elem[1] == "NN" or elem[1] == "NNS"):
            sostantivi.append(elem[0])#Usando il POS fatto prima controllo il secondo elemento (parola,POS) del vettore per vedere se è un sostantivo, quindi lo inserisco in una lista
    freq = nltk.FreqDist(sostantivi)#FreqDist ritorna una forma di dizionario con frequenza da cui poi posso ottenere le frequenze degli elementi più comuni
    return freq.most_common(20)#Ritorno i 20 sostantivi che si ripetono di più

#Ritorna i 20 aggettivi più frequenti
def freqAgge(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    tagged = nltk.pos_tag(tokens)#POS
    aggettivi = list()#Creo lista vuota di aggettivi che verrà usata dopo
    for elem in tagged:
        if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS"):
            aggettivi.append(elem[0])#Inserisco aggettivi nella lista confrontando con quanto riportato dall'attività di POS
    freq = nltk.FreqDist(aggettivi)#FreqDist ritorna una forma di dizionario con frequenza da cui poi posso ottenere le frequenze degli elementi più comuni
    return freq.most_common(20)#Ritorno i 20 aggettivi che si ripetono di più

#Ritorna i 20 bigrammi di token più frequenti che non contengono punteggiatura, articoli e congiunzioni
def freqBigrTK(simTxt):
    punteggiatura = ["|","!","?","/","//",".",",",";",":","-","–","_","(",")","[","]","{","}","<",">","\"","'","^","’","...","``","''","“","”","‘","*"]
    congiunzioni = ["and","or"]
    articoli = ["the","a","an"]#Preparo delle liste di punteggiatura, articoli e congiunzioni da usare per escludere gli elementi dopo
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    clearTokens = list()
    for elem in tokens:
        if(elem not in punteggiatura and elem not in congiunzioni and elem not in articoli):
            clearTokens.append(elem)#Inserisco dentro la lista clearTokens tutti gli elementi che non sono articoli cong. e punteggiatura
    freq = nltk.FreqDist(nltk.bigrams(clearTokens))#FreqDist ritorna una forma di dizionario con frequenza da cui poi posso ottenere le frequenze dei bigrammi più comuni
    return freq.most_common(20)#Ritorno i 20 bigrammi che si ripetono di più

#Ritorna i 10 part of speech più frequenti
def freqPOS(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    tagged = nltk.pos_tag(tokens)#Part of Speech tagging
    freq = nltk.FreqDist(tagged)#Frequenza elementi del testo già con POS
    return freq.most_common(10)#I 10 più comuni

#I 10 bigrammi di PoS più frequenti
def freqPOSBigr(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    tagged = nltk.pos_tag(tokens)#Part of Speech tagging
    freq = nltk.FreqDist(nltk.bigrams(tagged))#frequenza distribuita dei bigrammi sul POS
    return freq.most_common(10)

#I 10 trigrammi di PoS più frequenti
def freqPOSTigr(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    tagged = nltk.pos_tag(tokens)#Part of Speech tagging
    freq = nltk.FreqDist(nltk.trigrams(tagged))#frequenza distribuita dei bigrammi sul POS
    return freq.most_common(10)

#Funzione usata per il sorted nella funzione di seguito
def tks(elem):
    return elem[1]#Voglio ordinare gli elementi nella lista in base al numero di frequenza (bigramma,freq)

#-------------------ANNULLATA---------------------Rimane solo per fornire commenti d'ausilio su twentyBigramAS2
#I 20 bigrammi, aggettivo più sostantivo con frequenza > 2, di cui relativa frequenza per token, probabilità congiunta, forza associativa massima
def twentyBigramAS(simTxt):
    tokens = nltk.word_tokenize(simTxt)#Tokenizzazione
    tagged = nltk.pos_tag(tokens)#POS
    punteggiatura = ["|","!","?","/","//",".",",",";",":","-","–","_","(",")","[","]","{","}","<",">","\"","'","^","’","...","``","''","“","”","‘","*"]
    alfabeto = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    lista = list()
    for elem in tagged:
        if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS" or elem[1] == "NN" or elem[1] == "NNS"):
            if(elem[0] not in punteggiatura and elem[0] not in alfabeto): #Forse sbaglio qualcosa con il PoS però in questo modo risolvo il problema ' non è un aggettivo e t non è un nome
                lista.append(elem[0]) #Inserisco in una nuova lista solo i PoS che sono aggettivi o sostantivi
    listBigr = list()
    bigrEl1=0
    bigrEl2=0
    #Ho ottenuto fin'ora una lista di parole e relativa corrispondenza PoS (topo,NN)
    bigrammi = nltk.bigrams(lista) #Poi ottengo i bigrammi dalla lista menzionata sopra
    for bigram in bigrammi: #Per ogni bigramma:
        for i in range(0,len(lista)): #Per ogni token che compone il bigramma conto quante volte si ripete nel testo
            if(bigram[0] == lista[i]):
                bigrEl1+=1
            if(bigram[1] == lista[i]):
                bigrEl2+=1
        if(bigrEl1>2 and bigrEl2>2): #Prima di inserirlo in questa lista controllo che si ripeta più di 2 volte come richiesto
            appendimi = [bigram[0],bigrEl1,bigram[1],bigrEl2] #Inserisco i token che compongono il bigramma con relativa frequenza in una lista
            listBigr.append(appendimi)#Lista di [(token,frequenza),(token,frequenza)] compone un bigramma
        bigrEl1=0 #Azzero i contatori della frequenza di un token dentro un bigramma
        bigrEl2=0
    count=0
    newList = list()
    for elem in listBigr: #Conto quante volte un bigramma si ripete dentro la lista dei bigrammi che ho creato prima
        for j in range(0,len(listBigr)):
            if(elem==listBigr[j]):
                count += 1
        if([elem,count] not in newList): #Evito l'inserimento di duplicati
            newList.append([elem,count]) #Inserisco il bigramma e la sua frequenza in una nuova lista
        count=0
    #Rimuovo i duplicati dalla nuova lista
    ordinati = sorted(newList, key=tks, reverse=True)#Ordinato per frequenza del bigramma
    #In ordinati[i][0] c'è: parola1,freqP1,parola2,freqP2
    #In ordinati[i][1] c'è la frequenza del bigramma
    toPrint = ""
    for i in range(0,20):
        #ProbCongBigr=(FrequenzaAggettivo/NumTotParoleneiBigr)*(FreqBigrammi/FreqAggettivo)
        #P(agg) = freqAgg/TotNumAggeSost
        #P(sost)= freqSost/TotNumAggeSost
        #P(bigramma)= freqBigr/TotBigr
        #Lista: contiene gli aggettivi e i sostantivi
        #listBigr: contiene i bigrammi con relativa frequenza
        prob = (ordinati[i][0][1]/len(lista))*(ordinati[i][1]/ordinati[i][0][1])#Br(vA,vB)->P(A∩B)=P(A)*P(B|A)=>(freq(vA)/totPar)*(freq(vA,vB)/freq(vA))
        mi = math.log2((ordinati[i][1]/len(listBigr))/((ordinati[i][0][1]/len(lista))*(ordinati[i][0][3]/len(lista))))#log(p(v1,v2)/p(v1)*p(v2)) Ex. dove v1="il" e v2="cane"
        #mi = log2[(FreqBigrammi/TotBigrammi)/(FreqAgg/TotParoleNeiBigrammi)*(FreqSost/TotParoleNeiBigrammi)] NOTA: Anche se nella lista gli aggettivi e i sostantivi s'invertono non cambia nulla
        toPrint += (str(ordinati[i])+" - "+str("%.6f"%prob)+" - "+str("%.4f"%mi)+"\n")
    return toPrint

#Con questa funzione la frequenza dei bigrammi e delle parole la calcolo su tutto il corpus mentre con l'altra sulla sezione aggettivi e sostantivi
#Ordino per frequenza i 20 bigrammi composti da aggettivo e sostantivo
def twentyBigramAS2(simTxt):
    tokens = nltk.word_tokenize(simTxt) #Ottengo delle parole dal corpus inclusa la punteggiatura
    wordFreq = collections.Counter(tokens) #Uso counter per sapere la frequenza di ogni parola nel testo
    bigrFreq = collections.Counter(nltk.bigrams(tokens)) #Anche qui counter per ogni bigramma del corpus
    tagged = nltk.pos_tag(tokens)#POS
    punteggiatura = ["|","!","?","/","//",".",",",";",":","-","–","_","(",")","[","]","{","}","<",">","\"","'","^","’","...","``","''","“","”","‘","*"]
    alfabeto = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    lista = list()
    for elem in tagged:
        if(elem[1] == "JJ" or elem[1] == "JJR"  or elem[1] == "JJS" or elem[1] == "NN" or elem[1] == "NNS"):
            if(elem[0] not in punteggiatura and elem[0] not in alfabeto):
                lista.append(elem[0]) #Inserisco in una nuova lista solo i PoS che sono aggettivi o sostantivi
    listBigr = list()
    bigrammi = nltk.bigrams(lista)#Ottengo i bigrammi dalla lista di aggettivi e sostantivi
    for bigramma in bigrammi:
        if(wordFreq[bigramma[0]]>2 and wordFreq[bigramma[1]]>2): #Inserisco il bigramma in lista solo se si ripete più di due volte
            inseriscimi = [[bigramma[0],wordFreq[bigramma[0]],bigramma[1],wordFreq[bigramma[1]]],bigrFreq[bigramma]]
            if(inseriscimi not in listBigr): #Evito i duplicati
                listBigr.append(inseriscimi)#<un,7,gatto,5>,10 per ogni bigramma inserisco nella lista un'oggetto simile
    ordinati = sorted(listBigr, key=tks, reverse=True)#Ordino per frequenza la lista di bigrammi
    toPrint = ""
    for i in range(0,20):
        prob = (ordinati[i][0][1]/len(tokens))*(ordinati[i][1]/ordinati[i][0][1])#Br(vA,vB)->P(A∩B)=P(A)*P(B|A)=>(freq(vA)/totPar)*(freq(vA,vB)/freq(vA))
        mi = math.log2((ordinati[i][1]/len(list(nltk.bigrams(tokens))))/((ordinati[i][0][1]/len(tokens))*(ordinati[i][0][3]/len(tokens))))#log(p(v1,v2)/p(v1)*p(v2)) Ex. dove v1="il" e v2="cane"
        #mi = log2[(FreqBigrammi/TotBigrammi)/(FreqAgg/TotParoleNelCorpus)*(FreqSost/TotParoleNelCorpus)]
        #prob = (FreqAgg/totWord)*(FreqBigr/FreqAgg)
        toPrint += (str(ordinati[i])+" - "+str("%.6f"%prob)+" - "+str("%.4f"%mi)+"\n")
    return toPrint

#Le due frasi con probabilità più alta
def funMarkov(simTxt):
    sentences = nltk.sent_tokenize(simTxt) #Ottengo delle frasi dal corpus
    totWord = nltk.word_tokenize(simTxt) #Ottengo delle parole dal corpus inclusa la punteggiatura
    wordFreq = collections.Counter(totWord) #Uso counter per sapere la frequenza di ogni parola nel testo
    bigrFreq = collections.Counter(nltk.bigrams(totWord)) #Anche qui counter per ogni bigramma del corpus
    frase1="Nessuna frase trovata"
    frase2="Nessuna frase trovata"
    maxMk0 = 0 #Setto a zero una variabile che userò per determinare la frase con probabilità più alta Markov ordine 0
    maxMk1 = 0 #Setto a zero una variabile che userò per determinare la frase con probabilità più alta Markov ordine 1
    for sentence in sentences: #Per ogni frase del corpus
        words = nltk.word_tokenize(sentence) #Ottengo le parole di quella frase
        mult = 1 #Setto a 1 una variabile per la moltiplicazione p(a)*p(b)*p(...
        violazione = False
        if(len(words)>=6 and len(words)<=8): #Se la lunghezza della frase rientra nelle specifiche più lunga di 6 e minore di 8 token allora
            for i in range(0,len(words)): #Per ogni parola nella frase
                freqToken = wordFreq[words[i]]#Accedo al counter wordFreq per sapere la frequenza della parola words[i]
                if(freqToken > 2): #Se frequenza token maggiore di 2 come richiesto procedo altrimenti no
                    mult *= (freqToken/len(totWord))#Moltiplico la sua probabilità con quella precedente per ottenere la probabilità della frase.
                #La probabilità di una parola la ottengo facendo freq.Parola/numeroTotaleDiParole con wordFreq di words[i] chiedo al counter la frequenza della parola
                else:
                    violazione = True
            #Adesso mult contiene la probabilità della frase secondo Markov ordine 0
            if(violazione == False): #Se tutti i token hanno frequenza maggiore di 2 altrimenti non salvo la frase
                if(mult>maxMk0): #Se mult è maggiore di maxMk0 la nuova frase ha probabilità maggiore della precedente
                    maxMk0 = mult
                    frase1 = sentence
            bigrams = nltk.bigrams(words)#Ottengo i bigrammi dalle parole
            multBigr = 1
            violazione = False
            ck=0 #counter istanziato per sapere quando calcolare la probabilità della prima parola p(A)*p(B|A)*p(C|B)...
            for bigram in bigrams:#Per ogni bigramma nella lista di bigrammi
                freqToken = wordFreq[bigram[0]] #<ciao,hola> da questo bigramma prendo la frequenza di ciao usando il counter wordFreq
                if(freqToken > 2):
                    if(ck==0):
                        multBigr = (freqToken/len(totWord))#probabilità p(A) prima parola del primo bigramma
                        ck+=1
                    multBigr *= (bigrFreq[bigram]/freqToken)#frequenzaToken diviso numero totale di token così ottengo la probabilità di quel token poi lo moltiplico per la frequenza del bigramma che ho diviso con la frequenza del token A
                    #Quindi ho fatto: multBigr = multBigr * (p(A)*p(B|A)*p(C|B)*p(D|C)...) Dove p(A) è già nella variabile multBigr
                else:
                    violazione = True
            if(violazione == False):
                if(multBigr>maxMk1): #Se multBigr è maggiore di maxMk1 la nuova frase ha probabilità maggiore della precedente
                    maxMk1 = multBigr
                    frase2 = sentence
    return ("Markov ordine 0: "+frase1+" "+str(maxMk0)+"\nMarkov ordine 1: "+frase2+" "+str(maxMk1))

def coolPrint(lista):
    n=0
    for item in lista:
        n+=1
        if(n<=9):
            stringa = str(n)+")  "+str(item[0])
            totSpazi = (55 - len(stringa))#Per mettere degli spazi dopo la stringa in modo che le frequenze vengano allineate verticalmente
            for i in range(0,totSpazi):
                stringa += " "
            print(stringa+str(item[1]))
        else:
            stringa = str(n)+") "+str(item[0])#C'è una differenza di spazio tra il nove e il dieci 9)  parola 10) parola
            totSpazi = (55 - len(stringa))#Ritengo che in questo programma non ci siano stringhe più lunghe di 55 caratteri
            for i in range(0,totSpazi):
                stringa += " "
            print(stringa+str(item[1]))

def main(file1,file2):
    if(os.path.isfile(file1) and os.path.isfile(file2)):
        inputFile = [file1,file2]
        for varFile in inputFile:
            corpusText = loadFile(varFile)
            print("Per il file " + varFile +": ")
            print("I venti token più frequenti escludendo la punteggiatura: ")
            coolPrint(freqToken(corpusText))
            print("I venti sostantivi più frequenti: ")
            coolPrint(freqNomi(corpusText))
            print("I venti aggettivi più frequenti: ")
            coolPrint(freqAgge(corpusText))
            print("I venti bigrammi di token più frequenti escludendo punteggiatura, articoli e congiunzioni: ")
            coolPrint(freqBigrTK(corpusText))
            print("Le dieci Part Of Speech più frequenti: ")
            coolPrint(freqPOS(corpusText))
            print("I dieci bigrammi di POS più frequenti: ")
            coolPrint(freqPOSBigr(corpusText))
            print("I dieci trigrammi di POS più frequenti: ")
            coolPrint(freqPOSTigr(corpusText))
            print("In ordine decrescente i 20 bigrammi composti da aggettivo e sostantivo con frequenza per token > 2: ")
            print("---Ordinati per frequenza di bigramma---")
            print("[[Parola1,frequenzaP1,Parola2,frequenzaP2],FreqBigramma] - Probabilità - ForzaAssociativa")
            print(twentyBigramAS2(corpusText))
            print("Le due frasi con probabilità più alta: ")
            print(funMarkov(corpusText))
    else:
        print("Inserire percorsi validi per i due file corpus!")

main(sys.argv[1],sys.argv[2])
#Alessandro Mazzeo