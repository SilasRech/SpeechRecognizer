import numpy as np
import pandas as pd
import re
from xml.dom import minidom
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle

polit = []

german1 = pd.read_csv('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\3-million-german-sentences\\3M.csv', header=None)

german2 = minidom.parse('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Polit\\AA.xml')
items2 = german2.getElementsByTagName('rohtext')
for elem in items2:
    polit.append(elem.childNodes[0].data)
#german3 = minidom.parse('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Polit\\BP.xml')
#items3 = german3.getElementsByTagName('rohtext')
#for elem in items3:
#    polit.append(elem.childNodes[0].data)
#german4 = minidom.parse('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Polit\\BTP.xml')
#items4 = german4.getElementsByTagName('rohtext')
#for elem in items4:
#    polit.append(elem.childNodes[0].data)
#german5 = minidom.parse('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Polit\\BR.xml')
#items5 = german5.getElementsByTagName('rohtext')
#for elem in items5:
#    polit.append(elem.childNodes[0].data)
#german6 = pd.read_json('C:\\Users\\silas\\Desktop\\Project PARTICU-Larry\\GemanTextDatabase\\Recipes\\recipes.json')

#german9 = german1.to_csv()
#german8 = german6.to_csv()
#final = german7 + german8 + german9

