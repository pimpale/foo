#Author: Govind Pimpale
import io
import cv2
import pandas
import pytesseract

img = cv2.imread('imggoeshere.png')

out_tsv = pytesseract.image_to_data(img) # pytesseract returns tsv file
out_dataframe = pandas.read_csv(io.StringIO(out_tsv), sep='\t', header=0)# convert file to pandas dataframe

for index, row in out_dataframe.iterrows():
    # if the row has confidence
    if row['conf'] > 0:
        x1 = row['left']
        y1 = row['top']
        w = row['width']
        h = row['height']
        print('word area:', w*h)
        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (255, 0, 0), 2)


cv2.imshow('win', img)
k = cv2.waitKey(0) # 0==wait forever
