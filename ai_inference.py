import sys
import xml.etree.ElementTree as ET
import json
from Layoutlmv3_inference.inference_handler import handle

def parse_bbox_and_words_to_json(xml_file):
    # XML-Datei parsen
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Listen für Bounding Boxen und Inhalte
    image_path = []
    bboxes = []
    words = []
    i=0

    # Iteriere durch alle 'page'-Elemente
    for page in root.iter('page'):
        page_bboxes = []
        page_words = []

        for word in page.iter('word'):
            if 'xMin' in word.attrib and 'yMin' in word.attrib and 'xMax' in word.attrib and 'yMax' in word.attrib:
                # Bounding Box extrahieren
                bbox = [
                    round(float(word.attrib['xMin'])),
                    round(float(word.attrib['yMin'])),
                    round(float(word.attrib['xMax'])),
                    round(float(word.attrib['yMax']))
                ]
                page_bboxes.append(bbox)

                # Textinhalt extrahieren (falls vorhanden)
                word_text = word.text.strip() if word.text else ""
                page_words.append(word_text)

        if len(root) == 1:
            image_path.append(xml_file.replace(".xml",".jpg"))
        else:
            image_path.append(xml_file.replace(".xml","-")+str(i)+".jpg")
        bboxes.append(page_bboxes)
        words.append(page_words)
        i+=1
    
    return {
        "image_path" : image_path,
        "bboxes"     : bboxes,
        "words"      : words
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("missing argument")
    
    if len(sys.argv) == 2:
        inference_batch = {}

        xml_filepath=sys.argv[1]
        inference_batch = parse_bbox_and_words_to_json(xml_filepath)
        inference_result = handle(inference_batch,{"model_dir": "/app/oracle/ocrai/pre_trained_layoutlmv3_pdftotext"})
        print(inference_result)
#        print(xml_filepath.replace('.xml','.json'))
        with open(xml_filepath.replace('.xml','.json'), 'w') as inf_out:
            inf_out.write(inference_result)
