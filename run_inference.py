import argparse
from asyncio.log import logger
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle
import logging
import os
import traceback

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--images_path", type=str)
        args, _ = parser.parse_known_args()
        images_path = args.images_path
        image_files = os.listdir(images_path)
        images_path = [images_path+f'/{image_file}' for image_file in image_files]
#        inference_batch = prepare_batch_for_inference(images_path)
        inference_batch = {
  'image_path': ['images/432799-1.jpg','images/432799-0.jpg','images/432799-2.jpg'], 
  'bboxes': [
[[344,30,360,38],[362,30,394,38],[344,40,374,48],[57,30,110,39],[112,30,135,39],[138,30,144,39],[60,68,79,76],[84,68,144,76],[373,68,402,76],[413,30,476,38],[413,40,456,48],[423,68,471,76],[495,68,550,76],[87,83,132,91],[135,83,237,91],[239,83,294,91],[62,114,75,122],[60,177,65,185],[87,114,120,122],[123,114,142,122],[145,114,210,122],[87,124,140,132],[142,124,166,132],[169,124,205,132],[207,124,295,132],[87,135,132,143],[135,135,160,143],[162,135,184,143],[186,135,239,143],[241,135,273,143],[275,135,278,143],[280,135,313,143],[316,135,339,143],[378,114,383,122],[385,114,398,122],[438,114,461,122],[463,114,468,122],[517,114,540,122],[542,114,547,122],[84,177,108,185],[111,177,130,185],[132,177,230,185],[232,177,235,185],[238,177,250,185],[252,177,305,185],[307,177,351,185],[353,177,373,185],[375,177,396,185],[62,200,75,208],[87,200,120,208],[123,200,142,208],[145,200,210,208],[87,211,140,219],[142,211,166,219],[169,211,205,219],[207,211,295,219],[87,222,132,230],[135,222,160,230],[162,222,184,230],[186,222,239,230],[241,222,273,230],[275,222,298,230],[300,222,303,230],[305,222,338,230],[341,222,364,230],[378,200,383,208],[385,200,398,208],[438,200,461,208],[463,200,468,208],[517,200,540,208],[542,200,547,208],[62,252,75,260],[87,252,120,260],[123,252,142,260],[145,252,210,260],[87,263,140,271],[142,263,166,271],[169,263,205,271],[207,263,295,271],[87,274,132,282],[135,274,237,282],[239,274,294,282],[378,252,383,260],[385,252,398,260],[438,252,461,260],[463,252,468,260],[517,252,540,260],[542,252,547,260],[28,298,32,302],[60,316,65,324],[84,316,108,324],[111,316,130,324],[132,316,225,324],[227,316,230,324],[233,316,245,324],[247,316,300,324],[302,316,346,324],[348,316,368,324],[370,316,391,324],[62,339,75,347],[87,339,120,347],[123,339,142,347],[145,339,210,347],[87,350,140,358],[142,350,166,358],[169,350,205,358],[207,350,295,358],[87,360,132,368],[135,360,160,368],[162,360,184,368],[186,360,239,368],[241,360,273,368],[275,360,278,368],[280,360,313,368],[378,339,383,347],[385,339,398,347],[438,339,461,347],[463,339,468,347],[517,339,540,347],[542,339,547,347],[62,391,75,399],[87,391,120,399],[123,391,142,399],[145,391,210,399],[87,402,140,410],[142,402,166,410],[169,402,205,410],[207,402,295,410],[87,413,132,421],[135,413,237,421],[239,413,294,421],[378,391,383,399],[385,391,398,399],[438,391,461,399],[463,391,468,399],[517,391,540,399],[542,391,547,399],[62,443,75,451],[87,443,120,451],[123,443,142,451],[145,443,210,451],[87,454,140,462],[142,454,166,462],[169,454,205,462],[207,454,295,462],[87,465,132,473],[135,465,160,473],[162,465,184,473],[186,465,239,473],[241,465,273,473],[275,465,278,473],[280,465,313,473],[378,443,383,451],[385,443,398,451],[438,443,461,451],[463,443,468,451],[517,443,540,451],[542,443,547,451],[62,495,75,503],[87,495,120,503],[123,495,142,503],[145,495,210,503],[87,506,140,514],[142,506,166,514],[169,506,205,514],[207,506,295,514],[87,517,132,525],[135,517,237,525],[239,517,294,525],[378,495,383,503],[385,495,398,503],[438,495,461,503],[463,495,468,503],[517,495,540,503],[542,495,547,503],[62,548,75,556],[87,548,120,556],[123,548,142,556],[145,548,210,556],[87,558,140,566],[142,558,166,566],[169,558,205,566],[207,558,295,566],[87,569,132,577],[135,569,160,577],[162,569,184,577],[186,569,239,577],[241,569,273,577],[275,569,278,577],[280,569,313,577],[316,569,339,577],[378,548,383,556],[385,548,398,556],[438,548,461,556],[463,548,468,556],[517,548,540,556],[542,548,547,556],[62,600,75,608],[87,600,120,608],[123,600,142,608],[145,600,210,608],[87,611,140,619],[142,611,166,619],[169,611,205,619],[207,611,295,619],[87,622,132,630],[135,622,237,630],[239,622,294,630],[378,600,383,608],[385,600,398,608],[438,600,461,608],[463,600,468,608],[517,600,540,608],[542,600,547,608],[438,686,461,694],[463,686,468,694],[517,686,540,694],[542,686,547,694],[28,596,32,600],[60,664,65,672],[62,686,75,694],[84,664,108,672],[111,664,130,672],[132,664,225,672],[227,664,230,672],[233,664,245,672],[247,664,300,672],[302,664,346,672],[348,664,368,672],[370,664,391,672],[87,686,120,694],[123,686,142,694],[145,686,210,694],[87,697,140,705],[142,697,166,705],[169,697,205,705],[207,697,295,705],[378,686,383,694],[385,686,398,694],[446,714,481,722],[512,714,540,722],[542,714,547,722],[278,744,299,752],[301,744,306,752],[309,744,324,752],[326,744,331,752],[57,763,84,769],[86,763,123,769],[124,763,135,769],[57,771,129,777],[57,780,96,786],[98,780,104,786],[106,780,125,786],[127,780,158,786],[57,788,71,794],[72,788,84,794],[86,788,98,794],[100,788,131,794],[57,796,79,802],[80,796,166,802],[57,805,73,811],[75,805,166,811],[213,763,265,769],[267,763,289,769],[291,763,316,769],[213,771,283,777],[286,771,291,777],[293,771,330,777],[332,771,365,777],[213,780,228,786],[230,780,249,786],[213,788,267,794],[270,788,315,794],[213,796,247,802],[250,796,297,802],[412,763,442,769],[443,763,459,769],[461,763,471,769],[412,771,444,777],[446,771,462,777],[463,771,479,777],[481,771,489,777],[412,780,427,786],[429,780,441,786],[442,780,454,786],[456,780,464,786],[412,788,430,794],[432,788,450,794],[451,788,467,794],[469,788,485,794],[487,788,503,794],[504,788,520,794],[522,788,530,794],[412,796,426,802],[427,796,477,802]],
[[344,100,369,108],[344,112,463,129],[344,147,373,155],[344,158,407,166],[344,169,360,177],[362,169,394,177],[57,150,79,156],[80,150,108,156],[110,150,119,156],[121,150,177,156],[180,150,182,156],[186,150,220,156],[221,150,224,156],[228,150,230,156],[233,150,250,156],[252,150,278,156],[57,164,95,173],[98,164,134,173],[57,176,113,185],[116,176,122,185],[57,188,85,197],[87,188,131,197],[417,147,462,155],[417,158,460,166],[417,169,447,177],[344,192,413,200],[417,192,452,200],[454,192,487,200],[344,202,372,210],[417,202,544,210],[57,256,129,264],[419,256,471,264],[60,285,79,293],[84,285,144,293],[373,285,402,293],[423,285,471,293],[495,285,550,293],[28,298,32,302],[60,305,65,313],[28,596,32,600],[84,305,108,313],[111,305,130,313],[132,305,225,313],[227,305,230,313],[233,305,245,313],[247,305,300,313],[302,305,346,313],[348,305,368,313],[370,305,391,313],[62,328,75,336],[87,328,120,336],[123,328,142,336],[145,328,210,336],[87,339,140,347],[142,339,166,347],[169,339,205,347],[207,339,295,347],[87,349,132,357],[135,349,160,357],[162,349,184,357],[186,349,239,357],[241,349,273,357],[275,349,298,357],[300,349,303,357],[305,349,338,357],[378,328,383,336],[385,328,398,336],[438,328,461,336],[463,328,468,336],[517,328,540,336],[542,328,547,336],[62,380,75,388],[87,380,120,388],[123,380,142,388],[145,380,210,388],[87,391,140,399],[142,391,166,399],[169,391,205,399],[207,391,295,399],[87,402,132,410],[135,402,237,410],[239,402,294,410],[378,380,383,388],[385,380,398,388],[438,380,461,388],[463,380,468,388],[517,380,540,388],[542,380,547,388],[62,432,75,440],[87,432,120,440],[123,432,142,440],[145,432,210,440],[87,443,140,451],[142,443,166,451],[169,443,205,451],[207,443,295,451],[87,454,132,462],[135,454,160,462],[162,454,184,462],[186,454,239,462],[241,454,273,462],[275,454,278,462],[280,454,313,462],[378,432,383,440],[385,432,398,440],[438,432,461,440],[463,432,468,440],[517,432,540,440],[542,432,547,440],[62,485,75,493],[87,485,120,493],[123,485,142,493],[145,485,210,493],[87,495,140,503],[142,495,166,503],[169,495,205,503],[207,495,295,503],[87,506,132,514],[135,506,237,514],[239,506,294,514],[378,485,383,493],[385,485,398,493],[438,485,461,493],[463,485,468,493],[517,485,540,493],[542,485,547,493],[62,537,75,545],[87,537,120,545],[123,537,142,545],[145,537,210,545],[87,548,140,556],[142,548,166,556],[169,548,205,556],[207,548,295,556],[87,558,132,566],[135,558,160,566],[162,558,184,566],[186,558,239,566],[241,558,273,566],[275,558,278,566],[280,558,313,566],[378,537,383,545],[385,537,398,545],[438,537,461,545],[463,537,468,545],[517,537,540,545],[542,537,547,545],[62,589,75,597],[87,589,120,597],[123,589,142,597],[145,589,210,597],[87,600,140,608],[142,600,166,608],[169,600,205,608],[207,600,295,608],[87,611,132,619],[135,611,237,619],[239,611,294,619],[378,589,383,597],[385,589,398,597],[438,589,461,597],[463,589,468,597],[517,589,540,597],[542,589,547,597],[62,641,75,649],[87,641,120,649],[123,641,142,649],[145,641,210,649],[87,652,140,660],[142,652,166,660],[169,652,205,660],[207,652,295,660],[87,663,132,671],[135,663,160,671],[162,663,184,671],[186,663,239,671],[241,663,273,671],[275,663,278,671],[280,663,313,671],[378,641,383,649],[385,641,398,649],[438,641,461,649],[463,641,468,649],[517,641,540,649],[542,641,547,649],[62,694,75,702],[87,694,120,702],[123,694,142,702],[145,694,210,702],[87,704,140,712],[142,704,166,712],[169,704,205,712],[207,704,295,712],[378,694,383,702],[385,694,398,702],[438,694,461,702],[463,694,468,702],[517,694,540,702],[542,694,547,702],[446,721,481,729],[512,721,540,729],[542,721,547,729],[278,744,299,752],[301,744,306,752],[309,744,324,752],[326,744,331,752],[57,763,84,769],[86,763,123,769],[124,763,135,769],[57,771,129,777],[57,780,96,786],[98,780,104,786],[106,780,125,786],[127,780,158,786],[57,788,71,794],[72,788,84,794],[86,788,98,794],[100,788,131,794],[57,796,79,802],[80,796,166,802],[57,805,73,811],[75,805,166,811],[213,763,265,769],[267,763,289,769],[291,763,316,769],[213,771,283,777],[286,771,291,777],[293,771,330,777],[332,771,365,777],[213,780,228,786],[230,780,249,786],[213,788,267,794],[270,788,315,794],[213,796,247,802],[250,796,297,802],[412,763,442,769],[443,763,459,769],[461,763,471,769],[412,771,444,777],[446,771,462,777],[463,771,479,777],[481,771,489,777],[412,780,427,786],[429,780,441,786],[442,780,454,786],[456,780,464,786],[412,788,430,794],[432,788,450,794],[451,788,467,794],[469,788,485,794],[487,788,503,794],[504,788,520,794],[522,788,530,794],[412,796,426,802],[427,796,477,802]],
[[344,30,360,38],[362,30,394,38],[344,40,374,48],[57,30,110,39],[112,30,135,39],[138,30,144,39],[60,68,79,76],[84,68,144,76],[373,68,402,76],[413,30,476,38],[413,40,456,48],[423,68,471,76],[495,68,550,76],[87,83,132,91],[135,83,160,91],[162,83,184,91],[186,83,239,91],[241,83,273,91],[275,83,298,91],[300,83,303,91],[305,83,338,91],[28,298,32,302],[62,114,75,122],[87,114,120,122],[123,114,142,122],[145,114,210,122],[87,124,140,132],[142,124,166,132],[169,124,205,132],[207,124,295,132],[87,135,132,143],[135,135,237,143],[239,135,294,143],[378,114,383,122],[385,114,398,122],[438,114,461,122],[463,114,468,122],[517,114,540,122],[542,114,547,122],[62,166,75,174],[87,166,120,174],[123,166,142,174],[145,166,210,174],[87,177,140,185],[142,177,166,185],[169,177,205,185],[207,177,295,185],[87,187,132,195],[135,187,160,195],[162,187,184,195],[186,187,239,195],[241,187,273,195],[275,187,278,195],[280,187,313,195],[378,166,383,174],[385,166,398,174],[438,166,461,174],[463,166,468,174],[517,166,540,174],[542,166,547,174],[62,218,75,226],[87,218,120,226],[123,218,142,226],[145,218,210,226],[87,229,140,237],[142,229,166,237],[169,229,205,237],[207,229,295,237],[87,240,132,248],[135,240,237,248],[239,240,294,248],[378,218,383,226],[385,218,398,226],[438,218,461,226],[463,218,468,226],[517,218,540,226],[542,218,547,226],[62,270,75,278],[87,270,120,278],[123,270,142,278],[145,270,210,278],[87,281,140,289],[142,281,166,289],[169,281,205,289],[207,281,295,289],[87,292,132,300],[135,292,160,300],[162,292,184,300],[186,292,239,300],[241,292,273,300],[275,292,278,300],[280,292,313,300],[316,292,339,300],[378,270,383,278],[385,270,398,278],[438,270,461,278],[463,270,468,278],[517,270,540,278],[542,270,547,278],[248,340,305,348],[307,340,334,348],[512,340,540,348],[542,340,547,348],[216,357,290,367],[293,357,333,367],[504,357,538,367],[541,357,547,367],[28,596,32,600],[278,744,299,752],[301,744,306,752],[309,744,324,752],[326,744,331,752],[57,763,84,769],[86,763,123,769],[124,763,135,769],[57,771,129,777],[57,780,96,786],[98,780,104,786],[106,780,125,786],[127,780,158,786],[57,788,71,794],[72,788,84,794],[86,788,98,794],[100,788,131,794],[57,796,79,802],[80,796,166,802],[57,805,73,811],[75,805,166,811],[213,763,265,769],[267,763,289,769],[291,763,316,769],[213,771,283,777],[286,771,291,777],[293,771,330,777],[332,771,365,777],[213,780,228,786],[230,780,249,786],[213,788,267,794],[270,788,315,794],[213,796,247,802],[250,796,297,802],[412,763,442,769],[443,763,459,769],[461,763,471,769],[412,771,444,777],[446,771,462,777],[463,771,479,777],[481,771,489,777],[412,780,427,786],[429,780,441,786],[442,780,454,786],[456,780,464,786],[412,788,430,794],[432,788,450,794],[451,788,467,794],[469,788,485,794],[487,788,503,794],[504,788,520,794],[522,788,530,794],[412,796,426,802],[427,796,477,802]]
],  
  'words': [
["""Ihre""","""Lief-Nr.:""","""196899""","""Fortsetzung""","""Seite""","""2""","""Pos.""","""Beschreibung""","""Menge""","""Rechnungs-Nr.:""","""ER210166""","""Einzelpreis""","""Gesamtpreis""","""04.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1.9""","""2""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""05.03.2021""","""[14,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""20:45""","""1""","""Stk""","""14,00""","""€""","""14,00""","""€""","""Reise""","""vom""","""09.03.2021-010.03.2021""","""-""","""PZ""","""Ludwigsfeld""","""(Deutsche""","""Post""","""DHL)""","""2.1""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""09.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""","""07:45""",""",""","""Ankunft:""","""20:45""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""2.2""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""09.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""__""","""3""","""Reise""","""vom""","""23.03.2021-25.03.2021""","""-""","""PZ""","""Ludwigsfeld""","""(Deutsche""","""Post""","""DHL)""","""3.1""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""23.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""3.2""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""23.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""3.3""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""24.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""3.4""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""24.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""3.5""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""25.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""21:00""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""3.6""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""25.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""28,00""","""€""","""28,00""","""€""","""__""","""4""","""4.1""","""Reise""","""vom""","""29.03.2021-31.03.2021""","""-""","""PZ""","""Ludwigsfeld""","""(Deutsche""","""Post""","""DHL)""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""1""","""Stk""","""Übertrag""","""426,00""","""€""","""Seite""","""2""","""von""","""3""","""Simatos""","""Consulting""","""UG""","""(haftungsbeschränkt)""","""Ackergrenze""","""8,""","""44287""","""Dortmund""","""Tel.:""","""+49""","""231""","""22130006""","""E-Mail:""","""info@simatos-consulting.de""","""URL:""","""https://simatos-consulting.de/""","""Geschäftsführer:""","""Ermias""","""Simatos""","""Handelsregistereintrag""","""in""","""Amtsgericht""","""Dortmund,""","""HRB""","""28796""","""Umsatzsteuer-ID:""","""DE309903332""","""Steuer-Nr.:""","""315/5769/1074""","""Deutsche""","""Bank""","""AG""","""Konto-Nr.:""","""0200""","""0529""","""00""","""BLZ:""","""440""","""700""","""50""","""IBAN:""","""DE94""","""4407""","""0050""","""0200""","""0529""","""00""","""BIC:""","""DEUTDEDB440"""],
["""Kopie""","""EK-Rechnung""","""Datum:""","""Rechnungs-Nr.:""","""Ihre""","""Lief-Nr.:""","""Simatos""","""Consulting""","""UG""","""(haftungsbeschränkt)""","""•""","""Ackergrenze""","""8""","""•""","""44287""","""Dortmund""","""Nicolaos""","""Simatos""","""Ackergrenze""","""8""","""44287""","""Dortmund""","""31.03.2021""","""ER210166""","""196899""","""Ansprechpartner:""","""Nicolaos""","""Simatos""","""E-Mail:""","""simatos@simatos-consulting.de""","""Referenz-Zeichen""","""Referenz-Nr.""","""Pos.""","""Beschreibung""","""Menge""","""Einzelpreis""","""Gesamtpreis""","""__""","""1""","""__""","""Reise""","""vom""","""01.03.2021-05.03.2021""","""-""","""PZ""","""Ludwigsfeld""","""(Deutsche""","""Post""","""DHL)""","""1.1""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""01.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""","""04:45""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""1.2""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""01.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""1.3""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""02.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""1.4""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""02.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""1.5""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""03.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""1.6""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""03.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""1.7""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""04.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""1.8""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""Übertrag""","""192,00""","""€""","""Seite""","""1""","""von""","""3""","""Simatos""","""Consulting""","""UG""","""(haftungsbeschränkt)""","""Ackergrenze""","""8,""","""44287""","""Dortmund""","""Tel.:""","""+49""","""231""","""22130006""","""E-Mail:""","""info@simatos-consulting.de""","""URL:""","""https://simatos-consulting.de/""","""Geschäftsführer:""","""Ermias""","""Simatos""","""Handelsregistereintrag""","""in""","""Amtsgericht""","""Dortmund,""","""HRB""","""28796""","""Umsatzsteuer-ID:""","""DE309903332""","""Steuer-Nr.:""","""315/5769/1074""","""Deutsche""","""Bank""","""AG""","""Konto-Nr.:""","""0200""","""0529""","""00""","""BLZ:""","""440""","""700""","""50""","""IBAN:""","""DE94""","""4407""","""0050""","""0200""","""0529""","""00""","""BIC:""","""DEUTDEDB440"""],
  ["""Ihre""","""Lief-Nr.:""","""196899""","""Fortsetzung""","""Seite""","""3""","""Pos.""","""Beschreibung""","""Menge""","""Rechnungs-Nr.:""","""ER210166""","""Einzelpreis""","""Gesamtpreis""","""29.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""","""06:30""",""",""","""Ankunft:""","""__""","""4.2""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""29.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""4.3""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""30.03.2021""","""[28,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""1""","""Stk""","""28,00""","""€""","""28,00""","""€""","""4.4""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""30.03.2021""","""Übernachtungspauschale""","""[Deutschland]""","""1""","""Stk""","""20,00""","""€""","""20,00""","""€""","""4.5""","""FAHRT-""","""UND""","""REISEKOSTEN""","""Angefallenen""","""Fahrt-""","""und/oder""","""Übernachtungskosten""","""31.03.2021""","""[14,00""","""EUR]""","""Deutschland,""","""Abfahrt:""",""",""","""Ankunft:""","""21:00""","""1""","""Stk""","""14,00""","""€""","""14,00""","""€""","""Gesamtbetrag""","""[Netto]""","""508,00""","""€""","""Gesamtbetrag""","""[Brutto]""","""508,00""","""€""","""__""","""Seite""","""3""","""von""","""3""","""Simatos""","""Consulting""","""UG""","""(haftungsbeschränkt)""","""Ackergrenze""","""8,""","""44287""","""Dortmund""","""Tel.:""","""+49""","""231""","""22130006""","""E-Mail:""","""info@simatos-consulting.de""","""URL:""","""https://simatos-consulting.de/""","""Geschäftsführer:""","""Ermias""","""Simatos""","""Handelsregistereintrag""","""in""","""Amtsgericht""","""Dortmund,""","""HRB""","""28796""","""Umsatzsteuer-ID:""","""DE309903332""","""Steuer-Nr.:""","""315/5769/1074""","""Deutsche""","""Bank""","""AG""","""Konto-Nr.:""","""0200""","""0529""","""00""","""BLZ:""","""440""","""700""","""50""","""IBAN:""","""DE94""","""4407""","""0050""","""0200""","""0529""","""00""","""BIC:""","""DEUTDEDB440"""]
]
}
        context = {"model_dir": args.model_path}
#        print(inference_batch)
        handle(inference_batch,context)
    except Exception as err:
        traceback.print_exc()
        os.makedirs('log', exist_ok=True)
        logging.basicConfig(filename='log/error_output.log', level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
        logger = logging.getLogger(__name__)
        logger.error(err)


