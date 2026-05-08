from functools import partial
import sys
import json
from app.src.pipelines import BiencoderPipeline, LookupPipeline
from app.src.format import PassthroughFormatter
negation = False
ner_version = 2

method2pipeline = {
    'biencoder': partial(BiencoderPipeline, negation=negation, ner_version=ner_version),
    "lookup": LookupPipeline,
}

cdm2formatter = {
    'none': PassthroughFormatter
}

def main():
    """
    this is just for testing purposes.
    """
    random_footer = {
        "patient_id": "1",
        "admission_id": "2",
        "admission_date": '2026-02-25T10:43:12.173783',
        "admission_type": "emergency",
        "record_id": "3",
        "record_type": "discharge summary",
        "record_format": "txt"
    }

    texts = [
        "Este es un texto de ejemplo.\ncon un paciente procedente de almería aunque nacido en guadalupe, méxico, con mucha tos, mocos, fiebre y la varicela con meningitis.",
        "Otro texto con covid y paracetamol para probar.\ncon más  muchos más síntomas interesantes como edemas y negaciones como que 100% no tiene gripe A.",
        "El paciente reporta que no se ducha y por eso vomita sangre, pero realmente es porque tiene cancer de pulmón, hígado y piel"
    ]
    footers = [random_footer] * len(texts)

    method = 'biencoder'
    lang = "es"
    entities = ["disease"]
    

    print("Importing Models...")
    # entities = ["disease"]
    pipe = method2pipeline[method](lang=lang, entities=entities)

    cdm = 'none'
    formatter = cdm2formatter[cdm]()
    
    print("Performing inference...")
    annotations = pipe.predict(texts)
    results = [formatter.serialize(text, ann, footer) for text, ann, footer in zip(texts, annotations, footers)]
    for res in results:
        print(json.dumps(res, ensure_ascii=False, indent=4))
    

if __name__ == "__main__":
    sys.exit(main())