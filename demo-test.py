from transformers import pipeline
import gradio as gr

ner_pipeline = pipeline("ner")

def ner(text):
    """
    Named Entity Recognition
    param: text: str
    return: dict
    ---
    >>> ner("hello, I'm John Smith")
    [{'entity': 'I-PER', 'score': 0.997681, 'index': 6, 'word': 'John', 'start': 11, 'end': 15}, {'entity': 'I-PER', 'score': 0.99473864, 'index': 7, 'word': 's', 'start': 16, 'end': 17}, {'entity': 'I-PER', 'score': 0.9801539, 'index': 8, 'word': '##mith', 'start': 17, 'end': 21}]
    """
    output = ner_pipeline(text)
    return {"text": text, "entities": output}    

if __name__ == "__main__":
    
    ner_pipeline = pipeline("ner")

    examples = [
        "Does Chicago have any stores and does Joe live here?",
    ]
    demo = gr.Interface(ner,
                gr.Textbox(placeholder="Enter sentence here..."), 
                gr.HighlightedText(),
                examples=examples)

    demo.launch(share=True)