import sys
from pipeline import run_pipeline
from pipeline.predict_on_single_document import predict_topics_on_a_single_document


def start():
    text = "Run Pipeline: Press 0\nPredict on a Text Document: Press 1\nEnter your Value: "
    user_input = int(input(text))
    if user_input==0:
        print("\nPipeline Started!\n")
        run_pipeline.run()
        print("Pipeline Successfully Executed!")
    elif user_input==1:
        text = "\nChoose the Text Model: ['lsa', 'lda', 'nmf', 'all']\nEnter your Model Choice: "
        model_name = input(text)
        predict_topics_on_a_single_document(model_name)
    else:
        sys.exit("Incorrect Value Entered! Please Re-Run!")


if __name__=="__main__":
    start()
    sys.exit()