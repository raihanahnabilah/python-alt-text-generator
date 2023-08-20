from flask import Flask, request
import os
from dotenv import load_dotenv
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Match flask name file to run
app = Flask(__name__)

# HTTP Route: Home route
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# HTTP Route: Testing route
@app.route("/test", methods=['POST'])
def test():
    image_path = request.get_json()
    return {
        "path": image_path['path']
    }

# HTTP Route: Generate post request
@app.route("/v1/generate", methods=['POST'])
def generate_text():
    # Get input data
    print("Getting input data")
    input_data = request.get_json()
    # Get the name of the file from the request
    image_path = input_data['path']    
    # Do the logic here
    print("Processing the logic...")
    image_loader = ImageCaptionLoader(path_images=[image_path])
    load_loader = image_loader.load()
    print("Indexing...")
    indexing = VectorstoreIndexCreator().from_loaders([image_loader])
    print("Returning result...")
    res = indexing.query('Describe this image in 1 sentence. This description will be used as an alt text in the website. Look for some guidance on how to write alt text, and make sure that the description you write adheres to that guidance.')
    return {
        "caption": res
    }

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    
# HTTP Route: Generate post request
@app.route("/v2/generate", methods=['POST'])
def generate_caption():
    # Get input data
    print("Getting input data")
    input_data = request.get_json()
    # Get the name of the file from the request
    image_path = input_data['path']    
    # Do the logic here
    print("Processing the logic...")
    #initialize the gent
    tools = [ImageCaptionTool(), ObjectDetectionTool()]
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    llm = ChatOpenAI(
        openai_api_key= OPENAI_API_KEY,
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        max_iterations=5,
        verbose=True,
        memory=conversational_memory,
        early_stopping_method='generate'
    )
    user_question = "generate an alt text for this image?"
    response = agent.run(f'{user_question}, this is the image path: {image_path}')
    return {
        "caption": response
    }

if __name__ == "__main__":
    app.run(port=5000, debug=True)