# python-alt-text-generator

## Objective
This was a task from my part time job as a Student Associate for Student Affairs Office at Yale-NUS College where I have to add alt text to several pictures on the college’s website. 

To utilize today's incredibly advanced technology in AI, I decided to use AI to generate the alt text caption for the photos. 

## Running the program

1. Create an OPEN AI account and get the `OPENAI_API_KEY`. Change the key on the .env file inside backend folder. 

2. Create your FLASK_APP environment variable by running this on terminal:

```
export FLASK_APP=main.py
```

3. Run the flask app
```
flask run --host=0.0.0.0
```

4. Make a POST request to `/v2/generate`. You can also do `/v1/generate`, but `/v2/generate` is more reliable (see Acknowledgement section). This should be the input body:
```
{
    "path": /path/to/photo or URL
}
```

You're good to go!

## Future Plan
Add a frontend so that users can easily upload a picture and generate the caption without making a POST request manually via POSTMAN or curl.

## Acknowledgement

The custom tools for processing the image was largely adopted from Plaban Nayak's <a href="https://nayakpplaban.medium.com/ask-questions-to-your-images-using-langchain-and-python-1aeb30f38751">Ask questions to your images using LangChain and Python</a>.